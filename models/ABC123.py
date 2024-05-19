import cv2
import numpy as np
import torch
import torchmetrics
from einops import rearrange
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import  StepLR
from torchvision import transforms as T

from models.backbone_vit import ViTExtractor
from models.counting_head import CountingHeadMultiLinearLayersSeperateSum
from models.matcher import HungarianMatcher
from data import denormalize, get_dataloader


class ABC123(LightningModule):
    def __init__(self, CFG):
        super().__init__()
        self.CFG = CFG

        self.train_MAE = torchmetrics.MeanAbsoluteError()
        self.train_MSE = torchmetrics.MeanSquaredError(squared=False)
        
        self.test_MAE = torchmetrics.MeanAbsoluteError()
        self.test_MSE = torchmetrics.MeanSquaredError(squared=False)
        self.test_SRE = torchmetrics.MeanMetric()
        self.test_NAE = torchmetrics.MeanAbsoluteError()

        self.get_model()
        self.matcher = HungarianMatcher(
            cost=self.CFG["matcher_cost_p_norm"],
            cost_power=self.CFG["matcher_cost_power"],
            normalize=self.CFG["normalize_matching"] and self.CFG["matching_type"] == "density",
        )
            

    def forward(self, feats_counting):     
        y_count, intermediate_image = self.counting_head(feats_counting)
        return y_count, intermediate_image

    def step(self, batch, tag, batch_idx):
        input, _, _, gt_density, gt_cnt, _ = batch

        if len(gt_cnt.shape) == 1:
            gt_cnt = gt_cnt.unsqueeze(1)

        feats = self.counting_backbone(input)


        y_count, y_density_map = self(feats)

        y_density_map, y_count_loss, gt_cnt_loss, = self.matching(gt_density, gt_cnt, y_count, y_density_map)

        y_density_loss = y_density_map[:, : gt_density.shape[1]]
        y_density_loss = rearrange(
            y_density_loss.clone(), "b n h w -> (b n) (h w)"
        )
        gt_density_loss = rearrange(
            gt_density.clone(), "b n h w -> (b n) (h w)"
        )

        # take all the predictions matched to with nonzero gt counts
        gt_cnt_nonz = torch.nonzero(gt_cnt_loss.flatten()).squeeze()
        y_count_loss = y_count_loss[:, : gt_cnt_loss.shape[1]]

        gt_cnt_loss = gt_cnt_loss.flatten()[gt_cnt_nonz]
        y_count_loss = y_count_loss.flatten()[gt_cnt_nonz]
        gt_density_loss = gt_density_loss[gt_cnt_nonz, :]
        y_density_loss = y_density_loss[gt_cnt_nonz, :]

        if self.CFG['gtd_scale'] != 1:
            gt_density_loss *= self.CFG['gtd_scale']
            y_count_loss = y_count_loss.clone() / self.CFG['gtd_scale']

        if self.CFG["counting_loss"] == "MAE":
            loss = torch.abs(y_count_loss - gt_cnt_loss).mean()
        elif self.CFG["counting_loss"] == "pixelwise_mae":               
            loss = (torch.abs(y_density_loss - gt_density_loss)).mean()
        elif self.CFG["counting_loss"] == "pixelwise_mse":
            loss = (
                torch.abs(y_density_loss - gt_density_loss) ** 2
            ).mean()

        else:
            print("Counting loss not defined")


        self.log(
            f"{tag}/loss",
            torch.mean(loss),
            on_epoch=True,
            sync_dist=True,
        )

        y_count_loss_flattened = torch.flatten(y_count_loss)
        gt_count_flattened = torch.flatten(gt_cnt_loss)
        if tag == "train":
            self.train_MAE.update(y_count_loss_flattened, gt_count_flattened)
            self.train_MSE.update(y_count_loss_flattened, gt_count_flattened)
            
        else:
            self.test_MAE.update(y_count_loss_flattened, gt_count_flattened)
            self.test_MSE.update(y_count_loss_flattened, gt_count_flattened)
            y_count_loss_flattened_normed = (
                y_count_loss_flattened / gt_count_flattened
            )  
            y_count_loss_flattened_square_normed = (
                y_count_loss_flattened - gt_count_flattened
            ) ** 2 / gt_count_flattened
            self.test_NAE.update(
                y_count_loss_flattened_normed,
                torch.ones_like(y_count_loss_flattened_normed),
            )
            self.test_SRE.update(y_count_loss_flattened_square_normed)
        return loss

    def matching(self, gt_density, gt_cnt, y_count, y_density_map):
        y_density = rearrange(
                y_density_map.clone(), "b n h w -> b n (h w)"
            )
        gtd = rearrange(gt_density.clone(), "b n h w -> b n (h w)")

        if self.CFG["matching_type"] == "count":
            matching_inds = self.matcher(
                    torch.sum((y_density/self.CFG['gtd_scale']).clone(),2).unsqueeze(2),
                    torch.sum(gtd.clone(),2).unsqueeze(2),
                )

        elif self.CFG["matching_type"] == "density":
            matching_inds = self.matcher(
                    y_density.clone(),
                    gtd.clone(),
                )
        y_count_permed = torch.zeros(
                (y_count.shape[0], y_count.shape[1]), device=y_count.device
            )

        y_density_permed = torch.zeros_like(y_density)
        gt_count_permed = torch.zeros_like(gt_cnt)
        gt_density_permed = torch.zeros_like(gtd)
        for (
                b_i,
                (m_inds_b, y_b, gt_c_b, y_den_b, gt_den_b),
            )        in enumerate(
                zip(
                    matching_inds,
                    y_count,
                    gt_cnt,
                    y_density,
                    gtd,
                )
            ):
            y_inds = m_inds_b[0]
            gt_inds = m_inds_b[1]

            gt_cnt_nonz = torch.nonzero(gt_c_b[gt_inds]).squeeze()
                
            y_b_p = y_b[y_inds]
            if gt_c_b.shape[0] == 1:
                    gt_b_p = gt_c_b
            else:
                gt_b_p = gt_c_b[gt_inds]

            y_den_b_p = y_den_b[y_inds]
            gt_den_b_p = gt_den_b[gt_inds]
            y_count_len_min = min(y_count_permed.shape[1], y_b_p.shape[0])
            y_count_permed[b_i, :y_count_len_min] = y_b_p[:y_count_len_min]
            y_density_permed[b_i, :y_count_len_min] = y_den_b_p[
                    :y_count_len_min
                ]

            y_count_loss = y_count_permed
            gt_cnt_loss = gt_count_permed

            gt_count_permed[b_i, : gt_b_p.shape[0]] = gt_b_p
            gt_density_permed[b_i, : gt_b_p.shape[0]] = gt_den_b_p

            y_density_map = rearrange(
                    y_density_permed.clone(),
                    "b n (h w) -> b n h w",
                    h=y_density_map.shape[2],
                )
            gt_density = rearrange(
                    gt_density_permed.clone(),
                    "b n (h w) -> b n h w",
                    h=y_density_map.shape[2],
                )


        return y_density_map, y_count_loss, gt_cnt_loss

    
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, "train", batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, self.CFG["test_split"], batch_idx)

    def training_epoch_end(self, _outputs):
        tr_mae = self.train_MAE.compute()
        tr_rmse = self.train_MSE.compute()

        if tr_mae.get_device() == 0:
            print(f"   train over all GPUS, MAE: {tr_mae:.2f}, RMSE: {tr_rmse:.2f}")

        self.logger.experiment.add_scalar(
            "train/MAE",
            tr_mae,
            global_step=self.current_epoch,
        )
        self.logger.experiment.add_scalar(
            "train/RMSE",
            tr_rmse,
            global_step=self.current_epoch,
        )

        self.train_MAE.reset()
        self.train_MSE.reset()


    def validation_epoch_end(self, _outputs):
        test_mae = self.test_MAE.compute()
        test_rmse = self.test_MSE.compute()
        test_sre = self.test_SRE.compute()
        test_nae = self.test_NAE.compute()

        if test_mae.get_device() == 0:
            print(
                f"   {self.CFG['test_split']} over all GPUS, MAE,RMSE,NAE,SRE {test_mae:.2f} & {test_rmse:.2f} & {test_nae:.2f} & {test_sre**0.5:.2f} "
            )

        self.log(self.CFG["test_split"] + "_MAE", test_mae)
       
        self.logger.experiment.add_scalar(
            self.CFG["test_split"] + "/MAE",
            test_mae,
            global_step=self.current_epoch,
        )

        self.logger.experiment.add_scalar(
            self.CFG["test_split"] + "/RMSE",
            test_rmse,
            global_step=self.current_epoch,
        )
        self.logger.experiment.add_scalar(
            self.CFG["test_split"] + "/NAE",
            test_nae,
            global_step=self.current_epoch,
        )

        self.logger.experiment.add_scalar(
            self.CFG["test_split"] + "/SRE",
            test_sre**0.5,
            global_step=self.current_epoch,
        )
        self.test_MAE.reset()
        self.test_MSE.reset()
        self.test_SRE.reset()
        self.test_NAE.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=float(self.CFG["learning_rate"]),
            weight_decay=self.CFG["weight_decay"],
        )

        if self.CFG["scheduler"] == "StepLR":
            scheduler = StepLR(
                optimizer,
                step_size=self.CFG["scheduler_steps"],
                gamma=self.CFG["scheduler_gamma"],
            )
            return ({"optimizer": optimizer, "lr_scheduler": scheduler},)

        return optimizer

    def get_model(self):
        number_to_unfreeze = self.CFG["counting_backbone_unfreeze_layers"]


        feature_dim = 384
        vit_config = {
            "base_model": "vit_small_patch8_224_dino",
            "facet": "token",
            "layer": int(11),
            "bin": False,
            "stride": int(8),
            "pretrained": True,
        }

        if not self.CFG["counting_backbone_pretrained"]:
            vit_config["pretrained"] = False

            print("Loading random values into backbone")
        else:
            print("Loading pretrained dino values into backbone")

        self.counting_backbone = ViTExtractor(vit_config)

        model_dict = list(
            {
                int(k.split(".")[1])
                for k, _ in self.counting_backbone.named_parameters()
                if k.startswith("blocks.")
            }
        )

        if number_to_unfreeze == -1:
            print("Not freezing any of the counting backbone")
        else:
            if number_to_unfreeze != 0:
                dont_freeze = model_dict[-number_to_unfreeze:]
                dont_freeze = ["blocks." + str(k) + "." for k in dont_freeze]
            else:
                dont_freeze = []

            dont_freeze_list = []
            for name, param in self.counting_backbone.named_parameters():
                in_dont_freeze = False
                for df in dont_freeze:
                    if name.startswith(df):
                        in_dont_freeze = True
                if not in_dont_freeze:
                    param.requires_grad = False
                else:
                    dont_freeze_list.append(name)

            dont_freeze_list = set(
                [v.split(".")[0] + v.split(".")[1] for v in dont_freeze_list]
            )
            print("Not freezing from counting backbone", dont_freeze_list)

        count_strategy = self.CFG["counting_head"].split("_")
    
        linear_project_then_sum = False
        channels = count_strategy
        self.ch = count_strategy[-1]
        self.counting_head = CountingHeadMultiLinearLayersSeperateSum(
            feature_dim,
            channels,
            28,
            self.CFG["upsample_padding_mode"],
            linear_project_then_sum,
        )

        if self.CFG["resume_path"] != "":
            self.load_pretrained_model()
        else:
            print("RANDOMLY INTIALSIING AS NO CHECKPOINT SPECIFIED")

    def load_pretrained_model(self):
        model_dict = self.state_dict()
   
        pretrained_dict_path = self.CFG["resume_path"]
        print("Loading model checkpoint: ", pretrained_dict_path)
        pretrained_dict = torch.load(pretrained_dict_path)["state_dict"]
      
        if pretrained_dict.keys() != model_dict.keys():
            print("LOADED MODEL AND CREATED MODEL NOT THE SAME")

        p_kys = pretrained_dict.keys()
        m_kys = model_dict.keys()
        in_p_not_m = list(set(p_kys) - set(m_kys))
        in_m_not_p = list(set(m_kys) - set(p_kys))

        if len(in_p_not_m) > 0:
            print("LAYERS IN THE CHECKPOINT BUT NOT IN THE MODEL", in_p_not_m)

        if len(in_m_not_p) > 0:
            print("LAYERS IN THE MODEL BUT NOT IN THE CHECKPOINT", in_m_not_p)

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        keys_all = pretrained_dict.keys()
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if v.shape == model_dict[k].shape
        }
        keys_same_size = pretrained_dict.keys()

        if set(keys_all) != set(keys_same_size):
            print(
                f"Layers not loaded as different size {set(keys_all)-set(keys_same_size)}"
            )

        model_dict.update(pretrained_dict)
        self.load_state_dict(pretrained_dict, strict=False)

    def val_dataloader(self):
        dataloader = get_dataloader(self.CFG, train=False)
        return dataloader

    def train_dataloader(self):
        dataloader = get_dataloader(self.CFG, train=True)
        return dataloader
