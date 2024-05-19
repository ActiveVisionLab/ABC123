import json
import logging
import os
import random

import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as trans_F
from einops import rearrange
from PIL import Image, ImageFile
import torch
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler)
from torchvision import transforms

logger = logging.getLogger(__name__)

IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]

Normalize_tensor = transforms.Compose(
    [transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)]
)


def denormalize(tensor, means=IM_NORM_MEAN, stds=IM_NORM_STD, clip_0_1=True):
    with torch.no_grad():
        denormalized = tensor.clone()

        for channel, mean, std in zip(denormalized, means, stds):
            channel.mul_(std).add_(mean)

            if clip_0_1:
                channel[channel < 0] = 0
                channel[channel > 1] = 1

        return denormalized

class MCAC_Dataset(Dataset):
    def __init__(self, CFG, train):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        self.CFG = CFG
        self.img_size = CFG["img_size"]
        self.img_channels = CFG["img_channels"]
        self.tag = "train" if train else CFG["test_split"]
      
        self.im_dir = f"{CFG['data_path']}{self.tag}"

        self.gs_file = f"_c_8"
        self.gs_file += "_occ_" + str(int(CFG["MCAC_occ_limit"])) if CFG["MCAC_occ_limit"] != -1 else ""
        self.gs_file += "_non_int"
        self.gs_file += f"_crop{CFG['MCAC_crop_size']}" if CFG["MCAC_crop_size"] != -1 else ""
        self.gs_file += "_np"
        self.im_ids = [
            f for f in os.listdir(self.im_dir) if os.path.isdir(self.im_dir + "/" + f)
        ]

        self.toten = transforms.ToTensor()
        self.resize_im = transforms.Resize((self.img_size[0], self.img_size[1]))
        
        self.bboxes_str = "bboxes"
        self.centers_str = "centers"
        self.occlusions_str = "occlusions"
        self.area_str = "area"
        self.json_p = f"info_with_occ_bbox.json"

        if self.CFG["MCAC_crop_size"] != -1:
            self.bboxes_str += f"_crop{self.CFG['MCAC_crop_size']}"
            self.centers_str += f"_crop{self.CFG['MCAC_crop_size']}"
            self.occlusions_str += f"_crop{self.CFG['MCAC_crop_size']}"


        if CFG["dataset"] == "MCAC-M1":
            CFG["MCAC_exclude_imgs_with_num_classes_over"] = 1
            print("USING MCAC-M1")

        if CFG["MCAC_exclude_imgs_with_num_classes_over"] != -1:
            self.exlude_images_num_class()

        if CFG["MCAC_exclude_imgs_with_counts_over"] != -1:
            self.exlude_images_counts()
            
        print(
            f"{self.tag} set, size:{len(self.im_ids)}")

    def __len__(self):
        return len(self.im_ids)

    def __getitem__(self, idx):
        im_id = self.im_ids[idx]
        image = Image.open(f"{self.im_dir}/{im_id}/img.png")
        image.load()
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = self.toten(image)

        if self.CFG["MCAC_crop_size"] != -1:
            crop_boundary_size_0 = int(
                (image.shape[1] - self.CFG["MCAC_crop_size"]) / 2
            )
            crop_boundary_size_1 = int(
                (image.shape[2] - self.CFG["MCAC_crop_size"]) / 2
            )
            image = image[
                :,
                crop_boundary_size_0:-crop_boundary_size_0,
                crop_boundary_size_1:-crop_boundary_size_1,
            ]

        with open(f"{self.im_dir}/{im_id}/{self.json_p}", "r") as f:
            img_info = json.load(f)

        dots = np.zeros((self.CFG["MCAC_max_num_classes"], self.CFG["MCAC_max_number_per_type"], 2)) - 1
        rects = np.zeros((self.CFG["MCAC_max_num_classes"], self.CFG["MCAC_max_number_per_type"], 2, 2)) - 1
        counts = []
        for c_i, c in enumerate(img_info["countables"]):
            bboxes = np.array(c[self.bboxes_str])
            centers = np.array(c[self.centers_str])[:, :2]

            # scale boxes and centers
            bboxes[:, :, 0] = bboxes[:, :, 0] / (image.shape[1] / self.img_size[0])
            bboxes[:, :, 1] = bboxes[:, :, 1] / (image.shape[2] / self.img_size[1])
            bboxes = np.clip(
                bboxes, 0, self.img_size[0] - 1
            )  

            centers[:, 0] = centers[:, 0] * self.img_size[0]
            centers[:, 1] = (self.img_size[0] - 1) - centers[:, 1] * self.img_size[0]
            centers = centers.astype(int)
            centers = np.clip(
                centers, 0, self.img_size[0] - 1
            ) 

            if self.CFG["MCAC_occ_limit"] == -1:
                cnt = len(c["inds"])
                counts.append(cnt)
            else:
                assert len(c[self.occlusions_str]) == len(c["inds"])
                cnt_np = np.array(c[self.occlusions_str])
                inds = cnt_np < self.CFG["MCAC_occ_limit"]
                cnt_np = cnt_np[inds]
                centers = centers[inds, :]
                bboxes = bboxes[inds, :]
                cnt = len(cnt_np)
                counts.append(cnt)

            dots[c_i, : centers.shape[0]] = centers
            rects[c_i, : bboxes.shape[0]] = bboxes

        gt_cnt = torch.zeros((self.CFG["MCAC_max_num_classes"],))
        density = torch.zeros((1, self.CFG["MCAC_max_num_classes"], self.img_size[0], self.img_size[1]))

        gt_cnt[: len(counts)] = torch.tensor(counts)
        gt_pth = (
            f"{self.im_dir}/{im_id}/gtdensity_{self.img_size[0]}{self.gs_file}.npy"
        )
        density_load = torch.tensor(np.load(gt_pth))
        density_load = rearrange(density_load, "h w c -> 1 c h w")  #
        if density.shape[1] < density_load.shape[1]:
            # if there was a failure to place emements during creation, so get rid of those zeros
            dl = rearrange(density_load.clone(), "1 c h w -> c (h w)")
            dl_sum = torch.sum(dl, dim=1)
            dl_sum_nz_idx = torch.nonzero(dl_sum.flatten()).squeeze()
            dl_sum_nz = dl_sum[dl_sum_nz_idx]
            density_load = density_load[:, dl_sum_nz_idx]
        density[:, : density_load.shape[1]] = density_load

        for i in range(density.shape[1]):
            if i < len(img_info["countables"]):
                if torch.sum(density[:, i]) != 0:
                    density[:, i] = density[:, i] * (
                        gt_cnt[i] / torch.sum(density[:, i])
                    )
            else:
                if gt_cnt[i] or torch.sum(density[:, i]):
                    print("Should be 0", torch.sum(density[:, i]), gt_cnt[i])

        if self.tag == "train" and "ref_rot" in self.CFG["image_transforms"]:
            image, dots, rects, density = self.ref_rot(image, dots, rects, density)

        image = self.resize_im(image)
        rects = torch.IntTensor(rects)

        if self.img_channels == 1:
            image = torch.mean(image, dim=0).unsqueeze(0)

        density = density.squeeze(0)
        dots = torch.tensor(dots)
        im_id = torch.tensor(int(im_id)).long()
        rects_new = torch.zeros_like(rects)
        dots_new = torch.zeros_like(dots)
        density_new = torch.zeros_like(density)
        gt_cnt_new = torch.zeros_like(gt_cnt)
        
        non_z_inds = torch.nonzero(gt_cnt)
        gt_cnt_new[:len(non_z_inds)] = gt_cnt[non_z_inds].squeeze()
        rects_new[:len(non_z_inds)] = rects[non_z_inds].squeeze()
        dots_new[:len(non_z_inds)] = dots[non_z_inds].squeeze()
        density_new[:len(non_z_inds)] = density[non_z_inds].squeeze()

        return (
            image,
            rects_new,
            dots_new,
            density_new,
            gt_cnt_new,
            im_id,
        )
 
    def exlude_images_num_class(self):
        new_im_ids = []
        for id in self.im_ids:
            with open(f"{self.im_dir}/{id}/{self.json_p}", "r") as f:
                img_info = json.load(f)
            num_countables = 0
            for c in img_info["countables"]:
                if self.CFG["MCAC_occ_limit"] != -1:
                    assert len(c[self.occlusions_str]) == len(c["inds"])
                    cnt_np = np.array(c[self.occlusions_str])
                    inds = cnt_np < self.CFG["MCAC_occ_limit"]
                    cnt_np = cnt_np[inds]
                    cnt = len(cnt_np)
                else:
                    cnt = len(c["inds"])

                if cnt >= 1:
                    num_countables += 1
            if (
                num_countables
                <= self.CFG["MCAC_exclude_imgs_with_num_classes_over"]
            ):
                new_im_ids.append(id)

        print(
            f"EXCLUDING OVER LIMIT: {self.CFG['MCAC_exclude_imgs_with_num_classes_over']} class, from:{len(self.im_ids)} to {len(new_im_ids)}"
        )
        self.im_ids = new_im_ids

    def exlude_images_counts(self):

        new_im_ids = []
        all_counts = []
        for id in self.im_ids:
            with open(f"{self.im_dir}/{id}/{self.json_p}", "r") as f:
                img_info = json.load(f)
            include = True
            for c in img_info["countables"]:
                if self.CFG["MCAC_occ_limit"] != -1:
                    assert len(c[self.occlusions_str]) == len(c["inds"])
                    cnt_np = np.array(c[self.occlusions_str])
                    inds = cnt_np < self.CFG["MCAC_occ_limit"]
                    cnt_np = cnt_np[inds]
                    cnt = len(cnt_np)
                else:
                    cnt = len(c["inds"])

                if cnt != 0:
                    all_counts.append(cnt)
                if cnt > self.CFG["MCAC_exclude_imgs_with_counts_over"]:
                    include = False
            if include:
                new_im_ids.append(id)

        print(
            f"EXCLUDING OVER LIMIT: {self.CFG['MCAC_exclude_imgs_with_counts_over']} count, from:{len(self.im_ids)} to {len(new_im_ids)}"
        )
        self.im_ids = new_im_ids

    def ref_rot(self, image, dots, rects, density):
        if random.random() > 0.5:
            image = trans_F.hflip(image)
            density = trans_F.hflip(density)
            dots = self.hflip_dots(dots)
            rects = self.hflip_bboxes(rects)

        if random.random() > 0.5:
            image = trans_F.vflip(image)
            density = trans_F.vflip(density)
            dots = self.vflip_dots(dots)
            rects = self.vflip_bboxes(rects)

        rotate_angle = int(random.random() * 4)
        if rotate_angle != 0:
            image = trans_F.rotate(image, rotate_angle * 90)
            density = trans_F.rotate(density, rotate_angle * 90)
            for _i in range(rotate_angle):
                dots = self.rotate_dots_90(dots)
                rects = self.rotate_bboxes_90(rects)
        return image, dots, rects, density

    def rotate_bboxes_90(self, rects):
        none_rects = rects == -1
        new_x_rects = rects[:, :, 0]
        new_y_rects = (self.img_size[1] - 1) - rects[:, :, 1]
        rects = np.stack((new_y_rects, new_x_rects), axis=-2)
        rects[none_rects] = -1
        return rects

    def rotate_dots_90(self, dots):
        none_dots = dots == -1
        new_x = dots[:, :, 1]
        new_y = (self.img_size[1] - 1) - dots[:, :, 0]
        dots = np.stack((new_x, new_y), axis=-1)
        dots[none_dots] = -1
        return dots

    def vflip_bboxes(self, rects):
        none_rects = rects == -1
        rects[:, :, 0] = (self.img_size[1] - 1) - rects[:, :, 0]
        rects[none_rects] = -1
        return rects

    def vflip_dots(self, dots):
        none_dots = dots == -1
        dots[:, :, 1] = (self.img_size[1] - 1) - dots[:, :, 1]
        dots[none_dots] = -1
        return dots

    def hflip_bboxes(self, rects):
        none_rects = rects == -1
        rects[:, :, 1] = (self.img_size[0] - 1) - rects[:, :, 1]
        rects[none_rects] = -1
        return rects

    def hflip_dots(self, dots):
        none_dots = dots == -1
        dots[:, :, 0] = (self.img_size[0] - 1) - dots[:, :, 0]
        dots[none_dots] = -1
        return dots
    

def get_loader_counting(CFG):
    test_loader = get_dataloader(CFG, train=False)
    train_loader = get_dataloader(CFG, train=True)
    return train_loader, test_loader


def get_dataloader(CFG, train):
    if CFG["dataset"] == "MCAC" or CFG["dataset"] == "MCAC-M1":
        dataset = MCAC_Dataset(CFG, train=train)

    if train:
        bs = CFG["train_batch_size"]
        sampler = RandomSampler(dataset)

    else:
        bs = CFG["eval_batch_size"]
        sampler = SequentialSampler(dataset)

    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=bs,
        num_workers=CFG["num_workers"],
        pin_memory=True,
        drop_last=CFG["drop_last"],
    )
    return loader