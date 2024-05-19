from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
import argparse
import torch
import yaml
from models.ABC123 import ABC123


def main():
    if CFG["seed"] != -1:
        seed_everything(CFG["seed"], workers=True)

    t_logger = TensorBoardLogger(
        CFG["log_dir"], name=CFG["name"], default_hp_metric=False
    )
    if CFG["test_split"] == "val":        
        model_checkpoint_MAE = ModelCheckpoint(
            monitor="val_MAE",
            save_last=True,
            save_top_k=3,
            every_n_epochs=1,
            filename="{epoch}_{val_MAE:.2f}",
        )
    else:
        model_checkpoint_MAE = ModelCheckpoint(
            monitor="test_MAE",
            save_last=True,
            save_top_k=3,
            every_n_epochs=1,
            filename="{epoch}_{test_MAE:.2f}",
        )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks = [lr_monitor, model_checkpoint_MAE]
    
    trainer = Trainer(
        gpus=-1,
        logger=t_logger,
        max_epochs=CFG["max_epochs"],
        max_steps=CFG["max_steps"],
        accelerator="gpu",
        strategy=DDPPlugin(
            find_unused_parameters=CFG["find_unused_parameters"],
        ),
        replace_sampler_ddp=True,
        callbacks=callbacks,
        log_every_n_steps=1,
        num_sanity_val_steps=CFG["num_sanity_val_steps"],
    )

    model = ABC123(CFG)
    trainer.fit(model)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Train a 3D reconstruction model.")
    PARSER.add_argument("--config", "-c", type=str, help="Path to config file.")
    ARGS = PARSER.parse_args()

    CFG = yaml.safe_load(open("configs/_DEFAULT.yml"))
    CFG_new = yaml.safe_load(open("configs/{}.yml".format(ARGS.config)))
    CFG.update(CFG_new)
    
    CFG["name"] = ARGS.config
    main()
