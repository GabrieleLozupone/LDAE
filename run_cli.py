# Copyright (c) 2025 Gabriele Lozupone (University of Cassino and Southern Lazio).
# All rights reserved.
# --------------------------------------------------------------------------------
#
# LICENSE NOTICE
# *************************************************************************************************************
# By downloading/using/running/editing/changing any portion of codes in this package you agree to the license.
# If you do not agree to this license, do not download/use/run/edit/change this code.
# Refer to the LICENSE file in the root directory of this repository for full details.
# *************************************************************************************************************
#
# Contact: Gabriele Lozupone at gabriele.lozupone@unicas.it
# -----------------------------------------------------------------------------

from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from src.utils.progress_bar import MyProgressBar
from src.data.datamodules import BrainMRDataModule
import os
import resource
import torch


class MyCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(MyProgressBar, nested_key='progress_bar')
        parser.add_lightning_class_args(EarlyStopping, nested_key='early_stopping')
        parser.add_lightning_class_args(ModelCheckpoint, nested_key='model_checkpoint')


def cli_main():
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"
    torch.set_float32_matmul_precision("high")
    cli = MyCLI(
        datamodule_class=BrainMRDataModule,
        trainer_defaults={
            "logger": {
                "class_path": "lightning.pytorch.loggers.WandbLogger",
                "init_args": {
                    "project": "<DEFAULT_PROJECT_ISTANCE>",
                    "name": "<DEFAULT_PROJECT_NAME>",
                }
            },
            "strategy": {
                "class_path": "lightning.pytorch.strategies.DDPStrategy",
                "init_args": {
                    "find_unused_parameters": True
                }
            }
        },
        parser_kwargs={"parser_mode": "omegaconf"},
        save_config_kwargs={"overwrite": True}
    )
    return cli


if __name__ == '__main__':
    cli_main()
