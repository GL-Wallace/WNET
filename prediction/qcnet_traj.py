import getpass
import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from configs import DATASET
from data.collate.traj_pred_collate import qcnet_collate
from data.dataset.traj_pred_dataset import ArgoverseIIDataset
from models.structure import QCNet
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader
from utils import create_directories_and_file

task_name = "qcnet_test_0813"
local_train = True
if "bucket" in os.listdir("/"):
    local_train = False
if local_train:
    device_ids = [6]
    user_name = getpass.getuser()
    base_path = f"/home/users/{user_name}/results/"
else:
    device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    base_path = "/job_data/models/"

# configs
epochs = 64
float_lr = 5e-4
train_num_workers, val_num_workers = 8, 8
weight_decay = 1e-4
batch_size_per_gpu = 1
T_max = 64
log_every_n_steps = 50

# Dataset configurations
dataset_name = "prediction"
train_data = "Argoverse2_train"
val_data = "Argoverse2_val"

train_path = DATASET[dataset_name][train_data]
val_path = DATASET[dataset_name][val_data]

make_train_dataset = False
make_val_dataset = False

device = "cuda" if torch.cuda.is_available() else "cpu"


def main(num_nodes: int) -> None:
    ckpt_dir, viz_dir, tensorboard_dir, file_path = (
        create_directories_and_file(base_path, task_name)
    )

    train_data = ArgoverseIIDataset(
        data_path=train_path,
        mode="train",
        dataset_size=199908,
        reprocess=make_train_dataset,
    )
    val_data = ArgoverseIIDataset(
        data_path=val_path,
        mode="val",
        dataset_size=24988,
        reprocess=make_val_dataset,
    )

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size_per_gpu,
        num_workers=train_num_workers,
        shuffle=False,
        collate_fn=qcnet_collate,
    )

    val_loader = DataLoader(
        dataset=val_data,
        batch_size=batch_size_per_gpu,
        num_workers=val_num_workers,
        shuffle=False,
        collate_fn=qcnet_collate,
    )

    qcnet = QCNet(
        ckpt_dir,
        tensorboard_dir,
        viz_dir,
        file_path,
        float_lr,
        weight_decay,
        T_max,
        log_every_n_steps,
    )

    model_checkpoint = ModelCheckpoint(
        monitor="val_minFDE_k6", save_top_k=6, mode="min"
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    if local_train:
        trainer = pl.Trainer(
            logger=TensorBoardLogger(tensorboard_dir, name=""),
            accelerator="gpu",
            devices=device_ids,
            strategy=DDPStrategy(
                find_unused_parameters=True, gradient_as_bucket_view=True
            ),
            callbacks=[model_checkpoint, lr_monitor],
            enable_progress_bar=True,
            max_epochs=epochs,
        )
    else:
        trainer = pl.Trainer(
            logger=TensorBoardLogger(tensorboard_dir, name=""),
            accelerator="gpu",
            devices=device_ids,
            num_nodes=num_nodes,
            strategy=DDPStrategy(
                find_unused_parameters=True, gradient_as_bucket_view=True
            ),
            callbacks=[model_checkpoint, lr_monitor],
            enable_progress_bar=False,
            max_epochs=epochs,
        )
    trainer.fit(qcnet, train_loader, val_loader)


if __name__ == "__main__":
    pl.seed_everything(2024, workers=True)
    parser = ArgumentParser()
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--devices", type=int, default=8)

    args = parser.parse_args()
    main(args.num_nodes)
