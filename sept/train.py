import os
import argparse
import pytorch_lightning as pl
from hydra.utils import instantiate
from hydra import compose, initialize
from datetime import datetime
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


local_train = True
if "bucket" in os.listdir("/"):
    local_train = False
if local_train:
    device_ids = [0]
    base_dir = os.path.join(os.getcwd(), "outputs")
else:
    device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    base_dir = "/horizon-bucket/carizon_pnp_jfs/guowei.zhang/model_outputs"


def parse_args():
    parser = argparse.ArgumentParser(description="Training script with Hydra")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--devices", type=int, help="Number of devices per node")
    return parser.parse_args()


# @hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf):
    pl.seed_everything(conf.seed, workers=True)
    now = datetime.now()
    phase = conf.model["phase"]
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    output_dir = os.path.join(base_dir, phase, date_str, time_str)

    args = parse_args()

    if conf.wandb != "disable":
        logger = WandbLogger(
            project="Forecast-MAE",
            name=conf.output,
            mode=conf.wandb,
            log_model="all",
            resume=conf.checkpoint is not None,
        )
    else:
        logger = TensorBoardLogger(save_dir=output_dir, name="logs")

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename="{epoch}",
            monitor=f"{conf.monitor}",
            mode="min",
            save_top_k=conf.save_top_k,
            save_last=True,
        ),
        RichModelSummary(max_depth=1),
        RichProgressBar(),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    print("Checkpoint path: ", os.path.join(output_dir, "checkpoints"))

    trainer = pl.Trainer(
        logger=logger,
        gradient_clip_val=conf.gradient_clip_val,
        gradient_clip_algorithm=conf.gradient_clip_algorithm,
        max_epochs=conf.epochs,
        accelerator="gpu",
        num_nodes=args.num_nodes,
        devices=device_ids,
        strategy="ddp" if len(device_ids) > 1 else "auto",
        callbacks=callbacks,
        limit_train_batches=conf.limit_train_batches,
        limit_val_batches=conf.limit_val_batches,
        sync_batchnorm=conf.sync_bn,
    )

    model = instantiate(conf.model.target)
    datamodule = instantiate(conf.datamodule)
    trainer.fit(model, datamodule, ckpt_path=conf.checkpoint)


if __name__ == "__main__":
    initialize(version_base=None, config_path="conf")
    conf = compose(config_name="config")
    main(conf)
