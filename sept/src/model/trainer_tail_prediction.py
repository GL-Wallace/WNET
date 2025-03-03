import pytorch_lightning as pl
import torch
import torch.nn as nn
import os
import time
from sept.src.utils.optim import WarmupCosLR
from .model_tail_prediction import ModelTailPrediction


class Trainer(pl.LightningModule):
    def __init__(
            self,
            dim=128,
            historical_steps=10,
            future_steps=30,
            encoder_depth=3,
            decoder_depth=3,
            num_heads=8,
            mlp_ratio=4.0,
            qkv_bias=False,
            drop_path=0.2,
            actor_mask_ratio: float = 0.5,
            lane_mask_ratio: float = 0.5,
            epochs: int = 60,
            warmup_epochs: int = 10,
            lr: float = 1e-3,
            loss_weight=[1.0, 1.0, 0.35],
            weight_decay: float = 1e-4,
    ) -> None:
        super(Trainer, self).__init__()
        self.validation_outputs = []
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        self.start_time = None

        self.net = ModelTailPrediction(
            embed_dim=dim,
            encoder_depth=encoder_depth,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_path=drop_path,
            actor_mask_ratio=actor_mask_ratio,
            lane_mask_ratio=lane_mask_ratio,
            history_steps=historical_steps,
            future_steps=future_steps,
            loss_weight=loss_weight,
        )

    def forward(self, data):
        return self.net(data)

    def training_step(self, data, batch_idx):
        B, N, _ = data["x_centers"].size()
        save_dir = self.logger.save_dir
        out = self(data)
        loss = out["loss"]
        if loss is None or torch.isnan(loss):
            print(f"Loss is NaN or None at batch {batch_idx}.")
            self._handle_exception(data, out["output_result"], batch_idx, save_dir)
            return None
        self.log("train_loss", loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=B, sync_dist=True)
        return loss

    def validation_step(self, data, batch_idx):
        out = self(data)
        self.validation_outputs.append({"val_loss": out["loss"]})
        self.log("val_loss", out["loss"], on_epoch=True, on_step=False)

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-6,
            warmup_epochs=self.warmup_epochs,
            epochs=self.epochs,
        )
        return [optimizer], [scheduler]

    def _handle_exception(self, data, out, batch_idx, save_dir):
        exception_dir = os.path.join(save_dir, "nan_exception")

        os.makedirs(exception_dir, exist_ok=True)

        dump_path = os.path.join(exception_dir, f"nan_data_batch_train_{batch_idx}.pt")
        output_path = os.path.join(exception_dir, f"nan_output_batch_train_{batch_idx}.pt")
        debug_model_path = os.path.join(exception_dir, "nan_train_debug_model.pth")

        print(f"Dumping data to: {dump_path}")

        torch.save(data, dump_path)
        torch.save(out, output_path)
        torch.save(self.state_dict(), debug_model_path)

    def on_train_epoch_start(self):
        self.start_time = time.time()

    def on_train_batch_end(
            self, data, batch, batch_idx: int
    ) -> None:
        if self.trainer.is_global_zero:
            total_batch = self.trainer.num_training_batches
            total_epoch = self.trainer.max_epochs
            if (batch_idx + 1) % self.trainer.log_every_n_steps == 0:
                epoch = self.current_epoch
                print(
                    f"Epoch: {epoch}/{total_epoch}, \
                      Batch: {batch_idx + 1}/{total_batch}"
                )

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_outputs]).mean()
        print(f"Validation Loss: {avg_loss.item()}")
