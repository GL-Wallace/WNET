# Copyright (c) Carizon. All rights reserved.

import csv
import os
from collections import OrderedDict
from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from losses import QCNetloss
from metrics import Brier, minADE, minFDE
from models.backbone.qcnet_backbone import QCNetBackbone
from models.head.qcnet_head import QCNetHead


class QCNet(pl.LightningModule):
    """HAT-based QCNet implementation."""

    def __init__(
        self,
        ckpt_dir: str,
        tensorboard_dir: str,
        viz_dir: str,
        file_path: str,
        float_lr: float,
        weight_decay: float,
        T_max: int,
        log_every_n_steps: int,
    ):
        """Initialize method.

        Args:
            ckpt_dir(str): The checkpoint save directory.
            tensorboard_dir(str): The tensorboard monitoring directory.
            viz_dir(str): The visualization monitoring directory.
            file_path(str): The metrics recording file saved directory.
            float_lr(float): Learning rate of the AdamW optimizer.
            weight_decay(float): Weight decay value of the AdamW optimizer.
            T_max(int): The number of epochs or steps after which the learning
                        rate will reach its minimum value.
            log_every_n_steps(int): The number of log frequency.
        """
        super(QCNet, self).__init__()
        self.ckpt_dir = ckpt_dir
        self.tensorboard_dir = tensorboard_dir
        self.viz_dir = viz_dir
        self.file_path = file_path
        self.float_lr = float_lr
        self.weight_decay = weight_decay
        self.best_fde_score = float("inf")
        self.validation_outputs = []
        self.T_max = T_max
        self.log_every_n_steps = log_every_n_steps

        self.qcnet_encoder = QCNetBackbone(
            num_agent_classes=10,
            num_historical_steps=50,
            num_neighbors_pl2pl=16,
            num_neighbors_a2pl=16,
            num_neighbors_a2a=4,
            hidden_dim=128,
            num_freq_bands=64,
            num_map_layers=1,
            num_agent_layers=2,
            num_heads=8,
            head_dim=16,
            dropout=0.1,
        )

        self.qcnet_head = QCNetHead(
            num_historical_steps=50,
            num_future_steps=60,
            num_recurrent_steps=3,
            num_modes=6,
            max_agents=100,
            num_neighbors_a2pl=8,
            num_neighbors_a2a=2,
            hidden_dim=128,
            num_freq_bands=64,
            num_layers=2,
            num_heads=8,
            head_dim=16,
            dropout=0.1,
        )

        self.losses = QCNetloss()
        self.fde_metric_k6 = minFDE(max_guesses=6)
        self.ade_metric_k6 = minADE(max_guesses=6)
        self.brier_metric = Brier(max_guesses=6)
        self.fde_metric_k1 = minFDE(max_guesses=1)
        self.ade_metric_k1 = minADE(max_guesses=1)

    def custom_data_preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Customize preprocess function after before entering forward."""
        return data

    def custom_head_out_postprocess(
        self, head_outs: Dict[str, Any], head_cls_name: str
    ) -> Dict[str, torch.tensor]:
        """Customize post process function after head out.

        Args:
            head_outs: the output of the head.
            head_cls_name: the class name of the head.

        Returns:
            dec_result: the output of custom_head_out_postprocess.
        """
        pred_trajs = head_outs["locs_stage2"]  # [B, K, N, FT, 2]
        B, K, N, FT, _ = pred_trajs.shape

        # Extract values for metrics
        agent_valid_masks = head_outs["agent_features"][
            "agent_valid_masks"
        ].reshape(
            B * N
        )  # [B*N]
        batch_idxs = (
            torch.arange(B, device=pred_trajs.device)[:, None]
            .repeat(1, N)
            .reshape(B * N)
        )  # [B*N]
        pred_trajs = pred_trajs.permute(0, 2, 1, 3, 4).reshape(
            B * N, K, FT, -1
        )  # [B*N, K, FT, 2]
        pred_scores = head_outs["scores"]  # [B, K, N]
        pred_scores = pred_scores.permute(0, 2, 1).reshape(B * N, K)[
            :, :, None, None
        ]  # [B*N, K, 1, 1]
        fut_gt_trajs = head_outs["agent_features"]["acs"]["poses"]["fut"][
            ..., :2
        ]  # [B, N, FT, 2]
        fut_gt_trajs = fut_gt_trajs.reshape(B * N, FT, -1)[
            :, None, :, :
        ]  # [B*N, 1, FT, 2]
        state_valid_masks = head_outs["agent_features"]["state_valid_masks"][
            "fut"
        ]  # [B, N, FT]
        agent_classes = head_outs["agent_features"]["agent_properties"][
            "classes"
        ].reshape(
            B * N
        )  # [B, N]
        state_valid_masks = state_valid_masks.reshape(B * N, FT)  # [B*N, FT]
        valid_track_ids = head_outs["agent_features"]["track_ids"].reshape(
            B * N
        )  # [B*N]

        # Filter out invalid agents
        batch_idxs = batch_idxs[agent_valid_masks]
        pred_trajs = pred_trajs[agent_valid_masks]
        pred_scores = pred_scores[agent_valid_masks]
        fut_gt_trajs = fut_gt_trajs[agent_valid_masks]
        state_valid_masks = state_valid_masks[agent_valid_masks]
        agent_classes = agent_classes[agent_valid_masks]
        valid_track_ids = valid_track_ids[agent_valid_masks]

        head_outs.update(
            {
                "predict_trajs": pred_trajs,
                "real_scores": pred_scores,
                "future_trajectories": fut_gt_trajs,
                "valid_masks": state_valid_masks,
                "classes": agent_classes,
                "valid_batch_idxs": batch_idxs.detach().cpu().numpy(),
                "valid_class": agent_classes.detach().cpu().numpy(),
                "valid_track_ids": valid_track_ids.detach().cpu().numpy(),
            }
        )
        return head_outs

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Forward.

        Args:
            data: the original data. We do not care its format just make sure
                  it is a dictionary. The user should perform all the data
                  pre-processing operation in `custom_data_preprocess` by
                  yourself.
        """
        data["scene_enc"] = self.qcnet_encoder(data)
        data["epoch_id"] = self.current_epoch
        # print("the scenario_id is:",data["scenario_id"])
        model_result = OrderedDict()
        output_result = OrderedDict()
        model_result.update(data)

        head_output = self.qcnet_head(data)
        head_cls_name = "QCNetHead"
        model_result.update(head_output)
        output_result = self.custom_head_out_postprocess(
            model_result, head_cls_name
        )
        return output_result

    def training_step(
        self, data: Dict[str, Any], batch_idx: int
    ) -> torch.Tensor:
        """Training step of QCNet in pytorch.lightning module.
        Args:
            data: The cached data. We do not care its format just make sure
                  it is a dictionary and compatible with current QCNet.
            batch_idx(int): The index of a batch.
        Returns:
            total_loss(torch.Tensor): The training loss of QCNet
        """
        output_result = self(data)

        B, N = data["agent_features"]["categories"].size()

        cur_loss = self.losses(output_result)

        if "total_loss" in output_result and "total_loss" in cur_loss:
            cur_loss["total_loss"] += output_result["total_loss"]
        output_result.update(cur_loss)

        self.log(
            "train_reg_loss_propose",
            output_result["reg_propose_loss"],
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=B,
            sync_dist=True,
        )
        self.log(
            "train_reg_loss_refine",
            output_result["reg_refine_loss"],
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=B,
            sync_dist=True,
        )
        self.log(
            "train_cls_loss",
            output_result["cls_loss"],
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=B,
            sync_dist=True,
        )

        return output_result["total_loss"]

    def on_train_batch_end(
        self, data: Dict[str, Any], batch, batch_idx: int
    ) -> None:
        if self.trainer.is_global_zero:
            total_batch = self.trainer.num_training_batches
            total_epoch = self.trainer.max_epochs
            if (batch_idx + 1) % self.log_every_n_steps == 0:
                epoch = self.current_epoch
                # batch = self.batch_idx
                print(
                    f"Epoch: {epoch}/{total_epoch}, \
                      Batch: {batch_idx+1}/{total_batch}"
                )

    def validation_step(self, data: Dict[str, Any], batch_idx: int) -> None:
        """Validataion step of QCNet in pytorch.lightning module.
        Args:
            data: The cached data. We do not care its format just make sure
                  it is a dictionary and compatible with current QCNet.
            batch_idx(int): The index of a batch.
        """
        output_result = self(data)

        cur_loss = self.losses(output_result)

        B, N = data["agent_features"]["categories"].size()
        if "total_loss" in output_result and "total_loss" in cur_loss:
            cur_loss["total_loss"] += output_result["total_loss"]
        output_result.update(cur_loss)
        self.log(
            "train_reg_loss_propose",
            output_result["reg_propose_loss"],
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=B,
            sync_dist=True,
        )
        self.log(
            "train_reg_loss_refine",
            output_result["reg_refine_loss"],
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=B,
            sync_dist=True,
        )
        self.log(
            "train_cls_loss",
            output_result["cls_loss"],
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=B,
            sync_dist=True,
        )

        pred, target, prob, valid_mask = (
            output_result["predict_trajs"].squeeze(),
            output_result["future_trajectories"].squeeze(),
            output_result["real_scores"].squeeze(),
            output_result["valid_masks"].squeeze(),
        )
        pi_eval = F.softmax(prob, dim=1)

        eval_mask = (data["agent_features"]["categories"] == 3).reshape(B * N)
        valid_pred = pred[eval_mask]
        valid_target = target[eval_mask]
        valid_prob = pi_eval[eval_mask]
        valid_mask_eval = valid_mask[eval_mask]

        self.brier_metric.update(
            valid_pred, valid_target, valid_prob, valid_mask_eval
        )
        self.fde_metric_k6.update(
            valid_pred, valid_target, valid_prob, valid_mask_eval
        )
        self.ade_metric_k6.update(
            valid_pred, valid_target, valid_prob, valid_mask_eval
        )
        self.fde_metric_k1.update(
            valid_pred, valid_target, valid_prob, valid_mask_eval
        )
        self.ade_metric_k1.update(
            valid_pred, valid_target, valid_prob, valid_mask_eval
        )

        self.log(
            "val_Brier",
            self.brier_metric,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=B,
        )
        self.log(
            "val_minFDE_k6",
            self.fde_metric_k6,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=B,
        )
        self.log(
            "val_minADE_k6",
            self.ade_metric_k6,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=B,
        )
        self.log(
            "val_minFDE_k1",
            self.fde_metric_k1,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=B,
        )
        self.log(
            "val_minADE_k1",
            self.ade_metric_k1,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=B,
        )

        self.validation_outputs.append({"val_loss": cur_loss["total_loss"]})

    def on_validation_epoch_end(self) -> None:
        """Compute the final metrics."""
        brier_score = self.brier_metric.compute()
        ade_score_k6 = self.ade_metric_k6.compute()
        fde_score_k6 = self.fde_metric_k6.compute()
        ade_score_k1 = self.ade_metric_k1.compute()
        fde_score_k1 = self.fde_metric_k1.compute()

        avg_total_loss = torch.stack(
            [x["val_loss"] for x in self.validation_outputs]
        ).mean()
        if self.trainer.is_global_zero:
            with open(self.file_path, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        self.current_epoch,
                        avg_total_loss.item(),
                        fde_score_k6.item(),
                        fde_score_k1.item(),
                        ade_score_k6.item(),
                        ade_score_k1.item(),
                        brier_score.item(),
                    ]
                )

            if fde_score_k6 < self.best_fde_score:
                self.best_fde_score = fde_score_k6
                model_path = os.path.join(self.ckpt_dir, "best_model.pth")
                torch.save(self.state_dict(), model_path)
                print(f"Model saved with FDE score(K=6): {fde_score_k6}")

            current_model_path = os.path.join(
                self.ckpt_dir, f"epoch{self.current_epoch}_model.pth"
            )
            torch.save(self.state_dict(), current_model_path)
            print(f"Model saved with FDE score(K=6): {fde_score_k6}")

        self.validation_outputs.clear()

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler for the model.

        This method sets up the AdamW optimizer with weight decay and a
        cosine annealing learning rate scheduler.The optimizer and scheduler
        are returned as lists, which is required for some training frameworks
        like PyTorch Lightning.

        Returns:
            Tuple[List[torch.optim.Optimizer],
                  List[torch.optim.lr_scheduler._LRScheduler]]: A tuple where
                  the first element is a list containing the AdamW optimizer,
                  and the second element is a list containing the
                  CosineAnnealingLR scheduler.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.float_lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.T_max, eta_min=0.0
        )
        return [optimizer], [scheduler]
