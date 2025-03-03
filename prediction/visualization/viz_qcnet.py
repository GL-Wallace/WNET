import argparse
import os
from copy import deepcopy
from typing import Tuple

import matplotlib.pyplot as plt
import torch
from data.collate import qcnet_collate
from data.dataset.traj_pred_dataset import ArgoverseIIDataset
from models.structure import QCNet
from torch.utils.data import DataLoader
from utils import count_parameters


def reformat_input_data(
    val_data: ArgoverseIIDataset,
    mirrored_data: ArgoverseIIDataset,
    index: int,
) -> ArgoverseIIDataset:
    """Reformat agent features along Time Axis from a clip by an index.

       Example: original gcs_poses of a Batch data is [1, N, 50, 3], if you
       want to index the first 10 time step [0, 10] and predict trajectories
       along Time [11, 61], just choose the index = 10.

    Args:
        val_data(ArgoverseIIDataset): Validation data you want to visualize.
        mirrored_data(ArgoverseIIDataset): Deepcopied validation data to be
            reformated.
        index(int): The frame index you want to reformated.
    """

    B, N, T, D = val_data["agent_features"]["gcs"]["poses"]["his"].shape
    T_idx = T - index

    val_data["agent_features"]["gcs"]["poses"]["fut"] = torch.cat(
        [
            mirrored_data["agent_features"]["gcs"]["poses"]["his"][
                :, :, index:, :
            ],
            mirrored_data["agent_features"]["gcs"]["poses"]["fut"][
                :, :, :index, :
            ],
        ],
        dim=2,
    )

    val_data["agent_features"]["gcs"]["poses"]["his"] = torch.cat(
        [
            mirrored_data["agent_features"]["gcs"]["poses"]["his"][
                :, :, :1, :
            ].expand(B, N, T_idx, D),
            mirrored_data["agent_features"]["gcs"]["poses"]["his"][
                :, :, :index, :
            ],
        ],
        dim=2,
    )
    val_data["agent_features"]["gcs"]["vels"]["his"] = torch.cat(
        [
            torch.zeros([B, N, T_idx, D]).float(),
            mirrored_data["agent_features"]["gcs"]["vels"]["his"][
                :, :, :index, :
            ],
        ],
        dim=2,
    )
    val_data["agent_features"]["state_valid_masks"]["his"] = torch.cat(
        [
            torch.zeros([B, N, T_idx]).bool(),
            mirrored_data["agent_features"]["state_valid_masks"]["his"][
                :, :, :index
            ],
        ],
        dim=2,
    )
    return val_data


def plot_trajectories(
    i: int,
    j: int,
    agent_gcs_hist: torch.tensor,
    agent_gcs_fut: torch.tensor,
    traj_eval: torch.tensor,
    map_points: torch.tensor,
    ego_hist: torch.tensor,
    save_dir: str = "./tmp/viz",
    fig_size: Tuple[int, int] = (20, 20),
) -> None:
    """Generate plots for QCNet visualization.

    Args:
        i(int): The ith clip to be visualized.
        j(int): The jth frame index in which the clip is reformated.
        agent_gcs_hist(torch.tensor): Agent historical trajectories to be
            visualized with shape [B, N, TH, 3].
        agent_gcs_fut(torch.tensor): Agent groundtruth to be
            visualized with shape [B, N, TF, 3].
        traj_eval(torch.tensor): Predicted agent trajectories to be
            visualized with shape [B, N, TF, 3].
        map_points(torch.tensor): Map centerlines to be
            visualized with shape [B, M, P, 3].
        ego_hist(torch.tensor): Ego vehicle historical trajectory to be
            visualized with shape [TF, 3].
        save_dir (str) = The path to save generated images.
        fig_size (Tuple[int, int]): The fixed figure size.
    """

    plt.figure(figsize=fig_size)

    # Plot all map points and update bounds
    for cl in map_points:
        valid_points = cl[cl[:, 0] != 0]
        if valid_points.numel() > 0:  # Ensure there are valid points
            plt.plot(
                valid_points[:, 0],
                valid_points[:, 1],
                color="grey",
                linewidth=3,
            )

    label_idx = 0
    for curr_gcs_his in agent_gcs_hist:
        label_idx += 1
        last_point = curr_gcs_his[-2]
        arrow_dx = curr_gcs_his[-1, 0] - curr_gcs_his[-2, 0]
        arrow_dy = curr_gcs_his[-1, 1] - curr_gcs_his[-2, 1]
        plt.plot(
            curr_gcs_his[:, 0],
            curr_gcs_his[:, 1],
            color="black",
            linewidth=3,
            label=(
                "Focal Agent Historical Trajectory (5s)"
                if label_idx == 1
                else None
            ),
        )
        plt.arrow(
            last_point[0],
            last_point[1],
            arrow_dx,
            arrow_dy,
            head_width=2,
            head_length=4,
            fc="black",
            ec="black",
        )

    for curr_gcs_pred in traj_eval[:, :3]:
        for idx, curr_mode in enumerate(curr_gcs_pred):
            plt.plot(
                curr_mode[:, 0],
                curr_mode[:, 1],
                linewidth=3,
                label=f"Agent Mode {idx + 1} Predicted Trajectory (6s)",
            )

    label_idx = 0
    for curr_gcs_gt in agent_gcs_fut:
        label_idx += 1
        plt.plot(
            curr_gcs_gt[:, 0],
            curr_gcs_gt[:, 1],
            color="purple",
            linestyle="--",
            linewidth=3,
            label=(
                "Focal Agent Groudtruth Trajectory (6s)"
                if label_idx == 1
                else None
            ),
        )

    last_point = ego_hist[-2]
    arrow_dx = ego_hist[-1, 0] - ego_hist[-2, 0]
    arrow_dy = ego_hist[-1, 1] - ego_hist[-2, 1]
    plt.plot(
        ego_hist[:, 0],
        ego_hist[:, 1],
        color="red",
        linewidth=3,
        label="Ego Historical Trajectory (5s)",
    )
    plt.arrow(
        last_point[0],
        last_point[1],
        arrow_dx,
        arrow_dy,
        head_width=2,
        head_length=4,
        fc="red",
        ec="red",
    )

    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.legend(fontsize=26)
    plt.axis("equal")
    save_path = os.path.join(save_dir, f"test_{i}_{j}.png")
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QCNet Visualization Parser")
    parser.add_argument("--argo_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--save_dir", type=str, default="./tmp/viz")

    args = parser.parse_args()

    val_data = ArgoverseIIDataset(
        data_path=args.argo_path,
        mode="val",
        dataset_size=24988,
        top_k_polygons=150,
        reprocess=False,
    )

    val_loader = DataLoader(
        dataset=val_data,
        batch_size=1,
        shuffle=False,
        collate_fn=qcnet_collate,
    )

    qcnet = QCNet(
        ckpt_dir="./",
        tensorboard_dir="./",
        viz_dir="./",
        file_path="./",
        float_lr=2e-4,
        weight_decay=1e-4,
        T_max=64,
        log_every_n_steps=50,
    ).to(args.device)

    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    qcnet.load_state_dict(checkpoint)
    qcnet.eval()
    num_params = count_parameters(qcnet)
    print(f"Number of parameters: {num_params}")
    with torch.no_grad():
        for i, val_data in enumerate(val_loader):
            if i < args.start:
                continue
            elif i > args.end:
                break

            mirrored_data = deepcopy(val_data)
            _, _, T, _ = val_data["agent_features"]["gcs"]["poses"][
                "his"
            ].shape

            for j in range(args.frame, T):
                val_data = reformat_input_data(val_data, mirrored_data, j)
                output = qcnet(val_data)

                pred, target, prob, valid_mask = (
                    output["predict_trajs"].squeeze(),
                    output["future_trajectories"].squeeze(),
                    output["real_scores"].squeeze(),
                    output["valid_masks"].squeeze(),
                )
                agent_valid_masks = val_data["agent_features"][
                    "agent_valid_masks"
                ].squeeze()
                agent_category = val_data["agent_features"][
                    "categories"
                ].squeeze()[agent_valid_masks]
                agent_gcs_hist = val_data["agent_features"]["gcs"]["poses"][
                    "his"
                ].squeeze()[agent_valid_masks]
                agent_gcs_fut = val_data["agent_features"]["gcs"]["poses"][
                    "fut"
                ].squeeze()[agent_valid_masks]
                agent_acs_fut = val_data["agent_features"]["acs"]["poses"][
                    "fut"
                ].squeeze()[agent_valid_masks]
                map_points = val_data["map_features"]["gcs"][
                    "pt_poses"
                ].squeeze()
                ego_hist = agent_gcs_hist[-1]
                eval_mask = agent_category >= 3

                # transpose the prediction into global coordinates
                traj_pred_eval = pred[eval_mask, :, :, :2]
                origin_eval = agent_gcs_hist[eval_mask, -1, :2]
                theta_eval = agent_gcs_hist[eval_mask, -1, 2]
                cos, sin = theta_eval.cos(), theta_eval.sin()
                rot_mat = torch.stack([cos, sin, -sin, cos], dim=-1).reshape(
                    -1, 2, 2
                )

                # transform prediction acs trajectories
                traj_eval = (
                    torch.matmul(
                        traj_pred_eval,
                        rot_mat[:, None, :, :].expand(
                            -1, traj_pred_eval.size(1), -1, -1
                        ),
                    )
                    + origin_eval[:, None, None, :]
                )
                plot_trajectories(
                    i,
                    j,
                    agent_gcs_hist[eval_mask, :, :2],
                    agent_gcs_fut[eval_mask, :, :2],
                    traj_eval,
                    map_points,
                    ego_hist,
                )
