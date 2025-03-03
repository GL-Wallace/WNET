import argparse
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from data.dataset import TrajPredLmdbDataset
from tqdm import tqdm

EGO_ID = 2147483647


def generate_lmdb_pack_viz(
    curr_pack_data: TrajPredLmdbDataset,
    start_idx: int,
    end_idx: int,
    interval: int,
    save_dir: str = "./tmp/viz",
    fig_size: Tuple[int, int] = (20, 20),
) -> None:
    """Generate visualizations of packed LMDB data.

    Args:
        curr_pack_data (TrajPredLMDBDataset): Dataset Interface between
            LMDB-QCNet model.
        start_idx (int): The specified start frame index of your pack data
            to be visualized.
        end_idx (int): The specified end frame index of your pack data
            to be visualized.
        interval (int): The specified sampling frequency of your pack data
            to be visualized.
        save_dir (str) = The path to save generated images.
        fig_size (Tuple[int, int]): The fixed figure size.
    """

    for idx in tqdm(range(start_idx, end_idx, interval)):
        map_annos = curr_pack_data[idx]["map_features"]
        obs_annos = curr_pack_data[idx]["agent_features"]
        fig_name = f"{idx}.png"
        save_path = os.path.join(save_dir, fig_name)
        plt.figure(figsize=fig_size)
        for poly_type, poly_feat in map_annos.items():
            if poly_type == "center":
                continue
            for i, waypoint in enumerate(poly_feat):
                if waypoint.ndim < 2:
                    continue
                plt.plot(
                    waypoint[:, 0],
                    waypoint[:, 1],
                    color={
                        "left": "khaki",
                        "center": "grey",
                        "right": "brown",
                        "vertex": "blue",
                    }[poly_type],
                    linestyle="--" if poly_type == "center" else "-",
                    linewidth=3,
                    label=(f"{poly_type.capitalize()}" if i == 0 else None),
                )

        for i, (obs_id, obs_feats) in enumerate(obs_annos.items()):
            obs_hist = obs_feats["hist"]
            obs_fut = obs_feats["fut"]
            plt.plot(
                obs_hist[:, 0],
                obs_hist[:, 1],
                color="green" if obs_id == EGO_ID else "red",
                linewidth=3,
                label=(
                    "Ego Vehicle Current Pose"
                    if obs_id == EGO_ID
                    else (
                        "Historical Trajectories of Obstacles"
                        if i == 0
                        else None
                    )
                ),
            )
            plt.arrow(
                obs_hist[-1, 0],
                obs_hist[-1, 1],
                np.cos(obs_hist[-1, -2]),
                np.sin(obs_hist[-1, -2]),
                head_width=2,
                head_length=4,
                fc="green" if obs_id == EGO_ID else "red",
                ec="green" if obs_id == EGO_ID else "red",
            )
            plt.plot(
                obs_fut[obs_fut[:, 0] != 0, 0],
                obs_fut[obs_fut[:, 0] != 0, 1],
                color="purple",
                linestyle="--",
                linewidth=3,
                label=(
                    "Ego Future Trajectories(6s)"
                    if obs_id == -2
                    else (
                        "Future Trajectories of Obstacles(6s)"
                        if i == 0
                        else None
                    )
                ),
            )

        plt.xticks(fontsize=26)
        plt.yticks(fontsize=26)
        plt.legend(fontsize=26, loc="lower left")
        plt.axis("equal")
        plt.savefig(save_path)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LMDB Visualizer Parser")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--obs_name", type=str, default="lmdb_obstacles")
    parser.add_argument("--vec_name", type=str, default="lmdb_vectormap")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--interval", type=int, required=True)
    parser.add_argument("--save_dir", type=str, default="./tmp/viz")

    args = parser.parse_args()

    curr_pack_data = TrajPredLmdbDataset(
        lmdb_root=args.root,
        obs_lmdb_name=args.obs_name,
        vectormap_lmdb_name=args.vec_name,
    )

    generate_lmdb_pack_viz(
        curr_pack_data=curr_pack_data,
        start_idx=args.start,
        end_idx=args.end,
        interval=args.interval,
        save_dir=args.save_dir,
    )
