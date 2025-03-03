from pathlib import Path
import os
import json
from zlib import decompress
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from typing import Tuple
import torch
from forecast_mae_prediction.src.datamodule.traj_pred_dataset import LmdbWrapper, TrajPredLmdbDataset
EGO_ID = 88888888
all_keys = {}
processed_annos = {}
def generate_source(data_dir: Path):
    obs_lmdb_name = "lmdb_obstacles"
    vectormap_lmdb_name = "lmdb_vectormap"
    obs_key_path = os.path.join(data_dir, obs_lmdb_name, "timestamp.txt")
    dynamic_lmdb_path = os.path.join(data_dir, obs_lmdb_name)
    static_lmdb_path = os.path.join(data_dir, vectormap_lmdb_name)

    dynamic_lmdb = LmdbWrapper(
        dynamic_lmdb_path, file_type="gt"
    )
    static_lmdb = LmdbWrapper(
        static_lmdb_path, file_type="gt"
    )

    data_store ={}
    data_store.setdefault("agent_features", {}).setdefault("map_features", {})
    obs_feats = []
    vectormap_feats = []

    with open(obs_key_path, "r", encoding="utf-8") as file:
        for line in file:
            stripped_line = line.strip()  # 移除行首和行尾的空白字符。
            if stripped_line:
                frame, ts = stripped_line.split(",")
                all_keys[ts] = int(frame)  # 将时间戳作为键，帧作为值，存储在 self.all_keys 字典中

    all_time_stamp = list(all_keys.keys())
    if len(all_time_stamp) == 0:
        return
    all_time_stamp = sorted(all_time_stamp, key=float)

    for idx in range(0, len(all_time_stamp)):
        timestamp = all_time_stamp[idx]
        obs_feats_temp = json.loads(decompress(dynamic_lmdb.read(timestamp)))
        # data_store["agent_features"] = obs_feats_temp
        # obs_feats.append(obs_feats_temp)
        map_temp = json.loads(decompress(static_lmdb.read(timestamp)))
        # data_store["map_features"] = map_temp
        # vectormap_feats.append(map_temp)
        _extract_lmdb_feats(timestamp, obs_feats_temp, map_temp)

    return obs_feats, vectormap_feats, data_store


def vis_1_ts(obs_feats, vectormap_feats, data_store):
    for vectormap in vectormap_feats:
        # 提取 pl_type = 0 的 polygon
        polygons = [poly for poly in vectormap['polygons'] if poly['pl_type'] == 0]

        for polygon in polygons:
            # 提取 lines 中 pt_type = 1 的数据
            lines = [line for line in polygon['lines'] if line['pt_type'] == 1]

            for line in lines:
                # 提取 pts 中的 xy yaw
                pts = line['pts']
                x = [pt['x'] for pt in pts]
                y = [pt['y'] for pt in pts]

                # 绘制线上的所有点
                plt.plot(x, y, label=f"polygon {polygon['id']}")

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Lines with pt_type = 1')
    plt.legend()
    plt.show()

def generate_lmdb_pack_viz(
    curr_pack_data,
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
        # for poly_type, poly_feat in map_annos.items():
        #     if poly_feat['pl_type'] != 'lane':
        #         continue
        #     for i, type_tem in enumerate(poly_feat):
        #         if type_tem != 'center':
        #             break
        #         # if coords.ndim < 2:
        #         #     continue
        #         array_values = poly_feat[type_tem]['coords']
        #         array_values = np.array(array_values)
        #         plt.plot(
        #             array_values[:, 0],
        #             array_values[:, 1],
        #             color={
        #                 "left": "khaki",
        #                 "center": "grey",
        #                 "right": "brown",
        #                 "vertex": "blue",
        #             }[type_tem],
        #             linestyle="--" if type_tem == "center" else "-",
        #             linewidth=3,
        #             label=(f"{type_tem.capitalize()}" if i == 0 else None),
        #         )
        for poly_type, poly_feat in map_annos.items():
            # if poly_feat['pl_type'] != 'lane':
            #     continue
            for i, type_tem in enumerate(poly_feat):
                # if type_tem != 'center':
                #     break
                # if coords.ndim < 2:
                #     continue
                if isinstance(poly_feat.get(type_tem), dict):
                    array_values = poly_feat[type_tem]['coords']
                array_values = np.array(array_values)
                plt.plot(
                    array_values[:, 0],
                    array_values[:, 1],
                    # color={
                    #     "left": "khaki",
                    #     "center": "grey",
                    #     "right": "brown",
                    #     "vertex": "blue",
                    # }[type_tem],
                    linestyle="--" if type_tem == "center" else "-",
                    linewidth=3,
                    # label=(f"{type_tem.capitalize()}" if i == 0 else None),
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
        # plt.savefig(save_path)
        plt.show()
        plt.close()

def _extract_lmdb_feats(timestamp, obs_feats, vector_feats):
    HIS_FRAMES = 10
    FUTURE_FRAMES = 30
    # obs_feats = json.loads(decompress(dynamic_lmdb.read(timestamp)))
    frame = all_keys[timestamp]
    map_ts = obs_feats["vecmap_ts"]

    curr_obs_anno = processed_annos.setdefault(frame, {}).setdefault(
        "agent_features", {}
    )
    """Extract raw vectormap and agent features from LMDB dataset."""

    frame = all_keys[timestamp]
    map_ts = timestamp

    curr_obs_anno = processed_annos.setdefault(frame, {}).setdefault(
        "agent_features", {}
    )
    for obs_feat in obs_feats.get("obstacles", []):
        obs_id = obs_feat["id"]
        hist_traj = torch.tensor(
            [
                [pt[k] for k in ["x", "y", "v", "yaw", "t"]]
                for pt in obs_feat.get("hist_traj", [])
            ],
            dtype=torch.float32,
        )
        fut_traj = torch.tensor(
            [
                [pt[k] for k in ["x", "y", "v", "yaw", "t"]]
                for pt in obs_feat.get("fut_traj", [])
            ],
            dtype=torch.float32,
        )

        # [yujun.zhang]
        his_frame = hist_traj.size(0)

        hist_mask = torch.zeros(HIS_FRAMES)
        hist_mask[-his_frame:] = 1
        padding_traj = torch.zeros(HIS_FRAMES - his_frame, 5)
        hist_traj = torch.cat((padding_traj, hist_traj))
        fut_frame = fut_traj.size(0)
        if fut_frame > FUTURE_FRAMES:
            continue
        fut_mask = torch.zeros(FUTURE_FRAMES)
        fut_mask[:fut_frame] = 1
        padding_traj = torch.zeros(
            FUTURE_FRAMES - fut_frame, 5
        )
        fut_traj = torch.cat((fut_traj, padding_traj))
        curr_obs_anno[obs_id] = {"hist_valid_mask": hist_mask}
        curr_obs_anno[obs_id]["fut_valid_masks"] = fut_mask
        # [yujun.zhang] add ends
        curr_obs_anno[obs_id]["hist"] = hist_traj
        curr_obs_anno[obs_id]["fut"] = fut_traj
        curr_obs_anno[obs_id]["type"] = obs_feat["type"]
        curr_obs_anno[obs_id]["category"] = obs_feat["category"]
        curr_obs_anno[obs_id]["lateral"] = obs_feat["lateral"]

    # start wrapping up map features
    # vector_feats = json.loads(decompress(static_lmdb.read(map_ts)))
    curr_map_anno = processed_annos[frame].setdefault(
        "map_features",
        {},
    )

    for curr_poly in vector_feats.get("polygons", []):
        curr_poly_id = curr_poly.get("id", "")
        curr_poly_type = curr_poly.get("pl_type", 0)
        curr_arrow_type = curr_poly.get("arrow_type", 0)
        curr_map_anno.setdefault(
            curr_poly_id,
            {},
        )
        for curr_line in curr_poly.get("lines", []):
            coords = np.array(
                [
                    [
                        pt["x"],
                        pt["y"],
                        pt["yaw"],
                        pt["magnitude"],
                    ]
                    for pt in curr_line.get("pts", [])
                ]
            )
            pt_type_str = [
                "left",
                "center",
                "right",
                "vertex",
                "road_bdy",
                "lane_bdy",
            ][curr_line["pt_type"]]
            curr_map_anno[curr_poly_id][pt_type_str] = {}
            curr_map_anno[curr_poly_id][pt_type_str]["coords"] = coords
            curr_patterns = curr_line.get("line_patterns", [])
            curr_map_anno[curr_poly_id][pt_type_str][
                "line_patterns"
            ] = curr_patterns

        curr_map_anno[curr_poly_id]["pl_type"] = [
            "lane",
            "junction",
            "crosswalk",
            "boundary",
            "lane_boundary",
        ][curr_poly_type]
        # 10.14 added
        curr_map_anno[curr_poly_id]["arrow_type"] = curr_arrow_type


if __name__ == "__main__":
    dataset_path = Path(
        "/home/user/Projects/raw_data/data/20240716-063853_72_2024_10_11_14_54_48_804")
    obs_feats, vectormap_feats, data_store = generate_source(dataset_path)

    root = "/home/user/Projects/raw_data/data/20240716-063853_72_2024_10_11_14_54_48_804"
    obs_name = "lmdb_obstacles"
    vec_name = "lmdb_vectormap"
    start = 0
    end = 100
    interval = 10
    save_dir = "../output/viz"
    # vis_1_ts(obs_feats, vectormap_feats, data_store)
    # 调用生成可视化的函数
    generate_lmdb_pack_viz(
        curr_pack_data = processed_annos,
        start_idx=start,
        end_idx=end,
        interval=interval,
        save_dir=save_dir,
    )
