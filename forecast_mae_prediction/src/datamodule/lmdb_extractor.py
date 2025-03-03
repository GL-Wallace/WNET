import traceback
import numpy as np
import torch
import json
import os
from zlib import decompress
from tqdm import tqdm
from pathlib import Path
from datamodule.traj_pred_dataset import LmdbWrapper


class lmdb_extractor:
    def __init__(
            self,
            radius: float = 150,
            save_path: Path = None,
            remove_outlier_actors: bool = True,
            obs_lmdb_name: str = "lmdb_obstacles",
            vectormap_lmdb_name: str = "lmdb_vectormap",
            key_file_name: str = "timestamp.txt",
            downsample_gap: int = 10,
    ) -> None:
        self.save_path = save_path
        self.obs_lmdb_name = obs_lmdb_name
        self.vectormap_lmdb_name = vectormap_lmdb_name
        self.key_file_name = key_file_name
        self.radius = radius
        self.remove_outlier_actors = remove_outlier_actors
        self.all_keys = {}
        self.FUTURE_FRAMES = 30
        self.HIS_FRAMES = 10
        self.EGO_ID = 88888888
        self.dynamic_lmdb = None
        self.static_lmdb = None
        self.downsample_gap = downsample_gap
        self.data = []
        self.valid_ts = []
        X, Y, V, YAW, self.T = 0, 1, 2, 3, 4

    def save(self, file: Path):
        assert self.save_path is not None
        try:
            self._preprocess_data(file)
        except Exception:
            print(traceback.format_exc())
            print("found error while extracting data from {}".format(file))
        
        for idx, current_data in enumerate(self.data):
            ts = self.valid_ts[idx]
            data_name = f"{ts}.pt"
            save_file = os.path.join(self.save_path, data_name)
            torch.save(current_data, save_file)


    def _preprocess_data(self, raw_path: str):
        " Preprocess LMDB dataset to cache data. "

        obs_key_path = os.path.join(raw_path, self.obs_lmdb_name, self.key_file_name)
        dynamic_lmdb_path = os.path.join(raw_path, self.obs_lmdb_name)
        static_lmdb_path = os.path.join(raw_path, self.vectormap_lmdb_name)

        assert os.path.exists(
            dynamic_lmdb_path
        ), f"LMDB file {dynamic_lmdb_path} does not exist."
        assert os.path.exists(
            static_lmdb_path
        ), f"LMDB file {static_lmdb_path} does not exist."

        self.dynamic_lmdb = LmdbWrapper(
            dynamic_lmdb_path, file_type="gt"
        )
        self.static_lmdb = LmdbWrapper(
            static_lmdb_path, file_type="gt"
        )
        assert os.path.exists(
            obs_key_path
        ), f"Timestamp key file {obs_key_path} does not exist."
        with open(obs_key_path, "r", encoding="utf-8") as file:
            for line in file:
                stripped_line = line.strip()  # 移除行首和行尾的空白字符。
                if stripped_line:
                    frame, ts = stripped_line.split(",")
                    self.all_keys[ts] = int(frame)
        all_time_stamp = list(self.all_keys.keys())
        if len(all_time_stamp) == 0:
            return
        all_time_stamp = sorted(all_time_stamp, key=float)

        for idx in tqdm(range(0, len(all_time_stamp), self.downsample_gap)):
            result = self._extract_lmdb_feats(all_time_stamp[idx])
            if result is not None:
                self.data.append(result)
                self.valid_ts.append(all_time_stamp[idx])

    def _extract_lmdb_feats(self, timestamp: str):
        """Extract raw vectormap and agent features from LMDB dataset."""
        data = self.dynamic_lmdb.read(timestamp)
        if data is None:
            print(f"No data found in this timestamp: {timestamp}, type: {type(timestamp)}")
            return None
        elif self.static_lmdb.read(timestamp) is None:
            return None
        map_ts = timestamp
        obs_feats = json.loads(decompress(data))
        agents = []
        all_hist_trajs = []
        all_hist_v = []
        all_hist_yaw = []
        all_fut_trajs = []
        all_fut_v = []
        all_fut_yaw = []
        x_attr_list = []
        x_padding_mask = []

        for obs_feat in obs_feats.get("obstacles", []):
            obs_id = obs_feat["id"]
            agents.append(obs_id)  # Store agent ID
            x_attr_list.append([obs_feat["type"]])
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

            his_frame = hist_traj.size(0)
            if his_frame > self.HIS_FRAMES:  # 如果帧数超过预定义的最大帧数，跳过此障碍物。
                continue
            hist_mask = torch.ones(self.HIS_FRAMES, dtype=torch.bool)
            hist_mask[-his_frame:] = False  # 这个是x——padding mask
            padding_traj = torch.zeros(self.HIS_FRAMES - his_frame, self.T + 1)
            hist_traj = torch.cat((padding_traj, hist_traj))  # 此处就有完整的帧数的数据

            fut_frame = fut_traj.size(0)
            if fut_frame > self.FUTURE_FRAMES:
                continue
            fut_mask = torch.ones(self.FUTURE_FRAMES, dtype=torch.bool)
            fut_mask[:fut_frame] = False
            padding_traj = torch.zeros(
                self.FUTURE_FRAMES - fut_frame, self.T + 1
            )
            fut_traj = torch.cat((fut_traj, padding_traj))

            x_padding_mask_temp = torch.cat((hist_mask, fut_mask))
            x_padding_mask.append(x_padding_mask_temp)

            hist_traj_filled = hist_traj[:, :2]
            hist_v_filled = hist_traj[:, 2:3]
            hist_yaw_filled = hist_traj[:, 3:4]
            fut_traj_filled = fut_traj[:, :2]
            fut_v_filled = fut_traj[:, 2:3]
            fut_yaw_filled = fut_traj[:, 3:4]

            # Store padded trajectories
            all_hist_trajs.append(hist_traj_filled)
            all_hist_v.append(hist_v_filled)
            all_hist_yaw.append(hist_yaw_filled)
            all_fut_trajs.append(fut_traj_filled)
            all_fut_v.append(fut_v_filled)
            all_fut_yaw.append(fut_yaw_filled)

        if all_hist_trajs and all_fut_trajs:  # If there are any trajectories
            all_hist_trajs_tensor = torch.stack(all_hist_trajs)
            all_hist_v_tensor = torch.stack(all_hist_v)
            all_hist_yaw_tensor = torch.stack(all_hist_yaw)
            all_fut_trajs_tensor = torch.stack(all_fut_trajs)
            all_fut_v_tensor = torch.stack(all_fut_v)
            all_fut_yaw_tensor = torch.stack(all_fut_yaw)
            x_padding_mask_tensor = torch.stack(x_padding_mask)
            x_attr_tensor = torch.tensor(x_attr_list, dtype=torch.int32)

        target_id = self.EGO_ID
        num_nodes = len(agents)
        if all_hist_trajs_tensor.size(0) == all_fut_trajs_tensor.size(0) == len(agents):
            # x_attr_tensor = torch.tensor(x_attr_list, dtype=torch.float32)  # Shape: [N, 3]
            if target_id in agents:
                target_idx = agents.index(target_id)
                # Swap the target agent's data to the 0th position
                all_hist_trajs_tensor[[0, target_idx]] = all_hist_trajs_tensor[[target_idx, 0]]
                all_hist_v_tensor[[0, target_idx]] = all_hist_v_tensor[[target_idx, 0]]
                all_hist_yaw_tensor[[0, target_idx]] = all_hist_yaw_tensor[[target_idx, 0]]
                all_fut_trajs_tensor[[0, target_idx]] = all_fut_trajs_tensor[[target_idx, 0]]
                agents[0], agents[target_idx] = agents[target_idx], agents[0]
                x_attr_tensor[[0, target_idx]] = x_attr_tensor[[target_idx, 0]]
                x_padding_mask_tensor[[0, target_idx]] = x_padding_mask_tensor[[target_idx, 0]]

        x_positions = all_hist_trajs_tensor
        x_angles = torch.cat((all_hist_yaw_tensor, all_fut_yaw_tensor), dim=1).squeeze(-1)
        x_velocity = torch.cat((all_hist_v_tensor, all_fut_v_tensor), dim=1).squeeze(-1)
        x_centers = x_positions[:, 9].clone()
        origin = x_centers[0]
        x_velocity_diff = x_velocity[:, :10].clone()
        x_velocity_diff[:, 1:10] = x_velocity_diff[:, 1:10] - x_velocity_diff[:, :9]
        x_velocity_diff[:, 0] = torch.zeros(num_nodes)
        theta = all_hist_yaw_tensor[0][9].clone().detach()

        vector_feats = json.loads(decompress(self.static_lmdb.read(map_ts)))

        # 将 lane_positions 转换为 PyTorch Tensor
        lane_positions = []
        lane_angles = []
        lane_attr_list = []
        line_point_counts = []
        line_centers = []

        for curr_poly in vector_feats.get("polygons", []):
            curr_poly_id = curr_poly.get("id", "")

            for curr_line in curr_poly.get("lines", []):
                # 1:"center line", 4:"road_boundary", 5:"lane_boundary",
                if curr_line["pt_type"] not in (1, 4, 5):
                    continue
                coords = np.array(
                    [
                        [pt["x"], pt["y"], pt["yaw"], pt["magnitude"]]
                        for pt in curr_line.get("pts", [])
                    ]
                )
                xy_coords = coords[:, :2]
                yaw_coords = coords[:, 2]

                lane_positions.append(xy_coords)
                lane_attr_list.append(curr_line["pt_type"])
                point_count = len(coords)
                line_point_counts.append(point_count)

                # 计算中心点坐标，以及中心点坐标
                if point_count % 2 == 1:
                    center_index = point_count // 2
                    center_x, center_y = xy_coords[center_index]
                    lane_angles.append(yaw_coords[center_index])
                else:
                    center_index1 = point_count // 2 - 1
                    center_index2 = point_count // 2
                    center_x = (xy_coords[center_index1, 0] + xy_coords[center_index2, 0]) / 2
                    center_y = (xy_coords[center_index1, 1] + xy_coords[center_index2, 1]) / 2
                    lane_angles.append(torch.atan2(
                        xy_coords[center_index1, 1] - xy_coords[center_index2, 1],
                        xy_coords[center_index1, 0] - xy_coords[center_index2, 0],
                    ))
                line_centers.append([center_x, center_y])

        lane_positions = torch.tensor(np.array(lane_positions), dtype=torch.float32)
        lane_attrs = torch.tensor(np.array(lane_attr_list), dtype=torch.int32)
        lanes_centers = torch.tensor(np.array(line_centers), dtype=torch.float32)
        lanes_angles = torch.tensor(np.array(lane_angles), dtype=torch.float32)
        padding_mask = torch.isnan(lane_positions[:, :, 0]) | torch.isnan(lane_positions[:, :, 1])

        return {
            "x": x_positions,
            "y": all_fut_trajs_tensor,
            "x_attr": x_attr_tensor,
            "x_positions": x_positions,
            "x_centers": x_centers,
            "x_angles": x_angles,
            "x_velocity": x_velocity,
            "x_velocity_diff": x_velocity_diff,
            "x_padding_mask": x_padding_mask_tensor,
            "track_id": target_id,
            "origin": origin,
            "theta": theta,
            "lane_positions": lane_positions,
            "lane_attr": lane_attrs,
            "lane_centers": lanes_centers,
            "lane_angles": lanes_angles,
            "lane_padding_mask": padding_mask
        }