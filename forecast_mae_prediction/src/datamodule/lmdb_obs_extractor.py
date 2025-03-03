import traceback
import numpy as np
import torch
import json
import os
from zlib import decompress
from tqdm import tqdm
from pathlib import Path
from forecast_mae_prediction.src.datamodule.traj_pred_dataset import LmdbWrapper


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
            for idx, solo_agent in enumerate(current_data):
                ts = solo_agent['timestamp']
                agent = solo_agent['agents'][idx]
                data_name = f"{ts}_{agent}.pt"
                save_file = os.path.join(self.save_path, data_name)
                torch.save(solo_agent, save_file)
        # for idx, current_data in enumerate(self.data[0]):
        #     ts = current_data['timestamp']
        #     agent = current_data['focal_id']
        #     data_name = f"{ts}_{agent}.pt"
        #     save_file = os.path.join(self.save_path, data_name)
        #     torch.save(current_data, save_file)

    def _preprocess_data(self, raw_path: str):
        " Preprocess LMDB dataset to cache data. "

        obs_key_path = os.path.join(raw_path, self.obs_lmdb_name, self.key_file_name)
        dynamic_lmdb_path = os.path.join(raw_path, self.obs_lmdb_name)
        static_lmdb_path = os.path.join(raw_path, self.vectormap_lmdb_name)
        fts_lmdb_path = os.path.join(raw_path, "lmdb_features")
        assert os.path.exists(dynamic_lmdb_path), f"LMDB file {dynamic_lmdb_path} does not exist."
        assert os.path.exists(static_lmdb_path), f"LMDB file {static_lmdb_path} does not exist."
        self.dynamic_lmdb = LmdbWrapper(dynamic_lmdb_path, file_type="gt")
        self.static_lmdb = LmdbWrapper(static_lmdb_path, file_type="gt")
        self.feature_lmdb = LmdbWrapper(uri=fts_lmdb_path, file_type="gt")
        assert os.path.exists(obs_key_path), f"Timestamp key file {obs_key_path} does not exist."

        with open(obs_key_path, "r", encoding="utf-8") as file:
            for line in file:
                stripped_line = line.strip()
                if stripped_line:
                    frame, ts = stripped_line.split(",")
                    self.all_keys[ts] = int(frame)
        all_time_stamp = list(self.all_keys.keys())
        if len(all_time_stamp) == 0:
            return
        all_time_stamp = sorted(all_time_stamp, key=float)

        for idx in tqdm(range(0, len(all_time_stamp), self.downsample_gap)):
            result = self._extract_lmdb_featsV1(all_time_stamp[idx])
            if result is not None:
                self.data.append(result)
                self.valid_ts.append(all_time_stamp[idx])

    def _extract_lmdb_feats(self, timestamp: str):
        """Extract raw vectormap and agent features from LMDB dataset."""

        obs_feats = json.loads(decompress(self.dynamic_lmdb.read(timestamp)))
        vector_feats = json.loads(decompress(self.static_lmdb.read(timestamp)))


        if self.static_lmdb.read(timestamp) is None or self.static_lmdb.read(timestamp) is None:
            print(f"No data found in this timestamp: {timestamp}, type: {type(timestamp)}")
            return None

        obs_data, agents = [], []
        all_hist_trajs, all_hist_v, all_hist_yaw = [], [], []
        all_fut_trajs, all_fut_v, all_fut_yaw = [], [], []
        x_attr_list, x_padding_mask = [], []

        for obs_feat in obs_feats.get("obstacles", []):
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
            fut_frame = fut_traj.size(0)
            if his_frame > self.HIS_FRAMES or fut_frame > self.FUTURE_FRAMES:
                continue

            agents.append(obs_feat["id"])
            hist_mask = torch.ones(self.HIS_FRAMES, dtype=torch.bool)
            hist_mask[-his_frame:] = False
            padding_traj = torch.zeros(self.HIS_FRAMES - his_frame, self.T + 1)
            hist_traj = torch.cat((padding_traj, hist_traj))

            fut_mask = torch.ones(self.FUTURE_FRAMES, dtype=torch.bool)
            fut_mask[:fut_frame] = False
            padding_traj = torch.zeros(self.FUTURE_FRAMES - fut_frame, self.T + 1)
            fut_traj = torch.cat((fut_traj, padding_traj))
            x_padding_mask_temp = torch.cat((hist_mask, fut_mask))

            # Extract trajectory components
            x_padding_mask.append(x_padding_mask_temp)
            all_hist_trajs.append(hist_traj[:, :2])
            all_hist_v.append(hist_traj[:, 2:3])
            all_hist_yaw.append(hist_traj[:, 3:4])
            all_fut_trajs.append(fut_traj[:, :2])
            all_fut_v.append(fut_traj[:, 2:3])
            all_fut_yaw.append(fut_traj[:, 3:4])

            all_hist_trajs_tensor = torch.stack(all_hist_trajs)
            all_hist_v_tensor = torch.stack(all_hist_v)
            all_hist_yaw_tensor = torch.stack(all_hist_yaw)
            all_fut_trajs_tensor = torch.stack(all_fut_trajs)
            all_fut_v_tensor = torch.stack(all_fut_v)
            all_fut_yaw_tensor = torch.stack(all_fut_yaw)
            x_padding_mask_tensor = torch.stack(x_padding_mask)
            x_attr_tensor = torch.tensor(x_attr_list, dtype=torch.int32)

        ego_id = self.EGO_ID

        # rotate
        for i, agent_id in enumerate(agents):
            focals = agents.copy()
            focal_id = agents[i]
            hist_traj_tensor = all_hist_trajs_tensor
            fut_traj_tensor = all_fut_trajs_tensor
            hist_yaw_tensor = all_hist_yaw_tensor
            fut_yaw_tensor = all_fut_yaw_tensor

            origin = torch.tensor([hist_traj_tensor[i, 9, 0], hist_traj_tensor[i, 9, 1]], dtype=torch.float32)
            theta = torch.tensor([hist_yaw_tensor[i][9]], dtype=torch.float)
            rotate_mat = torch.tensor(
                [
                    [torch.cos(theta), -torch.sin(theta)],
                    [torch.sin(theta), torch.cos(theta)],
                ],
            )
            hist_yaw_tensor = (hist_yaw_tensor - theta + np.pi) % (2 * np.pi) - np.pi
            fut_yaw_tensor = (fut_yaw_tensor - theta + np.pi) % (2 * np.pi) - np.pi

            # Create mask for hist_traj_tensor
            hist_mask = ~x_padding_mask_tensor[:, :10].unsqueeze(-1)  # [N, 10, 1]
            hist_traj_tensor = torch.where(
                hist_mask,
                torch.matmul(hist_traj_tensor - origin, rotate_mat),
                hist_traj_tensor
            )

            # Create mask for fut_traj_tensor
            fut_mask = ~x_padding_mask_tensor[:, 10:40].unsqueeze(-1)  # [N, 30, 1]
            fut_traj_tensor = torch.where(
                fut_mask,
                torch.matmul(fut_traj_tensor - origin, rotate_mat),
                fut_traj_tensor
            )

            # Swap the target agent's data to the 0th position
            target_idx = agents.index(ego_id)
            all_hist_trajs_tensor[[0, target_idx]] = all_hist_trajs_tensor[[target_idx, 0]]
            all_hist_v_tensor[[0, target_idx]] = all_hist_v_tensor[[target_idx, 0]]
            all_fut_v_tensor[[0, target_idx]] = all_fut_v_tensor[[target_idx, 0]]
            all_hist_yaw_tensor[[0, target_idx]] = all_hist_yaw_tensor[[target_idx, 0]]
            all_fut_yaw_tensor[[0, target_idx]] = all_fut_yaw_tensor[[target_idx, 0]]
            all_fut_trajs_tensor[[0, target_idx]] = all_fut_trajs_tensor[[target_idx, 0]]
            focals[0], focals[target_idx] = focals[target_idx], focals[0]
            x_attr_tensor[[0, target_idx]] = x_attr_tensor[[target_idx, 0]]
            x_padding_mask_tensor[[0, target_idx]] = x_padding_mask_tensor[[target_idx, 0]]

            x_positions = hist_traj_tensor
            x_angles = torch.cat((hist_yaw_tensor, fut_yaw_tensor), dim=1).squeeze(-1)
            x_velocity = torch.cat((all_hist_v_tensor, all_fut_v_tensor), dim=1).squeeze(-1)
            x_centers = x_positions[:, 9].clone()
            x_velocity_diff = x_velocity[:, :10].clone()
            x_velocity_diff[:, 1:10] = x_velocity_diff[:, 1:10] - x_velocity_diff[:, :9]
            x_velocity_diff[:, 0] = torch.zeros(len(agents))

            lane_positions, lane_attr, lane_centers, lane_angles, padding_mask = self.get_lane_features(vector_feats,
                                                                                                        origin,
                                                                                                        rotate_mat)
            data = {
                "x": x_positions,
                "y": fut_traj_tensor,
                "x_attr": x_attr_tensor,
                "x_positions": x_positions,
                "x_centers": x_centers,
                "x_angles": x_angles,
                "x_velocity": x_velocity,
                "x_velocity_diff": x_velocity_diff,
                "x_padding_mask": x_padding_mask_tensor,
                "focal_id": focal_id,
                "track_id": ego_id,
                "origin": origin,
                "theta": theta,
                "lane_positions": lane_positions,
                "lane_attr": lane_attr,
                "lane_centers": lane_centers,
                "lane_angles": lane_angles,
                "lane_padding_mask": padding_mask,
                "agents": focals,
                "timestamp": timestamp
            }
            obs_data.append(data)

        return obs_data

    def get_lane_features(self, vector_feats, origin, rotate_mat):
        lane_positions, lane_angles, lane_attr_list = [], [], []

        for curr_poly in vector_feats.get("polygons", []):
            curr_poly_id = curr_poly.get("id", "")

            for curr_line in curr_poly.get("lines", []):
                # 1:"center line", 4:"road_boundary", 5:"lane_boundary",
                if curr_line["pt_type"] not in (0, 1, 2, 4, 5):
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
                lane_angles.append(yaw_coords)
                lane_attr_list.append(curr_line["pt_type"])

        lane_positions = torch.tensor(np.array(lane_positions), dtype=torch.float32)
        lane_positions = torch.matmul(lane_positions - origin, rotate_mat)
        lane_angles = torch.atan2(
            lane_positions[:, 16, 1] - lane_positions[:, 15, 1],
            lane_positions[:, 16, 0] - lane_positions[:, 15, 0]
        )
        lane_centers = lane_positions[:, 15, :2]
        lane_attr = torch.tensor(np.array(lane_attr_list), dtype=torch.int32)
        lanes_angles = torch.tensor(np.array(lane_angles), dtype=torch.float32)
        padding_mask = torch.isnan(lane_positions[:, :, 0]) | torch.isnan(lane_positions[:, :, 1])

        return lane_positions, lane_attr, lane_centers, lanes_angles, padding_mask


    def _extract_lmdb_featsV1(self, timestamp: str):
        """Extract raw vectormap and agent features from LMDB dataset."""

        obs_feats = json.loads(decompress(self.dynamic_lmdb.read(timestamp)))
        vector_feats = json.loads(decompress(self.static_lmdb.read(timestamp)))
        raw_feats = json.loads(decompress(self.feature_lmdb.read(timestamp)))

        if self.static_lmdb.read(timestamp) is None or self.static_lmdb.read(timestamp) is None:
            print(f"No data found in this timestamp: {timestamp}, type: {type(timestamp)}")
            return None

        obs_data, agents = [], []
        all_hist_trajs, all_hist_v, all_hist_yaw = [], [], []
        all_fut_trajs, all_fut_v, all_fut_yaw = [], [], []
        x_attr_list, x_padding_mask = [], []

        for obs_feat in obs_feats.get("obstacles", []):
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
            fut_frame = fut_traj.size(0)
            if his_frame > self.HIS_FRAMES or fut_frame > self.FUTURE_FRAMES:
                continue

            agents.append(obs_feat["id"])
            hist_mask = torch.ones(self.HIS_FRAMES, dtype=torch.bool)
            hist_mask[-his_frame:] = False
            padding_traj = torch.zeros(self.HIS_FRAMES - his_frame, self.T + 1)
            hist_traj = torch.cat((padding_traj, hist_traj))

            fut_mask = torch.ones(self.FUTURE_FRAMES, dtype=torch.bool)
            fut_mask[:fut_frame] = False
            padding_traj = torch.zeros(self.FUTURE_FRAMES - fut_frame, self.T + 1)
            fut_traj = torch.cat((fut_traj, padding_traj))
            x_padding_mask_temp = torch.cat((hist_mask, fut_mask))

            # Extract trajectory components
            x_padding_mask.append(x_padding_mask_temp)
            all_hist_trajs.append(hist_traj[:, :2])
            all_hist_v.append(hist_traj[:, 2:3])
            all_hist_yaw.append(hist_traj[:, 3:4])
            all_fut_trajs.append(fut_traj[:, :2])
            all_fut_v.append(fut_traj[:, 2:3])
            all_fut_yaw.append(fut_traj[:, 3:4])

            all_hist_trajs_tensor = torch.stack(all_hist_trajs)
            all_hist_v_tensor = torch.stack(all_hist_v)
            all_hist_yaw_tensor = torch.stack(all_hist_yaw)
            all_fut_trajs_tensor = torch.stack(all_fut_trajs)
            all_fut_v_tensor = torch.stack(all_fut_v)
            all_fut_yaw_tensor = torch.stack(all_fut_yaw)
            x_padding_mask_tensor = torch.stack(x_padding_mask)
            x_attr_tensor = torch.tensor(x_attr_list, dtype=torch.int32)

        ego_id = self.EGO_ID

        # rotate
        for i, agent_id in enumerate(agents):
            focals = agents.copy()
            focal_id = agents[i]
            hist_traj_tensor = all_hist_trajs_tensor
            fut_traj_tensor = all_fut_trajs_tensor
            hist_yaw_tensor = all_hist_yaw_tensor
            fut_yaw_tensor = all_fut_yaw_tensor

            origin = torch.tensor([hist_traj_tensor[i, 9, 0], hist_traj_tensor[i, 9, 1]], dtype=torch.float32)
            theta = torch.tensor([hist_yaw_tensor[i][9]], dtype=torch.float)
            rotate_mat = torch.tensor(
                [
                    [torch.cos(theta), -torch.sin(theta)],
                    [torch.sin(theta), torch.cos(theta)],
                ],
            )
            hist_yaw_tensor = (hist_yaw_tensor - theta + np.pi) % (2 * np.pi) - np.pi
            fut_yaw_tensor = (fut_yaw_tensor - theta + np.pi) % (2 * np.pi) - np.pi

            # Create mask for hist_traj_tensor
            hist_mask = ~x_padding_mask_tensor[:, :10].unsqueeze(-1)  # [N, 10, 1]
            hist_traj_tensor = torch.where(
                hist_mask,
                torch.matmul(hist_traj_tensor - origin, rotate_mat),
                hist_traj_tensor
            )

            # Create mask for fut_traj_tensor
            fut_mask = ~x_padding_mask_tensor[:, 10:40].unsqueeze(-1)  # [N, 30, 1]
            fut_traj_tensor = torch.where(
                fut_mask,
                torch.matmul(fut_traj_tensor - origin, rotate_mat),
                fut_traj_tensor
            )

            # Swap the target agent's data to the 0th position
            target_idx = agents.index(ego_id)
            all_hist_trajs_tensor[[0, target_idx]] = all_hist_trajs_tensor[[target_idx, 0]]
            all_hist_v_tensor[[0, target_idx]] = all_hist_v_tensor[[target_idx, 0]]
            all_fut_v_tensor[[0, target_idx]] = all_fut_v_tensor[[target_idx, 0]]
            all_hist_yaw_tensor[[0, target_idx]] = all_hist_yaw_tensor[[target_idx, 0]]
            all_fut_yaw_tensor[[0, target_idx]] = all_fut_yaw_tensor[[target_idx, 0]]
            all_fut_trajs_tensor[[0, target_idx]] = all_fut_trajs_tensor[[target_idx, 0]]
            focals[0], focals[target_idx] = focals[target_idx], focals[0]
            x_attr_tensor[[0, target_idx]] = x_attr_tensor[[target_idx, 0]]
            x_padding_mask_tensor[[0, target_idx]] = x_padding_mask_tensor[[target_idx, 0]]

            x_positions = hist_traj_tensor
            x_angles = torch.cat((hist_yaw_tensor, fut_yaw_tensor), dim=1).squeeze(-1)
            x_velocity = torch.cat((all_hist_v_tensor, all_fut_v_tensor), dim=1).squeeze(-1)
            x_centers = x_positions[:, 9].clone()
            x_velocity_diff = x_velocity[:, :10].clone()
            x_velocity_diff[:, 1:10] = x_velocity_diff[:, 1:10] - x_velocity_diff[:, :9]
            x_velocity_diff[:, 0] = torch.zeros(len(agents))

            lane_positions, lane_attr, lane_centers, lane_angles, padding_mask = self.get_lane_features(vector_feats,
                                                                                                        origin,
                                                                                                        rotate_mat)
            data = {
                "x": x_positions,
                "y": fut_traj_tensor,
                "x_attr": x_attr_tensor,
                "x_positions": x_positions,
                "x_centers": x_centers,
                "x_angles": x_angles,
                "x_velocity": x_velocity,
                "x_velocity_diff": x_velocity_diff,
                "x_padding_mask": x_padding_mask_tensor,
                "focal_id": focal_id,
                "track_id": ego_id,
                "origin": origin,
                "theta": theta,
                "lane_positions": lane_positions,
                "lane_attr": lane_attr,
                "lane_centers": lane_centers,
                "lane_angles": lane_angles,
                "lane_padding_mask": padding_mask,
                "agents": focals,
                "timestamp": timestamp
            }
            obs_data.append(data)

        return obs_data

    def get_rpe_features(self, input_feats):
        r_pt2pl = torch.tensor(
            input_feats["r_pt2pl"],
            dtype=torch.float32,
        )
        r_a2a = torch.tensor(
            input_feats["r_a2a"],
            dtype=torch.float32,
        )
        r_a2t = torch.tensor(
            input_feats["r_a2t"],
            dtype=torch.float32,
        )
        r_a2pl = torch.tensor(
            input_feats["r_a2pl"],
            dtype=torch.float32,
        )
        rpe_m2pl = torch.tensor(
            input_feats["rpe_m2pl"],
            dtype=torch.float32,
        )
        rpe_m2t = torch.tensor(
            input_feats["rpe_m2t"],
            dtype=torch.float32,
        )
        rpe_m2a = torch.tensor(
            input_feats["rpe_m2a"],
            dtype=torch.float32,
        )

        invalid_mask_pt2pl = torch.tensor(
            input_feats["invalid_mask_pt2pl"],
            dtype=torch.bool,
        )
        invalid_mask_pl2pl = torch.tensor(
            input_feats["invalid_mask_pl2pl"],
            dtype=torch.bool,
        )

        a2a_invalid_mask = torch.tensor(
            input_feats["a2a_invalid_mask"],
            dtype=torch.bool,
        )
        a2pl_invalid_mask = torch.tensor(
            input_feats["a2pl_invalid_mask"],
            dtype=torch.bool,
        )
        a2t_invalid_mask = torch.tensor(
            input_feats["a2t_invalid_mask"],
            dtype=torch.bool,
        )

        invalid_mask_m2t = torch.tensor(
            input_feats["invalid_mask_m2t"],
            dtype=torch.bool,
        )
        invalid_mask_m2pl = torch.tensor(
            input_feats["invalid_mask_m2pl"],
            dtype=torch.bool,
        )
        invalid_mask_m2a = torch.tensor(
            input_feats["invalid_mask_m2a"],
            dtype=torch.bool,
        )

        return {
            "r_pt2pl": r_pt2pl,
            "r_a2a": r_a2a,
            "r_a2t": r_a2t,
            "r_a2pl": r_a2pl,
            "rpe_m2pl": rpe_m2pl,
            "rpe_m2t": rpe_m2t,
            "rpe_m2a": rpe_m2a,
            "invalid_mask_pt2pl": invalid_mask_pt2pl,
            "invalid_mask_pl2pl": invalid_mask_pl2pl,
            "a2a_invalid_mask": a2a_invalid_mask,
            "a2pl_invalid_mask": a2pl_invalid_mask,
            "a2t_invalid_mask": a2t_invalid_mask,
            "invalid_mask_m2t": invalid_mask_m2t,
            "invalid_mask_m2pl": invalid_mask_m2pl,
            "invalid_mask_m2a": invalid_mask_m2a
        }
