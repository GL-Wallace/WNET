###################################################################
#   Copyright (C) 2024 Carizon. All rights reserved.
#
#   Filename    : traj_pred_dataset.py
#   Author      : yunchang.zhang
#   Date        : July 2024
#   Description : Motion Prediction Dataset Base Scripts.
###################################################################

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from zlib import decompress
import math

import torch

import lmdb
import numpy as np
import pandas as pd
import torch
from av2.geometry.interpolate import compute_midpoint_line, interp_arc
from av2.map.map_api import ArgoverseStaticMap
# from timeout_decorator import timeout
from tqdm import tqdm
# from utils import side_to_directed_lineseg, wrap_angle


class LmdbWrapper:
    T = 1024**4
    G = 1024**3
    M = 1024**2
    K = 1024

    DEFAULT_MAPSIZE_DICT = {
        "sync_file": 10 * M,
        "ego_pose": 10 * M,
        "image": 1 * G,
        "gt": 10 * M,
    }

    def __init__(
        self, uri, file_type, mapsize_config=DEFAULT_MAPSIZE_DICT
    ) -> None:
        """
        Initialization method for a generalized LMDB read and write class for
        Carizon driving data.

        Args:
            uri(str): The path of LMDB database.
            file_type(str): LMDB file types to be read, usually we choose "gt".
        """
        self.uri = uri
        self.file_type = file_type
        self.mapsize_config = mapsize_config
        assert self.file_type in self.mapsize_config

        self.kwargs = {
            "readonly": True,
            "lock": False,
            "map_size": self.mapsize_config[self.file_type],
        }
        self.env = self.open_lmdb()
        self.txn = self.env.begin()

    # @timeout(seconds=1800)
    def read(self, idx: Union[int, str]) -> bytes:
        """
        Reading dataset value by a key index

        Args:
            idx(Union[int, str]): The index to a value from LMDB database.
        Returns:
            value(bytes): Encoded LMDB values in bytes.
        """

        idx = "{}".format(idx).encode("ascii")
        return self.txn.get(idx)

    # @timeout(seconds=1800)
    def open_lmdb(self) -> lmdb.Environment:
        """
        Open an LMDB databased and create an environment.

        Returns:
            lmdb_env(lmdb.Environment): An opened lmdb.Environment class.
        """
        return lmdb.open(self.uri, **self.kwargs)

    def close(self) -> None:
        """
        Disable an LMDB environment.

        """
        if self.env is not None:
            self.env.close()
            self.env = None
            self.txn = None

    def __len__(self) -> int:
        """
        The length of a database.
        Returns:
            lmdb_len(int): The length of a LMDB database.
        """
        idx = "{}".format("__len__").encode("ascii")
        lmdb_len = self.txn.get(idx)
        if lmdb_len is None:
            lmdb_len = self.txn.stat()["entries"]
        else:
            lmdb_len = int(lmdb_len)
        return lmdb_len

    def get_keys(self) -> List[Union[str, bytes]]:
        """Get all keys.

        Returns:
            keys(List[Union[str, bytes]]): The list of keys to be queried with.
        """
        try:
            idx = "{}".format("__len__").encode("ascii")
            return range(int(self.txn.get(idx)))
        except Exception:
            # traversal may be slow while too much keys
            keys = []
            for key, _ in self.txn.cursor():
                keys.append(key)
            return keys


class TrajPredLmdbDataset(torch.utils.data.Dataset):
    """A wrapper Dataset to extract obstacle and map features from LMDB."""

    FUTURE_FRAMES = 150
    FPS = 25

    def __init__(
        self,
        lmdb_root: str,
        obs_lmdb_name: str = "lmdb_obstacles",
        vectormap_lmdb_name: str = "lmdb_vectormap",
        key_file_name: str = "timestamp.txt",
    ) -> None:
        """Initialize the dataset.

        Args:
            lmdb_root: The lmdb root directory of valid data/
            obs_lmdb_name: The lmdb file includes basic info of obstacles.
            vectormap_lmdb_name: The lmdb file includes basic info of
                vectormap.
            key_file_name: The text file include mapping of frame and
                timestamp.
        """
        self.processed_annos = {}
        self.dynamic_lmdb_path = os.path.join(lmdb_root, obs_lmdb_name)
        self.static_lmdb_path = os.path.join(lmdb_root, vectormap_lmdb_name)

        assert os.path.exists(
            self.dynamic_lmdb_path
        ), f"LMDB file {self.dynamic_lmdb_path} does not exist."
        assert os.path.exists(
            self.static_lmdb_path
        ), f"LMDB file {self.static_lmdb_path} does not exist."

        self.dynamic_lmdb = LmdbWrapper(
            uri=self.dynamic_lmdb_path, file_type="gt"
        )
        self.static_lmdb = LmdbWrapper(
            uri=self.static_lmdb_path, file_type="gt"
        )
        self.all_keys = {}
        self.invalid_frame = 0
        self._load_keys(lmdb_root, obs_lmdb_name, key_file_name)
        self._preprocess_data()

    def _load_keys(
        self, lmdb_root: str, obs_lmdb_name: str, key_file_name: str
    ) -> None:
        """Load keys for a LMDB dataset.

        Args:
            lmdb_root: The lmdb root directory of valid data/
            obs_lmdb_name: The lmdb file includes basic info of obstacles.
            key_file_name: The text file include mapping of frame and
            timestamp.
        """

        obs_key_path = os.path.join(lmdb_root, obs_lmdb_name, key_file_name)
        assert os.path.exists(
            obs_key_path
        ), f"Timestamp key file {obs_key_path} does not exist."
        with open(obs_key_path, "r", encoding="utf-8") as file:
            for line in file:
                stripped_line = line.strip()
                if stripped_line:
                    frame, ts = stripped_line.split(",")
                    self.all_keys[ts] = int(frame)

    def _preprocess_data(self) -> None:
        """Preprocess LMDB dataset to cache data."""
        for timestamp in tqdm(self.all_keys.keys()):
            time_key = (
                timestamp.decode()
                if not isinstance(timestamp, str)
                else timestamp
            )
            self._extract_lmdb_feats(time_key)

        self._extract_agent_fut_gts()

    def _extract_lmdb_feats(self, timestamp: str) -> None:
        """Extract vectormap and agent features from LMDB dataset."""
        obs_feats = json.loads(decompress(self.dynamic_lmdb.read(timestamp)))
        frame = self.all_keys[timestamp]
        map_ts = obs_feats["vecmap_ts"]
        if self.static_lmdb.read(map_ts) is None:
            self.invalid_frame += 1
            return
        curr_obs_anno = self.processed_annos.setdefault(frame, {}).setdefault(
            "agent_features", {}
        )

        for obs_feat in obs_feats.get("obstacles", []):
            obs_id = obs_feat["id"]
            hist_traj = torch.tensor(
                [
                    [pt[k] for k in ["x", "y", "vx", "vy", "yaw", "t"]]
                    for pt in obs_feat.get("hist_traj", [])
                ],
                dtype=torch.float32,
            )
            curr_obs_anno.setdefault(obs_id, {})["hist"] = hist_traj

        vector_feats = json.loads(decompress(self.static_lmdb.read(map_ts)))
        curr_map_anno = self.processed_annos[frame].setdefault(
            "map_features",
            {"left": [], "center": [], "right": [], "vertex": []},
        )
        for curr_poly in vector_feats.get("polygons", []):
            for curr_line in curr_poly.get("lines", []):
                coords = np.array(
                    [[pt["x"], pt["y"]] for pt in curr_line.get("pts", [])]
                )
                curr_map_anno[
                    ["left", "center", "right", "vertex"][curr_line["pt_type"]]
                ].append(coords)

    def _extract_agent_fut_gts(self) -> None:
        """Extract agent future groundtruth trajectories from LMDB dataset."""
        for frame, annos in tqdm(self.processed_annos.items()):
            agent_features = annos.get("agent_features", {})
            fut_data = torch.zeros((self.FUTURE_FRAMES, 6))

            for agent_id in agent_features:
                agent_features[agent_id]["fut"] = fut_data.clone()

            for next_frame in range(frame + 1, frame + self.FUTURE_FRAMES + 1):
                if next_frame in self.processed_annos:
                    next_frame_features = self.processed_annos[next_frame].get(
                        "agent_features", {}
                    )
                    for agent_id, next_feat in next_frame_features.items():
                        if agent_id in agent_features:
                            last_hist = next_feat["hist"][-1]
                            agent_features[agent_id]["fut"][
                                next_frame - frame - 1, :
                            ] = last_hist

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.processed_annos)

    def __getitem__(self, idx: int) -> dict:
        """Get an indexed item of the dataset."""
        return self.processed_annos[idx]


class ArgoverseIIDataset(torch.utils.data.Dataset):
    """
    Argoverse II motion prediction dataset. This one is to analyze the
    """

    HISTORICAL_STEPS = 50
    FUTURE_STEPS = 60

    # agent features
    AGENT_CATEGORIES = [
        "TRACK_FRAGMENT",
        "UNSCORED_TRACK",
        "SCORED_TRACK",
        "FOCAL_TRACK",
    ]

    AGENT_TYPES = [
        "vehicle",
        "pedestrian",
        "motorcyclist",
        "cyclist",
        "bus",
        "static",
        "background",
        "construction",
        "riderless_bicycle",
        "unknown",
    ]

    # polygon features
    POLYGON_TYPES = ["VEHICLE", "BIKE", "BUS", "PEDESTRIAN"]
    POLYGON_IS_INTERSECT = [True, False, None]
    POLYGON_TO_POLYGON_TYPES = ["NONE", "PRED", "SUCC", "LFET", "RIGHT"]

    # point features
    POINT_TYPES = [
        "DASH_SOLID_YELLOW",
        "DASH_SOLID_WHITE",
        "DASHED_WHITE",
        "DASHED_YELLOW",
        "DOUBLE_SOLID_YELLOW",
        "DOUBLE_SOLID_WHITE",
        "DOUBLE_DASH_YELLOW",
        "DOUBLE_DASH_WHITE",
        "SOLID_YELLOW",
        "SOLID_WHITE",
        "SOLID_DASH_WHITE",
        "SOLID_DASH_YELLOW",
        "SOLID_BLUE",
        "NONE",
        "UNKNOWN",
        "CROSSWALK",
        "CENTERLINE",
    ]

    POINT_SIDES = ["LEFT", "RIGHT", "CENTER"]

    def __init__(
        self,
        data_path: str = "/home/user/dev/argoverse2",
        mode: str = "train",
        dataset_size: int = 1000,
        top_k_points: int = 20,
        top_k_polygons: int = 75,
        transform: Optional[Callable] = None,
        reprocess: bool = True,
    ) -> None:
        """
        Initialization method.

        Args:
            data_path: The path of the parent directory of data.
                    One data_path could contain many specific datasets,
                    such as train, valid and test datasets.
            mode: The name of the dataset directory (train, val, and, test).
            dataset_size: The sample number to be used.
            top_k_points: Upsample or downsample the fixed number of polyline
                        points.
            top_k_polygons: choose the fixed number of polygons by KNN search.
            transform: Optional[Callable] = None,
            reprocess: bool = True,
            transforms: A function transform that takes input sample and its
                        target as entry and returns a transformed version.
        """
        super(ArgoverseIIDataset, self).__init__()

        self.data_path = data_path
        self.mode = mode
        self.dataset_size = dataset_size
        self.transform = transform

        self.process_dir = os.path.join(self.data_path, self.mode, "processed")
        self.num_steps = self.HISTORICAL_STEPS + self.FUTURE_STEPS
        self.dim = 3
        self.top_k_points = top_k_points
        self.top_k_polygons = top_k_polygons
        if not reprocess and (
            not os.path.exists(self.process_dir)
            or len(os.listdir(self.process_dir)) == 0
        ):
            assert "the processed data path is not exitst"
        if reprocess:
            self.raw_dir = os.path.join(self.data_path, self.mode, "raw")
            self.scenario_ids = os.listdir(self.raw_dir)
            os.makedirs(self.process_dir, exist_ok=True)
            self.process()
        self.processed_ids = os.listdir(self.process_dir)

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.processed_ids)

    def __getitem__(self, idx):
        """Get an indexed item of the dataset."""
        processed_id = self.processed_ids[idx]
        data = torch.load(os.path.join(self.process_dir, processed_id))
        return data

    def process(self):
        """Process raw data to cache."""
        for i, raw_file_name in tqdm(enumerate(self.scenario_ids)):
            if i > self.dataset_size:
                break
            df = pd.read_parquet(
                os.path.join(
                    self.raw_dir,
                    raw_file_name,
                    f"scenario_{raw_file_name}.parquet",
                )
            )
            map_dir = Path(os.path.join(self.raw_dir, raw_file_name))

            map_api = ArgoverseStaticMap.from_map_dir(
                map_dir, build_raster=False
            )

            data = dict()
            data["scenario_id"] = raw_file_name
            data["agent_features"] = self.get_agent_features(df)
            data["map_features"] = self.get_map_features(map_api)
            data = self.update_map_feature_dims(data)

            torch.save(
                data,
                os.path.join(self.process_dir, f"{raw_file_name}" + ".pt"),
            )

    def update_map_feature_dims(
        self,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Unify map feature dimenstions across all batches.

        This function applies a KNN algorithm to choose top k nearest
        polygons of a vectormap based on distances between an ego vehicle and
        all polygons, if the number of polygons is greater than k. Otherwise,
        it will patch all zeros to K.

        Args:
            data(Dict[str, Any]): Original map features.
        Returns:
            data_fixed(Dict[str, Any]): Map features with fixed dimension.
        """
        av_index = data["agent_features"]["av_index"]
        av_origin = data["agent_features"]["position"][
            av_index, self.HISTORICAL_STEPS, :2
        ]
        polygon_features = data["map_features"]["map_polygon"]["position"]
        polygon_types = data["map_features"]["map_polygon"]["type"]
        polygon_masks = data["map_features"]["map_polygon"]["valid_mask"]

        point_features = data["map_features"]["map_point"]["position"]
        point_heights = data["map_features"]["map_point"]["height"]
        point_magnitudes = data["map_features"]["map_point"]["magnitude"]
        point_masks = data["map_features"]["map_point"]["valid_mask"]

        polygon_coords = polygon_features[..., :2]
        dist_poly = torch.norm(
            polygon_coords - av_origin,
            p=2,
            dim=-1,
        )

        _, topk_ind = torch.topk(-dist_poly, self.top_k_polygons)

        data["map_features"]["map_polygon"]["position"] = polygon_features[
            topk_ind
        ]
        data["map_features"]["map_polygon"]["type"] = polygon_types[topk_ind]
        data["map_features"]["map_polygon"]["valid_mask"] = polygon_masks[
            topk_ind
        ]
        data["map_features"]["map_point"]["position"] = point_features[
            topk_ind
        ]
        data["map_features"]["map_point"]["magnitude"] = point_magnitudes[
            topk_ind
        ]
        data["map_features"]["map_point"]["height"] = point_heights[topk_ind]
        data["map_features"]["map_point"]["valid_mask"] = point_masks[topk_ind]
        return data

    def get_map_features(
        self,
        map_api: ArgoverseStaticMap,
    ):
        """Extract map features from map-api.
        Args:
            map_api: The av2 map api to be queried.
        Returns:
            data: Map features in dictionary format.
        """
        lane_segment_ids = map_api.get_scenario_lane_segment_ids()
        cross_walk_ids = list(map_api.vector_pedestrian_crossings.keys())
        polygon_ids = lane_segment_ids + cross_walk_ids
        num_polygons = max(
            len(lane_segment_ids) + len(cross_walk_ids) * 2,
            self.top_k_polygons,
        )

        # initialization
        polygon_position = torch.zeros(
            num_polygons, self.dim, dtype=torch.float
        )
        polygon_mask = torch.zeros(num_polygons, dtype=torch.uint8)
        polygon_height = torch.zeros(num_polygons, dtype=torch.float)
        polygon_type = torch.zeros(num_polygons, dtype=torch.uint8)
        point_position = torch.zeros(
            num_polygons, self.top_k_points, self.dim, dtype=torch.float
        )
        point_mask = torch.zeros(
            num_polygons, self.top_k_points, dtype=torch.uint8
        )
        point_magnitude = torch.zeros(
            num_polygons, self.top_k_points, dtype=torch.float
        )
        point_height = torch.zeros(
            num_polygons, self.top_k_points, dtype=torch.float
        )
        polygon_mask[: len(lane_segment_ids) + len(cross_walk_ids) * 2] = 1
        point_mask[
            : len(lane_segment_ids) + len(cross_walk_ids) * 2,
            : self.top_k_points,
        ] = 1

        for lane_segment in map_api.get_scenario_lane_segments():
            lane_segment_idx = polygon_ids.index(lane_segment.id)
            centerline = torch.from_numpy(
                map_api.get_lane_segment_centerline(lane_segment.id)
            )

            norm_centerline = interp_arc(self.top_k_points + 1, centerline)

            # polygon vector features
            polygon_position[lane_segment_idx, :2] = norm_centerline[0, :2]
            polygon_position[lane_segment_idx, 2] = torch.atan2(
                norm_centerline[1, 1] - norm_centerline[0, 1],
                norm_centerline[1, 0] - norm_centerline[0, 0],
            )

            polygon_height[lane_segment_idx] = (
                norm_centerline[1, 2] - norm_centerline[0, 2]
            )

            polygon_type[lane_segment_idx] = self.POLYGON_TYPES.index(
                lane_segment.lane_type.value
            )

            center_vectors = norm_centerline[1:] - norm_centerline[:-1]
            point_poses = norm_centerline[:-1, :2]
            point_heading = torch.atan2(
                center_vectors[:, 1], center_vectors[:, 0]
            )
            point_position[lane_segment_idx] = torch.cat(
                [
                    point_poses,
                    point_heading.unsqueeze(-1),
                ],
                dim=-1,
            )

            point_magnitude[lane_segment_idx] = torch.norm(
                center_vectors[:, :2], p=2, dim=-1
            )
            point_height[lane_segment_idx] = center_vectors[:, 2]

        for crosswalk in map_api.get_scenario_ped_crossings():
            crosswalk_idx = polygon_ids.index(crosswalk.id)
            edge1 = torch.from_numpy(crosswalk.edge1.xyz).float()
            edge2 = torch.from_numpy(crosswalk.edge2.xyz).float()

            start_position = (edge1[0] + edge2[0]) / 2
            end_position = (edge1[-1] + edge2[-1]) / 2

            polygon_position[crosswalk_idx, :2] = start_position[:2]
            polygon_position[crosswalk_idx, 2] = torch.atan2(
                (end_position - start_position)[1],
                (end_position - start_position)[0],
            )

            polygon_position[crosswalk_idx + len(cross_walk_ids), :2] = (
                end_position[:2]
            )
            polygon_position[crosswalk_idx + len(cross_walk_ids), 2] = (
                torch.atan2(
                    (start_position - end_position)[1],
                    (start_position - end_position)[0],
                )
            )

            polygon_height[crosswalk_idx] = end_position[2] - start_position[2]
            polygon_height[crosswalk_idx + len(cross_walk_ids)] = (
                start_position[2] - end_position[2]
            )

            # crosswalk polygon type features
            polygon_type[crosswalk_idx] = self.POLYGON_TYPES.index(
                "PEDESTRIAN"
            )
            polygon_type[crosswalk_idx + len(cross_walk_ids)] = (
                self.POLYGON_TYPES.index("PEDESTRIAN")
            )

            # crosswalk point vector features
            if (
                side_to_directed_lineseg(
                    (edge1[0] + edge1[-1]) / 2, start_position, end_position
                )
                == "LEFT"
            ):
                left_boundary, right_boundary = edge1, edge2
            else:
                left_boundary, right_boundary = edge2, edge1

            centerline = torch.from_numpy(
                compute_midpoint_line(
                    left_boundary.numpy(),
                    right_boundary.numpy(),
                    num_interp_pts=self.top_k_points + 1,
                )[0]
            ).float()

            center_vectors = centerline[1:] - centerline[:-1]

            crosswalk_point_pose = centerline[:-1, :2]
            crosswalk_point_heading = torch.atan2(
                center_vectors[:, 1], center_vectors[:, 0]
            )

            crosswalk_point_reverse_pose = centerline.flip(dims=[0])[:-1, :2]
            crosswalk_point_reverse_heading = torch.atan2(
                -center_vectors.flip(dims=[0])[:, 1],
                -center_vectors.flip(dims=[0])[:, 0],
            )

            point_position[crosswalk_idx] = torch.cat(
                [
                    crosswalk_point_pose,
                    crosswalk_point_heading.unsqueeze(-1),
                ],
                dim=-1,
            )
            point_position[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
                [
                    crosswalk_point_reverse_pose,
                    crosswalk_point_reverse_heading.unsqueeze(-1),
                ],
                dim=-1,
            )

            point_magnitude[crosswalk_idx] = torch.norm(
                center_vectors[:, :2], p=2, dim=-1
            )
            point_magnitude[crosswalk_idx + len(cross_walk_ids)] = torch.norm(
                -center_vectors.flip(dims=[0])[:, :2], p=2, dim=-1
            )

            point_height[crosswalk_idx] = center_vectors[:, 2]
            point_height[crosswalk_idx + len(cross_walk_ids)] = (
                -center_vectors.flip(dims=[0])[:, 2]
            )

        num_points = torch.tensor(
            [point.size(0) for point in point_position], dtype=torch.long
        )
        map_data = {
            "map_polygon": {},
            "map_point": {},
        }
        map_data["map_polygon"]["num_nodes"] = num_polygons
        map_data["map_polygon"]["position"] = polygon_position
        map_data["map_polygon"]["valid_mask"] = polygon_mask.bool()
        if self.dim == 3:
            map_data["map_polygon"]["height"] = polygon_height
        map_data["map_polygon"]["type"] = polygon_type
        if len(num_points) == 0:
            map_data["map_point"]["num_nodes"] = 0
            map_data["map_point"]["position"] = torch.tensor(
                [], dtype=torch.float
            )
            map_data["map_point"]["magnitude"] = torch.tensor(
                [], dtype=torch.float
            )
            map_data["map_point"]["valid_mask"] = torch.tensor(
                [], dtype=torch.bool
            )
            if self.dim == 3:
                map_data["map_point"]["height"] = torch.tensor(
                    [], dtype=torch.float
                )

        else:
            map_data["map_point"]["num_nodes"] = num_points.sum().item()
            map_data["map_point"]["position"] = point_position
            map_data["map_point"]["magnitude"] = point_magnitude
            if self.dim == 3:
                map_data["map_point"]["height"] = point_height
            map_data["map_point"]["valid_mask"] = point_mask.bool()
        return map_data

    def get_agent_features(self, df: pd.DataFrame) -> Dict[str, torch.tensor]:
        """Extract agent features from dataframe.

        Args:
            df: The dataframe that containing agent features in parquet format.
        Returns:
            data: Processed agent features in dictionary format.
        """

        agent_ids = list(
            df[df["timestep"] == self.HISTORICAL_STEPS - 1][
                "track_id"
            ].unique()
        )
        num_agents = len(agent_ids)
        av_index = agent_ids.index("AV")

        # initialization
        valid_mask = torch.zeros(num_agents, self.num_steps, dtype=torch.bool)
        curr_valid_mask = torch.zeros(num_agents, dtype=torch.bool)
        predict_mask = torch.zeros(
            num_agents, self.num_steps, dtype=torch.bool
        )
        agent_id = torch.zeros(num_agents, dtype=torch.long)
        agent_type = torch.zeros(num_agents, dtype=torch.uint8)
        agent_category = torch.zeros(num_agents, dtype=torch.uint8)

        position = torch.zeros(
            num_agents, self.num_steps, self.dim, dtype=torch.float
        )
        acs_gt = torch.zeros(
            num_agents, self.FUTURE_STEPS, self.dim, dtype=torch.float
        )
        heading = torch.zeros(num_agents, self.num_steps, dtype=torch.float)
        velocity = torch.zeros(
            num_agents, self.num_steps, self.dim, dtype=torch.float
        )

        for track_id, track_df in df.groupby("track_id"):
            if track_id not in agent_ids:
                continue
            agent_idx = agent_ids.index(track_id)
            agent_steps = track_df["timestep"].values
            valid_mask[agent_idx, agent_steps] = True
            predict_mask[agent_idx, agent_steps] = True
            curr_valid_mask[agent_idx] = valid_mask[
                agent_idx, self.HISTORICAL_STEPS - 1
            ]
            valid_mask[agent_idx, 1 : self.HISTORICAL_STEPS] = (
                valid_mask[agent_idx, : self.HISTORICAL_STEPS - 1]
                & valid_mask[agent_idx, 1 : self.HISTORICAL_STEPS]
            )
            valid_mask[agent_idx, 0] = False
            predict_mask[agent_idx, : self.HISTORICAL_STEPS] = False
            # not predicting the future if current frame is not available
            if not curr_valid_mask[agent_idx]:
                predict_mask[agent_idx, self.HISTORICAL_STEPS :] = False
            agent_id[agent_idx] = int(track_id) if track_id != "AV" else 1
            agent_type[agent_idx] = self.AGENT_TYPES.index(
                track_df["object_type"].values[0]
            )
            agent_category[agent_idx] = track_df["object_category"].values[0]
            position[agent_idx, agent_steps, :2] = torch.from_numpy(
                np.stack(
                    [
                        track_df["position_x"].values,
                        track_df["position_y"].values,
                    ],
                    axis=-1,
                )
            ).float()
            position[agent_idx, agent_steps, 2] = torch.from_numpy(
                track_df["heading"].values
            ).float()

            heading[agent_idx, agent_steps] = torch.from_numpy(
                track_df["heading"].values
            ).float()

            velocity[agent_idx, agent_steps, :2] = torch.from_numpy(
                np.stack(
                    [
                        track_df["velocity_x"].values,
                        track_df["velocity_y"].values,
                    ],
                    axis=-1,
                )
            ).float()
        origin = position[:, self.HISTORICAL_STEPS - 1, :2]
        theta = position[:, self.HISTORICAL_STEPS - 1, 2]
        cos, sin = theta.cos(), theta.sin()
        rot_mat = theta.new_zeros(num_agents, 2, 2)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos
        acs_gt[..., :2] = torch.matmul(
            position[:, self.HISTORICAL_STEPS :, :2]
            - origin[:, :2].reshape(-1, 1, 2),
            rot_mat,
        )
        acs_gt[..., 2] = wrap_angle(
            heading[:, self.HISTORICAL_STEPS :] - theta.unsqueeze(-1)
        )
        return {
            "num_nodes": num_agents,
            "av_index": av_index,
            "valid_mask": valid_mask,
            "predict_mask": predict_mask,
            "id": agent_id,
            "type": agent_type,
            "category": agent_category,
            "position": position,
            "acs_gt": acs_gt,
            "velocity": velocity,
        }
# Copyright (c) Carizon. All rights reserved.



def angle_between_2d_vectors(
    ctr_vector: torch.Tensor, nbr_vector: torch.Tensor
) -> torch.Tensor:
    """Calculate the angle between two 2D vectors in radiant.

    Args:
        ctr_vector(torch.Tensor): The 2D vector chosen to be centered.
        nbr_vector(torch.Tensor): The 2D vector chosen to be the neighbor.
    Returns:
        torch.Tensor: The angle between the vectors in radians.
    """
    return torch.atan2(
        ctr_vector[..., 0] * nbr_vector[..., 1]
        - ctr_vector[..., 1] * nbr_vector[..., 0],
        (ctr_vector[..., :2] * nbr_vector[..., :2]).sum(dim=-1),
    )


def angle_between_3d_vectors(
    ctr_vector: torch.Tensor, nbr_vector: torch.Tensor
) -> torch.Tensor:
    """Calculate the angle between two 3D vectors in radiant.

    Args:
        ctr_vector(torch.Tensor): The 3D vector chosen to be centered.
        nbr_vector(torch.Tensor): The 3D vector chosen to be the neighbor.
    Returns:
        torch.Tensor: The angle between the vectors in radians.
    """
    return torch.atan2(
        torch.cross(ctr_vector, nbr_vector, dim=-1).norm(p=2, dim=-1),
        (ctr_vector * nbr_vector).sum(dim=-1),
    )


def side_to_directed_lineseg(
    query_point: torch.Tensor,
    start_point: torch.Tensor,
    end_point: torch.Tensor,
) -> str:
    """
    Determine the relative position of a query point to a directed
    line segment.

    This function calculates the orientation of the `query_point` with
    respect to the directed line segment from `start_point` to `end_point`.
    It uses the cross product of vectors to determine if the point is
    to the left, right, or on the line segment.

    Args:
        query_point (torch.Tensor): The point whose position is being
            determined.
        start_point (torch.Tensor): The starting point of the line segment.
        end_point (torch.Tensor): The ending point of the line segment.

    Returns:
        str: "LEFT" if the query point is to the left of the line segment,
             "RIGHT" if it is to the right,
             "CENTER" if it is on the line segment.
    """
    cond = (end_point[0] - start_point[0]) * (
        query_point[1] - start_point[1]
    ) - (end_point[1] - start_point[1]) * (query_point[0] - start_point[0])
    if cond > 0:
        return "LEFT"
    elif cond < 0:
        return "RIGHT"
    else:
        return "CENTER"


def wrap_angle(
    angle: torch.Tensor, min_val: float = -math.pi, max_val: float = math.pi
) -> torch.Tensor:
    """
    Wrap an angle to be within the specified range [min_val, max_val).

    This function ensures that the angle remains within the given bounds
    by wrapping it around. The result will be in the range [min_val, max_val).

    Args:
        angle (torch.Tensor): The angle(s) to be wrapped.
        min_val (float, optional): The minimum value of the wrapping range.
            Default is -math.pi.
        max_val (float, optional): The maximum value of the wrapping range.
            Default is math.pi.

    Returns:
        torch.Tensor: The wrapped angle(s) within the range [min_val, max_val).
    """
    return min_val + (angle + max_val) % (max_val - min_val)
