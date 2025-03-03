from enum import Enum
from typing import Dict

import torch
from shapely.geometry import LineString, Point, Polygon
from prediction.utils import centerline_to_polygon

X_COORD, Y_COORD, VELOCITY, YAW, TIME = 0, 1, 2, 3, 4
FPS = 5
EGO_ID = 88888888


class LateralTagger(Enum):
    untagged = 0
    straight = 1
    left_turn = 2
    right_turn = 3
    uturn = 4
    left_lane_change = 5
    right_lane_change = 6
    curve = 7
    stop = 8


class TrackTagger(Enum):
    untagged = 0
    unscored = 1
    scored = 2
    focal = 3


class EgoCriticalTagger(Enum):
    unscored = 0
    junction_crossing = 1
    lane_diverging = 3
    unsmoothed_boundary = 3


class FilterCriticalObstacle:
    """Categorize critical obstacle, critical events for PnP Model based on
    a set of rules.
    """

    def __init__(
        self,
        is_junction: bool = True,
        scaling_factor: float = 3.0,
        is_viz: bool = False,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.is_junction = is_junction
        self.is_viz = is_viz

    def __call__(
        self,
        agent_features: Dict[str, Dict[str, torch.Tensor]],
        **kwargs,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Tag the obstacle lateral behavior.
        Args:
            agent_features(Dict[int, Any]): Preprocessed annos include
                agent_features:
                -- agent_ids:
                    -- hist
                    -- fut
                    -- type
                    -- category
                    -- track(to-be-added): use _classify_focal_agents
        Returns
        """
        # agent_features = self._classify_focal_agents(
        #     agent_features, self.scaling_factor
        # )
        map_features = kwargs.get("map_features", [])
        agent_features = self.classify_ego_critical_events(
            agent_features, map_features
        )
        return agent_features

    def classify_ego_critical_events(
        self,
        agent_features: Dict[str, Dict[str, torch.Tensor]],
        map_features: Dict[str, Dict[str, torch.Tensor]],
    ):
        ego_feat = agent_features[EGO_ID]
        ego_lateral = ego_feat["lateral"]
        # initialize an empty tagger
        agent_features[EGO_ID]["critical"] = EgoCriticalTagger.unscored
        if not self.is_junction:
            # --------- normal scenario ---------
            # if ego_lateral in [1, 7] and len(ego_feat["fut"]) > 1:
            #     if (
            #         ego_feat["fut"][-1, X_COORD] > 20
            #         and abs(ego_feat["fut"][-1, Y_COORD])
            #         / ego_feat["fut"][-1, X_COORD]
            #         < 1 / 1.2
            #     ):
            #         agent_features[EGO_ID][
            #             "critical"
            #         ] = EgoCriticalTagger.junction_crossing
            agent_features[EGO_ID][
                "critical"
            ] = EgoCriticalTagger.junction_crossing
        else:
            # --------- junction scenario ---------
            if ego_lateral in [1, 7] and len(ego_feat["fut"]) > 1:
                ego_start_pt = Point(ego_feat["fut"][0, [X_COORD, Y_COORD]])
                ego_end_pt = Point(ego_feat["fut"][-1, [X_COORD, Y_COORD]])
                mid_idx = (
                    len(ego_feat["fut"]) // 2 - 1
                    if len(ego_feat["fut"]) > 4
                    else 0
                )
                ego_mid_pt = Point(
                    ego_feat["fut"][mid_idx, [X_COORD, Y_COORD]]
                )
                # critical event: junction scenario
                for _, poly_feat in map_features.items():
                    if (
                        poly_feat["pl_type"] != "junction"
                        or len(poly_feat["vertex"]["coords"]) < 1
                    ):
                        continue

                    junction_polygon = Polygon(
                        poly_feat["vertex"]["coords"][:, [X_COORD, Y_COORD]]
                    )
                    if (
                        junction_polygon.contains(ego_start_pt)
                        or junction_polygon.contains(ego_end_pt)
                        or junction_polygon.contains(ego_mid_pt)
                    ):
                        if (
                            ego_feat["fut"][-1, X_COORD] > 20
                            and abs(ego_feat["fut"][-1, Y_COORD])
                            / ego_feat["fut"][-1, X_COORD]
                            < 1 / 1.2
                        ):
                            agent_features[EGO_ID][
                                "critical"
                            ] = EgoCriticalTagger.junction_crossing
                            break
        if self.is_viz:
            agent_features[EGO_ID][
                "critical"
            ] = EgoCriticalTagger.junction_crossing
        return agent_features

    @staticmethod
    def _classify_focal_agents(
        agent_features: Dict[str, Dict[str, torch.Tensor]],
        scaling_factor: float = 3,
    ):
        ego_feat = agent_features[EGO_ID]
        ego_fut = ego_feat["fut"]
        ego_yaw_end = ego_fut[-1, YAW]

        if ego_fut.shape[0] < 2:
            return
        ego_query_traj = ego_fut[:, [X_COORD, Y_COORD]].clone()
        ego_query_traj[0, X_COORD] -= 5.0
        if agent_features[EGO_ID]["lateral"] == 8:
            ego_query_traj[-1, X_COORD] += 15.0 * torch.cos(ego_yaw_end)
            ego_query_traj[-1, Y_COORD] += 15.0 * torch.sin(ego_yaw_end)
        else:
            ego_query_traj[-1, X_COORD] += 5.0 * torch.cos(ego_yaw_end)
            ego_query_traj[-1, Y_COORD] += 5.0 * torch.sin(ego_yaw_end)
        ego_safe_polygon = centerline_to_polygon(
            ego_query_traj.numpy(), scaling_factor
        )
        ego_safe_polygon = Polygon(ego_safe_polygon)

        for agent_idx, agent_feat in agent_features.items():
            agent_features[agent_idx]["track"] = TrackTagger.untagged
            if agent_feat["category"] in [
                2,
                5,
                6,
            ]:
                continue
            agent_fut = agent_feat["fut"]
            agent_hist = agent_feat["hist"]

            agent_end_idx = agent_fut.any(dim=-1).sum()
            if agent_end_idx > 2:
                agent_fut_traj = LineString(
                    agent_fut[:agent_end_idx, [X_COORD, Y_COORD]]
                )
                if agent_fut_traj.intersects(ego_safe_polygon):
                    agent_features[agent_idx]["track"] = TrackTagger.focal
                    continue
            if len(agent_hist) > 2:
                agent_hist_traj = LineString(agent_hist[:, [X_COORD, Y_COORD]])
                if agent_hist_traj.intersects(ego_safe_polygon):
                    agent_features[agent_idx]["track"] = TrackTagger.scored
                    continue

            agent_features[agent_idx]["track"] = TrackTagger.unscored
        return agent_features


class FilterCriticalObstacleV2:
    """Categorize critical obstacle, critical events for PnP Model based on
    a set of rules.
    """

    def __init__(
        self,
        is_junction: bool = True,
        scaling_factor: float = 3.0,
        is_viz: bool = False,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.is_junction = is_junction
        self.is_viz = is_viz

    def __call__(
        self,
        sample: Dict[str, Dict[str, torch.Tensor]],
        **kwargs,
    ) -> EgoCriticalTagger:
        """Tag the obstacle lateral behavior.
        Args:
            sample(Dict[int, Any]): Preprocessed sample include
                agent_features:
        Returns:
            scenarios(bool)
        """

        map_features = kwargs.get("map_features", [])
        sample["ego_tagger"] = self.classify_ego_critical_events(
            sample, map_features
        )
        sample = self.classify_cutin_events(sample)
        return sample

    def classify_cutin_events(
        self,
        sample: Dict[str, Dict[str, torch.Tensor]],
    ):
        all_lateral_tags = sample["lateral"]
        all_fut_trajs = sample["agent_features"]["gcs"]["fut_pose"]
        all_track_ids = sample["agent_features"]["track_ids"]
        ego_traj = all_fut_trajs[all_track_ids == EGO_ID].squeeze()
        ego_fut_traj = LineString(ego_traj[:, [X_COORD, Y_COORD]])
        for idx, curr_lateral_tag in enumerate(all_lateral_tags):
            # Skip if not a lane-changing tag or is the ego vehicle
            if all_track_ids[idx] == EGO_ID or curr_lateral_tag.item() not in [
                5,
                6,
            ]:
                continue

            # Check trajectory points
            fut_traj = all_fut_trajs[idx]
            y_positions = fut_traj[:, 1]  # Lateral positions
            x_positions = fut_traj[:, 0]  # Longitudinal positions
            agent_fut_traj = LineString(fut_traj[:, [X_COORD, Y_COORD]])
            # Conditions for cut-in behavior
            if (
                x_positions[0]
                > -2.0  # Starts in front of the ego within a threshold
                and abs(y_positions[0]) > 1.5
                and torch.all(
                    torch.diff(x_positions) > 0
                )  # Gradual forward movement
                and (
                    self.is_trending_toward_zero(y_positions)
                    or self.is_trending_toward_zero(y_positions[:15])
                    or agent_fut_traj.intersects(ego_fut_traj)
                )
                and self.is_y_close_to_zero(y_positions)  # Proximity to Y = 0
            ):
                # Classify as cut-in
                sample["lateral"][idx] = 9

        return sample

    def is_trending_toward_zero(self, y_positions: torch.Tensor) -> bool:
        """
        Check if y_positions exhibit a trend toward 0 (allowing noise).
        """
        abs_y = torch.abs(y_positions)
        avg_diff = torch.mean(torch.diff(abs_y))
        return avg_diff < 0

    def is_y_close_to_zero(
        self,
        y_positions: torch.Tensor,
        threshold: float = 1.0,
    ) -> bool:
        """
        Check if the Y trajectory remains reasonably close to 0.
        """
        return torch.any(torch.abs(y_positions) < threshold)

    def classify_ego_critical_events(
        self,
        sample: Dict[str, Dict[str, torch.Tensor]],
        map_features: Dict[str, Dict[str, torch.Tensor]],
    ):
        all_track_ids = sample["agent_features"]["track_ids"]
        ego_idx = all_track_ids == EGO_ID
        ego_tag = sample["lateral"][ego_idx]
        ego_feat = sample["agent_features"]["gcs"]["fut_pose"][ego_idx]
        ego_mask = sample["agent_features"]["fut_valid_masks"][ego_idx]
        ego_fut = ego_feat[ego_mask][:, [0, 1]]
        curr_tag = EgoCriticalTagger.unscored
        if not self.is_junction:
            # --------- normal + junction scenario ---------
            if int(ego_tag) in [1, 7] and len(ego_fut) > 1:
                if (
                    ego_fut[-1, X_COORD] > 20
                    and abs(ego_fut[-1, Y_COORD]) / ego_fut[-1, X_COORD]
                    < 1 / 1.2
                ):
                    curr_tag = EgoCriticalTagger.junction_crossing
        # --------- junction scenario ---------
        else:
            if int(ego_tag) in [1, 7] and len(ego_fut) > 1:
                ego_start_pt = Point(ego_fut[0, [X_COORD, Y_COORD]])
                ego_end_pt = Point(ego_fut[-1, [X_COORD, Y_COORD]])
                mid_idx = len(ego_fut) // 2 - 1 if len(ego_fut) > 4 else 0
                ego_mid_pt = Point(ego_fut[mid_idx, [X_COORD, Y_COORD]])
                # critical event 1
                for _, poly_feat in map_features.items():
                    if (
                        poly_feat["pl_type"] != "junction"
                        or len(poly_feat["vertex"]["coords"]) < 1
                    ):
                        continue

                    junction_polygon = Polygon(
                        poly_feat["vertex"]["coords"][:, [X_COORD, Y_COORD]]
                    )
                    if (
                        junction_polygon.contains(ego_start_pt)
                        or junction_polygon.contains(ego_end_pt)
                        or junction_polygon.contains(ego_mid_pt)
                    ):
                        if (
                            ego_fut[-1, X_COORD] > 20
                            and abs(ego_fut[-1, Y_COORD])
                            / ego_fut[-1, X_COORD]
                            < 1 / 1.2
                        ):
                            curr_tag = EgoCriticalTagger.junction_crossing
        if self.is_viz:
            curr_tag = EgoCriticalTagger.junction_crossing
        return curr_tag
