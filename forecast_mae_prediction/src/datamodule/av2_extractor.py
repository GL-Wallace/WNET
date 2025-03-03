import traceback
from pathlib import Path
from typing import List
import av2.geometry.interpolate as interp_utils
import numpy as np
import torch
from av2.map.map_api import ArgoverseStaticMap

from .av2_data_utils import (
    OBJECT_TYPE_MAP,
    OBJECT_TYPE_MAP_COMBINED,
    LaneTypeMap,
    load_av2_df,
)


class Av2Extractor:
    def __init__(
            self,
            radius: float = 150,  # 为什么是150？
            save_path: Path = None,
            mode: str = "train",
            ignore_type: List[int] = [5, 6, 7, 8, 9],
            remove_outlier_actors: bool = True,
    ) -> None:
        self.save_path = save_path
        self.mode = mode
        self.radius = radius
        self.remove_outlier_actors = remove_outlier_actors
        self.ignore_type = ignore_type

    def save(self, file: Path):
        assert self.save_path is not None

        try:
            data = self.get_data(file)
        except Exception:
            print(traceback.format_exc())
            print("found error while extracting data from {}".format(file))
        save_file = self.save_path / (file.stem + ".pt")
        torch.save(data, save_file)

    def get_data(self, file: Path):
        return self.process(file)

    def process(self, raw_path: str, agent_id=None):
        df, am, scenario_id = load_av2_df(raw_path)
        city = df.city.values[0]

        agent_id = df["focal_track_id"].values[0]

        # 筛选出 track_id 为 agent Id 的所有行的信息
        local_df = df[df["track_id"] == agent_id].iloc

        # 把agent_id的 time——step【49】的位置转成 tensor _> 保存成origin
        origin = torch.tensor(
            [local_df[49]["position_x"], local_df[49]["position_y"]], dtype=torch.float
        )

        # 通过当前 focal agent的时间步T的航向角, 生成一个基于该航向角的旋转矩阵，目的是将地图坐标系转换成 agent centric 坐标系.
        theta = torch.tensor([local_df[49]["heading"]], dtype=torch.float)
        rotate_mat = torch.tensor(
            [
                [torch.cos(theta), -torch.sin(theta)],
                [torch.sin(theta), torch.cos(theta)],
            ],
        )

        # 先通过取出所有唯一的时间步序列， 取到当前时间步的所有object -> df，
        timestamps = list(np.sort(df["timestep"].unique()))
        cur_df = df[df["timestep"] == timestamps[49]]
        # 提取这个时间步中的所有的actor ids + position
        actor_ids = list(cur_df["track_id"].unique())
        cur_pos = torch.from_numpy(cur_df[["position_x", "position_y"]].values).float()

        # self.radius = 150, 当前时间步agent的位置在focal agent 150范围以外的，要删掉。
        out_of_range = np.linalg.norm(cur_pos - origin, axis=1) > self.radius
        actor_ids = [aid for i, aid in enumerate(actor_ids) if not out_of_range[i]]
        # 删除focal agent id，然后把它加到第一位置
        actor_ids.remove(agent_id)
        actor_ids = [agent_id] + actor_ids
        num_nodes = len(actor_ids)  # 有多少个agents

        df = df[df["track_id"].isin(actor_ids)]

        # initialization
        x = torch.zeros(num_nodes, 110, 2, dtype=torch.float)
        x_attr = torch.zeros(num_nodes, 3,
                             dtype=torch.uint8)  #？？what？ 有三列，1, object type，2, object_category 3 OBJECT_TYPE_MAP_COMBINED
        x_heading = torch.zeros(num_nodes, 110, dtype=torch.float)
        x_velocity = torch.zeros(num_nodes, 110, dtype=torch.float)
        x_track_horizon = torch.zeros(num_nodes, dtype=torch.int)
        padding_mask = torch.ones(num_nodes, 110, dtype=torch.bool)  # 这是什么用处

        # 通过groupby每个track_id， 得到每个actor的df
        for actor_id, actor_df in df.groupby("track_id"):
            # 这个actor 在整个actor-ids中的位置，（所有的actor，都一一个已经定好的位置）
            node_idx = actor_ids.index(actor_id)

            # 找出这个actor所有所在的时间步， 然后找到这些时间步对应的索引位置
            # timestamps 是所有的时间步的排序序列，其中的时间步是唯一的索引
            # actor_df【timestep】是 返回该actor的所有时间步
            # 然后将每个时间步的位置转换为在全局时间步序列中的索引位置
            node_steps = [timestamps.index(ts) for ts in actor_df["timestep"]]

            object_type = OBJECT_TYPE_MAP[actor_df["object_type"].values[0]]

            x_attr[node_idx, 0] = object_type
            x_attr[node_idx, 1] = actor_df["object_category"].values[0]
            x_attr[node_idx, 2] = OBJECT_TYPE_MAP_COMBINED[  # 这是什么意思？
                actor_df["object_type"].values[0]
            ]

            # 记录每个actor的时间步长是多少； 记录在这条actor所对应的index位置上
            x_track_horizon[node_idx] = node_steps[-1] - node_steps[0]

            # actor所在多有time step位置， padding mask的信息标记为false
            padding_mask[node_idx, node_steps] = False
            # 如果 第50步，当前时间，为ture（就是actor在这没有信息），或者是忽略的object 类型，把后面的所有padding mask 都更改为True
            if padding_mask[node_idx, 49] or object_type in self.ignore_type:
                padding_mask[node_idx, 50:] = True

            #把该 actor 所有的位置信息 存到 pos——xy中
            pos_xy = torch.from_numpy(
                np.stack(
                    [actor_df["position_x"].values, actor_df["position_y"].values],
                    axis=-1,
                )
            ).float()
            # 朝向角
            heading = torch.from_numpy(actor_df["heading"].values).float()
            # x，y 方向的速度
            velocity = torch.from_numpy(
                actor_df[["velocity_x", "velocity_y"]].values
            ).float()
            # 通过L2 范数（欧几里得范数）计算速度大小
            velocity_norm = torch.norm(velocity, dim=1)

            # （ 所在的每个时间步上的actor位置减去focal-origin的位置） * 转换位置向量，成为新的位置
            x[node_idx, node_steps, :2] = torch.matmul(pos_xy - origin, rotate_mat)
            # agent在每个时间步上的 heading，
            x_heading[node_idx, node_steps] = (heading - theta + np.pi) % (
                    2 * np.pi
            ) - np.pi
            # 按照 x， index，和ts 存储 速度

            x_velocity[node_idx, node_steps] = velocity_norm

        (
            lane_positions,
            is_intersections,
            lane_ctrs,
            lane_angles,
            lane_attr,
            lane_padding_mask,
        ) = self.get_lane_features(am, origin, origin, rotate_mat, self.radius)

        if self.remove_outlier_actors:
            # ::用于指定步长。步长决定了在选取元素时的间隔。 ：：1 表示步长为 1，这意味着从头到尾每一个元素都被选中。实际上，::1 可以简化为 :，因为默认步长就是 1。
            # view(-1,2) 将其转换成2D， -1 表示自动推断新的第一个维度的大小，2 表示新的第二个维度大小为 2。
            # lane_positions[93,20,2] _> lane_samples[1860,2]
            lane_samples = lane_positions[:, ::1, :2].view(-1, 2)
            #  nearest_dist = torch.cdist(x[:, 49, :2], lane_samples).min(dim=1).values
            # 计算每个车辆在第50步上到其最近的中线的最小距离，要是距离小于5, 则把valid actor_mask 设置为true
            nearest_dist = torch.cdist(x[:, 49, :2], lane_samples).min(dim=1).values
            valid_actor_mask = nearest_dist < 5
            valid_actor_mask[0] = True  # always keep the target agent

            # 再次筛选一遍数据，要求在第50步上，actor的位置离中线的距离要在5以内。
            x = x[valid_actor_mask]
            x_heading = x_heading[valid_actor_mask]
            x_velocity = x_velocity[valid_actor_mask]
            x_attr = x_attr[valid_actor_mask]
            padding_mask = padding_mask[valid_actor_mask]
            num_nodes = x.shape[0]  # 重新获取有多少个actor

        x_ctrs = x[:, 49, :2].clone()
        x_positions = x[:, :50, :2].clone()
        x_velocity_diff = x_velocity[:, :50].clone()

        x[:, 50:] = torch.where(  #
            # 50步以后的步数，
            # 真 1. TS_current 没位置数据， padding mask 50 以后 也没数据 2. 有位置数据，但是后面的TS没有数据，3. 没有current ls， 后面的有信息， 那么，就创造一个全0的，6s的空位置信息 复制给x
            # 假， 如果，current ts 有数据，后面所有的ts都有数据，改成相对于TS【49】位置的 x，y坐标了
            (padding_mask[:, 49].unsqueeze(-1) | padding_mask[:, 50:]).unsqueeze(-1),
            torch.zeros(num_nodes, 60, 2),
            x[:, 50:] - x[:, 49].unsqueeze(-2),
        )

        x[:, 1:50] = torch.where(  # 这个是在做什么操作？？
            # T 1 - 49的部分，，如果
            # 真： 0-48都没数据，或者，1-49 没数据， 前面一段每数据，都用0填充
            # 假： 前面都有数据，就用下一时间步的位置减去上一时间步的位置
            (padding_mask[:, :49] | padding_mask[:, 1:50]).unsqueeze(-1),
            torch.zeros(num_nodes, 49, 2),
            x[:, 1:50] - x[:, :49],
        )
        # 所有actor 第0个时间步上的位置设置为0
        x[:, 0] = torch.zeros(num_nodes, 2)  # 把第0时间步的数据

        # ts-1-49, V
        # 真： 前0-48为true，（没有数据）， 或者前 1-49 也没有数据， 前所有ts， V = 0
        # 假： 前面都有数据，就用下一时间步的速度减去上一时间步的速度
        x_velocity_diff[:, 1:50] = torch.where(
            (padding_mask[:, :49] | padding_mask[:, 1:50]),
            torch.zeros(num_nodes, 49),
            x_velocity_diff[:, 1:50] - x_velocity_diff[:, :49],
        )
        # 所有actor， 第0个时间步上的速度设置为0
        x_velocity_diff[:, 0] = torch.zeros(num_nodes)

        y = None if self.mode == "test" else x[:, 50:]

        return {
            "x": x[:, :50],
            "y": y,
            "x_attr": x_attr,
            "x_positions": x_positions,
            "x_centers": x_ctrs,
            "x_angles": x_heading,
            "x_velocity": x_velocity,
            "x_velocity_diff": x_velocity_diff,
            "x_padding_mask": padding_mask,
            "lane_positions": lane_positions,
            "lane_centers": lane_ctrs,
            "lane_angles": lane_angles,
            "lane_attr": lane_attr,
            "lane_padding_mask": lane_padding_mask,
            "is_intersections": is_intersections,
            "origin": origin.view(-1, 2),
            "theta": theta,
            "scenario_id": scenario_id,
            "track_id": agent_id,
            "city": city,
        }

    @staticmethod
    def get_lane_features(
            am: ArgoverseStaticMap,
            query_pos: torch.Tensor,  # origin 把agent_id的 time——step【49】的位置转成 tensor _> 保存成origin 也就是中心点？
            origin: torch.Tensor,  # origin
            rotate_mat: torch.Tensor,
            radius: float,
    ):

        # get_nearby_lane_segments: 通过中心点和搜索的半径.
        # 1. vector lane segments.- list 2. 如果一个 if [list] lane_segment.waypoints - query_center <= radius return lane segment
        lane_segments = am.get_nearby_lane_segments(query_pos.numpy(), radius)

        lane_positions, is_intersections, lane_attrs = [], [], []

        for segment in lane_segments:
            lane_centerline, lane_width = interp_utils.compute_midpoint_line(
                left_ln_boundary=segment.left_lane_boundary.xyz,
                right_ln_boundary=segment.right_lane_boundary.xyz,
                num_interp_pts=20,  # 用于插值的中点数为什么是20？
            )

            # 提取中线的xy， 把中线的瓦窑，转变成 focal centric的角度
            lane_centerline = torch.from_numpy(lane_centerline[:, :2]).float()
            lane_centerline = torch.matmul(lane_centerline - origin, rotate_mat)

            is_intersection = am.lane_is_in_intersection(segment.id)

            lane_positions.append(lane_centerline)
            is_intersections.append(is_intersection)

            # get lane attrs
            lane_type = LaneTypeMap[segment.lane_type]
            attribute = torch.tensor(
                [lane_type, lane_width, is_intersection], dtype=torch.float
            )
            lane_attrs.append(attribute)

        # 对 lane_positions进行堆叠，lane positon本来是一个包含着多个tensor的list，
        # 例子 假设每条车道有 15 个点，每个点有 2 个坐标 (x, y)
        # lane1 = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11], [12, 12], [13, 13], [14, 14]])
        # lane2 = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15]])
        # lane3 = torch.tensor([[1, 0], [2, 1], [3, 2], [4, 3], [5, 4], [6, 5], [7, 6], [8, 7], [9, 8], [10, 9], [11, 10], [12, 11], [13, 12], [14, 13], [15, 14]])
        # 将这些张量放入一个列表
        # lane_positions = [lane1, lane2, lane3]
        # 对 lane_positions 进行stack操作后，会生成一个新的张量 stacked_lane_positions，其形状为 (3, 20, 2)。
        # 具体来说： 3 表示车道的数量segment number。 20 表示每条车道的点的数量 20【插值点数】。2 表示每个点的坐标 (x, y)。
        lane_positions = torch.stack(lane_positions)
        # 计算中心点： 因为插值为20, 取【9】,【10】的位置坐标，
        lanes_ctr = lane_positions[:, 9:11].mean(dim=1)

        # 计算每条车道的第 10 个点和第 9 个点在 x，y 方向上的差异。
        # 计算每条车道的第 9 个点和第 10 个点之间的角度。？ 这个有什么用？
        # 计算给定的 y 和 x 的反正切值，返回每条车道的第 9 和第 10 个点之间的角度。
        lanes_angle = torch.atan2(
            lane_positions[:, 10, 1] - lane_positions[:, 9, 1],
            lane_positions[:, 10, 0] - lane_positions[:, 9, 0],
        )

        is_intersections = torch.Tensor(is_intersections)
        lane_attrs = torch.stack(lane_attrs, dim=0)

        x_max, x_min = radius, -radius
        y_max, y_min = radius, -radius

        # 标记出，每个center line如果有一个waypoint上的xy坐标超出150, 则为true
        padding_mask = (
                (lane_positions[:, :, 0] > x_max)
                | (lane_positions[:, :, 0] < x_min)
                | (lane_positions[:, :, 1] > y_max)
                | (lane_positions[:, :, 1] < y_min)
        )
        # 对每一行取反操作,一个segment中的所有waypoint的点超过了150 才表示成真, false表示没有超过, true表示超过
        # all 是生成一个布尔掩码 invalid_mask，用于标识 padding_mask 中每一行是否所有元素都为 True。如果某一行的所有元素都是 True，则 invalid_mask 对应位置为 True；否则为 False。
        invalid_mask = padding_mask.all(dim=-1)
        lane_positions = lane_positions[~invalid_mask]
        is_intersections = is_intersections[~invalid_mask]
        lane_attrs = lane_attrs[~invalid_mask]
        lanes_ctr = lanes_ctr[~invalid_mask]
        lanes_angle = lanes_angle[~invalid_mask]
        padding_mask = padding_mask[~invalid_mask]

        # 1. padding_mask[..., None]： 对padding mask 进行广播，将维度扩展为lane_positions一样的维度
        # 2. where 判断， padding_mask位置上为True的值的位置，返回全0的值，false返回相对应的lane_positions 中 waypoint的值。
        lane_positions = torch.where(
            padding_mask[..., None], torch.zeros_like(lane_positions), lane_positions
        )

        return (
            lane_positions,
            is_intersections,
            lanes_ctr,
            lanes_angle,
            lane_attrs,
            padding_mask,
        )
