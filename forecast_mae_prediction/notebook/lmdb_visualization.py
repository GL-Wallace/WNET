from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import os


def generate_scenario(data_dir: Path):
    # all_scenario_files = sorted(data_dir.rglob("*.parquet"))
    file_list = sorted(list(data_dir.glob("*.pt")))
    for index in (400, len(file_list) - 1):
        data = torch.load(file_list[index], weights_only=False)
    return data
def visualization_scenario_xy_flip(data):
    x_positions = data['x_positions']
    y_positions = data['y']
    heading = data['x_angles']
    x_centers = data['x_centers']
    lane_positions = data['lane_positions']
    lane_attr = data['lane_attr']

    plt.figure(figsize=(20, 20), dpi=300)

    # 绘制车道中心线
    for lane_id in range(len(lane_positions)):
        lane_pos = lane_positions[lane_id]
        if lane_attr[lane_id] == 1:
            plt.plot(lane_pos[:, 1], lane_pos[:, 0], '--', linewidth=1, color='purple', label='Center line' if lane_id == 0 else "")
        else:
            plt.plot(lane_pos[:, 1], lane_pos[:, 0], 'k-', linewidth=1, label='Lane' if lane_id == 0 else "")

    for actor_id in range(len(x_positions)):
        if actor_id == 0:
            label1 = 'Focal Agent Future Trajectory'
            agentLabel = 'Focal Agent History Trajectory'
        elif actor_id == 1:
            label1 = 'Other Agent Future Trajectory'
            agentLabel = 'Other Agent History Trajectory'
        else:
            label1 = None
            agentLabel = None
        color = 'red' if actor_id == 0 else 'blue'
        actor_pos = x_positions[actor_id]
        future_points = y_positions[actor_id]
        actor_center = x_centers[actor_id]
        actor_heading = heading[actor_id][9]

        plt.plot(actor_pos[actor_pos[:, 0] != 0, 1], actor_pos[actor_pos[:, 1] != 0, 0], color=color, linewidth=1,
                 label=agentLabel)
        plt.plot(future_points[future_points[:, 0] != 0, 1], future_points[future_points[:, 1] != 0, 0],
                 color=color, linewidth=1, linestyle='--', label=label1)

        x, y = float(actor_center[1]), float(actor_center[0])
        dx = np.cos(actor_heading)
        dy = np.sin(actor_heading)
        plt.arrow(x, y, float(dy), float(dx), head_width=2, head_length=4, fc=color, ec=color)

    plt.legend()
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.legend(fontsize=26, loc="lower left")
    plt.xlim(-50, 50)

def visualization_scenario(data):

    x_positions = data['x_positions']
    y_positions = data['y']
    heading = data['x_angles']
    x_centers = data['x_centers']
    lane_positions = data['lane_positions']
    lane_attr = data['lane_attr']

    plt.figure(figsize=(20, 20), dpi=300)

    # 绘制车道中心线
    for lane_id in range(len(lane_positions)):
        lane_pos = lane_positions[lane_id]
        if lane_attr[lane_id] == 1:
            plt.plot(lane_pos[:, 0], lane_pos[:, 1], '--', linewidth=1, color='purple', label='Center line' if lane_id == 0 else "")
        else:
            plt.plot(lane_pos[:, 0], lane_pos[:, 1], 'k-', linewidth=1, label='Lane' if lane_id == 0 else "")

    for actor_id in range(len(x_positions)):
        if actor_id == 0:
            label1 = 'Focal Agent Future Trajectory'
            agentLabel = 'Focal Agent History Trajectory'
        elif actor_id == 1:
            label1 = 'Other Agent Future Trajectory'
            agentLabel = 'Other Agent History Trajectory'
        else:
            label1 = None
            agentLabel = None
        color = 'red' if actor_id == 0 else 'blue'
        actor_pos = x_positions[actor_id]
        future_points = y_positions[actor_id]
        actor_center = x_centers[actor_id]
        actor_heading = heading[actor_id][9]

        plt.plot(actor_pos[actor_pos[:, 0] != 0, 0], actor_pos[actor_pos[:, 1] != 0, 1], color=color, linewidth=1,
                 label=agentLabel)
        plt.plot(future_points[future_points[:, 0] != 0, 0], future_points[future_points[:, 1] != 0, 1],
                 color=color, linewidth=1, linestyle='--', label=label1)

        x, y = float(actor_center[0]), float(actor_center[1])
        dx = np.cos(actor_heading)
        dy = np.sin(actor_heading)
        plt.arrow(x, y, float(dx), float(dy), head_width=2, head_length=4, fc=color, ec=color)

    plt.legend()
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.legend(fontsize=26, loc="lower left")
    plt.axis("equal")



def visualization_scenario_scene_flipping(data):

    x_positions = data['x_positions']
    y_positions = data['y']
    heading = data['x_angles']
    x_centers = data['x_centers']
    lane_positions = data['lane_positions']
    agents = data['agents']
    lane_attr = data['lane_attr']
    plt.figure(figsize=(20, 20), dpi=300)

    # 绘制车道中心线
    for lane_id in range(len(lane_positions)):
        lane_pos = lane_positions[lane_id]
        if lane_attr[lane_id] == 1:
            plt.plot(lane_pos[:, 0], lane_pos[:, 1], '--', linewidth=1.5, color='purple', label='Center line' if lane_id == 0 else "")
        else:
            plt.plot(lane_pos[:, 0], lane_pos[:, 1], 'k-', linewidth=1.5, label='Lane' if lane_id == 0 else "")

    for actor_id in range(len(agents)):
        if agents[actor_id] == data['track_id']:
            label1 = 'Ego Agent Future Trajectory'
            agentLabel = 'Ego Agent History Trajectory'
            color = 'red'
        elif agents[actor_id] == data['focal_id']:
            label1 = 'Centric Agent Future Trajectory'
            agentLabel = 'Centric Agent History Trajectory'
            color = 'brown'
        else:
            label1 = None
            agentLabel = None
            color = 'blue'

        actor_pos = x_positions[actor_id]
        future_points = y_positions[actor_id]
        actor_center = x_centers[actor_id]
        actor_heading = heading[actor_id][9]

        plt.plot(actor_pos[actor_pos[:, 0] != 0, 0], actor_pos[actor_pos[:, 1] != 0, 1], color=color, alpha=1, linewidth=2,
                 label=agentLabel)
        plt.plot(future_points[future_points[:, 0] != 0, 0], future_points[future_points[:, 1] != 0, 1], alpha=1,
                 color=color, linewidth=2, linestyle='--', label=label1)

        # 绘制航向箭头
        x, y = float(actor_center[0]), float(actor_center[1])
        dx = np.cos(actor_heading)
        dy = np.sin(actor_heading)
        plt.arrow(x, y, float(dx), float(dy), head_width=2, head_length=4, fc=color, ec=color)
    plt.title(data["focal_id"])
    plt.legend()
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.legend(fontsize=26, loc="lower left")
    plt.axis("equal")

    # fig.savefig('Visualization_scenario_raw.png', dpi=800)


if __name__ == "__main__":
    dataset_path = Path(
        "/home/user/Projects/raw_data/data/20240716-063853_72_2024_10_11_14_54_48_804/forecast-mae/train")

    # collect dataset_path
    save_dir = "/home/user/Documents/pt_files/_0"

    # all_pickle_files = os.listdir(save_dir)
    all_valid_pickles = []
    for root, dirs, files in os.walk(save_dir):
        for file in files:
            if (file.startswith("172")):
                file_path = os.path.join(root, file)
                if os.path.getsize(file_path) > 0:  # 检查文件大小是否大于0
                    all_valid_pickles.append(file_path)

    for path in all_valid_pickles:
        path = Path(path)
        scenario_data = torch.load(path, weights_only=False)
        visualization_scenario(scenario_data)

    plt.show()
    plt.close()