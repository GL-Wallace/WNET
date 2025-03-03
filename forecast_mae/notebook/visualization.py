from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from typing import Optional
from sept.src.datamodule.av2_dataset import collate_fn
# from forecast_mae.src.model.trainer_forecast import Trainer as Model
from sept.src.model.trainer_sept import Trainer as Model
from forecast_mae.src.metrics.utils import sort_predictions

def generate_scenario(data_dir: Path):
    # all_scenario_files = sorted(data_dir.rglob("*.parquet"))
    file_list = sorted(list(data_dir.glob("*.pt")))
    for index in (0):
        data = torch.load(file_list[index], weights_only=False)
    return data

def visualization_scenario(data, save_dir: Optional[Path]=None):

    x_positions = data['x_positions']
    y_positions = data['y']
    heading = data['x_angles']
    x_centers = data['x_centers']
    lane_positions = data['lane_positions']

    plt.figure(figsize=(20, 20), dpi=300)

    for lane_id in range(len(lane_positions)):
        lane_pos = lane_positions[lane_id]
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
        actor_heading = heading[actor_id][49]
        value_1 = actor_pos[49, 0]
        value_2 = actor_pos[49, 1]
        plt.plot(actor_pos[actor_pos[:, 0] != 0, 0], actor_pos[actor_pos[:, 1] != 0, 1], color=color, linewidth=1,
                 label=agentLabel)
        plt.plot(future_points[future_points[:, 0] != 0, 0] + value_1, future_points[future_points[:, 1] != 0, 1] + value_2,
                 color=color, linewidth=1, linestyle='--', label=label1)

        dx = np.cos(actor_heading)
        dy = np.sin(actor_heading)
        plt.arrow(value_1, value_2, float(dx), float(dy), head_width=1, head_length=1.5, fc=color, ec=color)

    plt.legend()
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.legend(fontsize=26, loc="lower left")
    # plt.axis("equal")
    if save_dir != None:
        plt.savefig(save_dir)

def vis_pred(data_source, save_dir, ckpt_path, ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Model.load_from_checkpoint(ckpt_path, pretrained_weights=None, strict=False)
    model = model.eval()
    model = model.to(device)
    data = collate_fn([data_source])

    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            data[key] = value.to(device)

    with torch.no_grad():
        prediction, prob  = model.predict(data)

    # pred, prob_ = sort_predictions(prediction, prob, k=6)

    visualization_scenario(data=data_source)

    pred = prediction.squeeze(0).cpu().numpy()

    for i in range(0, len(pred)):
        trajectory = pred[i]
        label_ = "Predicted Trajectory " + str(i)
        plt.plot(trajectory[:, 0], trajectory[:, 1],  label=label_, linestyle='-',  linewidth=2)
        print(i)

    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.legend(fontsize=26, loc="lower left")
    plt.savefig(save_dir, bbox_inches='tight')
    plt.show()



if __name__ == "__main__":

    save_dir = '/home/user/Projects/pnp_research/forecast_mae_prediction/outputs/pre.png'
    data_folder = Path("/home/user/Documents/argoverse2_forecast_mae/forecast-mae/train")

    file_list = sorted(list(data_folder.glob("*.pt")))
    data = torch.load(file_list[3], weights_only=False)
    # data = collate_fn([data])
    visualization_scenario(data=data)
    # for file in file_list:
    #     data = torch.load(file, weights_only=False)
    #     # data = collate_fn([data])
    #     visualization_scenario(data=data, save_dir=save_dir)

    # ckpt = '/horizon-bucket/carizon_pnp_jfs/guowei.zhang/model_outputs/sept_fine_tune/2024-12-26/13-27-26/checkpoints/epoch=57.ckpt'
    # vis_pred(data, save_dir=save_dir, ckpt_path=ckpt)
    plt.show()
    plt.close()