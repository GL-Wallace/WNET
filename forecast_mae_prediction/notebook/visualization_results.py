import torch
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
from forecast_mae_prediction.src.datamodule.lmdb_dataset import collate_fn
from forecast_mae_prediction.src.model.trainer_forecast import Trainer as Model
from forecast_mae_prediction.src.metrics.utils import sort_predictions
from forecast_mae_prediction.notebook.lmdb_visualization import visualization_scenario

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_folder = Path("/home/user/Documents/pkl_files/1")  # Path to the datase
ckpt = '/home/user/Downloads/epoch=86.ckpt'
# data_folder = Path(
#     "/horizon-bucket/carizon_pnp_jfs/guowei.zhang/pkl/2024-10-28-10:49:36_pkl_intersection_normal_new/intersection_7_1")  # Path to the datase
# ckpt = "/horizon-bucket/carizon_pnp_jfs/guowei.zhang/model_outputs/sept_fine_tune/2024-12-09/11-08-23/checkpoints/epoch=195.ckpt"
save_dir = 'forecast_mae_prediction/outputs/pred_results'

file_list = sorted(list(data_folder.glob("*.pt")))[0]
data_source = torch.load(file_list)
model = Model.load_from_checkpoint(ckpt, pretrained_weights=None, strict=False)
model = model.eval()
model = model.to(device)
data = collate_fn([data_source])

# 移动每个Tensor到GPU
for key, value in data.items():
    if isinstance(value, torch.Tensor):
        data[key] = value.to(device)

# 关闭梯度计算以提高推理速度
with torch.no_grad():
    start_time = time.time()
    prediction, prob  = model.predict(data)
    end_time = time.time()
inference_time_ms = (end_time - start_time) * 1000
print(f"Inference time: {inference_time_ms:.6f} milliseconds")

pred, prob_ = sort_predictions(prediction, prob, k=3)

visualization_scenario(data_source)

pred = pred.squeeze(0).cpu().numpy()

# print(prob_)
for i in range(0, len(pred)):
    trajectory = pred[i]
    label_ = "Predicted Trajectory " + str(i)
    plt.plot(trajectory[:, 0], trajectory[:, 1],  label=label_, linestyle='-',  linewidth=2)
    print(i)

plt.legend()
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.legend(fontsize=26, loc="lower left")
# plt.axis("equal")


pic_title = str(len(pred)) + 'trajectories.png'
file_path = os.path.join(save_dir, pic_title)
os.makedirs(save_dir, exist_ok=True)
plt.savefig(file_path, bbox_inches='tight')
plt.savefig(file_path)
plt.show()
