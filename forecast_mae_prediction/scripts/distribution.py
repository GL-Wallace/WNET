import numpy as np
from scipy.stats import multivariate_normal
import torch
import time
from pathlib import Path
import matplotlib.pyplot as plt
from forecast_mae_prediction.src.datamodule.lmdb_dataset import collate_fn
from forecast_mae_prediction.src.model.trainer_forecast import Trainer as Model
from forecast_mae_prediction.src.metrics.utils import sort_predictions
from forecast_mae_prediction.notebook.lmdb_visualization import visualization_scenario, visualization_scenario_xy_flip


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_folder = Path("/home/user/Documents/pkl_files/3")  # Path to the dataset
ckpt = '/home/user/Downloads/epoch=86.ckpt'
save_dir = '../notebook/forecast_mae_prediction/outputs/pred_results'

file_list = sorted(list(data_folder.glob("*.pt")))
pred_list = []
idx = 1
for file in file_list:
    start_time_pic = time.time()
    data_source = torch.load(file)
    model = Model.load_from_checkpoint(ckpt, pretrained_weights=None, strict=False)
    model = model.eval()
    model = model.to(device)
    data = collate_fn([data_source])

    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            data[key] = value.to(device)

    with torch.no_grad():
        start_time = time.time()
        pred, prob = model.predict(data)
        end_time = time.time()
        inference_time_ms = (end_time - start_time) * 1000
        print(f"Prediction time: {inference_time_ms:.3f} ms")
        pred_list.append(pred)
        visualization_scenario_xy_flip(data_source)
        plt.savefig(f'/home/user/Documents/temp_1/trajectory_plot_{idx}.png', bbox_inches='tight')
        pred = pred.squeeze(0).cpu().numpy()
        for i in range(0, len(pred)):
            trajectory = pred[i]
            label_ = "Predicted Trajectory " + str(i)
            plt.plot(trajectory[:, 1], trajectory[:, 0], label=label_, linestyle='-', linewidth=2)

    # Save the plot to a file
    plt.legend(fontsize=26, loc="lower left")
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.xlim(-50, 50)
    plt.savefig(f'/home/user/Documents/temp/trajectory_plot_{idx}.png', bbox_inches='tight')
    plt.close()
    end_time_pic = time.time()
    pkl_pic_time = end_time_pic - start_time_pic
    print(f"pkl_pic_time: {pkl_pic_time:.3f} s")
    idx +=1

data = torch.stack(pred_list)
data = data.reshape(-1, 30, 2).cpu()
mean = data.mean(dim=(0, 1)).numpy()
mean_flipped = mean[::-1]
covariance = np.cov(data.reshape(-1, 2), rowvar=False)
covariance_flipped = covariance[::-1, ::-1]


reshaped_pred = data.reshape(-1, 2)
num = reshaped_pred.size(0)

x_pos = reshaped_pred[:, 0]
y_pos = reshaped_pred[:, 1]

# 绘制散点图，交换 x 和 y
plt.scatter(y_pos, x_pos, alpha=0.5)
plt.title('Scatter Plot of (x, y) Coordinates')
plt.ylim(0, 100)
plt.xlim(-15, 15)
plt.grid(True)
plt.show()
plt.close()

# 创建一个网格
x, y = np.meshgrid(np.linspace(-15, 15, num), np.linspace(0, 100, num))
pos = np.dstack((x, y))

# 计算二维正态分布的概率密度值
pdf_values = multivariate_normal.pdf(pos, mean=mean_flipped, cov=covariance_flipped)

# 绘制概率密度图像
plt.figure(figsize=(8, 8))
plt.contourf(x, y, pdf_values, cmap='viridis')
plt.colorbar()
plt.xlabel('Y')
plt.ylabel('X')
plt.title('2D Gaussian Probability Density')
plt.show()


# 创建一个网格
x, y = np.meshgrid(np.linspace(-15, 15, num), np.linspace(0, 100, num))
pos = np.dstack((x, y))

# 计算二维正态分布的概率密度值
pdf_values = multivariate_normal.pdf(pos, mean=mean_flipped, cov=covariance_flipped)

# 绘制三维概率密度图像
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, pdf_values, cmap='viridis')
ax.set_xlabel('Y')
ax.set_ylabel('X')
# ax.set_zlabel('PDF Value')
ax.set_title('3D Gaussian Probability Density')

plt.show()