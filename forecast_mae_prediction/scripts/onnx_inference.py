import torch
import onnx
import onnxruntime as ort
import numpy as np
from forecast_mae_prediction.src.model.trainer_forecast import Trainer as TorchModel
from forecast_mae_prediction.src.metrics.utils import sort_predictions
from forecast_mae_prediction.src.datamodule.lmdb_dataset import collate_fn
from gen_onnx import ForecastMAEONNX


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

sample_input = "/home/user/Documents/pt_files/0/1722513825.834000_60.pt"
ckpt_path = '/home/user/Downloads/epoch=86.ckpt'

# load Pytorch model
with open(sample_input, "rb") as f:
    sample_data = torch.load(f)
data = collate_fn([sample_data])

# model for Torchmodel
# Print all keys in the model
torchModel = TorchModel.load_from_checkpoint(ckpt_path, pretrained_weights=None, strict=False )
torchModel = torchModel.eval().to(device)
for key, value in data.items():
    if isinstance(value, torch.Tensor):
        data[key] = value.to(device)
with torch.no_grad():
    predictionTorch, prob = torchModel.predict(data)
    predictionTorch = predictionTorch.cpu()


# 使用 ONNX Runtime 加载模型
onnx_file_pth = "/home/user/Projects/pnp_research/forecast_mae_prediction/outputs/onnx/forecast_mae.onnx"
model = onnx.load(onnx_file_pth)
onnx.checker.check_model(model)
ort_session = ort.InferenceSession(onnx_file_pth)

input_names = [input.name for input in ort_session.get_inputs()]

# 准备输入字典
ort_inputs = {
    input_names[0]: data["x_padding_mask"],
    input_names[1]: data["x_key_padding_mask"],
    input_names[2]: data["x"],
    input_names[3]: data["x_velocity_diff"],
    input_names[4]: data["lane_padding_mask"],
    input_names[5]: data["lane_positions"],
    input_names[6]: data["lane_centers"],
    input_names[7]: data["x_angles"],
    input_names[8]: data["lane_angles"],
    input_names[9]: data["lane_key_padding_mask"],
    input_names[10]: data["x_centers"],
    input_names[11]: data["x_attr"],
}

# 确保所有输入都是 numpy 数组
for key in ort_inputs:
    if hasattr(ort_inputs[key], 'numpy'):
        ort_inputs[key] = ort_inputs[key].cpu().numpy()
onnx_output = ort_session.run(None, ort_inputs)[0]
onnx_output = torch.from_numpy(onnx_output)
print(
    "Difference of TRT and Torch Output Trajectories in 0.5 meters: ",
    torch.allclose(
        predictionTorch,
        onnx_output,
        atol=9,
    )
)
