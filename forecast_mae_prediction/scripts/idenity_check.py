import tensorrt as trt
import torch.nn as nn
import torch
import onnx
import onnxruntime as ort
import numpy as np
from forecast_mae_prediction.src.model.trainer_forecast import Trainer as TorchModel
from forecast_mae_prediction.src.metrics.utils import sort_predictions
from forecast_mae_prediction.src.datamodule.lmdb_dataset import collate_fn
from gen_onnx import ForecastMAEONNX


#封装 TensorRT 推理的过程
class TRTInference(nn.Module):
    def __init__(self, engine_path):
        super(TRTInference, self).__init__()
        # 日志接口，TensorRT通过该接口报告错误、警告和信息性消息
        # 正确的初始化方式
        self.logger = trt.Logger(trt.Logger.VERBOSE)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def forward(self, inputs):
        bindings = [None] * self.engine.num_bindings
        output_dict = {}
        stream = torch.cuda.current_stream().cuda_stream

        # Allocate device memory for inputs and outputs
        for i in range(self.engine.num_bindings):
            binding_name = self.engine.get_binding_name(i)
            if self.engine.binding_is_input(i):
                # Input binding
                bindings[i] = inputs[binding_name].contiguous().data_ptr()
            else:
                # Output binding
                output_shape = tuple(self.context.get_binding_shape(i))
                dtype = (
                    torch.float32
                    if self.engine.get_binding_dtype(i) == trt.DataType.FLOAT
                    else torch.int32
                )
                output = torch.empty(
                    size=output_shape, dtype=dtype, device="cuda"
                )
                bindings[i] = output.data_ptr()
                output_dict[binding_name] = output

        # Measure inference time
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()  # Record the start time

        # Execute the model
        self.context.execute_async_v2(bindings=bindings, stream_handle=stream)

        end_event.record()  # Record the end time

        # Wait for the events to complete
        torch.cuda.synchronize()

        # Calculate the elapsed time
        inference_time = start_event.elapsed_time(end_event)  # Time in milliseconds
        print(f"Inference time: {inference_time:.2f} ms")

        # Return the output tensors directly
        return output_dict

if __name__ == "__main__":
    sample_input = "/home/user/Documents/pt_files/0/1722513825.834000_60.pt"
    ckpt_path = '/home/user/Downloads/epoch=86.ckpt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    trt_engine_path = "/home/user/Projects/pnp_research/forecast_mae_prediction/outputs/onnx/forecast_mae.trt"
    trt_model = TRTInference(trt_engine_path).to(device)

    with open(sample_input, "rb") as f:
        sample_data = torch.load(f)
    data = collate_fn([sample_data])

    tensorrt_inputs = {
        "x_padding_mask": data["x_padding_mask"],
        "x_key_padding_mask": data["x_key_padding_mask"],
        "x": data["x"],
        "x_velocity_diff": data["x_velocity_diff"],
        "lane_padding_mask": data["lane_padding_mask"],
        "lane_positions": data["lane_positions"],
        "lane_centers": data["lane_centers"],
        "x_angles":  data["x_angles"],
        "lane_angles": data["lane_angles"],
        "lane_key_padding_mask": data["lane_key_padding_mask"],
        "x_centers": data["x_centers"],
        "x_attr": data["x_attr"],
    }
    tensorrt_inputs = {key: tensor.to(device) for key, tensor in tensorrt_inputs.items()}
    outputs = trt_model(tensorrt_inputs)
    tensorrt_output = {}
    tensorrt_output["prediction"] = outputs["prediction"].detach().cpu()
    tensorrt_output["prob"] = outputs["prob"].detach().cpu()

    # 使用 ONNX Runtime 加载模型
    onnx_file_pth = "/home/user/Projects/pnp_research/forecast_mae_prediction/outputs/onnx/forecast_mae.onnx"
    model = onnx.load(onnx_file_pth)
    onnx.checker.check_model(model)
    ort_session = ort.InferenceSession(onnx_file_pth)

    input_names = [input.name for input in ort_session.get_inputs()]

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
            tensorrt_output["prediction"],
            onnx_output,
            atol=0.05,
        )
    )
