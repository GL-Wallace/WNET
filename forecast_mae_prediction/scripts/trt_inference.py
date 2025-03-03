import tensorrt as trt
import torch
import torch.nn as nn
from gen_onnx import ForecastMAEONNX
from forecast_mae_prediction.src.model.trainer_forecast import Trainer as TorchModel
from forecast_mae_prediction.src.metrics.utils import sort_predictions
from forecast_mae_prediction.src.datamodule.lmdb_dataset import collate_fn

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


# Create the TRT model
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    trt_engine_path = "/home/user/Projects/pnp_research/forecast_mae_prediction/outputs/onnx/forecast_mae.trt"
    trt_model = TRTInference(trt_engine_path).to(device)

    sample_input = "/home/user/Documents/pt_files/0/1722513825.834000_60.pt"
    ckpt_path = '/home/user/Downloads/epoch=86.ckpt'
    ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    state_dict = {
        k.replace("net.", "encoder."): v for k, v in ckpt.items() if k.startswith("net.")
    }
    model = ForecastMAEONNX()
    model.load_state_dict(state_dict=state_dict, strict=False)
    model.eval().to(device)

    # prepare data for models: tensorRT, onnx (CPU)
    with open(sample_input, "rb") as f:
        sample_data = torch.load(f)
    data = collate_fn([sample_data])

    # model for Torchmodel
    torchModel = TorchModel.load_from_checkpoint(ckpt_path, pretrained_weights=None, strict=False)
    torchModel = torchModel.eval().to(device)
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            data[key] = value.to(device)
    with torch.no_grad():
        predictionTorch, prob = torchModel.predict(data)
        predictionTorch = predictionTorch.cpu()

    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            data[key] = value.to(device)

    # tensorrt_inputs_int32 = {}
    # for key, tensor in data.items():
    #     if isinstance(tensor, list):
    #         tensor = torch.tensor(tensor)
    #     tensorrt_inputs_int32[key] = tensor.to(torch.int32)

    x_padding_mask = data["x_padding_mask"]
    x_key_padding_mask = data["x_key_padding_mask"]
    x = data["x"]
    x_velocity_diff = data["x_velocity_diff"]
    lane_padding_mask = data["lane_padding_mask"]
    lane_positions = data["lane_positions"]
    lane_centers = data["lane_centers"]
    x_angles = data["x_angles"]
    lane_angles = data["lane_angles"]
    lane_key_padding_mask = data["lane_key_padding_mask"]
    x_centers = data["x_centers"]
    x_attr = data["x_attr"]
    lane_centers = data["lane_centers"]

    torch_output = model(x_padding_mask, x_key_padding_mask, x, x_velocity_diff, lane_padding_mask, lane_positions,
                         lane_centers, x_angles, lane_angles, lane_key_padding_mask, x_centers, x_attr)
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

    # # Create a dictionary of inputs
    # tensorrt_inputs = {
    #     "x_padding_mask": tensorrt_inputs_int32["x_padding_mask"],
    #     "x_key_padding_mask": tensorrt_inputs_int32["x_key_padding_mask"],
    #     "x": tensorrt_inputs_int32["x"],
    #     "x_velocity_diff": tensorrt_inputs_int32["x_velocity_diff"],
    #     "lane_padding_mask": tensorrt_inputs_int32["lane_padding_mask"],
    #     "lane_positions": tensorrt_inputs_int32["lane_positions"],
    #     "lane_centers": tensorrt_inputs_int32["lane_centers"],
    #     "x_angles":  tensorrt_inputs_int32["x_angles"],
    #     "lane_angles": tensorrt_inputs_int32["lane_angles"],
    #     "lane_key_padding_mask": tensorrt_inputs_int32["lane_key_padding_mask"],
    #     "x_centers": tensorrt_inputs_int32["x_centers"],
    #     "x_attr": tensorrt_inputs_int32["x_attr"],
    # }

    # Perform inference
    outputs = trt_model(tensorrt_inputs)
    tensorrt_output = {}

    tensorrt_output["prediction"] = outputs["prediction"].detach().cpu()
    tensorrt_output["prob"] = outputs["prob"].detach().cpu()

    predictions = torch_output[0].detach().cpu()
    prob=torch_output[1].detach().cpu()

    # Print the outputs for comparison
    print("TensorRT Prediction:")
    print(tensorrt_output["prediction"])
    print("\nTorch Prediction:")
    print(predictions)
    print("\nTensorRT Probability:")
    print(tensorrt_output["prob"])
    print("\nTorch Probability:")
    print(prob)

    print(
        "Difference of TRT and Torch Output Trajectories in 0.05 meters: ",
        torch.allclose( # 用于比较两个张量是否在指定的绝对公差 (atol) 和相对公差 (rtol) 范围内近似相等
            predictionTorch,
            predictions,
            atol=0.05,
        )
    )
    print(
        "Difference of TRT and Torch Output Trajectories in 0.5 meters: ",
        torch.allclose( # 用于比较两个张量是否在指定的绝对公差 (atol) 和相对公差 (rtol) 范围内近似相等
            tensorrt_output["prediction"],
            predictions,
            atol=9,
        )
    )
