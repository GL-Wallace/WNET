import getpass

import torch
import torch.nn.functional as F
from configs import DATASET
from data.collate import qcnet_collate
from data.dataset.traj_pred_dataset import ArgoverseIIDataset
from metrics import Brier, minADE, minFDE
from models.structure import QCNet
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import (
    count_parameters,
    create_directories_and_file,
    recursive_to_device,
)

device_ids = [3, 6, 7]
dataset_name = "prediction"
task_name = "qcnet_test_0813"
epochs = 64
float_lr = 5e-4
train_num_workers, val_num_workers = 0, 8
weight_decay = 1e-4
batch_size_per_gpu = 1
T_max = 64
log_every_n_steps = 50
user_name = getpass.getuser()
base_path = f"/home/users/{user_name}/results/"
dataset_name = "prediction"

device = "cuda" if torch.cuda.is_available() else "cpu"

train_data = "Argoverse2_train"
val_data = "Argoverse2_val"

train_path = DATASET[dataset_name][train_data]
val_path = DATASET[dataset_name][val_data]

val_data = ArgoverseIIDataset(
    data_path=val_path,
    mode="val",
    dataset_size=5,
    reprocess=False,
)

val_loader = DataLoader(
    dataset=val_data,
    batch_size=1,
    shuffle=False,
    collate_fn=qcnet_collate,
)

checkpoint_path = (
    "/home/users/huajiang.liu/results/qcnet_0813_aidi/"
    + "checkpoint/best_model.pth"
)
checkpoint = torch.load(checkpoint_path, map_location="cpu")

ckpt_dir, viz_dir, tensorboard_dir, file_path = create_directories_and_file(
    base_path, task_name
)

qcnet = QCNet(
    ckpt_dir,
    tensorboard_dir,
    viz_dir,
    file_path,
    float_lr,
    weight_decay,
    T_max,
    log_every_n_steps,
).to(device)
qcnet.load_state_dict(checkpoint)

qcnet.eval()


num_params = count_parameters(qcnet)
print(f"Number of parameters: {num_params}")

fde_metric = minFDE(max_guesses=6, device=device)
ade_metric = minADE(max_guesses=6, device=device)
brier_metric = Brier(max_guesses=6, device=device)
fde_metric_k1 = minFDE(max_guesses=1, device=device)
ade_metric_k1 = minADE(max_guesses=1, device=device)


for idx, val_data in enumerate(tqdm(val_loader)):
    with torch.no_grad():
        val_data = recursive_to_device(val_data, device)
        output = qcnet(val_data)
        pred, target, prob, valid_mask = (
            output["predict_trajs"].squeeze(),
            output["future_trajectories"].squeeze(),
            output["real_scores"].squeeze(),
            output["valid_masks"].squeeze(),
        )

        pi_eval = F.softmax(prob, dim=1)

        B, N = val_data["agent_features"]["categories"].size()
        eval_mask = (val_data["agent_features"]["categories"] == 3).reshape(
            B * N
        )
        valid_pred = pred[eval_mask]
        valid_target = target[eval_mask]
        valid_prob = pi_eval[eval_mask]
        valid_mask_eval = valid_mask[eval_mask]

        brier_metric.update(
            valid_pred, valid_target, valid_prob, valid_mask_eval
        )
        fde_metric.update(
            valid_pred, valid_target, valid_prob, valid_mask_eval
        )
        ade_metric.update(
            valid_pred, valid_target, valid_prob, valid_mask_eval
        )
        fde_metric_k1.update(
            valid_pred, valid_target, valid_prob, valid_mask_eval
        )
        ade_metric_k1.update(
            valid_pred, valid_target, valid_prob, valid_mask_eval
        )


# Compute the final metrics
brier_score = brier_metric.compute()
fde_score = fde_metric.compute()
ade_score = ade_metric.compute()
ade_score_k1 = ade_metric_k1.compute()
fde_score_k1 = fde_metric_k1.compute()

print(
    f"Brier Score: {brier_score.item()} ",
    f"FDE(K=6) Score: {fde_score.item()} ",
    f"FDE(K=1) Score: {fde_score_k1.item()} ",
    f"ADE(K=6) Score: {ade_score.item()} "
    f"ADE(K=1) Score: {ade_score_k1.item()} ",
)
