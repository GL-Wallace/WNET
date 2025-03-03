from pathlib import Path
import torch
import pickle
import os
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class LmdbDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        mode: str = None,
    ):
        super(LmdbDataset, self).__init__()
        self.file_list = []
        
        if mode == "train":
            data_list_path = os.path.join(data_root, mode + "_data_list.pkl")
        elif mode == "val":
            data_list_path = os.path.join(data_root, mode + "_data_list.pkl")
        else:
            data_list_path = os.path.join(data_root, "data_list.pkl")
        
        # Load the data list for this root
        with open(data_list_path, "rb") as f:
                self.file_list.extend(pickle.load(f))

        print(
            f"data root: {data_root}, total number of files: {len(self.file_list)}"
        )

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int):
        try:
            data = torch.load(self.file_list[index])
        except EOFError:
            print(f"EOFError: Failed to load {self.file_list[index]}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None
            
        return data


def collate_fn(batch):
    # 过滤掉 None 值
    batch = [b for b in batch if b is not None]

    # 确保批次不为空
    if not batch:
        return None
    data = {}

    for key in [
        "x",
        "x_attr",
        "x_positions",
        "x_centers",
        "x_angles",
        "x_velocity",
        "x_velocity_diff",
        "lane_positions",
        "lane_centers",
        "lane_angles",
        "lane_attr"
    ]:
        data[key] = pad_sequence([b[key].float() for b in batch], batch_first=True)

    if batch[0]["y"] is not None:
        data["y"] = pad_sequence([b["y"] for b in batch], batch_first=True)

    for key in ["x_padding_mask", "lane_padding_mask"]:
        data[key] = pad_sequence(
            [b[key] for b in batch], batch_first=True, padding_value=True
        )

    data["x_key_padding_mask"] = data["x_padding_mask"].all(-1)
    data["lane_key_padding_mask"] = data["lane_padding_mask"].all(-1)
    data["num_actors"] = (~data["x_key_padding_mask"]).sum(-1)
    data["num_lanes"] = (~data["lane_key_padding_mask"]).sum(-1)
    data["track_id"] = [b["track_id"] for b in batch]
    data["origin"] = torch.cat([b["origin"] for b in batch], dim=0)
    data["theta"] = torch.cat([b["theta"] for b in batch])

    return data
