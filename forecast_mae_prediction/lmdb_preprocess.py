from pathlib import Path
from typing import List
from argparse import ArgumentParser
import ray
# from src.datamodule.lmdb_extractor import lmdb_extractor
from src.datamodule.lmdb_obs_extractor import lmdb_extractor
from src.datamodule.av2_extractor_multiagent import Av2ExtractorMultiAgent
from src.utils.ray_utils import ActorHandle,ProgressBar
import os

ray.init(num_cpus=16)

class Args:
    def __init__(self, data_root, save_path, multiagent=False):
        self.data_root = data_root
        self.save_path = save_path
        self.multiagent = multiagent

def preprocess_batch(extractor: lmdb_extractor, file_list: List[Path], pb: ActorHandle):
    for file in file_list:
        extractor.save(file)
        pb.update.remote(1)

def preprocess(data_root, save_path):
    save_path = Path(save_path) 
    save_path.mkdir(exist_ok=True, parents=True) 
    if args.multiagent:
        extractor = Av2ExtractorMultiAgent(save_path = save_path)
    else:
        extractor = lmdb_extractor(save_path = save_path)

    save_path.mkdir(exist_ok=True, parents=True)
    extractor.save(data_root)
    return len(extractor.data)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_root", "-d", type=str, required=False, default="/home/user/Projects/raw_data/data/")
    parser.add_argument("--save_path", "-s", type=str, required=False, default="/home/user/Documents/pt_files")
    parser.add_argument("--multiagent", "-m", type=bool, default=False)
    args = parser.parse_args()
  
    scenario = os.path.basename(args.data_root)
    max_pkl_per_folder = 5000
    current_sum_num = 0
    dir_index = 0
    pack_dirs = [
        d
        for d in os.listdir(args.data_root)
        if os.path.isdir(os.path.join(args.data_root, d))
    ]
    sub_save_path = os.path.join(args.save_path, f"{scenario}_{dir_index}")
    print(sub_save_path)
    for pack_dir in pack_dirs:
        if pack_dir == "pkl":
            continue
        print(pack_dir)
        if current_sum_num >= max_pkl_per_folder:
            dir_index += 1
            current_sum_num = 0
            sub_save_path = os.path.join(args.save_path, f"{scenario}_{dir_index}")
        if pack_dir != "output":
            curr_pack_path = os.path.join(args.data_root, pack_dir)
            pt_count = preprocess(curr_pack_path, sub_save_path)
            if pt_count:
                current_sum_num += pt_count
    # args = Args("/home/sy/workspace/Pnp/pnp_model/data/EN8610_20241001-144232-188", "/home/sy/workspace/Pnp/pnp_model/data/EN8610_20241001-144232-188/pkl", multiagent=False)
    