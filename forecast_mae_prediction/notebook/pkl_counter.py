import pickle
import random
from argparse import ArgumentParser
import os
import shutil


# -------------- model data dump module --------------

def replicate_pt(src_dir, dest_dir, file_limit=500):
    # duplicate n=200 files into /home/users/huajiang.liu/intern.guowei.zhang/pkl_files
    os.makedirs(dest_dir, exist_ok=True)

    total_copied = 0

    for root, dirs, files in os.walk(src_dir):
        for subdir in dirs:
            if total_copied >= file_limit:
                break

            subdir_path = os.path.join(root, subdir)
            files_in_subdir = os.listdir(subdir_path)

            for file_name in files_in_subdir:
                if total_copied < file_limit:
                    src_file = os.path.join(subdir_path, file_name)
                    dest_file = os.path.join(dest_dir, file_name)
                    shutil.copy2(src_file, dest_file)
                    total_copied += 1
                else:
                    break

            if total_copied >= file_limit:
                break

    print(f"Copied a total of {total_copied} files to {dest_dir}")


def gen_dataset_pkl(args):
    print(args.is_viz)
    save_dir = args.save_dir

    train_data_list_path = os.path.join(save_dir, "train_data_list.pkl")
    val_data_list_path = os.path.join(save_dir, "val_data_list.pkl")
    # all_pickle_files = os.listdir(save_dir)
    all_valid_pickles = []
    for root, dirs, files in os.walk(save_dir):
        for file in files:
            if (file.startswith("172") or file.startswith("171") or file.startswith("170")):
                file_path = os.path.join(root, file)
                if os.path.getsize(file_path) > 0:  # 检查文件大小是否大于0
                    all_valid_pickles.append(file_path)
    # all_valid_pickles = sorted(
    #     all_valid_pickles,
    #     key=lambda x: (
    #         int(x.split("_")[0]),
    #         int(x.split("_")[1].split(".")[0]),
    if args.is_viz:
        train_ratio = 0
    else:
        random.shuffle(all_valid_pickles)
        train_ratio = args.train_ratio

    # * set the pkl number to what you want it to be!
    if args.total_pkl_len == 0:
        pkl_num = len(all_valid_pickles)
    else:
        pkl_num = args.total_pkl_len
    print("the actuary length of total data is:", len(all_valid_pickles))
    print("The pkl length we set is", pkl_num)

    train_num = int(pkl_num * train_ratio)
    train_data_list = [
        os.path.join(save_dir, train_data)
        for train_data in all_valid_pickles[:train_num]
    ]
    val_data_list = [
        os.path.join(save_dir, val_data)
        for val_data in all_valid_pickles[train_num:pkl_num]
    ]

    with open(train_data_list_path, "wb") as f:
        print("len of train data is:", len(all_valid_pickles[:train_num]))
        pickle.dump(train_data_list, f)
    with open(val_data_list_path, "wb") as f:
        print("len of val data is:", len(all_valid_pickles[train_num:pkl_num]))
        pickle.dump(val_data_list, f)

    n = 5  # Number of items to print, just checking

    with open(train_data_list_path, "rb") as f:
        loaded_train_data_list = pickle.load(f)
        print("First {} items in train_data_list:".format(n))
        for item in loaded_train_data_list[:n]:
            print(item)

    with open(val_data_list_path, "rb") as f:
        loaded_val_data_list = pickle.load(f)
        print("First {} items in val_data_list:".format(n))
        for item in loaded_val_data_list[:n]:
            print(item)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_dir", type=str,
                        default='/home/user/Documents/pt_files')
    parser.add_argument("--is_viz", action="store_false", default=False)
    parser.add_argument("--train_ratio", type=float, default=0.80)
    parser.add_argument("--total_pkl_len", type=int, default=0)  # 0 is for default actuary data length in buckets.
    args = parser.parse_args()
    # args = args("/home/user/Projects/raw_data/argoverse2/raw_argo_partly/raw",multiagent=False)
    # replicate_pt(src_dir='/horizon-bucket/carizon_pnp_jfs/pkl/2024-10-28-10:49:36_pkl_intersection_normal_gwz/', dest_dir='/home/users/huajiang.liu/intern.guowei.zhang/pkl_files/intersection_01')
    gen_dataset_pkl(args)
