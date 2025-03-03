# PnP (Prediction and Planning) Module

This repository implements data-driven prediction and planning algorithms. Documentations can be found: https://carizon.feishu.cn/wiki/MKWhwS1AVih5orkxCIhcbqHwnOf.

## Introduction
PnP aims at developing efficient and user-friendly AI(Deep Learning) data-driven prediction and planning algorithms(based on Pytorch APIs) for Carizon projects.

## Installation
See [installation instructions](https://carizon.feishu.cn/wiki/GwM0wpJjniIAJ0kigZ0cCprnnCb) for running algorithms locally or using AIDI platform.

## Getting Started
To create an conda environment locally, simply run
```
conda create -n py38cu118
conda activate py38cu118
cd prediction
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Train Models
Refer to the sub_task.sh to train your model locally or online.

## Visualization
The project provides scripts for visualizing preprocessed batch data as well as model inference results.
### Data Pipeline Visualization
You can visualize the packed LDBD data using the following script:  
```
cd /path/to/your/pnp_research/prediction
export PYTHONPATH=./
python3 ./visualization/viz_lmdb.py --root /home/user/dev/lmdb/20240812-210838 --start 1000 --end 2000 --interval 50 --save_dir ./tmp/viz
```

### QCNet Prediction Visualization
Before you start, ensure the parameters in viz_qcnet.py is correctly configured. Then, you can visualize the predicted results of QCNet using the following script:  
```
cd /path/to/your/pnp_research/prediction
export PYTHONPATH=./
python3 ./visualization/viz_qcnet.py --argo_path /home/user/dev/argoverse2 --ckpt_path /home/user/dev/best_model.pth --start 5 --end 10 --frame 49 --save_dir ./tmp/viz
```