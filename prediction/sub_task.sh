export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
export PNP_DIR="/home/users/huajiang.liu/dev/pnp_research"
export PYTHONPATH=${PNP_DIR}:${PYTHONPATH}

export NUM_MACHINES=4
export NUM_GPUS_PER_MACHINE=8
export MAX_JOB_TIME_MINUTES=14400
export CLUSTER="carizon-4090-pnc"
export JOB_NAME="qcnet_traj_1.0"

# 1. For trajectory prediction: local training
# python3 -W ignore ${PNP_DIR}/prediction/qcnet_traj.py

# 2. For trajectory prediction: local validation
# python3 -W ignore ${PNP_DIR}/prediction/validation.py

# 3. For trajectory prediction: train on aidi
export RUN_CMD='python3 prediction/qcnet_traj.py \
--num_nodes ${NUM_MACHINES} \
--devices ${NUM_GPUS_PER_MACHINE}'

python3 -W ignore k8s_submit/submit.py \
--config k8s_submit/k8s_config.py --cluster $CLUSTER --job-name $JOB_NAME