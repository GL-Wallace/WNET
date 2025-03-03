import os

job_password = "newk8s666"

# aidi training setting
launcher = "torch"
priority = 5
project_id = "LIT24040-pnc"
docker_image = (
    "cr-aidi-harbor-cn-shanghai-selfdriving-vecps.cr.autodriving."
    + "volcengine.com/bev_perception/bev_perception:ubuntu20.04-gcc9.4-py3.8-cuda11.3-torch1.11.0_mae"
)
# cr-aidi-harbor-cn-shanghai-selfdriving-vecps.cr.autodriving.volcengine.com/dlp/base:centos7.6-gcc5.4-py3.8-cuda11.1-horizon-v1.2.2 

num_machines = int(os.getenv("NUM_MACHINES", "1"))
num_gpus_per_machine = int(os.getenv("NUM_GPUS_PER_MACHINE", "8"))
max_jobtime = int(os.getenv("MAX_JOB_TIME_MINUTES", "14400"))
cluster = os.getenv("CLUSTER")

# aidi file setting
upload_folder_name = "k8s_job"
folder_list = [
    "forecast_mae_prediction",
    "k8s_submit",
    "k8s_submit/url2IP.py",
    "k8s_submit/ssh_launcher.py",
]
input_bucket = "carizon_pnp_jfs"
output_bucket = "carizon_pnp_jfs"

custom_cmds_before_job_list = [
    "export NUM_MACHINES=%d" % num_machines,
    "export NUM_GPUS_PER_MACHINE=%d" % num_gpus_per_machine,
]
# startswith 'python' command
job_list = [os.getenv("RUN_CMD")]
