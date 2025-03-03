import argparse
import logging
import os
import re
import subprocess
import sys
from datetime import datetime

from aidisdk import AIDIClient
from aidisdk.compute.job_abstract import (
    JobMountType,
    MountItem,
    MountMode,
    RunningResourceConfig,
    StartUpConfig,
)
from aidisdk.compute.package_abstract import (
    CodePackageConfig,
    LocalPackageItem,
)

from k8s_submit.config_wrapper import Config
from k8s_submit.generate_submit_files import (
    generate_bash_file,
    generate_upload_folder,
)

sys.path.append(".")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DIR_PATH = os.path.dirname(__file__)


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model-type", type=str, required=True)
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--cluster",
        type=str,
        required=True,
        help="queue name, please set up queue name of current cluster",
    )
    parser.add_argument(
        "--dag-name",
        default=None,
        type=str,
        required=False,
        help="if set, will override dag_name in config",
    )
    parser.add_argument(
        "--job-name",
        default=None,
        type=str,
        required=False,
        help="if set, will override job_name in config",
    )
    parser.add_argument(
        "--num-machines",
        default=None,
        type=int,
        required=False,
        help="if set, will override num_machines in config",
    )
    parser.add_argument(
        "--num-gpus-per-machine",
        default=None,
        type=int,
        required=False,
        help="if set, will override num_gpus_per_machine in config",
    )
    parser.add_argument(
        "--project-id",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--job-type",
        default="train",
        type=str,
        required=False,
        help="job type, use `aidisdk.compute.job_abstract.JobType` enum",
    )
    parser.add_argument("--upload-folder", type=str, default="./")
    parser.add_argument(
        "--save-upload-folder",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--sleep",
        dest="sleep",
        action="store_true",
        help="sleep in cluster for debug",
    )

    parser.add_argument(
        "--experiment-name",
        default=None,
        type=str,
        required=False,
        help="AIDI MLOPs Exmeriment name",
    )
    parser.add_argument(
        "--enable-tracking",
        action="store_true",
        default=False,
        help="use AIDI MLOPs Exmeriment and enable tracking",
    )

    known_args, unknown_args = parser.parse_known_args()
    return known_args, unknown_args
    # args = parser.parse_args()
    # return args


# def submit(
#     model_type,
#     config,
#     cluster,
#     dag_name,
#     job_name,
#     num_machines,
#     num_gpus_per_machine,
#     project_id,
#     job_type,
#     upload_folder,
#     save_upload_folder,
#     sleep,
#     experiment_name,
#     enable_tracking,
#     max_jobtime,
# ):
def submit(
    config: str,
    cluster: str,
    dag_name: str = None,
    job_name: str = None,
    num_machines: int = None,
    num_gpus_per_machine: int = None,
    job_type: str = "train",
    upload_folder: str = "./",
    save_upload_folder: bool = False,
    sleep: bool = False,
    experiment_name: str = None,
    enable_tracking: bool = False,
    args_env: list = None,
):
    """Submit cluster function.

    Args:
        config: Config file path.
        cluster: queue name of the cluster being used.
        dag_name: dag name, If None, use default in config.
        job_name: job name, if None, use default in config.
        num_machines: number of machines, if None, use default in config.
        num_gpus_per_machine: number of gpus per machine.
            If None, use default in config.
        job_type: job type, default train.
        upload_folder: Upload folder path.
        save_upload_folder: Wheather save the upload folder.
        sleep: Sleep in cluster for debug.
        experiment_name: AIDI MLOPs Experiment name.
        enable_tracking: Use AIDI MLOPs Exmeriment and enable tracking.
        args_env: The args will be set in env.
    """
    cfg = Config.fromfile(config)
    if "k8s_config" in cfg:
        cfg = Config(cfg["k8s_config"])

    upload_folder = generate_upload_folder(
        cfg, upload_folder, cfg.upload_folder_name
    )

    generate_bash_file(cfg, upload_folder, sleep)
    # generate_bash_file(cfg, upload_folder, 1)

    try:
        client = AIDIClient()
        job_name = job_name if job_name is not None else cfg.job_name
        num_machines = (
            num_machines if num_machines is not None else cfg.num_machines
        )
        num_gpu = (
            num_gpus_per_machine
            if num_gpus_per_machine is not None
            else cfg.num_gpus_per_machine
        )

        mount_list = []
        # input bucket
        if cfg.get("input_bucket") is not None:
            bucket_list = cfg.input_bucket.split(",")
            for bucket in bucket_list:
                mount_item = MountItem(
                    mount_type=JobMountType.BUCKET,
                    name=bucket,
                    mode=MountMode.READ_ONLY,
                )
                mount_list.append(mount_item)
        # output bucket
        if cfg.get("output_bucket") is not None:
            bucket_list = cfg.output_bucket.split(",")
            for bucket in bucket_list:
                mount_item = MountItem(
                    mount_type=JobMountType.BUCKET,
                    name=bucket,
                    mode=MountMode.READ_AND_WRITE,
                )
                mount_list.append(mount_item)

        time_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")

        def _check_name(value, re_exp="\W"):  # noqa W605
            res = re.findall(re_exp, value)
            if res:
                logger.warning(
                    f"Task name must match ^[a-zA-Z]+[a-zA-Z0-9_]*$, but get "
                    f"{res} in your task name, which will be replaced by `_`."
                )
                for c in res:
                    value = value.replace(c, "_")
            return value

        dag_name = (
            dag_name if dag_name is not None else cfg.get("dag_name", None)
        )
        dag_name = (
            f"{dag_name}_{time_suffix}"
            if dag_name is not None
            else f"DAG_{job_name}_{time_suffix}"
        )

        dag = client.dag.new_dag(
            name=_check_name(dag_name),
            project_id=cfg.project_id,
            queue_name=cluster,
            code_package=CodePackageConfig(
                raw_package=LocalPackageItem(
                    lpath=upload_folder,
                    encrypt_passwd=cfg.job_password,
                    follow_softlink=True,
                ).set_as_startup_dir(),
                # git_packages=[hat_git],
            ),
        )

        _ = dag.new_job(
            name=_check_name(job_name + f"_{time_suffix}"),
            job_type=job_type,
            queue_name=cluster,
            running_resource=RunningResourceConfig(
                docker_image=cfg.docker_image,
                instance=num_machines,
                gpu=num_gpu,
                walltime=cfg.max_jobtime,
            ),
            mount=mount_list,
            startup=StartUpConfig(
                command="${WORKING_PATH}/job.sh",
            ),
            code_package=CodePackageConfig(
                raw_package=LocalPackageItem(
                    lpath=upload_folder,
                    encrypt_passwd=cfg.job_password,
                    follow_softlink=True,
                ).set_as_startup_dir(),
                # git_packages=[hat_git],
            ),
            priority=cfg.priority,
        )

        if experiment_name is not None:
            if client.experiment.get_experiment(experiment_name) is None:
                client.experiment.create_experiment(
                    name=experiment_name,
                    project_id=cfg.get("project_id", None),
                )
            with client.experiment.init(
                experiment_name=experiment_name,
                run_name=f"DAG-{job_type}",
                enabled=enable_tracking,
            ) as run:
                run.log_runtime(runtime=dag, config_file=config)
                client.dag.submit_dag(dag)
        else:
            client.dag.submit_dag(dag)

        print(dag)
        print("Submit dag successfully.")
    except Exception as e:
        logger.error("submit failed! " + str(e))
        raise e
    finally:
        if not save_upload_folder:
            if os.path.exists(upload_folder):
                subprocess.check_call(["rm", "-rf", upload_folder])


if __name__ == "__main__":
    args, args_env = parse_args()

    submit(
        config=args.config,
        cluster=args.cluster,
        dag_name=args.dag_name,
        job_name=args.job_name,
        num_machines=args.num_machines,
        num_gpus_per_machine=args.num_gpus_per_machine,
        job_type=args.job_type,
        upload_folder=args.upload_folder,
        save_upload_folder=args.save_upload_folder,
        sleep=args.sleep,
        experiment_name=args.experiment_name,
        enable_tracking=args.enable_tracking,
        args_env=args_env,
    )

    # submit(**vars(args))
