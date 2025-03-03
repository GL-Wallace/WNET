import os
import random
import subprocess


def generate_upload_folder(cfg, upload_folder, upload_folder_name):
    upload_folder = os.path.join(upload_folder, upload_folder_name)
    subprocess.check_call(["mkdir", "-p", upload_folder])

    assert "folder_list" in cfg
    for path in cfg.folder_list:
        if not os.path.exists(path):
            print(f"{path} not exists, skip")
            continue
        # support converting soft links to files.
        subprocess.check_call(["rsync", "-aL", path, upload_folder])
        print("copy %s to %s" % (path, upload_folder))

    return upload_folder


def write_mpi_multi_machines_cmd(fn, cfg, job):
    """Write job to fn."""
    command = "mpirun -n %d -ppn %d --hostfile %s %s %s" % (
        cfg.num_machines * cfg.num_gpus_per_machine,
        cfg.num_gpus_per_machine,
        "/job_data/mpi_hosts",
        job,
        "--dist-url tcp://$dis_url:8000 --launcher mpi ",
    )
    fn.write("%s\n" % command)


def write_torch_multi_machines_cmd(fn, cfg, job):
    """Write job to fn."""
    if job.startswith("python3"):
        cmd = (
            "torchrun --nnodes=%d --nproc_per_node=%d --rdzv_id=%d"
            "--rdzv_backend=c10d --rdzv_endpoint=$HOST_NODE_ADDR %s"
            % (
                cfg.num_machines,
                cfg.num_gpus_per_machine,
                random.randint(0, 100000),
                job.replace("python3 ", ""),
            )
        )
    else:
        cmd = job
    command = "python3 ssh_launcher.py -n %d -g %d -H %s '%s'" % (
        cfg.num_machines,
        cfg.num_gpus_per_machine,
        "/job_data/mpi_hosts",
        cmd,
    )
    fn.write("%s\n" % command)


def write_multi_machines_cmd(fn, cfg, job, launcher):
    if launcher == "mpi":
        write_mpi_multi_machines_cmd(fn, cfg, job)
    elif launcher == "torch":
        write_torch_multi_machines_cmd(fn, cfg, job)
    else:
        raise ValueError("launcher only supports %s, %s" % ("mpi", "torch"))


def generate_bash_file(cfg, upload_folder, run_in_sleep):
    bash_file = os.path.join(upload_folder, "job.sh")
    with open(bash_file, "w") as fn:
        fn.write("set -e\n")
        fn.write("export PYTHONPATH=${WORKING_PATH}:$PYTHONPATH\n")
        fn.write("date\n")
        fn.write("env\n")
        fn.write("pip3 list\n")
        fn.write("export PYTHONUNBUFFERED=0\n")

        fn.write("cd ${WORKING_PATH}\n")
        if run_in_sleep:
            fn.write("sleep 1000000m\n")

        for cmd in getattr(cfg, "prefix_cmds_on_master", []):
            fn.write(f"{cmd}\n")

        if cfg.num_machines > 1:  # multi-machines
            fn.write("python3 url2IP.py\n")
            fn.write("cat /job_data/mpi_hosts\n")
            fn.write("dis_url=$(head -n +1 /job_data/mpi_hosts)\n")
            if hasattr(cfg, "custom_cmds_before_job_list"):
                cmds_file = os.path.join(
                    upload_folder, "custom_cmds_before_job_list.sh"
                )
                with open(cmds_file, "w") as cus:
                    for cmd in cfg.custom_cmds_before_job_list:
                        cus.write("%s\n" % cmd)
                job = "bash ${WORKING_PATH}/custom_cmds_before_job_list.sh"
                write_multi_machines_cmd(fn, cfg, job, cfg.launcher)

            for job in cfg.job_list:
                write_multi_machines_cmd(fn, cfg, job, cfg.launcher)

            if hasattr(cfg, "custom_cmds_after_job_list"):
                cmds_file = os.path.join(
                    upload_folder, "custom_cmds_after_job_list.sh"
                )
                with open(cmds_file, "w") as cus:
                    for cmd in cfg.custom_cmds_after_job_list:
                        cus.write("%s\n" % cmd)
                job = "bash ${WORKING_PATH}/custom_cmds_after_job_list.sh"
                write_multi_machines_cmd(fn, cfg, job, cfg.launcher)
        else:
            if hasattr(cfg, "custom_cmds_before_job_list"):
                for cmd in cfg.custom_cmds_before_job_list:
                    fn.write("%s\n" % cmd)
            for job in cfg.job_list:
                if cfg.launcher == "torch":
                    job = (
                        "torchrun --nproc_per_node=%d "
                        % cfg.num_gpus_per_machine
                        + job.replace("python3 ", "")
                    )
                fn.write("%s\n" % job)
            if hasattr(cfg, "custom_cmds_after_job_list"):
                for cmd in cfg.custom_cmds_after_job_list:
                    fn.write("%s\n" % cmd)

        for cmd in getattr(cfg, "suffix_cmds_on_master", []):
            fn.write(f"{cmd}\n")

    subprocess.check_call(["chmod", "777", bash_file])
