#!/bin/bash

# sbatch train-multi.slurm

#SBATCH --job-name=train-multi
#SBATCH --comment="Xling-SemDiv train-multi"

# Trixie
#SBATCH --partition=TrixieMain,JobTesting
#SBATCH --account=dt-mtp
# On GPSC7
##SBATCH --partition=gpu_a100
##SBATCH --account=nrc_ict__gpu_a100

#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=6G

#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@30
#SBATCH --output=%x-%j.out

# Fix SLURM environment variables.
SLURM_JOB_CPUS_PER_NODE=${SLURM_JOB_CPUS_PER_NODE%%(*)}   # '24(x2)' => '24'
SLURM_JOB_CPUS_PER_NODE_PACK_GROUP_0=${SLURM_JOB_CPUS_PER_NODE_PACK_GROUP_0%%(*)}   # '24(x2)' => '24'
SLURM_STEP_TASKS_PER_NODE=${SLURM_STEP_TASKS_PER_NODE%%(*)}   # '4(x2)' => '4'
SLURM_TASKS_PER_NODE=${SLURM_TASKS_PER_NODE%%(*)}   # '4(x2)' => '4'

# NOTE: We set OMP_NUM_THREADS or else we get the following Warning:
# WARNING:torch.distributed.run:
# *****************************************
# Setting OMP_NUM_THREADS environment variable for each process to be 1 in
# default, to avoid your system being overloaded, please further tune the
# variable for optimal performance in your application as needed.
# *****************************************
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-$(nproc)}

# Requeueing on Trixie
# [source](https://www.sherlock.stanford.edu/docs/user-guide/running-jobs/)
# [source](https://hpc-uit.readthedocs.io/en/latest/jobs/examples.html#how-to-recover-files-before-a-job-times-out)
function _requeue {
   echo "BASH - trapping signal 10 - requeueing $SLURM_JOBID"
   date
   # This would allow to generically requeue any job but since we are using XLM
   # which is slurm aware, XLM could save its model before requeueing.
   scontrol requeue "$SLURM_JOBID"
}

if [[ -n "$SLURM_JOBID" ]]; then
  declare -a FORMAT_STRING=(
    JobID
    Submit
    Start
    End
    Elapsed
    ExitCode
    State
    CPUTime
    MaxRSS
    MaxVMSize
    MaxDiskRead
    MaxDiskWrite
    AllocCPUs
    AllocTRES%-50
    NodeList
    JobName%-30
    Comment%-80
  )
  SACCT_FORMAT=$(IFS=","; echo "${FORMAT_STRING[*]}")
  export SACCT_FORMAT
  trap "sacct --jobs $SLURM_JOBID --format=$SACCT_FORMAT" 0
  trap _requeue USR1
  unset FORMAT_STRING
fi

head -n 123123 "$0" >&2


# Output debugging information
function debug_info {
  echo "DEBUGGING INFO" >&2

  {
    hostname
    uname --all
    cat /etc/issue
    pwd
    which python
    #env
    # [How to list variables declared in script in bash?](https://stackoverflow.com/a/1305273)
    # In section SHELL BUILTIN COMMANDS (in the set section) it says: "In
    # posix mode, only shell variables are listed."
    ( set -o posix; set; )
    tr ":" "\n" <<< "$LD_LIBRARY_PATH"
    # What conda environment are we in?
    # Record all package versions.
    uv pip list
    uv pip freeze
    conda env export
  } >&2

  {
    # What GPU id is assigned to this job.
    nvidia-smi --list-gpus
    nvidia-smi
  } | sed 's/^/   /' >&2

  echo "DEBUGGING INFO END"
  echo;echo;echo
}

source /gpfs/projects/DT/mtp/pkgs/miniforge3/bin/activate ""
conda activate semdiv
#export https_proxy=http://webproxy.science.gc.ca:8888
#export http_proxy=http://webproxy.science.gc.ca:8888
export TQDM_MININTERVAL=30
export HF_HOME=/gpfs/projects/DT/mtp/models/HuggingFace
head_node_ip=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
head_node_port=29507

debug_info

command time --portability bash train-multi.sh en fr
