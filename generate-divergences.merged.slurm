#!/bin/bash

# parallel generate-divergences.merged.slurm en fr

#SBATCH --job-name=generate-divergences.merged
#SBATCH --comment="Xling-SemDiv generate-divergences"

# Trixie
#SBATCH --partition=TrixieMain,JobTesting
#SBATCH --account=dt-mtp
# On GPSC7
##SBATCH --partition=gpu_a100
##SBATCH --account=nrc_ict__gpu_a100

#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G

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
head_node_port=29501

debug_info


# ==== Set directories
# readonly root_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
readonly root_dir="/home/larkins/git/CAISI/xling-SemDiv"
readonly data_dir=$root_dir/data
readonly scripts_dir=$root_dir/source
readonly seed_file=WikiMatrix.$1-$2.tsv.filtered_sample_50000.moses.seed
readonly seeds=$data_dir/wikimatrix/$seed_file
readonly job_id=${SLURM_JOBID:+-$SLURM_JOBID}


# === Set dependencies
export NLTK_DATA=$data_dir/nltk_data
python -m spacy download 'en_core_web_sm'

cut -f 1-2 "$seeds" > "${seeds}_exclude_align"

echo $'\n> Prepare divergence ranking for sentence-level divergentmBERT'
command time --portability python "$scripts_dir/build_bert_training_data.py" \
  --path_to_unlabeled "${seeds}_exclude_align" \
  --path_to_divergences "$root_dir/synthetic/from_$seed_file" \
  --divergent_list rdpg \
  --contrastive \
  --learn-to-rank \
  --divergence-ranking

echo $'\n> Prepare divergence ranking for multi-task divergentmBERT'
command time --portability python "$scripts_dir/build_bert_training_data.py" \
  --path_to_unlabeled "${seeds}_exclude_align" \
  --path_to_divergences "$root_dir/synthetic/from_$seed_file" \
  --divergent_list rdpg \
  --contrastive \
  --learn-to-rank \
  --divergence-ranking \
  --multi-task


readonly label_file=$root_dir/for_divergentmBERT/from_$seed_file/contrastive_multi_hard/rdpg/labels.txt
echo "O" > "$label_file"
echo "D" >> "$label_file"
