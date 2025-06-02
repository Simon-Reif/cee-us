#!/bin/bash

# Sample Slurm job script for Galvani 

#SBATCH -J planner_stack_data_collection               # Job name
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=8          # Number of CPU cores per task
#SBATCH --nodes=1                  # Ensure that all cores are on the same machine with nodes=1
#SBATCH --partition=a100-galvani   # Which partition will run your job
#SBATCH --time=3-00:00             # Allowed runtime in D-HH:MM
#SBATCH --gres=gpu:1               # (optional) Requesting type and number of GPUs
#SBATCH --mem=50G                  # Total memory pool for all cores (see also --mem-per-cpu); exceeding this number will cause your job to fail.
#SBATCH --output=/mnt/lustre/work/martius/mot363/cee-us/slurm-outputs/%x-%j.out       # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=/mnt/lustre/work/martius/mot363/cee-us/slurm-outputs/%x-%j.err        # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=END            # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=simonreif@t-online.de   # Email to which notifications will be sent

# Diagnostic and Analysis Phase - please leave these in.
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi # only if you requested gpus
LUST_WORK=/mnt/lustre/work/martius/mot363
SING_PATH=$LUST_WORK/Singularity/ubuntu2204.sif
PROJECT_PATH=$LUST_WORK/cee-us
#ls $WORK # not necessary just here to illustrate that $WORK is available here

# Setup Phase
# add possibly other setup code here, e.g.
# - copy singularity images or datasets to local on-compute-node storage like /scratch_local
# - loads virtual envs, like with anaconda
# - set environment variables
# - determine commandline arguments for `srun` calls
cd $PROJECT_PATH
singularity exec --nv $SING_PATH python mbrl/main.py experiments/cee_us/settings/construction/zero_shot_generalization/fixed_goal_positions/stack.yaml
# Compute Phase
#srun python3 runfile.py  # srun will automatically pickup the configuration defined via `#SBATCH` and `sbatch` command line arguments  