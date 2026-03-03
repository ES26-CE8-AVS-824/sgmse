#!/bin/bash

#SBATCH --job-name=mambattention_inference
#SBATCH --output=mambattention_inference.out
#SBATCH --error=mambattention_inference.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

singularity exec --nv \
     -B /ceph/project/es26-ce8-avs-824/whispers-in-the-storm/extern/MambAttention/mambattention_venv:/scratch/mambattention_venv \
     -B $HOME/.singularity:/scratch/singularity \
     /ceph/container/pytorch/pytorch_26.01.sif \
     /bin/bash -c "export TMPDIR=/scratch/singularity/tmp && \
                   export TRITON_LIBCUDA_PATH=/.singularity.d/libs && \
                   source /scratch/mambattention_venv/bin/activate && \
                   python inference.py \
                     --input_folder=data/inference_input \
                     --output_folder=data/inference_output \
                     --checkpoint_file=checkpoints/seed3441.pth \
                     --config=checkpoints/MambAttention_seed3441_VB-DemandEx.yaml"
