#!/bin/bash

#SBATCH --job-name=build_basic
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=out.%x
#SBATCH --error=err.%x

export SINGULARITY_TMPDIR=$HOME/.singularity/tmp
export SINGULARITY_CACHEDIR=$HOME/.singularity/cache
mkdir -p $SINGULARITY_CACHEDIR $SINGULARITY_TMPDIR

# URI of container to pull
uri="nvcr.io/nvidia/pytorch:24.01-py3"

# Name of the resulting container
output_sif="pytorch_24.01.sif"

# Call the build instructions
singularity pull $output_sif docker://$uri