srun singularity exec --nv --writable-tmpfs \
     -B /ceph/project/es26-ce8-avs-824/whispers-in-the-storm/extern/sgmse/sgmse_venv:/scratch/sgmse_venv \
     -B $HOME/.singularity:/scratch/singularity \
     /ceph/project/es26-ce8-avs-824/whispers-in-the-storm/extern/sgmse/pytorch_24.01.sif \
     /bin/bash -c "export TMPDIR=/scratch/singularity/tmp && \
                   pip list"