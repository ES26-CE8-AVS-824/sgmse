#srun singularity exec --nv --writable-tmpfs \
#    /ceph/project/es26-ce8-avs-824/whispers-in-the-storm/extern/sgmse/pytorch_24.01.sif \
#    pip install --user virtualenv

#srun singularity exec --nv \
#    /ceph/project/es26-ce8-avs-824/whispers-in-the-storm/extern/sgmse/pytorch_24.01.sif \
#    bash -c "python -m virtualenv --system-site-packages \
#             /ceph/project/es26-ce8-avs-824/whispers-in-the-storm/extern/sgmse/sgmse_venv"

srun singularity exec --nv --writable-tmpfs \
     -B /ceph/project/es26-ce8-avs-824/whispers-in-the-storm/extern/sgmse/sgmse_venv:/scratch/sgmse_venv \
     -B $HOME/.singularity:/scratch/singularity \
     /ceph/project/es26-ce8-avs-824/whispers-in-the-storm/extern/sgmse/pytorch_24.01.sif \
     /bin/bash -c "export TMPDIR=/scratch/singularity/tmp && \
                   source /scratch/sgmse_venv/bin/activate && \
                   pip install -r requirements_version.txt --no-cache-dir"
