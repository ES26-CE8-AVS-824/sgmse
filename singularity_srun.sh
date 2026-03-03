srun singularity exec --nv --writable-tmpfs \
     -B /ceph/project/es26-ce8-avs-824/whispers-in-the-storm/extern/sgmse/sgmse_venv:/scratch/sgmse_venv \
     -B $HOME/.singularity:/scratch/singularity \
     /ceph/container/python/python_3.11.sif \
     /bin/bash -c "export TMPDIR=/scratch/singularity/tmp && \
                   source /scratch/sgmse_venv/bin/activate && \
                   pip install --no-cache-dir torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
                   pip install --no-cache-dir -r requirements_marko.txt"