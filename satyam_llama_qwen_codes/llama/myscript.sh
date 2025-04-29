#!/bin/bash
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --ntasks-per-node=1
#SBATCH -t 7:00:00
#SBATCH --gres=gpu:V100:1
#SBATCH --mem-per-gpu=32G
#SBATCH -J gemma-v1
#SBATCH -o ./slurm_%j.out

echo "SLURM job started"

# 1. Load modules
module purge
module load anaconda3/2022.05.0.1
echo "Modules loaded"

which python
python --version
which conda
conda --version

# 2. Activate venv
# source /path/to/anaconda3/2022.05.0.1/etc/profile.d/conda.sh
source /usr/local/pace-apps/manual/packages/anaconda3/2022.05.0.1/etc/profile.d/conda.sh
conda activate /home/hice1/ssingh934/scratch/dl_project
echo "Conda environment activated"


which python
python --version
which conda
conda --version
which python3
python --version

# 3. Move to your project directory
cd /home/hice1/ssingh934/scratch/deep-learning/project
echo "Changed to project directory"

echo "Uninstall started"
pip uninstall unsloth unsloth-zoo -y
echo "Uninstall ended"

echo "New install started"
pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git git+https://github.com/unslothai/unsloth-zoo.git
echo "New install ended"

# 4. Run training script
echo "Starting Python training script"
python3 training.py
echo "Python script finished"


# 5. Deactivate (optional)
conda deactivate
echo "Conda environment deactivated"

echo "SLURM job completed"
