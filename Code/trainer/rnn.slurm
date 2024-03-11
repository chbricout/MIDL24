#!/bin/bash
#SBATCH --job-name=res_cnn
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=60G
#SBATCH --gpus-per-node=1
#SBATCH --time=30:00:00
#SBATCH --account=rrg-ebrahimi
#SBATCH --mail-user=bricout.charles@outlook.com
#SBATCH --mail-type=ALL
#SBATCH -o ./logs/output-%x.%j.out # STDOUT

# Load Python module (assuming it's installed on your cluster)
module load python
module load cuda
module load httpproxy

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Activate bowl virtual environment
source ~/bowl/bin/activate

srun python trainer/rnn_cnn_trainer.py --raw_images --use_decoder --max_epochs 1000  --learning_rate 1e-6 --batch_train 16 --beta 0.01
# Deactivate the virtual environment
deactivate