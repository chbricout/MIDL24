
dos2unix trainer/*

sbatch trainer/baseline.slurm
sbatch trainer/baseline_no_decoder.slurm

sbatch trainer/aug_baseline.slurm
sbatch trainer/aug_baseline_no_dec.slurm

sbatch trainer/naive_res.slurm
sbatch trainer/naive_res_no_dec.slurm

sbatch trainer/rnn.slurm
sbatch trainer/rnn_no_decoder.slurm