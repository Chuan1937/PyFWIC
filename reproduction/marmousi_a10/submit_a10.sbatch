#!/usr/bin/env bash
#SBATCH --job-name=marmousi-a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=marmousi-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?请从 PyFWIC 仓库根目录提交}"
nvidia-smi -L
exec reproduction/marmousi_a10/run_a10.sh "$@"
