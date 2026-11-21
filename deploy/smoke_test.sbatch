#!/usr/bin/env bash
#SBATCH --job-name=pear-smoke
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gpus-per-node=4
#SBATCH --mem=450G
#SBATCH --time=00:20:00
#SBATCH --time-min=00:15:00
#SBATCH --output=/scratch/%u/pear_work/logs/%x-%j.out
#SBATCH --error=/scratch/%u/pear_work/logs/%x-%j.err

# Submit from the repository root; an explicit PEAR_ROOT argument also works.
set -euo pipefail
ROOT=${PEAR_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
for argument in "$@"; do
  case "$argument" in PEAR_ROOT=*) ROOT=${argument#*=} ;; esac
done
exec bash "$ROOT/deploy/train.sbatch" --smoke "$@"
