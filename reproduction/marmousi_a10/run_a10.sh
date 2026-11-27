#!/usr/bin/env bash
set -euo pipefail

case_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$case_dir/../.." && pwd)"
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
exec "${PYFWI_PYTHON:-python3}" "$case_dir/run.py" "$@"
