#!/usr/bin/env bash
# 双卡拆分：一张卡跑 DV 分支，另一张跑 PCS 分支，结束后合并结果。
# 用法: run_split.sh [--out DIR] [run.py 的其他参数]
set -euo pipefail

case_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$case_dir/../.." && pwd)"
python="${PYFWI_PYTHON:-python3}"

gpu_dv="${PYFWI_GPU_DV:-0}"
gpu_pcs="${PYFWI_GPU_PCS:-1}"
cpu_dv="${PYFWI_CPUS_DV:-0-3}"
cpu_pcs="${PYFWI_CPUS_PCS:-4-7}"

args=()
out=""
while (($#)); do
  case "$1" in
    --out) out="$2"; shift 2 ;;
    *) args+=("$1"); shift ;;
  esac
done
if [[ -z "$out" ]]; then
  echo "用法: run_split.sh --out DIR [--profile full|smoke] [--case marmousi|layered] ..." >&2
  exit 2
fi

work="$out/branches"
mkdir -p "$work/logs"
echo "DV  分支: GPU $gpu_dv, CPU $cpu_dv -> $work/dv"
echo "PCS 分支: GPU $gpu_pcs, CPU $cpu_pcs -> $work/pcs"

CUDA_VISIBLE_DEVICES="$gpu_dv" taskset -c "$cpu_dv" \
  "$case_dir/run_a10.sh" --branch dv "${args[@]}" --out "$work/dv" \
  >"$work/logs/dv.log" 2>&1 &
pid_dv=$!
CUDA_VISIBLE_DEVICES="$gpu_pcs" taskset -c "$cpu_pcs" \
  "$case_dir/run_a10.sh" --branch pcs "${args[@]}" --out "$work/pcs" \
  >"$work/logs/pcs.log" 2>&1 &
pid_pcs=$!

status=0
wait "$pid_dv" || status=1
wait "$pid_pcs" || status=1
if ((status)); then
  echo "至少一个分支失败，日志在 $work/logs" >&2
  exit 1
fi

export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$python" "$case_dir/combine.py" --dv "$work/dv" --pcs "$work/pcs" --out "$out"
