#!/usr/bin/env bash
# Profile Genesis Franka 30k-env physics with rocprofv3.
#
# Do NOT use `rocprofv3 -- python ...`. PyTorch ships its own librocprofiler-sdk
# and the wrapper dies with:
#   error code 16: Configuration request occurred outside of valid rocprofiler
#   configuration period
# Instead load the rocprofv3 *tool* via ROCP_TOOL_LIBRARIES so Torch's SDK
# configures it on first import.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
source /mnt/dcgpuval/younsroh/projects/3d/.venv/bin/activate

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="${OUT:-$ROOT/out/rocprofv3/franka-30k-$STAMP}"
mkdir -p "$OUT"

ROCM_TOOL="${ROCM_TOOL:-/opt/rocm/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so}"
export ROCPROFILER_LIBRARY_CTOR=1
export ROCP_TOOL_LIBRARIES="$ROCM_TOOL"
export ROCPROF_OUTPUT_FILE_NAME="${ROCPROF_OUTPUT_FILE_NAME:-genesis_franka_30k}"
export ROCPROF_OUTPUT_PATH="$OUT"
export ROCPROF_OUTPUT_FORMAT=csv
export ROCPROF_KERNEL_TRACE=1
export ROCPROF_MEMORY_COPY_TRACE=1
export ROCPROF_STATS=1
export ROCPROF_STATS_SUMMARY=1
export ROCPROF_TRUNCATE_KERNELS=1

export GS_N_ENVS="${GS_N_ENVS:-30000}"
export GS_WARMUP_STEPS="${GS_WARMUP_STEPS:-10}"
export GS_PROFILE_STEPS="${GS_PROFILE_STEPS:-80}"
export GS_ATTACH_WAIT=0

echo "Profiling -> $OUT"
python -u "$ROOT/profile_franka_30k.py"
echo "kernel stats: $OUT/${ROCPROF_OUTPUT_FILE_NAME}_kernel_stats.csv"
