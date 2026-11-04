#!/usr/bin/env bash
# P4 training launcher:
#   Single process: bash run_train.sh --model gpt-tiny --dry-run ...
#   Distributed:    bash run_train.sh --nproc-per-node 4 --model gpt-125m ...
#
# --nproc-per-node belongs to this launcher, not train.py.
# Other arguments pass through unchanged.

set -euo pipefail
source /workspace/p4/p4env.sh

backend=auto
compiled=0
nproc=""
args=()

while (( $# > 0 )); do
  case "$1" in
    --nproc-per-node|--nproc_per_node)
      if (( $# < 2 )); then
        echo "ERROR: --nproc-per-node needs a value" >&2
        exit 2
      fi
      nproc="$2"
      shift 2
      ;;
    --nproc-per-node=*|--nproc_per_node=*)
      nproc="${1#*=}"
      shift
      ;;
    --compile)
      compiled=1
      args+=("$1")
      shift
      ;;
    --backend)
      if (( $# < 2 )); then
        echo "ERROR: --backend needs a value" >&2
        exit 2
      fi
      backend="$2"
      args+=("$1" "$2")
      shift 2
      ;;
    --backend=*)
      backend="${1#*=}"
      args+=("$1")
      shift
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done

case "$backend" in
  auto|flash|cudnn|efficient|math) ;;
  *)
    echo "ERROR: unsupported backend: $backend" >&2
    exit 2
    ;;
esac

if [[ -n "$nproc" ]] && ! [[ "$nproc" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --nproc-per-node must be a positive integer" >&2
  exit 2
fi

if (( compiled )); then
  export TORCHINDUCTOR_CACHE_DIR="${P4_INDUCTOR_CACHE_ROOT:-/workspace/.cache/p4-inductor}/$backend"
  mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
  echo "[p4] compile backend=$backend cache=$TORCHINDUCTOR_CACHE_DIR"
fi

if [[ -n "$nproc" ]]; then
  echo "[p4] launching torchrun with $nproc processes"
  exec torchrun --standalone --nproc_per_node="$nproc" train.py "${args[@]}"
else
  echo "[p4] launching single-process training"
  exec python train.py "${args[@]}"
fi
