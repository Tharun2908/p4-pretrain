#!/usr/bin/env bash
# P4 training launcher. Use this for compiled runs instead of `python train.py`.
# Chooses a per-backend Inductor cache before the Python process imports torch.
# Examples:
#   bash run_train.sh --model gpt-tiny --steps 40 --compile --backend cudnn --dry-run
#   bash run_train.sh --model gpt-125m --steps 200 --backend flash --compile ...
# Noncompiled runs are also supported; all arguments pass through to train.py.

set -euo pipefail
source /workspace/p4/p4env.sh

backend=auto
compiled=0
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  case "${args[i]}" in
    --compile) compiled=1 ;;
    --backend)
      if (( i+1 >= ${#args[@]} )); then
        echo "ERROR: --backend requires a value" >&2
        exit 2
      fi
      backend="${args[i+1]}"
      ((i+=1))
      ;;
    --backend=*) backend="${args[i]#--backend=}" ;;
  esac
done

case "$backend" in
  flash|cudnn|efficient|math|auto) ;;
  *) echo "ERROR: unsupported attention backend: $backend" >&2; exit 2 ;;
esac

if (( compiled )); then
  # Use a fresh process for each run; Inductor reads this variable during import.
  # Explicitly isolate caches so a compiled Flash graph is never reused for cuDNN.
  export TORCHINDUCTOR_CACHE_DIR="${P4_INDUCTOR_CACHE_ROOT:-/workspace/.cache/p4-inductor}/$backend"
  mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
  echo "[p4] torch.compile backend=$backend; cache=$TORCHINDUCTOR_CACHE_DIR"
fi
exec python train.py "$@"
