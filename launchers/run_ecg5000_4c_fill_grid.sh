#!/usr/bin/env bash
# Last gaps in the 4-class ECG5000 grid (2 budgets x balanced {T, F} x baselines
# + {vendi-q2, bandit-vq2} x use_grad {T, F} x 7 kernels, 6 seeds) on wandb
# 2026-10-18: 42 settings with no runs, never launched. Runs the two launchers
# one after another, detached (setsid + nohup).
#
#   bash launchers/run_ecg5000_4c_fill_grid.sh   # returns at once
#   tail -f experiments/activelearning/ecg5000_4c_chain_fill_<date>.log
#   kill -- -<pgid>                               # stop the whole chain (pgid is in the chain log)
#
# Stages:
#   1. exp_ecg5000_basic_tune_kernel_bandit.py   84 runs  feature bandit-vq2, balanced init, low + med
#   2. exp_ecg5000_basic_tune_kernel.py         168 runs  grad bandit-vq2 + vq2, low, balanced {T, F}
# --jobs 5: 5 x 6.9 GB = 35 GB at worst, ~70% of the ~50 GB available on
# 2026-10-18. 2026-10-17 timing at --jobs 5: 84 feature-bandit runs 1.4 h,
# 168 grad runs 2.9 h; ~0.2 GB disk per run (~55 GB in all, 119 GB free).

REPO=/home/jing/Desktop/realistic-al
D=$REPO/experiments/activelearning
STAMP=$(date +%Y%m%d)
CHAIN_LOG=$D/ecg5000_4c_chain_fill_$STAMP.log
MIN_FREE_GB=50  # don't start a stage below this

if [[ "$1" != "--detached" ]]; then
    ( setsid nohup bash "$(readlink -f "$0")" --detached >> "$CHAIN_LOG" 2>&1 < /dev/null & )
    echo "chain started, log: $CHAIN_LOG"
    exit 0
fi

source /home/jing/anaconda3/etc/profile.d/conda.sh
conda activate real-al
cd "$REPO" || exit 1
source .env

echo "[$(date)] chain pid $$, pgid $(ps -o pgid= $$ | tr -d ' ')"

run_stage() {  # run_stage <log name> <launcher file> <jobs>
    local name=$1 launcher=$2 jobs=$3 free_gb
    free_gb=$(df --output=avail -BG "$REPO" | tail -1 | tr -dc 0-9)
    if (( free_gb < MIN_FREE_GB )); then
        echo "[$(date)] STOP before $name: ${free_gb}G free < ${MIN_FREE_GB}G"
        exit 1
    fi
    echo "[$(date)] start $name: $launcher --jobs $jobs (${free_gb}G free)"
    python "launchers/$launcher" --jobs "$jobs" > "$D/ecg5000_4c_${name}_$STAMP.log" 2>&1
    echo "[$(date)] end $name (exit $?)"
}

run_stage bandit_feat_bal exp_ecg5000_basic_tune_kernel_bandit.py 5
run_stage grad_low exp_ecg5000_basic_tune_kernel.py 5
echo "[$(date)] chain finished"
