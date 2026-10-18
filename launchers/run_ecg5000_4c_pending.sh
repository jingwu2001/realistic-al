#!/usr/bin/env bash
# Pending 4-class ECG5000 launchers after the 2026-10-14 OOM and the
# 2026-10-15 disk-full crash, run one after another, detached (setsid + nohup)
# so the chain survives the terminal / Claude session.
#
#   bash launchers/run_ecg5000_4c_pending.sh     # returns at once
#   tail -f experiments/activelearning/ecg5000_4c_chain_<date>.log
#   kill -- -<pgid>                               # stop the whole chain (pgid is in the chain log)
#
# Stages, small fill-ins first so the existing seed sets complete early:
#   1. exp_ecg5000_basic_tune_kernel_balanced.py        9 runs  feature vq2           ~4.8 GB/run
#   2. exp_ecg5000_basic_tune_kernel_missing_seeds.py   5 runs  feature vq2 + bandit
#   3. exp_ecg5000_basic_tune_kernel_bandit.py         84 runs  feature bandit (vq2 arm)
#   4. exp_ecg5000_basic_tune_kernel.py               168 runs  grad bandit + vq2     ~5.8-6.9 GB/run
# --jobs 5 everywhere: 5 x 6.9 GB = 35 GB at worst, ~70% of the ~51 GB
# available on 2026-10-17 (host is shared). Keep num_workers at 16, see the
# compute-budget note: it changes the RNG streams that pair seeds across runs.

REPO=/home/jing/Desktop/realistic-al
D=$REPO/experiments/activelearning
STAMP=$(date +%Y%m%d)
CHAIN_LOG=$D/ecg5000_4c_chain_$STAMP.log
MIN_FREE_GB=50  # don't start a stage below this; ~0.2 GB per finished run

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

run_stage balanced_rerun exp_ecg5000_basic_tune_kernel_balanced.py 5
run_stage missing_seeds exp_ecg5000_basic_tune_kernel_missing_seeds.py 5
run_stage bandit_feat exp_ecg5000_basic_tune_kernel_bandit.py 5
run_stage grad exp_ecg5000_basic_tune_kernel.py 5
echo "[$(date)] chain finished"
