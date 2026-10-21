#!/usr/bin/env bash
# 25-condition TS-only LSTM phenotyping (random initial set), queued behind the
# running single-task TS-only tune_vendi launcher, run one after another,
# detached (setsid + nohup) so the chain survives the terminal / Claude session.
#
#   bash launchers/run_mimic3_pheno25_lstm_nodemo.sh [wait_pid]   # returns at once
#   tail -f experiments/activelearning/mimic3_pheno25_lstm_nodemo_chain_<date>.log
#   kill -- -<pgid>                               # stop the whole chain (pgid is in the chain log)
#
# Stages:
#   0. wait until <wait_pid> exits (2026-10-20: the single-task
#      sweep_exp_mimic3_pheno_single_task_lstm_nodemo_tune_vendi.py launcher)
#   1. sweep_exp_mimic3_pheno_lstm_nodemo_baselines.py    30 runs
#   2. sweep_exp_mimic3_pheno_lstm_nodemo_tune_vendi.py  120 runs
# --jobs 4: LSTM runs peak at ~7-7.5 GB RAM (the 25-label grad bandit smoke
# test included), < 1 GB VRAM; ~83 MB disk per finished run (~12.5 GB in all).

REPO=/home/jing/Desktop/realistic-al
D=$REPO/experiments/activelearning
STAMP=$(date +%Y%m%d)
CHAIN_LOG=$D/mimic3_pheno25_lstm_nodemo_chain_$STAMP.log
MIN_FREE_GB=15  # don't start a stage below this
JOBS=4
WAIT_PID=${1:-}

if [[ "$2" != "--detached" ]]; then
    ( setsid nohup bash "$(readlink -f "$0")" "$WAIT_PID" --detached >> "$CHAIN_LOG" 2>&1 < /dev/null & )
    echo "chain started, log: $CHAIN_LOG"
    exit 0
fi

source /home/jing/anaconda3/etc/profile.d/conda.sh
conda activate real-al
cd "$REPO" || exit 1
source .env

echo "[$(date)] chain pid $$, pgid $(ps -o pgid= $$ | tr -d ' ')"

if [[ -n "$WAIT_PID" ]]; then
    echo "[$(date)] waiting for pid $WAIT_PID: $(ps -o args= -p "$WAIT_PID")"
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
    echo "[$(date)] pid $WAIT_PID exited"
fi

run_stage() {  # run_stage <log name> <launcher file> <jobs>
    local name=$1 launcher=$2 jobs=$3 free_gb
    free_gb=$(df --output=avail -BG "$REPO" | tail -1 | tr -dc 0-9)
    if (( free_gb < MIN_FREE_GB )); then
        echo "[$(date)] STOP before $name: ${free_gb}G free < ${MIN_FREE_GB}G"
        exit 1
    fi
    echo "[$(date)] start $name: $launcher --jobs $jobs (${free_gb}G free)"
    python "launchers/$launcher" --jobs "$jobs" > "$D/mimic3_pheno25_lstm_nodemo_${name}_$STAMP.log" 2>&1
    echo "[$(date)] end $name (exit $?)"
}

run_stage baselines sweep_exp_mimic3_pheno_lstm_nodemo_baselines.py $JOBS
run_stage tune_vendi sweep_exp_mimic3_pheno_lstm_nodemo_tune_vendi.py $JOBS
echo "[$(date)] chain finished"
