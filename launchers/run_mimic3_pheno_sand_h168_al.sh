#!/usr/bin/env bash
# Full AL sweep for the leading random-grid setting (SAND x STraTS 129 variables x
# 168 h x demographics), queued behind the LSTM whole-stay chain
# (run_mimic3_pheno_lstm_full_random.sh):
#   1. wait for that chain to exit (pass or fail: these runs don't depend on it)
#      and for no src/main.py run to be left
#   2. sweep_exp_mimic3_pheno_sand_h168_baselines.py --jobs 6   (24 runs)
#   3. sweep_exp_mimic3_pheno_sand_h168_vendi.py --jobs 6       (120 runs)
# random is not rerun: its runs come from the SAND window grid (20261025-22f904d09).
# Run detached:
#   setsid bash launchers/run_mimic3_pheno_sand_h168_al.sh > LOG 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
source .env

echo "$(date '+%F %T') waiting for the LSTM whole-stay chain"
while pgrep -f "run_mimic3_pheno_lstm_full_random.sh" > /dev/null; do sleep 60; done
while pgrep -f "src/main.py" > /dev/null; do sleep 60; done

for part in baselines vendi; do
    echo "$(date '+%F %T') launching sweep_exp_mimic3_pheno_sand_h168_$part.py"
    conda run --no-capture-output -n real-al python launchers/sweep_exp_mimic3_pheno_sand_h168_$part.py --jobs 6
    echo "$(date '+%F %T') $part exit=$?"
done
echo "$(date '+%F %T') all done"
