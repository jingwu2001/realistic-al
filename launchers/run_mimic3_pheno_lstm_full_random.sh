#!/usr/bin/env bash
# LSTM x STraTS 129 variables x whole stay, random query (6 runs), queued behind
# the SAND window grid:
#   1. wait for sweep_exp_mimic3_pheno_sand_window_random.py to exit and for no
#      src/main.py run to be left (the build peaks at tens of GB: never overlap it)
#   2. build $DATA_ROOT/mimic3_sand_full (preprocess_mimic3_sand.py --hours full);
#      the build checks the first 24 h against mimic3_sand and fails on a mismatch
#   3. launchers/sweep_exp_mimic3_pheno_lstm_full_random.py --jobs 3
# Any failing step stops the chain. Run detached:
#   setsid bash launchers/run_mimic3_pheno_lstm_full_random.sh > LOG 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/.."
source .env
GRID_LOG=$EXPERIMENT_ROOT/activelearning/launch_sand_window_random_20261025.log

echo "$(date '+%F %T') waiting for the SAND window grid"
until grep -q "^exit=" "$GRID_LOG"; do sleep 60; done
while pgrep -f "src/main.py" > /dev/null; do sleep 60; done
echo "$(date '+%F %T') grid done ($(grep '^exit=' "$GRID_LOG")), building mimic3_sand_full"

conda run --no-capture-output -n real-al python src/data/preprocess_mimic3_sand.py --hours full
echo "$(date '+%F %T') build done, launching the LSTM whole-stay random runs"

conda run --no-capture-output -n real-al python launchers/sweep_exp_mimic3_pheno_lstm_full_random.py --jobs 3
echo "$(date '+%F %T') all done"
