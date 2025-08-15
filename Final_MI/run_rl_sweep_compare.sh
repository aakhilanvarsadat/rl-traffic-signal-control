#!/usr/bin/env bash
set -euo pipefail
# Run comparison: DQN vs Double DQN vs Dueling DQN
# Requires scenarios_out/{routes_train.txt,routes_val.txt,routes_test.txt}

SUMO_TRAIN_BIN="${SUMO_TRAIN_BIN:-sumo}"
SUMO_EVAL_BIN="${SUMO_EVAL_BIN:-sumo}"
NET="${NET:-hourhold_src.net.xml}"
PORT="${PORT:-8899}"
SHORT_EPISODES="${SHORT_EPISODES:-40}"
LONG_EPISODES="${LONG_EPISODES:-200}"
EP_SECONDS="${EP_SECONDS:-3600}"
DECISION="${DECISION:-5}"
EXTEND_S="${EXTEND_S:-5}"
MIN_GREEN="${MIN_GREEN:-10}"
MAX_GREEN="${MAX_GREEN:-60}"
CLIP_Q="${CLIP_Q:-15}"
SWITCH_PENALTY="${SWITCH_PENALTY:-0.05}"
SEED="${SEED:-146}"
OUTDIR="${OUTDIR:-compare_out}"
DEVICE="${DEVICE:-cpu}"
SUMO_EXTRA="${SUMO_EXTRA:-}"
SCEN_DIR="${SCEN_DIR:-scenarios_out}"

python -u rl_sweep_split_compare.py \
  --sumo-train-bin "$SUMO_TRAIN_BIN" --sumo-eval-bin "$SUMO_EVAL_BIN" \
  --net "$NET" \
  --routes-train-list "$SCEN_DIR/routes_train.txt" \
  --routes-val-list   "$SCEN_DIR/routes_val.txt" \
  --routes-test-list  "$SCEN_DIR/routes_test.txt" \
  --port "$PORT" --short-episodes "$SHORT_EPISODES" --long-episodes "$LONG_EPISODES" \
  --ep-seconds "$EP_SECONDS" --decision "$DECISION" --extend-s "$EXTEND_S" \
  --min-green "$MIN_GREEN" --max-green "$MAX_GREEN" \
  --clip-q "$CLIP_Q" --switch-penalty "$SWITCH_PENALTY" \
  --seed "$SEED" --outdir "$OUTDIR" --device "$DEVICE" \
  ${SUMO_EXTRA:+--sumo-extra $SUMO_EXTRA}
