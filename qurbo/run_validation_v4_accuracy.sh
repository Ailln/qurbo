#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-./}"
DATA="${DATA:-$ROOT/data/final-validation}"
OUT="${OUT:-$ROOT/validation_results_v4_accuracy/results}"
LOG="${LOG:-$ROOT/validation_results_v4_accuracy/logs}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$OUT" "$LOG"

COMMON_ARGS=(
  --iterations 500
  --time-limit-seconds 1800
  --eta-ema 0.3
  --eta-resc 0.5
  --qaoa-opt-steps 80
  --qaoa-opt-steps-large 0
  --qaoa-multistart 4
  --qaoa-depth-small 3
  --qaoa-depth-medium 3
  --qaoa-depth-large 1
  --qaoa-rhobeg 0.5
  --shots-small 2048
  --shots-large 1024
  --top-k 30
  --device GPU
  --seed 42
  --cache-size 10000
  --temperature-scale 0.03
  --cooling-rate 0.97
  --temperature-min 0.001
  --elite-size 30
  --no-improve-restart-threshold 15
  --no-improve-expand-threshold 8
  --q-growth-step 2
  --q-shrink-step 1
  --late-break-fraction 0.7
  --exact-init-limit 15
  --init-random-small 15
  --init-random-large 15
)

run_case() {
  local name="$1"
  local input="$2"
  local output="$3"
  local seconds="$4"
  shift 4

  local solver_log="$LOG/${name}.log"
  local verify_log="$LOG/${name}_verify.log"

  echo "[RUN] $(date "+%F %T") $name -> $output" | tee -a "$LOG/run_accuracy.log"
  set +e
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" miqp_hybrid_v4.py \
    --input "$input" \
    --output "$output" \
    "${COMMON_ARGS[@]}" \
    "$@" > "$solver_log" 2>&1
  local code=$?
  set -e

  echo "[RUN] $(date "+%F %T") $name exit_code=$code" | tee -a "$LOG/run_accuracy.log"
  if [ "$code" -eq 124 ]; then
    echo "[WARN] $name timeout after ${seconds}s" | tee -a "$LOG/run_accuracy.log"
    return 124
  fi
  if [ "$code" -ne 0 ]; then
    tail -n 80 "$solver_log" | tee -a "$LOG/run_accuracy.log"
    return "$code"
  fi

  "$PYTHON_BIN" verify_solution.py --input "$input" --solution "$output" > "$verify_log" 2>&1
  tail -n 30 "$verify_log" | tee -a "$LOG/run_accuracy.log"
}

# This profile implements the 30-minute tuning note where it is effective in
# the current V4 code, with environment-calibrated safeguards:
# - test_1 is exactly solvable in initialization, so it keeps short QAOA audit calls.
# - test_2..4 stay at q=17/18 to spend time on high-throughput certified LNS.
# - test_5 uses q=12 because prior q=18 runs spent too much wall time per move.
run_case test_1 "$DATA/miqp_test_1.npz" "$OUT/solution_test_1_v4_accuracy.npz" 420 \
  --iterations 8 --time-limit-seconds 300 \
  --q-max 10 --qaoa-qubits 10 --initial-sub-size 10 --min-sub-size 10 \
  --qaoa-opt-steps 12 --qaoa-multistart 1 --shots-small 1024

run_case test_2 "$DATA/miqp_test_2.npz" "$OUT/solution_test_2_v4_accuracy.npz" 1950 \
  --q-max 18 --qaoa-qubits 18 --initial-sub-size 18 --min-sub-size 17

run_case test_3 "$DATA/miqp_test_3.npz" "$OUT/solution_test_3_v4_accuracy.npz" 1950 \
  --q-max 18 --qaoa-qubits 18 --initial-sub-size 18 --min-sub-size 17

run_case test_4 "$DATA/miqp_test_4.npz" "$OUT/solution_test_4_v4_accuracy.npz" 1950 \
  --q-max 18 --qaoa-qubits 18 --initial-sub-size 18 --min-sub-size 17

# test_5 uses q=12 as an accuracy-throughput profile because q=18 consumed too
# much wall time per neighborhood in prior validation and reduced total coverage.
run_case test_5 "$DATA/miqp_test_5.npz" "$OUT/solution_test_5_v4_accuracy.npz" 1950 \
  --q-max 12 --qaoa-qubits 12 --initial-sub-size 12 --min-sub-size 10 \
  --qaoa-opt-steps 0 --qaoa-multistart 1

echo "[DONE] $(date "+%F %T") all accuracy validation cases finished" | tee -a "$LOG/run_accuracy.log"
