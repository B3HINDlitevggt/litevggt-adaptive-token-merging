#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEEDS="${SEEDS:-0 1 2 3 4}"
EXP_NAME="${EXP_NAME:-ga_sweep}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/ga_sweeps}"
RUN_EVAL="$ROOT_DIR/scripts/run_eval_dtu.sh"

mkdir -p "$LOG_DIR"

SUMMARY_TSV="$LOG_DIR/${EXP_NAME}_summary.tsv"
SUMMARY_TXT="$LOG_DIR/${EXP_NAME}_summary.txt"

printf "experiment\tseed\tauc30\tauc15\tauc5\tauc3\tlog\n" > "$SUMMARY_TSV"

printf "Experiment: %s\n" "$EXP_NAME" | tee "$SUMMARY_TXT"
printf "Seeds: %s\n" "$SEEDS" | tee -a "$SUMMARY_TXT"
printf "Logs: %s\n\n" "$LOG_DIR" | tee -a "$SUMMARY_TXT"

printf "%-18s %-6s %-10s %-10s %-10s %-10s\n" "experiment" "seed" "AUC@30" "AUC@15" "AUC@5" "AUC@3" | tee -a "$SUMMARY_TXT"
printf "%-18s %-6s %-10s %-10s %-10s %-10s\n" "----------" "----" "------" "------" "-----" "-----" | tee -a "$SUMMARY_TXT"

for seed in $SEEDS; do
  export SEED="$seed"
  log_file="$LOG_DIR/${EXP_NAME}_seed${seed}.log"

  printf "Running %s seed %s ... " "$EXP_NAME" "$seed"
  if bash "$RUN_EVAL" > "$log_file" 2>&1; then
    mean_line="$(grep '^Mean AUC:' "$log_file" | tail -1 || true)"
    if [[ -z "$mean_line" ]]; then
      printf "no Mean AUC line found. See %s\n" "$log_file"
      printf "%s\t%s\tNA\tNA\tNA\tNA\t%s\n" "$EXP_NAME" "$seed" "$log_file" >> "$SUMMARY_TSV"
      continue
    fi

    read -r auc30 auc15 auc5 auc3 < <(
      python3 - "$mean_line" <<'PY'
import re
import sys

line = sys.argv[1]
vals = re.findall(r"([0-9]+\.[0-9]+)", line)
if len(vals) >= 4:
    print(vals[0], vals[1], vals[2], vals[3])
else:
    print("NA NA NA NA")
PY
    )

    printf "AUC@30=%s AUC@15=%s\n" "$auc30" "$auc15"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$EXP_NAME" "$seed" "$auc30" "$auc15" "$auc5" "$auc3" "$log_file" >> "$SUMMARY_TSV"
    printf "%-18s %-6s %-10s %-10s %-10s %-10s\n" "$EXP_NAME" "$seed" "$auc30" "$auc15" "$auc5" "$auc3" | tee -a "$SUMMARY_TXT"
  else
    printf "failed. See %s\n" "$log_file"
    tail -40 "$log_file" || true
    printf "%s\t%s\tFAIL\tFAIL\tFAIL\tFAIL\t%s\n" "$EXP_NAME" "$seed" "$log_file" >> "$SUMMARY_TSV"
  fi
done

python3 - "$SUMMARY_TSV" <<'PY' | tee -a "$SUMMARY_TXT"
import csv
import statistics
import sys

path = sys.argv[1]
rows = []
with open(path, newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        try:
            rows.append({
                "auc30": float(row["auc30"]),
                "auc15": float(row["auc15"]),
                "auc5": float(row["auc5"]),
                "auc3": float(row["auc3"]),
            })
        except ValueError:
            pass

print()
if not rows:
    print("No successful runs to summarize.")
    raise SystemExit

def mean(key):
    return statistics.mean(row[key] for row in rows)

def stdev(key):
    if len(rows) < 2:
        return 0.0
    return statistics.stdev(row[key] for row in rows)

print("Aggregate")
print(f"runs   : {len(rows)}")
print(f"AUC@30 : mean={mean('auc30'):.4f}, std={stdev('auc30'):.4f}")
print(f"AUC@15 : mean={mean('auc15'):.4f}, std={stdev('auc15'):.4f}")
print(f"AUC@5  : mean={mean('auc5'):.4f}, std={stdev('auc5'):.4f}")
print(f"AUC@3  : mean={mean('auc3'):.4f}, std={stdev('auc3'):.4f}")
PY

printf "\nSummary TSV: %s\n" "$SUMMARY_TSV"
printf "Summary TXT: %s\n" "$SUMMARY_TXT"
