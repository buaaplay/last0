#!/usr/bin/env bash

set -euo pipefail

STATS_PATH="${1:-/home/robot/zhy/last0/weights/Pretrain/stats_data.json}"
TASK_SUITE_NAME="${2:-libero_spatial}"

python - <<'PY' "$STATS_PATH" "$TASK_SUITE_NAME"
import json
import sys

stats_path = sys.argv[1]
task_suite_name = sys.argv[2]

with open(stats_path, "r") as f:
    data = json.load(f)

keys = list(data.keys())
print(f"stats_path={stats_path}")
print("available_keys=")
for key in keys:
    print(f"  - {key}")

print(f"task_suite_name={task_suite_name}")
if task_suite_name in data:
    print(f"recommended_unnorm_key={task_suite_name}")
else:
    print("recommended_unnorm_key=NOT_FOUND")
PY
