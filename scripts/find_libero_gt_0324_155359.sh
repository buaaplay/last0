#!/usr/bin/env bash

set -euo pipefail

python - <<'PY'
import os
from pathlib import Path

print("== import-based lookup ==")
try:
    from libero.libero import get_libero_path
    datasets_path = get_libero_path("datasets")
    bddl_path = get_libero_path("bddl_files")
    print(f"libero_datasets_path={datasets_path}")
    print(f"libero_bddl_path={bddl_path}")
except Exception as e:
    print(f"libero_import_failed={type(e).__name__}: {e}")

print()
print("== filesystem candidates ==")

candidates = [
    "/Disk1/zhy/last0/LIBERO/datasets",
    "/home/robot/zhy/last0/LIBERO/datasets",
    "/Disk1/zhy/last0",
    "/home/robot/zhy/last0",
    "/home/robot/play/last0",
]

def looks_like_libero_episode_tree(root: Path):
    npy_files = list(root.rglob("*.npy"))
    if not npy_files:
        return False, []
    sample = [str(p) for p in npy_files[:5]]
    return True, sample

for item in candidates:
    path = Path(item)
    if not path.exists():
        continue
    ok, sample = looks_like_libero_episode_tree(path)
    if ok:
        print(f"candidate_root={path}")
        print("sample_npy_files=")
        for s in sample:
            print(f"  - {s}")
        print()

print("== task-folder hints ==")
task_root_candidates = [
    "/Disk1/zhy/last0/LIBERO/datasets",
    "/home/robot/zhy/last0/LIBERO/datasets",
]
for item in task_root_candidates:
    path = Path(item)
    if not path.exists():
        continue
    task_dirs = [p.name for p in path.iterdir() if p.is_dir()]
    if task_dirs:
        print(f"task_root={path}")
        for name in task_dirs[:20]:
            print(f"  - {name}")
        print()
PY
