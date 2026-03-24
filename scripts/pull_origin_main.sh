#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_NAME="${1:-origin}"
BRANCH_NAME="${2:-main}"

if [[ -n "$(git -C "$ROOT_DIR" status --porcelain)" ]]; then
  echo "Refusing to pull: repository has uncommitted changes."
  git -C "$ROOT_DIR" status --short
  exit 1
fi

echo "==> Fetching $REMOTE_NAME/$BRANCH_NAME"
git -C "$ROOT_DIR" fetch "$REMOTE_NAME" "$BRANCH_NAME"

echo "==> Pulling $REMOTE_NAME/$BRANCH_NAME"
git -C "$ROOT_DIR" pull --ff-only "$REMOTE_NAME" "$BRANCH_NAME"

echo "==> Current commit"
git -C "$ROOT_DIR" rev-parse --short HEAD
