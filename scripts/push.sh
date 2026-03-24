#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMMIT_MSG="${1:-Update last0 workspace}"
BRANCH_NAME="${2:-main}"

cd "$ROOT_DIR"

if command -v zsh >/dev/null 2>&1 && [[ -f ~/.zshrc ]]; then
  PROXY_VARS="$(zsh -lc 'source ~/.zshrc >/dev/null 2>&1; if command -v proxy_on >/dev/null 2>&1; then proxy_on >/dev/null 2>&1; printf "%s\n%s\n" "${http_proxy:-}" "${https_proxy:-}"; fi')" || true
  if [[ -n "$PROXY_VARS" ]]; then
    export http_proxy="$(printf '%s\n' "$PROXY_VARS" | sed -n '1p')"
    export https_proxy="$(printf '%s\n' "$PROXY_VARS" | sed -n '2p')"
  fi
fi

git add .gitignore experiments/open_loop/libero scripts

if [[ -n "$(git status --porcelain)" ]]; then
  git commit -m "$COMMIT_MSG"
else
  echo "Nothing to commit. Pushing current HEAD."
fi

git push origin "$BRANCH_NAME"
