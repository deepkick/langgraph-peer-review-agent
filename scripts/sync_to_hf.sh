#!/usr/bin/env bash
# scripts/sync_to_hf.sh
# Sync deploy artifacts (app.py + examples/replays/) from this GitHub repo
# to the sibling HF Space repo, then commit + push.
#
# Usage:
#   ./scripts/sync_to_hf.sh "<commit message>"
#
# Prerequisites:
#   - Run from within this GitHub repo (langgraph-peer-review-agent/)
#   - HF Space repo cloned as sibling: ../langgraph-peer-review-agent-hf/
#   - HF write token configured (hf auth login)

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 \"<commit message>\"" >&2
  echo "Example: $0 \"Update theme: system fonts\"" >&2
  exit 1
fi

COMMIT_MSG="$1"

# Resolve paths
GITHUB_REPO="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "$GITHUB_REPO" ]; then
  echo "ERROR: Not in a git repository. Run from within the GitHub repo." >&2
  exit 1
fi

HF_REPO="$(dirname "$GITHUB_REPO")/langgraph-peer-review-agent-hf"
if [ ! -d "$HF_REPO/.git" ]; then
  echo "ERROR: HF Space repo not found at: $HF_REPO" >&2
  echo "Clone it first:" >&2
  echo "  cd $(dirname "$GITHUB_REPO")" >&2
  echo "  git clone https://huggingface.co/spaces/deepkick/langgraph-peer-review-agent \\" >&2
  echo "    langgraph-peer-review-agent-hf" >&2
  exit 1
fi

# Branch sanity check
cd "$GITHUB_REPO"
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
  read -rp "WARN: not on main (current: $CURRENT_BRANCH). Continue? [y/N] " ok
  case "$ok" in
    y|Y) ;;
    *) echo "Aborted."; exit 1 ;;
  esac
fi

echo "→ GitHub repo:  $GITHUB_REPO (branch: $CURRENT_BRANCH)"
echo "→ HF Space repo: $HF_REPO"
echo

# Sync files
echo "→ Syncing app.py..."
cp "$GITHUB_REPO/app.py" "$HF_REPO/app.py"

echo "→ Syncing examples/replays/..."
mkdir -p "$HF_REPO/examples"
rsync -av --delete \
  "$GITHUB_REPO/examples/replays/" \
  "$HF_REPO/examples/replays/"

# Diff preview
cd "$HF_REPO"
echo
echo "→ Changes in HF Space repo:"
echo "─────────────────────────────────"
git status --short
echo "─────────────────────────────────"

if git diff --quiet && git diff --cached --quiet \
   && [ -z "$(git ls-files --others --exclude-standard)" ]; then
  echo "✓ No changes. Already in sync."
  exit 0
fi

echo
read -rp "Commit and push to HF Space? [y/N] " confirm
case "$confirm" in
  y|Y) ;;
  *) echo "Aborted. Files synced but not committed."; exit 1 ;;
esac

# Commit + push
git add .
git commit -m "$COMMIT_MSG"
echo
echo "→ Pushing to HF Space..."
git push origin main

echo
echo "✓ Deployed to HF Space"
echo "  Watch build: https://huggingface.co/spaces/deepkick/langgraph-peer-review-agent"
