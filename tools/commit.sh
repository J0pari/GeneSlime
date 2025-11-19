#!/bin/bash

MESSAGE="${1:-}"

echo "=== COMMIT PIPELINE ==="
echo

echo "[1/5] Staging all changes..."
rm -f nul 2>/dev/null || true
git add -A
echo "✓ Staged $(git diff --cached --numstat | wc -l) file changes"
echo

echo "[2/5] Creating commit with web-flow author..."
GIT_AUTHOR_NAME="web-flow" \
GIT_AUTHOR_EMAIL="noreply@github.com" \
GIT_COMMITTER_NAME="web-flow" \
GIT_COMMITTER_EMAIL="noreply@github.com" \
git commit --allow-empty-message -m "$MESSAGE" || { echo "✗ Commit failed"; exit 1; }
COMMIT_HASH=$(git rev-parse --short HEAD)
echo "✓ Created commit $COMMIT_HASH"
echo

echo "[3/5] Verifying remote..."
git remote get-url origin >/dev/null 2>&1 || {
    echo "  Setting remote to https://github.com/J0pari/GeneSlime.git"
    git remote add origin https://github.com/J0pari/GeneSlime.git
}
REMOTE_URL=$(git remote get-url origin)
echo "✓ Remote: $REMOTE_URL"
echo

CURRENT_BRANCH=$(git branch --show-current)
echo "[4/5] Pushing branch '$CURRENT_BRANCH' to remote..."

if git push -u origin "$CURRENT_BRANCH" 2>&1; then
    echo "✓ Push successful (upstream set)"
elif git push 2>&1; then
    echo "✓ Push successful (upstream already set)"
else
    echo "  Conflict detected, attempting rebase..."
    git pull --rebase || { echo "✗ Rebase failed"; exit 1; }
    git push || { echo "✗ Push failed after rebase"; exit 1; }
    echo "✓ Rebased and pushed"
fi
echo

echo "[5/5] Summary..."
echo "  Commit: $COMMIT_HASH"
echo "  Branch: $CURRENT_BRANCH"
echo "  Message: $MESSAGE"
echo
echo "=== COMPLETE ==="
