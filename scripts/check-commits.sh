#!/usr/bin/env bash
# Validate that every commit in the range BASE..HEAD is GPG/SSH signed
# and carries a DCO Signed-off-by trailer.
set -e

BASE_BRANCH="${1:-upstream/main}"
BASE=$(git merge-base "$BASE_BRANCH" HEAD)
RANGE="$BASE..HEAD"

ERR_SIG=""
ERR_DCO=""

for sha in $(git rev-list --first-parent "$RANGE"); do
  short=$(git rev-parse --short "$sha")
  # Bypass SSH/GPG verification engine; only check that a signature header exists.
  if ! git cat-file commit "$sha" | grep -q '^gpgsig'; then
    ERR_SIG="$ERR_SIG $short"
  fi
  if ! git log -1 --format=%B "$sha" | grep -q '^Signed-off-by:'; then
    ERR_DCO="$ERR_DCO $short"
  fi
done

if [ -z "$ERR_SIG" ] && [ -z "$ERR_DCO" ]; then
  echo "All commits in $RANGE are signed and signed-off."
  exit 0
fi

[ -n "$ERR_SIG" ] && echo "Commits missing signature:$ERR_SIG"
[ -n "$ERR_DCO" ] && echo "Commits missing Signed-off-by:$ERR_DCO"
echo
echo "Fix with:"
echo "  git rebase --exec 'git commit --amend --no-edit -S -s' -i $BASE_BRANCH"
exit 1
