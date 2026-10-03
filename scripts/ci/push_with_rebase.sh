#!/usr/bin/env bash
# Push a scoring job's data commit. If the branch moved after the job checked out (another writer
# pushed meanwhile), rebase the commit onto the new tip and push again. A rebase CONFLICT means
# another writer changed the same generated files: abort, say so, and fail -- never force.
#
# 2026-10-03: a dispatched hurricane run waited in the scoring concurrency queue, checked out the
# commit main had when it was TRIGGERED, and its push was rejected; its forecasts were lost.
# The workflows now check out the branch tip at job start; this is the second line of defence.
set -u
branch="${GITHUB_REF_NAME:-main}"
remote="${PUSH_REMOTE:-origin}"
for attempt in 1 2 3; do
  if git push "$remote" "HEAD:${branch}"; then
    exit 0
  fi
  echo "push rejected (${branch} moved); rebasing onto ${remote}/${branch} (attempt ${attempt})"
  if ! git pull --rebase "$remote" "$branch"; then
    git rebase --abort || true
    echo "::error::rebase conflict: another writer changed the same generated files; nothing was pushed"
    exit 1
  fi
done
echo "::error::push still rejected after 3 rebases"
exit 1
