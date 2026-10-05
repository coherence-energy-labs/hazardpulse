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
  # files the job changed but does not commit must not block the rebase: on 2026-10-03 an earthquake run
  # lost its forecast to "cannot pull with rebase: You have unstaged changes", reported as a conflict.
  # --autostash sets them aside and puts them back; the list is printed so the cause is never hidden.
  git status --short | grep -v '^[AMDR] ' | head -20 || true
  if ! git pull --rebase --autostash "$remote" "$branch"; then
    git rebase --abort || true
    echo "::error::rebase conflict: another writer changed the same generated files; nothing was pushed"
    git status --short | head -40 || true
    exit 1
  fi
done
echo "::error::push still rejected after 3 rebases"
exit 1
