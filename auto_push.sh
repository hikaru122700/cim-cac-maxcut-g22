#!/usr/bin/env bash
# auto_push.sh
# 10 秒ごとに pull / add / commit / push を繰り返す。
# 出力は 1 イテレーションにつき 1 行。
#
# 使い方:
#   bash scripts/auto_push.sh
#   INTERVAL=5 bash scripts/auto_push.sh   # 間隔を 5 秒に変更
#
# 停止: Ctrl+C
#
# 注意:
#   - git add -A を使うので .gitignore を整備しておくこと（.env 等を弾く）。
#   - 認証は事前に設定しておく（SSH key or credential helper）。

set -u

INTERVAL="${INTERVAL:-10}"
BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)" || {
  echo "not a git repository" >&2
  exit 1
}

trap 'printf "\nstopped\n"; exit 0' INT TERM

while true; do
  ts="$(date '+%Y-%m-%d %H:%M:%S')"

  # pull (rebase + autostash で衝突を最小化)
  if git pull --rebase --autostash origin "$BRANCH" >/dev/null 2>&1; then
    pull_status="ok"
  else
    pull_status="fail"
  fi

  # stage
  git add -A >/dev/null 2>&1

  # commit + push
  commit_info="no-changes"
  push_status="skip"

  if ! git diff --cached --quiet; then
    msg="auto: $ts"
    if git commit -m "$msg" >/dev/null 2>&1; then
      sha="$(git rev-parse --short HEAD)"
      files="$(git diff-tree --no-commit-id --name-only -r HEAD | wc -l | tr -d ' ')"
      commit_info="commit=$sha files=$files"
      if git push origin "$BRANCH" >/dev/null 2>&1; then
        push_status="ok"
      else
        push_status="fail"
      fi
    else
      commit_info="commit=fail"
    fi
  fi

  printf '[%s] pull=%s %s push=%s\n' "$ts" "$pull_status" "$commit_info" "$push_status"

  sleep "$INTERVAL"
done
