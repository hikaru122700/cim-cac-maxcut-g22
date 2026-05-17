#!/usr/bin/env bash
# auto_push.sh
# 一定間隔で pull / add / commit / push を繰り返す自動同期スクリプト。
#
# 使い方:
#   bash auto_push.sh
#   INTERVAL=30 bash auto_push.sh      # 間隔を 30 秒に変更（デフォルト 10）
#   VERBOSE=1 bash auto_push.sh        # 変更が無い tick もログに出す
#
# 停止: Ctrl+C
#
# 注意:
#   - git add -A を使うので .gitignore を整備しておくこと（.env 等を弾く）。
#   - 認証は事前に設定しておく（SSH key or credential helper）。
#   - .claude/worktrees/ 配下は add 対象から除外する（ネスト worktree 対策）。

set -u

INTERVAL="${INTERVAL:-10}"
VERBOSE="${VERBOSE:-0}"
BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)" || {
  echo "not a git repository" >&2
  exit 1
}

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

trap 'printf "\nstopped\n"; exit 0' INT TERM

while true; do
  # index.lock が残っていたら今回はスキップ（並行操作との競合回避）
  if [ -f .git/index.lock ]; then
    log "skip: .git/index.lock exists"
    sleep "$INTERVAL"
    continue
  fi

  # pull (rebase + autostash で衝突を最小化)
  pull_err=$(git pull --rebase --autostash origin "$BRANCH" 2>&1 >/dev/null) || {
    log "pull=fail: $(echo "$pull_err" | tr '\n' ' ' | cut -c1-200)"
    sleep "$INTERVAL"
    continue
  }

  # stage（ネスト worktree を除外、エラーは表示）
  add_err=$(git add -A -- . ':!.claude/worktrees' 2>&1 >/dev/null) || {
    log "add=fail: $(echo "$add_err" | tr '\n' ' ' | cut -c1-200)"
    sleep "$INTERVAL"
    continue
  }

  # 変更が無ければ静かに次へ
  if git diff --cached --quiet; then
    [ "$VERBOSE" = "1" ] && log "no-changes"
    sleep "$INTERVAL"
    continue
  fi

  # コミットメッセージに先頭ファイル名を含める
  files=$(git diff --cached --name-only)
  n_files=$(printf '%s\n' "$files" | wc -l | tr -d ' ')
  head_files=$(printf '%s\n' "$files" | head -3 | tr '\n' ' ' | sed 's/ $//')
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  msg="auto: $ts (${n_files} files) ${head_files}"

  if ! commit_err=$(git commit -m "$msg" 2>&1 >/dev/null); then
    log "commit=fail: $(echo "$commit_err" | tr '\n' ' ' | cut -c1-200)"
    sleep "$INTERVAL"
    continue
  fi

  sha="$(git rev-parse --short HEAD)"
  if push_err=$(git push origin "$BRANCH" 2>&1 >/dev/null); then
    log "push=ok sha=$sha files=$n_files"
  else
    log "push=fail sha=$sha: $(echo "$push_err" | tr '\n' ' ' | cut -c1-200)"
  fi

  sleep "$INTERVAL"
done
