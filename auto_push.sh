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
  # 失敗してもローカル commit は試みる（push だけ skip）。
  # ネット瞬断時に変更が宙ぶらりんになるのを防ぐ。
  pull_online=1
  if pull_err=$(git pull --rebase --autostash origin "$BRANCH" 2>&1 >/dev/null); then
    : # ok
  else
    log "pull=fail: $(echo "$pull_err" | tr '\n' ' ' | cut -c1-200)"
    pull_online=0
  fi

  # ネット復帰後の catch-up: 過去 tick で offline 中に作った未 push コミットを
  # ここでまとめて push する（新しい diff が無くても push される）。
  if [ "$pull_online" -eq 1 ]; then
    ahead=$(git rev-list --count '@{u}..HEAD' 2>/dev/null || echo 0)
    if [ "$ahead" -gt 0 ]; then
      if catchup_err=$(git push origin "$BRANCH" 2>&1 >/dev/null); then
        log "push=ok (catchup $ahead commits)"
      else
        log "push=fail (catchup): $(echo "$catchup_err" | tr '\n' ' ' | cut -c1-200)"
      fi
    fi
  fi

  # stage（ネスト worktree と SQLite 補助ファイルを除外、エラーは表示）
  # SQLite の *.db-journal / *.db-wal / *.db-shm は瞬間的に生成・消滅するため
  # git add -A と race する。.gitignore でも弾いているが pathspec でも除外する。
  add_pathspec=(
    '--'
    '.'
    ':!.claude/worktrees'
    ':!*.db-journal'
    ':!*.db-wal'
    ':!*.db-shm'
  )
  if ! add_err=$(git add -A "${add_pathspec[@]}" 2>&1 >/dev/null); then
    # 「unable to stat」系は transient な可能性が高いので 1 回だけリトライ
    if echo "$add_err" | grep -q "unable to stat"; then
      sleep 1
      if ! add_err2=$(git add -A "${add_pathspec[@]}" 2>&1 >/dev/null); then
        log "add=fail (after retry): $(echo "$add_err2" | tr '\n' ' ' | cut -c1-200)"
        sleep "$INTERVAL"
        continue
      fi
    else
      log "add=fail: $(echo "$add_err" | tr '\n' ' ' | cut -c1-200)"
      sleep "$INTERVAL"
      continue
    fi
  fi

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
  if [ "$pull_online" -eq 0 ]; then
    # pull が失敗したサイクルでは push もほぼ確実に失敗するので、無駄打ちを避ける。
    # 次に pull 成功する tick でまとめて push される。
    log "commit=ok sha=$sha files=$n_files push=skip(offline)"
  elif push_err=$(git push origin "$BRANCH" 2>&1 >/dev/null); then
    log "push=ok sha=$sha files=$n_files"
  else
    log "push=fail sha=$sha: $(echo "$push_err" | tr '\n' ' ' | cut -c1-200)"
  fi

  sleep "$INTERVAL"
done
