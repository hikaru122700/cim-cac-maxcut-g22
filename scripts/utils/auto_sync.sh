#!/usr/bin/env bash
# 10 秒ごとに自動 commit + pull --rebase + push を行う同期スクリプト。
#
# 動作:
#   1. ローカル差分があれば `git add -A && git commit -m "auto_sync: ..."`
#      (注: `add -A` なので .env など機密ファイルがあれば .gitignore で除外しておくこと)
#   2. `git pull --rebase --no-edit origin <branch>`
#      衝突したら `git rebase --abort` して衝突ログを出すだけで継続。
#      手動で解決が必要な場合は別ターミナルで作業し、その後ループに戻る。
#   3. `git push origin <branch>`
#
# Usage:
#   bash scripts/utils/auto_sync.sh
#   (プロジェクトルートから、または任意のサブディレクトリから実行可)
#
# Ctrl-C で停止。

set -u

INTERVAL=10
LOG_PREFIX="[auto_sync]"

log() {
    echo "$LOG_PREFIX $(date '+%Y-%m-%d %H:%M:%S') $*"
}

if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
    log "ERROR: git リポジトリではありません。"
    exit 1
fi

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"
BRANCH=$(git rev-parse --abbrev-ref HEAD)

log "起動: root=$ROOT, branch=$BRANCH, interval=${INTERVAL}s"
log "Ctrl-C で停止"

trap 'log "停止しました"; exit 0' INT TERM

while true; do
    # --- 1. ローカル差分があれば commit ---
    has_changes=0
    if ! git diff --quiet || ! git diff --cached --quiet; then
        has_changes=1
    elif [ -n "$(git ls-files --others --exclude-standard)" ]; then
        has_changes=1
    fi

    if [ "$has_changes" -eq 1 ]; then
        git add -A
        if ! git diff --cached --quiet; then
            msg="auto_sync: $(date '+%Y-%m-%d %H:%M:%S')"
            if commit_out=$(git commit -m "$msg" 2>&1); then
                log "commit: $msg"
            else
                log "WARN: commit 失敗 (pre-commit hook など?):"
                echo "$commit_out" | sed "s/^/$LOG_PREFIX /"
            fi
        fi
    fi

    # --- 2. pull --rebase ---
    pull_out=$(git pull --rebase --no-edit origin "$BRANCH" 2>&1)
    pull_ec=$?
    if [ $pull_ec -ne 0 ]; then
        if [ -d "$ROOT/.git/rebase-merge" ] || [ -d "$ROOT/.git/rebase-apply" ]; then
            log "CONFLICT during rebase — abort して再試行待ち:"
            echo "$pull_out" | sed "s/^/$LOG_PREFIX /"
            git rebase --abort >/dev/null 2>&1 || true
        else
            log "WARN: pull --rebase 失敗 (network 等?):"
            echo "$pull_out" | sed "s/^/$LOG_PREFIX /"
        fi
        sleep "$INTERVAL"
        continue
    fi

    # --- 3. push ---
    # ahead がなければ push 自体を省略
    ahead=$(git rev-list --count "@{u}..HEAD" 2>/dev/null || echo "0")
    if [ "$ahead" -gt 0 ]; then
        push_out=$(git push origin "$BRANCH" 2>&1)
        push_ec=$?
        if [ $push_ec -ne 0 ]; then
            log "WARN: push 失敗:"
            echo "$push_out" | sed "s/^/$LOG_PREFIX /"
        else
            log "push 完了 ($ahead commit)"
        fi
    fi

    sleep "$INTERVAL"
done
