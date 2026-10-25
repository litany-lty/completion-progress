# completion-progress 共享状态设置
# 默认使用 ~/.local/state/completion-progress（同机多 agent 自动共用）
# 若存在沙箱隔离，可 source env.sh 或 eval "$(python3 install.py --print-env)" 指向本仓库 state/
_CP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export COMPLETION_PROGRESS_STATE="${COMPLETION_PROGRESS_STATE:-$_CP_ROOT/state}"
unset _CP_ROOT
