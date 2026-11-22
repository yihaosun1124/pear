#!/usr/bin/env bash
# Keep an interactive remote channel alive on this Mila login node.
set -euo pipefail
cd -- "$(dirname -- "$0")"
session=pear-cluv-access
if [[ ${1:-} == --check ]]; then
  failed=0
  for cluster in tamia fir rorqual trillium-gpu nibi; do
    printf '%s: ' "$cluster"
    ssh -o BatchMode=yes -o ConnectTimeout=15 -O check "$cluster" 2>&1 || failed=1
  done
  exit "$failed"
fi
if (($# == 0)); then
  exec tmux attach-session -t "$session"
fi
if (($# != 1)); then
  echo 'Usage: ./login [tamia|fir|rorqual|trillium-gpu|nibi|--check]' >&2
  exit 2
fi
cluster=$1
case "$cluster" in tamia|fir|rorqual|trillium-gpu|nibi) ;; *) exit 2 ;; esac
if ssh -o BatchMode=yes -o ConnectTimeout=15 -O check "$cluster" >/dev/null 2>&1; then
  printf '%s: existing authenticated connection is available on %s\n' "$cluster" "$(hostname)"
  exit 0
fi
if ! tmux has-session -t "$session" 2>/dev/null; then
  tmux new-session -d -s "$session" -n "$cluster" -c "$PWD" "ssh -tt $cluster"
elif ! tmux list-windows -t "$session" -F '#{window_name}' | grep -Fxq "$cluster"; then
  tmux new-window -t "$session" -n "$cluster" -c "$PWD" "ssh -tt $cluster"
fi
tmux select-window -t "$session:$cluster"
if [[ -n ${TMUX:-} ]]; then
  exec tmux switch-client -t "$session:$cluster"
fi
exec tmux attach-session -t "$session:$cluster"
