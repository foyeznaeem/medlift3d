#!/usr/bin/env bash
# Drive a rented GPU box (vast.ai, RunPod, Lambda, any SSH host) from here.
#
# The pattern throughout: long jobs run DETACHED on the remote under tmux, with
# output tee'd to a log. Nothing depends on the SSH session staying up, so a
# dropped connection, a closed laptop or a killed terminal costs nothing.
#
#   ./scripts/remote.sh keygen
#   export MEDLIFT_HOST='-p 12345 root@ssh5.vast.ai'
#   ./scripts/remote.sh setup
#   ./scripts/remote.sh sync
#   ./scripts/remote.sh run  'python scripts/run_gates.py'
#   ./scripts/remote.sh launch train 'python scripts/train_prior.py --data ... --out ...'
#   ./scripts/remote.sh watch  train
#   ./scripts/remote.sh fetch
#   ./scripts/remote.sh cost
set -euo pipefail

KEY="${MEDLIFT_KEY:-$HOME/.ssh/medlift3d_vast}"
HOST="${MEDLIFT_HOST:-}"
REMOTE_DIR="${MEDLIFT_REMOTE_DIR:-/workspace/medlift3d}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS="${MEDLIFT_RESULTS:-$LOCAL_DIR/remote_results}"

die() { echo "error: $*" >&2; exit 1; }

need_host() {
  [ -n "$HOST" ] || die "set MEDLIFT_HOST first, e.g.
  export MEDLIFT_HOST='-p 12345 root@ssh5.vast.ai'
(copy the connection string from the vast.ai instance card)"
}

# shellcheck disable=SC2086
SSH() { need_host; ssh -i "$KEY" -o StrictHostKeyChecking=accept-new \
          -o ServerAliveInterval=30 -o ServerAliveCountMax=6 $HOST "$@"; }

# rsync needs the ssh flags bundled into one -e argument.
rsh_flags() { need_host; echo "ssh -i $KEY -o StrictHostKeyChecking=accept-new ${HOST% *}"; }
host_only() { need_host; echo "${HOST##* }"; }

cmd_keygen() {
  if [ -f "$KEY" ]; then echo "key already exists: $KEY"; else
    mkdir -p "$(dirname "$KEY")"
    # No passphrase: every command here is non-interactive. Acceptable because
    # this key is dedicated to ephemeral rented boxes and used nowhere else --
    # do not reuse it for anything that matters.
    ssh-keygen -t ed25519 -N '' -C "medlift3d-vast" -f "$KEY" >/dev/null
    echo "created $KEY"
  fi
  echo
  echo "Paste this public key into vast.ai -> Account -> SSH Keys:"
  echo "------------------------------------------------------------"
  cat "$KEY.pub"
  echo "------------------------------------------------------------"
}

cmd_setup() {
  echo "== remote environment =="
  SSH 'set -e
    echo "host   : $(hostname)"
    echo "gpu    :"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "  NO GPU VISIBLE"
    echo "disk   : $(df -h /workspace 2>/dev/null | tail -1 || df -h / | tail -1)"
    echo "python : $(python3 -V 2>&1)"
    command -v tmux >/dev/null || { echo "installing tmux..."; (apt-get update -qq && apt-get install -y -qq tmux rsync) >/dev/null 2>&1 || true; }
    echo "tmux   : $(tmux -V 2>/dev/null || echo MISSING)"
    python3 -c "import torch;print(f\"torch  : {torch.__version__} cuda={torch.cuda.is_available()}\")" 2>/dev/null || echo "torch  : not installed"
  '
}

cmd_sync() {
  need_host
  echo "== syncing $LOCAL_DIR -> $(host_only):$REMOTE_DIR =="
  SSH "mkdir -p '$REMOTE_DIR'"
  rsync -az --delete --info=stats1 \
    --exclude '.venv/' --exclude '.git/' --exclude '__pycache__/' \
    --exclude '*.pyc' --exclude 'data/' --exclude 'runs/' \
    --exclude 'remote_results/' --exclude '*.egg-info/' \
    -e "$(rsh_flags)" \
    "$LOCAL_DIR/" "$(host_only):$REMOTE_DIR/"
  echo "== installing package =="
  SSH "cd '$REMOTE_DIR' && pip install -q -e . 2>&1 | tail -3; python3 -c 'import medlift3d;print(\"medlift3d\",medlift3d.__version__,\"ready\")'"
}

cmd_run() {
  [ $# -ge 1 ] || die "usage: remote.sh run '<command>'"
  SSH "cd '$REMOTE_DIR' && export PYTHONUNBUFFERED=1 && $*"
}

cmd_launch() {
  [ $# -ge 2 ] || die "usage: remote.sh launch <name> '<command>'"
  local name="$1"; shift
  local log="$REMOTE_DIR/logs/$name.log"
  SSH "set -e
    mkdir -p '$REMOTE_DIR/logs'
    if tmux has-session -t '$name' 2>/dev/null; then
      echo 'session \"$name\" already running -- stop it first or pick another name'; exit 1
    fi
    cd '$REMOTE_DIR'
    tmux new-session -d -s '$name' \"export PYTHONUNBUFFERED=1; $* 2>&1 | tee '$log'; echo EXIT=\\\$? | tee -a '$log'\"
    sleep 2
    echo \"launched '$name' -> $log\"
    tail -5 '$log' 2>/dev/null || true"
  echo
  echo "detached. it keeps running if this connection drops."
  echo "  ./scripts/remote.sh watch $name"
}

cmd_watch() {
  local name="${1:-}"; [ -n "$name" ] || die "usage: remote.sh watch <name> [lines]"
  local n="${2:-40}"
  SSH "tail -n $n '$REMOTE_DIR/logs/$name.log' 2>/dev/null || echo 'no log for \"$name\" yet'
    echo '---'
    tmux has-session -t '$name' 2>/dev/null && echo 'STATUS: running' || echo 'STATUS: finished or never started'"
}

cmd_follow() {
  local name="${1:-}"; [ -n "$name" ] || die "usage: remote.sh follow <name>"
  echo "following $name (Ctrl-C to detach; the job keeps running)"
  SSH "tail -f '$REMOTE_DIR/logs/$name.log'"
}

cmd_stop() {
  local name="${1:-}"; [ -n "$name" ] || die "usage: remote.sh stop <name>"
  SSH "tmux kill-session -t '$name' 2>/dev/null && echo 'stopped $name' || echo 'no session $name'"
}

cmd_status() {
  SSH "echo '== gpu =='
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null || echo 'no gpu'
    echo; echo '== jobs =='
    tmux ls 2>/dev/null || echo '(none running)'
    echo; echo '== logs =='
    ls -la '$REMOTE_DIR/logs' 2>/dev/null | tail -10 || echo '(none)'
    echo; echo '== outputs =='
    du -sh '$REMOTE_DIR'/{data,runs} 2>/dev/null || echo '(none yet)'
    echo; echo '== disk =='
    df -h '$REMOTE_DIR' | tail -1"
}

cmd_fetch() {
  need_host
  mkdir -p "$RESULTS"
  echo "== fetching results -> $RESULTS =="
  # Checkpoints are large and re-downloadable; grab metrics, figures and the
  # small stuff by default. Use 'fetch-all' when you want the weights too.
  rsync -az --info=stats1 \
    --include='*/' \
    --include='*.csv' --include='*.json' --include='*.png' \
    --include='*.nii.gz' --include='*.log' \
    --exclude='*' \
    -e "$(rsh_flags)" \
    "$(host_only):$REMOTE_DIR/runs/" "$RESULTS/" || true
  echo "fetched into $RESULTS"
  find "$RESULTS" -type f | head -20
}

cmd_fetch_all() {
  need_host
  mkdir -p "$RESULTS"
  echo "== fetching EVERYTHING including checkpoints (may be large) =="
  rsync -az --info=progress2 -e "$(rsh_flags)" \
    "$(host_only):$REMOTE_DIR/runs/" "$RESULTS/"
}

cmd_cost() {
  local rate="${MEDLIFT_RATE:-}"
  SSH "echo \"uptime: \$(uptime -p 2>/dev/null || uptime)\"" || true
  if [ -n "$rate" ]; then
    echo "MEDLIFT_RATE=\$$rate/hr -> multiply the uptime above to estimate spend."
  else
    echo
    echo "set MEDLIFT_RATE to your instance's \$/hr to get a spend estimate:"
    echo "  export MEDLIFT_RATE=0.40"
  fi
  echo
  echo "REMINDER: vast.ai bills while the instance EXISTS, not just while it"
  echo "computes. Fetch results, then DESTROY the instance from the web UI."
}

case "${1:-help}" in
  keygen)    shift; cmd_keygen "$@";;
  setup)     shift; cmd_setup "$@";;
  sync)      shift; cmd_sync "$@";;
  run)       shift; cmd_run "$@";;
  launch)    shift; cmd_launch "$@";;
  watch)     shift; cmd_watch "$@";;
  follow)    shift; cmd_follow "$@";;
  stop)      shift; cmd_stop "$@";;
  status)    shift; cmd_status "$@";;
  fetch)     shift; cmd_fetch "$@";;
  fetch-all) shift; cmd_fetch_all "$@";;
  cost)      shift; cmd_cost "$@";;
  *) sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//';;
esac
