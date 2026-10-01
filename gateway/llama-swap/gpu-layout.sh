#!/bin/sh
# Decides which model containers run on the B70s. layouts.conf says which cards each
# container uses and, per layout, which containers to keep running in priority order.
# On-demand models (llama-swap's, such as qwen-image) come first; then each layout
# member starts if all its cards are free. Everything else that layouts.conf lists is
# stopped. Every command is idempotent and holds a lock, so overlapping calls from
# llama-swap and `make mode` can't interleave.
#   gpu-layout reset         forget on-demand models, then reconcile (container start)
#   gpu-layout reconcile     apply the current layout (after `make mode` or `make up`)
#   gpu-layout load NAME     add an on-demand model, evicting what shares its cards
#   gpu-layout unload NAME   drop it and give its cards back to the layout
#   gpu-layout status        layout, on-demand models and container states
set -eu
DIR=/config/layouts
CONF=$DIR/layouts.conf
# On-demand models llama-swap has loaded. In /tmp, so a restart starts empty, which
# matches llama-swap's own view after a restart.
ACTIVE=/tmp/gpu-layout.active

log() { echo "gpu-layout: $*" >&2; }
cards_of() { awk -v n="$1" '$1 == "card" && $2 == n { for (i = 3; i <= NF; i++) printf "%s ", $i }' "$CONF"; }
managed() { awk '$1 == "card" { print $2 }' "$CONF"; }
mode() { cat "$DIR/mode" 2>/dev/null || echo split; }
members() { awk -v m="$1" '$1 == "layout" && $2 == m { for (i = 3; i <= NF; i++) print $i; f = 1 } END { exit !f }' "$CONF"; }
running() { [ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" = true ]; }

reconcile() {
  layout=$(mode)
  if ! list=$(members "$layout"); then
    log "unknown layout '$layout' in $DIR/mode; see $CONF"
    return 1
  fi
  want=" " used=" "
  for c in $(cat "$ACTIVE" 2>/dev/null) $list; do
    case "$want" in *" $c "*) continue ;; esac
    cards=$(cards_of "$c")
    if [ -z "$cards" ]; then
      log "$c has no card line in $CONF; skipping it"
      continue
    fi
    fits=true
    for g in $cards; do
      case "$used" in *" $g "*) fits=false ;; esac
    done
    if $fits; then
      want="$want$c " used="$used$cards"
    elif grep -qx "$c" "$ACTIVE" 2>/dev/null; then
      log "on-demand $c needs cards held by another on-demand model; not starting it"
    fi
  done
  # Stop before start: the cards must be free before the next container loads.
  for c in $(managed); do
    case "$want" in *" $c "*) continue ;; esac
    if running "$c"; then
      log "stop $c"
      docker stop -t 30 "$c" >/dev/null
    fi
  done
  for c in $want; do
    if ! running "$c"; then
      log "start $c"
      docker start "$c" >/dev/null
    fi
  done
}

status() {
  echo "layout: $(mode)"
  echo "on-demand: $(tr '\n' ' ' < "$ACTIVE" 2>/dev/null)"
  for c in $(managed); do
    printf '  %-16s cards %-5s %s\n' "$c" "$(cards_of "$c")" "$(running "$c" && echo running || echo stopped)"
  done
}

cmd=${1:-status}
name=${2:-}
case "$cmd" in
  status) status; exit ;;
  reset | reconcile) ;;
  load | unload) [ -n "$name" ] || { log "usage: gpu-layout $cmd NAME"; exit 2; } ;;
  *) log "unknown command '$cmd'"; exit 2 ;;
esac

exec 9> /tmp/gpu-layout.lock
flock 9
touch "$ACTIVE"
case "$cmd" in
  reset) : > "$ACTIVE" ;;
  load) grep -qx "$name" "$ACTIVE" || echo "$name" >> "$ACTIVE" ;;
  unload) grep -vx "$name" "$ACTIVE" > "$ACTIVE.new" || true; mv "$ACTIVE.new" "$ACTIVE" ;;
esac
reconcile
