#!/bin/sh
# llama-swap `cmd` for a model that lives in its own container: claim its cards through
# gpu-layout (which stops whatever shares them and starts fallbacks on the cards left
# over), then stay in the foreground until the container stops, as llama-swap requires.
# Uses only start, stop and inspect because that's all the docker-api proxy allows (no
# attach or wait).
#   run-container NAME
set -eu
name="$1"
gpu-layout load "$name"
trap 'exit 0' TERM INT
while [ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" = true ]; do
  sleep 2 &
  wait $!
done
echo "run-container: $name stopped" >&2
# If it stopped by itself (a crash), give its cards back to the layout. After cmdStop
# this is a no-op.
gpu-layout unload "$name"
