#!/bin/sh
# llama-swap assumes none of its models is loaded when it starts. Forget any on-demand
# model and apply the layout, which stops one left running (by a restart of this
# container, or started by hand) and starts the layout's containers on the free cards.
set -eu
gpu-layout reset
exec llama-swap -config /config/config.yaml -listen :8080 -watch-config
