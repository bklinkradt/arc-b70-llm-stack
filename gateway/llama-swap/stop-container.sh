#!/bin/sh
# llama-swap `cmdStop`: stop a model container and give its cards back to the layout
# (see gpu-layout).
#   stop-container NAME
set -eu
gpu-layout unload "$1"
