#!/usr/bin/env bash
# run.sh — Build and run the OpenCL MC option pricer
# All command-line arguments are forwarded to the binary.
#
# Examples:
#   ./run.sh                        # default European call
#   ./run.sh -p                     # European put
#   ./run.sh -S 150 -K 145 -v 0.30 # custom parameters
#   ./run.sh -n 2000000 -s 500     # more paths & steps
#   ./run.sh --no-cpu               # skip CPU baseline
#   ./run.sh -h                     # show help

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Building OpenCL MC Pricer ==="
make -j"$(nproc)" all
echo ""

echo "=== Running ==="
./mc_pricer_ocl "$@"
