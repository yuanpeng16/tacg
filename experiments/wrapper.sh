#!/usr/bin/env bash

if [ $# = 2 ]; then
  TASK=$1
  MODEL=$2
else
  echo model.
  exit
fi

SCRIPT="$(readlink -f "$0")"
ABS_PATH=$(readlink -f "${SCRIPT}")
cd "$(dirname "$(dirname "${ABS_PATH}")")" || exit

for RANDOM_SEED in $(seq 5); do
  sh experiments/script.sh "${TASK}" "${MODEL}" "${RANDOM_SEED}"
done
