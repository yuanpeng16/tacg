#!/usr/bin/env bash

if [ $# = 2 ]; then
  MODEL=$1
  RANDOM_SEED=$2
else
  echo model and random_seed.
  exit
fi

SCRIPT="$(readlink -f "$0")"
ID=${MODEL}/${RANDOM_SEED}
ABS_PATH=$(readlink -f "${SCRIPT}")
cd "$(dirname "$(dirname "${ABS_PATH}")")" || exit

LOG_DIR="logs/${ID}"
mkdir -p "${LOG_DIR}"
cp "${ABS_PATH}" "${LOG_DIR}"

CUDA_VISIBLE_DEVICES="" \
  python3 -u main.py \
  --model "${MODEL}" \
  --data_random_seed "${RANDOM_SEED}" \
  --parameter_random_seed "${RANDOM_SEED}" |
  tee "${LOG_DIR}"/log.txt
