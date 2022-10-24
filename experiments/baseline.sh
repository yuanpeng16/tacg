SCRIPT="$(readlink -f "$0")"
ID=$(basename "${SCRIPT}" | sed "s/.sh$//g")
ABS_PATH=$(readlink -f "${SCRIPT}")
cd "$(dirname "$(dirname "${ABS_PATH}")")" || exit

LOG_DIR="logs/${ID}"
mkdir -p "${LOG_DIR}"
cp "${ABS_PATH}" "${LOG_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  python3 -u main.py \
  --parameter_random_seed 8 |
  tee "${LOG_DIR}"/log.txt
