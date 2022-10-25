SCRIPT="$(readlink -f "$0")"
ID=$(basename "${SCRIPT}" | sed "s/.sh$//g")
ABS_PATH=$(readlink -f "${SCRIPT}")
cd "$(dirname "$(dirname "${ABS_PATH}")")" || exit

LOG_DIR="logs/${ID}"
mkdir -p "${LOG_DIR}"
cp "${ABS_PATH}" "${LOG_DIR}"

CUDA_VISIBLE_DEVICES= \
  python3 -u main.py \
  --model no_encoder |
  tee "${LOG_DIR}"/log.txt
