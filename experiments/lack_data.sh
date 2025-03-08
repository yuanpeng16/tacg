SCRIPT="$(readlink -f "$0")"
ABS_PATH=$(readlink -f "${SCRIPT}")
cd "$(dirname "$(dirname "${ABS_PATH}")")" || exit

sh experiments/wrapper.sh lack_data
