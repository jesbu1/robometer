#!/bin/bash
set -u

ROOT=/gpfs/scrubbed/jessezha/robometer_jesse
LOG_DIR=/gpfs/scrubbed/jessezha
SO_LOG=${LOG_DIR}/armnet_tiled_so101_full.log
BI_LOG=${LOG_DIR}/armnet_tiled_bimanual_full.log
WATCH_LOG=${LOG_DIR}/armnet_tiled_watchdog.log
SO_PID=${1:?single-arm PID required}
BI_PID=${2:?bimanual PID required}

alert() {
  message="$1"
  printf '%s %s\n' "$(date -Is)" "$message" >> "${WATCH_LOG}"
  if [[ -n "${ALERT_EMAIL:-}" ]] && command -v mail >/dev/null 2>&1; then
    printf '%s\n' "$message" | mail -s "Armnet tiled conversion watchdog" "${ALERT_EMAIL}"
  fi
  if [[ -n "${ALERT_WEBHOOK_URL:-}" ]] && command -v curl >/dev/null 2>&1; then
    curl -fsS -X POST -H 'Content-Type: application/json' \
      --data "{\"text\":\"${message//\"/\\\"}\"}" "${ALERT_WEBHOOK_URL}" >/dev/null || true
  fi
}

alert "watchdog started for PIDs ${SO_PID} and ${BI_PID}"
while kill -0 "${SO_PID}" 2>/dev/null || kill -0 "${BI_PID}" 2>/dev/null; do
  sleep 60
done

so_done=0
bi_done=0
grep -q "Dataset conversion complete!" "${SO_LOG}" 2>/dev/null && so_done=1
grep -q "Dataset conversion complete!" "${BI_LOG}" 2>/dev/null && bi_done=1
if [[ "${so_done}" == 1 && "${bi_done}" == 1 ]]; then
  touch "${LOG_DIR}/armnet_tiled_conversions.done"
  alert "Armnet tiled conversions completed; inspect Hub upload messages in ${SO_LOG} and ${BI_LOG}"
else
  touch "${LOG_DIR}/armnet_tiled_conversions.failed"
  alert "Armnet tiled conversion failed; inspect ${SO_LOG} and ${BI_LOG}"
fi
