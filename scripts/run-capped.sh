#!/bin/bash
# Run a heavy dopploc command in its own cgroup scope with a hard memory cap, so an overrun
# kills only this job (not the terminal / desktop session). Prints the peak memory afterwards.
#   scripts/run-capped.sh 24G dopploc search cases/oyster.toml
set -uo pipefail
cap="$1"; shift
unit="dopploc-$$-$(date +%s)"
peakfile=$(mktemp)
(
  cg=""
  for _ in $(seq 50); do
    cg=$(systemctl --user show -p ControlGroup --value "$unit.scope" 2>/dev/null)
    [ -n "$cg" ] && break
    sleep 0.2
  done
  while [ -n "$cg" ] && [ -r "/sys/fs/cgroup$cg/memory.peak" ]; do
    cat "/sys/fs/cgroup$cg/memory.peak" > "$peakfile"
    sleep 2
  done
) &
mon=$!
systemd-run --user --scope --quiet --unit="$unit" -p MemoryMax="$cap" -p MemorySwapMax=1G "$@"
rc=$?
wait "$mon" 2>/dev/null
peak=$(cat "$peakfile" 2>/dev/null || echo 0)
rm -f "$peakfile"
echo "exit=$rc peak_memory=$(numfmt --to=iec "${peak:-0}")"
exit $rc
