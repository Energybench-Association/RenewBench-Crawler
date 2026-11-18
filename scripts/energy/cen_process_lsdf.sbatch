#!/bin/bash
# Process CEN data on the FTP cluster and publish it to EnergyBench through the LSDF.
#
# Publishes on the LSDF route: the repository is written straight into the bucket on
# the LSDF (lsdf.dst_dir_data), and the STAC records are handed over through the inbox
# (lsdf.dst_dir_inbox), which the VM publishes from every ten minutes. Only the time periods
# in PERIODS (years or months, e.g. "2024 2025" or "2025-03") are processed (s. --periods);
# the run's outputs and logs go to its run directory WORK.
#
# Usage, from the crawler checkout on a login node:
#   export CEN_API_KEY=...            # never put it in this file
#   RELEASE=v0.1.0 bash scripts/energy/cen_process_lsdf.sbatch
# RELEASE is required (v<major>.<minor>.<patch>, v0.1.0 for the first full run).
# Run with bash, it creates logs/ and submits itself with its Slurm log there (Slurm
# won't create the folder); the crawler's own detailed log is in $WORK/logs/.
# Override any setting below the same way, e.g. PERIODS="2024 2025" RELEASE=v0.1.0 bash ...

#SBATCH --job-name=cen-lsdf
#SBATCH --partition=intel-spr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000mb
#SBATCH --time=24:00:00

set -euo pipefail

: "${CEN_API_KEY:?export CEN_API_KEY before submitting}"
: "${RELEASE:?set RELEASE, e.g. RELEASE=v0.1.0 for the first full run}"

# Not yet in a job: make the log folder, then submit this script with its log there.
if [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p logs
    exec sbatch --output=logs/%x-%j.out "$0" "$@"
fi

SYSOP=cen
LSDF="${LSDF:-/lsdf/kit/scc/projects/renewbench}"
RAW="${RAW:-$LSDF/raw/energy/$SYSOP}"
RESOURCES="${RESOURCES:-$LSDF/coordinate_resources}"
PERIODS="${PERIODS:-$(ls "$RAW/1h" | cut -c1-4 | grep -E '^[0-9]{4}$' | grep -vx "$(date +%Y)" | sort -u | tr '\n' ' ')}"  # every year in RAW except the current one
ENERGYBENCH="${ENERGYBENCH:-$LSDF/energybench}"   # the VM's /mnt/lsdf-data
WORK="${WORK:-$LSDF/work/$SYSOP/run-$RELEASE}"       # run directory, outputs and logs
CRAWLER="${CRAWLER:-$PWD}"

# Group-writable, so the VM can read the data and move the release out of the inbox.
umask 0002

echo "== processing release $RELEASE for: $PERIODS"
cd "$CRAWLER"
# shellcheck disable=SC1091
source .venv/bin/activate
# shellcheck disable=SC2086  # PERIODS splits into one argument per period
python -m scripts.energy.cen_process \
    --release "$RELEASE" \
    --periods $PERIODS \
    -o paths.dst_dir_raw="$RAW" \
    -o paths.dst_dir_proc="$WORK" \
    -o paths.resources_dir="$RESOURCES" \
    -o access.api_key="$CEN_API_KEY" \
    -o route=lsdf \
    -o lsdf.dst_dir_data="$ENERGYBENCH/data" \
    -o lsdf.dst_dir_inbox="$ENERGYBENCH/inbox"

echo "== handed over; the VM publishes it within ten minutes:"
ls -la "$ENERGYBENCH/inbox"
