#!/bin/bash
# Process CEN data on the FTP cluster and publish it to EnergyBench through the LSDF.
#
# Publishes on the LSDF route: the repository is written straight into the bucket on
# the LSDF (lsdf.dst_dir_data), and the STAC records are handed over through the inbox
# (lsdf.dst_dir_inbox), which the VM publishes from every ten minutes. Only the months
# in MONTHS are processed: their raw files are linked into a run directory under WORK.
#
# Usage, from the crawler checkout on a login node:
#   export CEN_API_KEY=...            # never put it in this file
#   RELEASE=v0.1.0 bash scripts/energy/cen_process_lsdf.sbatch
# RELEASE is required (v<major>.<minor>.<patch>, v0.1.0 for the first full run).
# Run with bash, it creates logs/ and submits itself with its Slurm log there (Slurm
# won't create the folder); the crawler's own detailed log is in
# $WORK/processed/cen/logs/. Override any setting below the same way, e.g.
# MONTHS="2024 2025" RELEASE=v0.1.0 bash ...

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
RAW="${RAW:-$LSDF/raw/energy/$SYSOP/1h}"
RESOURCES="${RESOURCES:-$LSDF/coordinate_resources}"
MONTHS="${MONTHS:-$(ls "$RAW" | cut -c1-4 | grep -E '^[0-9]{4}$' | grep -vx "$(date +%Y)" | sort -u | tr '\n' ' ')}"  # every year in RAW except the current one
ENERGYBENCH="${ENERGYBENCH:-$LSDF/energybench}"   # the VM's /mnt/lsdf-data
WORK="${WORK:-$LSDF/work/$SYSOP/run-$RELEASE}"       # run directory, links to the raw files, logs
CRAWLER="${CRAWLER:-$PWD}"

# Group-writable, so the VM can read the data and move the release out of the inbox.
umask 0002

echo "== raw files for: $MONTHS"
mkdir -p "$WORK/raw/$SYSOP/1h"
find "$WORK/raw/$SYSOP/1h" -maxdepth 1 -type l -delete
for month in $MONTHS; do
    ln -s "$RAW/$month"-*.csv "$WORK/raw/$SYSOP/1h/"
done
find "$WORK/raw/$SYSOP/1h" -maxdepth 1 -type l | wc -l

echo "== processing release $RELEASE"
cd "$CRAWLER"
# shellcheck disable=SC1091
source .venv/bin/activate
python -m scripts.energy.cen_process \
    --release "$RELEASE" \
    -o paths.dst_dir_raw="$WORK/raw/$SYSOP" \
    -o paths.dst_dir_proc="$WORK/processed/$SYSOP" \
    -o paths.resources_dir="$RESOURCES" \
    -o access.api_key="$CEN_API_KEY" \
    -o route=lsdf \
    -o lsdf.dst_dir_data="$ENERGYBENCH/data" \
    -o lsdf.dst_dir_inbox="$ENERGYBENCH/inbox"

echo "== handed over; the VM publishes it within ten minutes:"
ls -la "$ENERGYBENCH/inbox"
