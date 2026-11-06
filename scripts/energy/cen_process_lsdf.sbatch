#!/bin/bash
# Process CEN data on the FTP cluster and publish it to energybench through the LSDF.
#
# The repository is written straight into the bucket on the LSDF (storage.local_root),
# and the STAC records are handed over through the inbox (stac_inbox), which the VM
# publishes from every ten minutes. Only the months in MONTHS are processed: their raw
# files are linked into a run directory under WORK.
#
# Usage, from the crawler checkout on a login node:
#   export CEN_API_KEY=...            # never put it in this file
#   bash scripts/energy/cen_process_lsdf.sbatch
# Run with bash, it creates logs/ and submits itself with its Slurm log there (Slurm
# won't create the folder); the crawler's own detailed log is in
# $WORK/processed/cen/logs/. Override any setting below the same way, e.g.
# RELEASE=v0.0.2 bash ...

#SBATCH --job-name=cen-lsdf
#SBATCH --partition=intel-spr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000mb
#SBATCH --time=04:00:00

set -euo pipefail

: "${CEN_API_KEY:?export CEN_API_KEY before submitting}"

# Not yet in a job: make the log folder, then submit this script with its log there.
if [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p logs
    exec sbatch --output=logs/%x-%j.out "$0" "$@"
fi

RELEASE="${RELEASE:-v0.0.1}"                      # test release; not v0.1.0
MONTHS="${MONTHS:-2025-01 2025-02 2025-03}"
LSDF="${LSDF:-/lsdf/kit/scc/projects/renewbench}"
RAW="${RAW:-$LSDF/raw/energy/cen/1h}"
RESOURCES="${RESOURCES:-$LSDF/coordinate_resources}"
ENERGYBENCH="${ENERGYBENCH:-$LSDF/energybench}"   # the VM's /mnt/lsdf-data
WORK="${WORK:-$HOME/energybench-cen-$RELEASE}"    # run directory, logs, local copies
CRAWLER="${CRAWLER:-$PWD}"

# Group-writable, so the VM can read the data and move the release out of the inbox.
umask 0002

echo "== raw files for: $MONTHS"
mkdir -p "$WORK/raw/cen/1h"
find "$WORK/raw/cen/1h" -maxdepth 1 -type l -delete
for month in $MONTHS; do
    ln -s "$RAW/$month"-*.csv "$WORK/raw/cen/1h/"
done
find "$WORK/raw/cen/1h" -maxdepth 1 -type l | wc -l

echo "== processing release $RELEASE"
cd "$CRAWLER"
# shellcheck disable=SC1091
source .venv/bin/activate
python -m scripts.energy.cen_process \
    -o paths.dst_dir_raw="$WORK/raw/cen" \
    -o paths.dst_dir_proc="$WORK/processed/cen" \
    -o paths.resources_dir="$RESOURCES" \
    -o access.api_key="$CEN_API_KEY" \
    -o release="$RELEASE" \
    -o data_base_url=https://www.energybench.org/files \
    -o storage.local_root="$ENERGYBENCH/data" \
    -o stac_inbox="$ENERGYBENCH/inbox"

echo "== handed over; the VM publishes it within ten minutes:"
ls -la "$ENERGYBENCH/inbox"
