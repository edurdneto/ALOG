#!/bin/bash
# Runs the ALOQ experiments for the four datasets.
# Results are written to Results/Experiment_<Dataset>/<Budget|Granularity|Numpoints>/
#
# Usage:
#   ./run.sh                 # all datasets
#   ./run.sh uniform porto   # only the datasets given (uniform, normal, geo, porto)

set -e
cd "$(dirname "$0")"

DATASETS=("$@")
if [ ${#DATASETS[@]} -eq 0 ]; then
    DATASETS=(uniform normal geo porto)
fi

for DATASET in "${DATASETS[@]}"; do
    PROFILE="profiles/${DATASET}.txt"
    if [ ! -f "$PROFILE" ]; then
        echo "Profile not found: $PROFILE" >&2
        exit 1
    fi
    echo "=== Running ALOQ on ${DATASET} (${PROFILE}) ==="
    python3 aloq.py "$PROFILE"
done

echo "All executions completed."
