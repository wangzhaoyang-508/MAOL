#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

mkdir -p data checkpoints result severity/results severity/results_predicted

echo "MAOL environment setup completed."
