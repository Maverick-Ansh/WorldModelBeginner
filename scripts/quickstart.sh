#!/usr/bin/env bash
# Fast CPU demo: train both agents briefly on the dependency-free toy hand.
# Extra args are forwarded as config overrides, e.g. ./scripts/quickstart.sh seed=1
set -euo pipefail
cd "$(dirname "$0")/.."

python -m worldmodel.train --algo dreamerv3 --env toy total_steps=3000 prefill_steps=300 "$@"
python -m worldmodel.train --algo tdmpc2    --env toy total_steps=3000 prefill_steps=300 "$@"

echo
echo "Done. Inspect training curves with:  tensorboard --logdir logs"
