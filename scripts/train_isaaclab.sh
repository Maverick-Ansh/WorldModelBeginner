#!/usr/bin/env bash
# Train on NVIDIA Isaac Lab (needs a GPU + Isaac Sim/Lab; see docs/sim2real.md).
#   ./scripts/train_isaaclab.sh [dreamerv3|tdmpc2] [extra overrides...]
# Env vars: NUM_ENVS (default 4096), TASK (default Shadow Hand cube reorient).
set -euo pipefail
cd "$(dirname "$0")/.."

ALGO="${1:-dreamerv3}"; shift || true
NUM_ENVS="${NUM_ENVS:-4096}"
TASK="${TASK:-Isaac-Repose-Cube-Shadow-Direct-v0}"

python -m worldmodel.train --algo "$ALGO" --env isaaclab_shadow_hand \
  env.task="$TASK" env.num_envs="$NUM_ENVS" \
  prefill_steps=50000 total_steps=50000000 "$@"
