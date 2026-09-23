#!/usr/bin/env bash
#SBATCH --job-name=v1-more
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --account=mzding
#SBATCH --qos=mzding
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80gb
#SBATCH --time=08:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/blue/mzding/yujunchen/projects/TRIBEv2_emotion_modeling}"
BASE_V1_RUN="${BASE_V1_RUN:-$(cat "$PROJECT_DIR/latest_v1_quick_run.txt")}"
DATASET_DIR="${DATASET_DIR:-$BASE_V1_RUN/dataset}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_DIR/runs_v1_more_models}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$RUN_ROOT/$RUN_NAME"

cd "$PROJECT_DIR"
mkdir -p logs "$RUN_DIR"

module load conda
module load cuda/12.8.1 || module load cuda/12.9.1 || module load cuda
source /blue/mzding/yujunchen/projects/TRIBEv2/.venv/bin/activate

echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Base V1 run: $BASE_V1_RUN"
echo "Dataset: $DATASET_DIR"
echo "Run dir: $RUN_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
python - <<'PY'
import sys, torch, sklearn, pandas, numpy
print("python", sys.version)
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.version.cuda)
print("sklearn", sklearn.__version__)
print("pandas", pandas.__version__)
print("numpy", numpy.__version__)
PY

python -m src.train_v1_more_models \
  --dataset-dir "$DATASET_DIR" \
  --out-dir "$RUN_DIR"

echo "$RUN_DIR" > "$PROJECT_DIR/latest_v1_more_models_run.txt"
echo "DONE: $RUN_DIR"
