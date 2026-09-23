#!/usr/bin/env bash
#SBATCH --job-name=emotion-models
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --account=mzding
#SBATCH --qos=mzding
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=120gb
#SBATCH --time=24:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/blue/mzding/yujunchen/projects/TRIBEv2_emotion_modeling}"
RESPONSES_DIR="${RESPONSES_DIR:-/blue/mzding/yujunchen/projects/TRIBEv2/outputs_ckvideo}"
METADATA_CSV="${METADATA_CSV:-/blue/mzding/yujunchen/data/original_ckvideo_data/CowenKeltnerEmotionalVideos.csv}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_DIR/runs}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$RUN_ROOT/$RUN_NAME"

cd "$PROJECT_DIR"
mkdir -p logs "$RUN_DIR"

module load conda
module load cuda/12.8.1 || module load cuda/12.9.1 || module load cuda

# Reuse the TRIBE v2 environment because it already has CUDA-enabled torch,
# pandas, sklearn, and the dependencies installed on B200-compatible CUDA.
source /blue/mzding/yujunchen/projects/TRIBEv2/.venv/bin/activate

echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Run dir: $RUN_DIR"
echo "Responses: $RESPONSES_DIR"
echo "Metadata: $METADATA_CSV"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
python - <<'PY'
import sys, torch, sklearn, pandas, numpy
print("python", sys.version)
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available(), torch.version.cuda)
print("sklearn", sklearn.__version__)
print("pandas", pandas.__version__)
print("numpy", numpy.__version__)
PY

python -m src.prepare_dataset \
  --responses-dir "$RESPONSES_DIR" \
  --metadata-csv "$METADATA_CSV" \
  --out-dir "$RUN_DIR/dataset" \
  --sample-mode timepoint

python -m src.train_classical \
  --dataset-dir "$RUN_DIR/dataset" \
  --out-dir "$RUN_DIR/classical"

python -m src.train_nn \
  --dataset-dir "$RUN_DIR/dataset" \
  --out-dir "$RUN_DIR/neural"

python -m src.summarize_results --run-dir "$RUN_DIR"

echo "$RUN_DIR" > "$PROJECT_DIR/latest_run.txt"
echo "DONE: $RUN_DIR"
