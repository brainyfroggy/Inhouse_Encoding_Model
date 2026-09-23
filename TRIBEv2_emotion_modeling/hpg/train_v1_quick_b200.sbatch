#!/usr/bin/env bash
#SBATCH --job-name=v1-emotion
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --account=mzding
#SBATCH --qos=mzding
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --time=03:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/blue/mzding/yujunchen/projects/TRIBEv2_emotion_modeling}"
RESPONSES_DIR="${RESPONSES_DIR:-/blue/mzding/yujunchen/projects/TRIBEv2/outputs_ckvideo}"
METADATA_CSV="${METADATA_CSV:-/blue/mzding/yujunchen/data/original_ckvideo_data/CowenKeltnerEmotionalVideos.csv}"
LH_V1_LABEL="${LH_V1_LABEL:-$PROJECT_DIR/atlases/fsaverage5/lh.V1_exvivo.label}"
RH_V1_LABEL="${RH_V1_LABEL:-$PROJECT_DIR/atlases/fsaverage5/rh.V1_exvivo.label}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_DIR/runs_v1_quick}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$RUN_ROOT/$RUN_NAME"

cd "$PROJECT_DIR"
mkdir -p logs "$RUN_DIR"

module load conda
module load cuda/12.8.1 || module load cuda/12.9.1 || module load cuda
source /blue/mzding/yujunchen/projects/TRIBEv2/.venv/bin/activate

echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Run dir: $RUN_DIR"
echo "Responses: $RESPONSES_DIR"
echo "Metadata: $METADATA_CSV"
echo "LH V1 label: $LH_V1_LABEL"
echo "RH V1 label: $RH_V1_LABEL"
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

python -m src.quick_v1_roi \
  --responses-dir "$RESPONSES_DIR" \
  --metadata-csv "$METADATA_CSV" \
  --lh-v1-label "$LH_V1_LABEL" \
  --rh-v1-label "$RH_V1_LABEL" \
  --out-dir "$RUN_DIR"

echo "$RUN_DIR" > "$PROJECT_DIR/latest_v1_quick_run.txt"
echo "DONE: $RUN_DIR"
