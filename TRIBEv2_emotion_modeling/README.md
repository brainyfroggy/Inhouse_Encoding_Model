# TRIBE v2 Emotion Modeling

This project uses TRIBE v2 predicted fMRI responses as `X` and CK video
valence/arousal annotations as two classification targets:

- `valence_class`: low, mid, high
- `arousal_class`: low, mid, high

Each TRIBE v2 time point is treated as one sample. For a video with multiple
predicted response time points, each time point becomes one `20484`-vertex
feature vector with that video's valence/arousal labels. Train/validation/test
splits are still assigned by video, so time points from the same video stay in
the same split.

## Data Sources On HiPerGator

TRIBE v2 responses:

```bash
/blue/mzding/yujunchen/projects/TRIBEv2/outputs_ckvideo
```

CK emotion metadata:

```bash
/blue/mzding/yujunchen/data/original_ckvideo_data/CowenKeltnerEmotionalVideos.csv
```

## Run On HiPerGator B200

```bash
cd /blue/mzding/yujunchen/projects/TRIBEv2_emotion_modeling
sbatch hpg/train_b200.sbatch
```

Monitor:

```bash
squeue -u yujunchen
tail -f logs/emotion-models-<jobid>.out
```

Outputs are written under:

```bash
/blue/mzding/yujunchen/projects/TRIBEv2_emotion_modeling/runs/<timestamp>
```

The run contains:

- `dataset/`: memmapped `X`, labels, split table, metadata
- `classical/`: linear and nonlinear scikit-learn model metrics
- `neural/`: PyTorch MLP hyperparameter sweep metrics
- `summary/`: merged result tables and best-model reports

## Label Definition

The raw `valence` and `arousal` columns are min-max normalized across videos:

```text
y_norm = (y - min(y)) / (max(y) - min(y))
```

Then divided into fixed thirds:

- low: `[0, 1/3)`
- mid: `[1/3, 2/3)`
- high: `[2/3, 1]`
