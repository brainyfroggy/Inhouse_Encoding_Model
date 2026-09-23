# Affective neural encoding research roadmap

> **Static-image consortium update:** the newer, image-only dataset census and TRIBE/MOSAIC training design are in [IMAGE_TRIBE_AFFECTIVE_FMRI_BLUEPRINT.md](N:/Experimental_Data/yujunchen/projects/Inhouse_encoding_model/IMAGE_TRIBE_AFFECTIVE_FMRI_BLUEPRINT.md). Use that document for the fMRI foundation-model build; this broader roadmap remains the reference for affect theory, EEG, and LPP.

**Prepared:** 2026-07-28  
**Scope:** BERG, BERG-main, TRIBE v2, local IAPS/NAPS/EmoSet/AffectNet/CK-video analyses, affective fMRI, affective EEG, LPP, and a path to an emotion-capable stimulus-to-brain model.  
**Literature cutoff:** 2026-07-28. The search prioritized primary papers, official dataset records, model papers/code, and the user's local analyses. Preprints are labeled as such.

## Executive recommendation

Do this as a two-stage research program rather than immediately training a large new model.

1. **Build an affective-validity benchmark for existing encoders.** Freeze BERG, BERG-main, and TRIBE v2 and ask whether their predicted neural responses reproduce *measured* affective neural phenomena after visual and semantic confounds are controlled. The primary result should distinguish early visual/expression information from sustained affective processing.
2. **Add a small, factorized affect adapter.** Start with frozen backbones and low-rank adapters trained on paired affective neural data. Use separate categorical, continuous valence/arousal, appraisal, and subject-residual heads. Add recurrent/top-down dynamics only after the benchmark identifies where and when the generic models fail.

The best theoretical frame is a **hybrid emotion-schema + motivated-attention/recurrent-gain account**:

- Feedforward visual/object/face systems rapidly extract affect-predictive cues.
- Objects, actions, faces, and scene context instantiate learned emotion schemas.
- Motivational relevance and appraisal recruit recurrent amplification, producing EPN/LPP and sustained sensory gain.
- Emotion is represented jointly as continuous affective geometry, graded categories, appraisals/action tendencies, and subject-specific state.

This frame makes falsifiable predictions without claiming that a feedforward null result proves reentry. It also fits recent results showing that object representations often predict everyday-scene emotion, while explicitly affect-trained representations help more for deliberately evocative stimuli ([Gao et al., 2025](https://doi.org/10.1038/s42003-025-08145-1)); that rich video emotion is high-dimensional and distributed ([Horikawa et al., 2020](https://doi.org/10.1016/j.isci.2020.101060)); and that hippocampus and vmPFC favor different categorical versus affective-space representations ([Ma & Kragel, 2026](https://doi.org/10.1038/s41467-025-68240-z)).

The strongest eventual claim is:

> Emotion-rich paired neural training improves prediction of real, held-out affective brain dynamics—especially sustained motivated-attention signals—on unseen stimuli and unseen people, beyond low-level vision, objects, faces, semantics, and the source backbone.

Avoid making the headline claim “emotion can be decoded from synthetic EEG/fMRI.” A decoder can recover stimulus semantics copied through a generator even when the generator is neurally wrong.

## 1. What the current results support

### 1.1 The face result is a useful positive control, not evidence of a general emotion code

The local AffectNet face contrasts are strongest over occipital channels around 100 ms, like the early peaks for IAPS and NAPS. Late performance is much weaker. This timing is consistent with separable eyes, mouth, pose, texture, and facial action geometry. It is not an LPP-like signature and does not show that the model represents a viewer's felt emotion.

Accordingly, the face benchmark must be:

- leave-identity-out;
- cross-dataset;
- pose-, intensity-, and background-controlled;
- compared with facial landmarks/FACS action units and an image-only backbone;
- evaluated separately in early visual, N170/EPN, and late windows.

Face emotion should remain as a positive control. NFED is a strong fMRI dataset for this role ([Scientific Data paper](https://doi.org/10.1038/s41597-024-04088-0), [OpenNeuro ds005047](https://openneuro.org/datasets/ds005047)). It should not be used as the main evidence for general scene affect.

### 1.2 The scene null is informative, but it does not prove reentry

The 2024 NEM deck shows a consistent pattern:

- object decoding is strong;
- emotion decoding falls toward chance after object balancing in IAPS/NAPS;
- face-expression contrasts decode;
- ViT/global-semantic features recover modest scene valence, but stimulus shortcuts remain plausible;
- EmoSet examples reveal strong category/object/style biases.

Several mechanisms can produce this pattern: training distribution, label noise, stimulus imbalance, insufficient subjective/contextual variables, readout limitations, missing late dynamics, or measurement noise. Therefore:

> “Feedforward model fails, therefore reentry is necessary” is not a valid causal inference.

Amygdala projections back to visual cortex do make recurrent gain biologically plausible ([Freese & Amaral, 2006](https://pmc.ncbi.nlm.nih.gov/articles/PMC2564872/)), but the hypothesis must be tested through matched feedforward/recurrent models, temporal EEG, context manipulations, or perturbation—not inferred from one model's failure.

### 1.3 BERG's current EEG architecture explains the LPP null

The official BERG EEG model is an ImageNet ViT-B/32 feature extractor followed by PCA and independent linear regressions for every sensor and time sample. It was trained on THINGS-EEG2 rapid object-viewing data, covers approximately -100 to +600 ms, and has no recurrent neural state. The same static image vector is read out separately at each latency.

Consequences:

- It is well suited to early visual/object responses.
- Its four outputs are fitted training repetitions, not independent biological subjects or realistic trial noise.
- It has no mechanism for task-, appraisal-, or attention-dependent late dynamics.
- It cannot model the full sustained LPP, which may continue well beyond 600 ms.
- A null LPP is expected and is not evidence that the human data lack an LPP.

Relevant local source evidence:

- model card: `BERG-main/BERG/berg/models/model_cards/eeg-things_eeg_2-vit_b_32.yaml`;
- training pipeline: `BERG-main/BERG/berg_creation_code/02_train_encoding_models/train_dataset-things_eeg_2/model-vit_b_32/train_encoding.py`;
- local result files under `BERG/outputs/IAPS_AI`, `BERG/outputs/IAPS_balanced`, `BERG/outputs/NAPS_balanced`, and `BERG/outputs/AffectNet`.

### 1.4 The local EEG analyses suggest that real late affective geometry may exist

The older IAPS EEG RSA slides show an arousal-related representational peak roughly in the 500–800 ms range and weaker/less stable valence geometry, especially after correction. This is compatible with the motivated-attention account: LPP amplitude is more reliably tied to emotional arousal/relevance than to ordered positive-versus-negative valence.

The canonical findings are:

- EPN: roughly 180–300 ms over posterior sensors, reflecting facilitated emotional selection ([Schupp et al., 2003](https://doi.org/10.1111/1467-9280.01411));
- LPP: begins near 300 ms, is centroparietal, and may persist for seconds; emotional pleasant and unpleasant stimuli usually exceed neutral ([Cuthbert et al., 2000](https://doi.org/10.1016/S0301-0511(99)00044-7));
- simultaneous EEG-fMRI links LPP variation to visual/temporal cortex, amygdala, orbitofrontal cortex, and insula, with limited pleasant-versus-unpleasant separation ([Liu et al., 2012](https://doi.org/10.1523/JNEUROSCI.3109-12.2012));
- task relevance interacts most clearly with emotion in the late potential ([Schindler et al., 2020](https://doi.org/10.1111/psyp.13585)).

Thus the primary EEG target should be **emotional/motivational salience and continuous arousal/relevance**, not discrete category accuracy alone.

### 1.5 TRIBE v2 is promising, but its current local results are not human group statistics

[TRIBE v2](https://arxiv.org/abs/2605.04326) is a 2026 multimodal fMRI foundation model. It combines video, audio, and language features and predicts average-subject high-resolution responses. The paper reports more than 1,000 hours across 720 participants, but the training subset is 25 participants and 451.6 fMRI hours; the larger totals include evaluation datasets. It was not trained with an explicit affective objective.

The local TRIBE results show widespread associations between predicted cortical response and IAPS/CK-video valence/arousal. These results are useful as *in-silico model characterization*, but:

- the responses are deterministic average-subject predictions;
- the current local outputs are fsaverage5 cortical surface responses;
- they are not a human-subject second-level analysis;
- thousands of FDR-significant vertices can reflect visual, object, face, action, audio, or semantic covariates;
- still images repeated into silent MP4s are out of distribution for a contextual movie model;
- mean-over-time responses can mix stimulus onset, clip edge, context-window, and hemodynamic effects.

The current three-class CK-video code does use video-disjoint 70/15/15 splits, which prevents direct frame/video leakage. However, treating all time points as samples weights long videos more heavily, correlated time points inflate apparent effective sample size, and model ranking on the test set is not valid. Predictions and inference should be aggregated or bootstrapped at video level, with one locked test set selected only once.

### 1.6 Local analysis issues to repair before publication

The audit found several fixable issues in the personal BERG notebooks:

- some pipelines convert emotional images to grayscale, removing potentially valid color-affect cues;
- baseline-corrected arrays are computed but the uncorrected arrays are averaged;
- time samples are mislabeled as 50-ms increments in places;
- a smoothing call uses `fs=1.0` although the model output is 200 Hz;
- one notebook subtracts the previous array condition as “deconvolution,” which is invalid unless array order is the true continuous presentation order;
- FDR is sometimes applied within sensor/comparison rather than across the complete sensor × time × contrast family;
- synthetic repeats/subjects are sometimes treated as if they were human replication.

Until corrected, significant 300–420 ms posterior points should not be called a replicated LPP.

The high fMRI decoding shown in the 2026 NSD-emotion deck is interesting, but 70–95% accuracy warrants a strict audit of repeated-image leakage, split level, stimulus selection, task labels, semantic imbalance, and whether image repetitions occur across folds.

## 2. Neuroscience theory to operationalize

### 2.1 Primary theory: emotion schemas plus motivated-attention recurrent gain

This theory predicts a temporal and representational hierarchy:

1. **80–180 ms:** luminance, contrast, spatial frequency, color, layout, face/body parts, and early object features dominate.
2. **180–350 ms:** objects, actions, expressions, and emotion-predictive semantics produce EPN/N170-like differentiation.
3. **300–1500+ ms:** appraisal of motivational relevance, arousal, goals, and context produces recurrent amplification, sustained LPP, and posterior alpha/low-beta suppression.
4. **Seconds to minutes:** temporal relations, memory, and state trajectories support category/context and continuous affect in transmodal cortex, vmPFC, and hippocampal systems.

This is consistent with:

- object and LOC representations predicting emotion ratings for everyday scenes ([Gao et al., 2025](https://doi.org/10.1038/s42003-025-08145-1));
- an affect-trained visual network predicting deliberately evocative emotion schemas and visual-cortical activity ([Kragel et al., 2019](https://doi.org/10.1126/sciadv.aaw4358));
- combined semantic, valence, and arousal features explaining natural-image occipito-temporal responses ([Abdel-Ghaffar et al., 2024](https://doi.org/10.1038/s41467-024-49073-8));
- 34-category features explaining unique transmodal fMRI variance beyond visual/semantic covariates ([Horikawa et al., 2020](https://doi.org/10.1016/j.isci.2020.101060));
- arousal generalizing across naturalistic movie datasets more reliably than valence ([Ke et al., 2025](https://doi.org/10.1371/journal.pcbi.1012994)).

### 2.2 Competing representational theories should become model heads, not verbal alternatives

Do not force the project to choose one ontology before testing it. Use a factorized latent with several supervised views:

- **dimensional:** valence, arousal, dominance/intensity;
- **categorical:** graded probabilities over 20–34 categories rather than a single hard label;
- **appraisal/component:** novelty, goal relevance, control, social norms, action tendency, bodily response, feeling;
- **semantic schema:** objects, actions, faces, bodies, scenes, relationships;
- **subject state:** participant-specific ratings, regulation instruction, physiology, and trait/state embedding.

The branches can compete through held-out variance partitioning. Recent fMRI supports this hybrid: hippocampal responses favor fine-grained category/conjunctive structure, whereas vmPFC better tracks integrated trajectories in affective space ([Ma & Kragel, 2026](https://doi.org/10.1038/s41467-025-68240-z)). Component-process features also map onto distributed neural networks ([Mohammadi et al., 2023](https://doi.org/10.1093/cercor/bhad093)).

### 2.3 What would count as evidence for recurrent/reentry processing?

At minimum, require all three:

1. A recurrent model beats a matched feedforward model with comparable backbone, capacity, data, and optimization.
2. Its advantage emerges in late measured EEG/EPN-LPP or context-sensitive fMRI, not only in early visual responses.
3. Removing the top-down/appraisal pathway selectively reduces the late/contextual effect while preserving early visual fidelity.

Stronger evidence would add a context or regulation manipulation using identical images, an ssVEP visual-gain paradigm, effective-connectivity data, laminar data, stimulation/lesion evidence, or a preregistered perturbation. Model ablation alone is computational support, not proof of a biological causal circuit.

## 3. Prioritized data plan

### 3.1 fMRI datasets

| Priority | Dataset | Scale and labels | Role in this program | Critical caveat |
|---|---|---|---|---|
| 1 | [Abdel-Ghaffar natural images](https://doi.org/10.1038/s41467-024-49073-8), [OSF](https://osf.io/b5pxu/) | 6 densely sampled participants; 1,440 training and 180 independent-test emotional images; subject valence/arousal; 23 semantic classes | Best static-image affective encoding benchmark and adapter set; directly supports semantic × affect variance partitioning | Some source images cannot be redistributed; preserve the authors' independent test and copyright restrictions |
| 1 | [Horikawa/Kamitani emotion videos](https://doi.org/10.1016/j.isci.2020.101060), [code/data](https://github.com/KamitaniLab/EmotionVideoNeuralRepresentation), [OpenNeuro ds002425](https://openneuro.org/datasets/ds002425) | 5 densely sampled participants; about 2,181 unique silent clips; 34 categories, 14 dimensions, visual and semantic covariates | Best high-dimensional affect benchmark and training set; large stimulus diversity | Small participant N; splits must be clip/source-disjoint |
| 1 | [Emo-FilM](https://doi.org/10.1038/s41597-025-04803-5), [ratings ds004872](https://openneuro.org/datasets/ds004872), [fMRI ds004892](https://openneuro.org/datasets/ds004892) | 30 participants; 14 films, more than 2.5 h; fMRI, physiology; 50 continuous emotion/component items from 44 independent raters | Best appraisal, temporal-context, and physiology-aware fine-tuning set | Only 14 films; leave-one-film-out or source-held-out validation is essential |
| 1 | [BOLD5000](https://doi.org/10.1038/s41597-019-0052-3) + [Gao ratings/code](https://osf.io/eks8u/) | 4 participants; 4,916 diverse images; 20-category ratings for 4,913 images from 300 raters | Tests the everyday object/schema regime and comparison against LOC/object features | Only 4 neural participants; normative ratings are not each scanned participant's felt state |
| 2 | [REELMO](https://doi.org/10.1038/s41597-025-05159-6), [release](https://doi.org/10.6084/m9.figshare.28255745) | 20 participants viewing a complete movie; second-wise reports for 20 states | Strong held-out naturalistic movie test | fMRI covers one film; temporal and story autocorrelation must be modeled |
| 2 | [StudyForrest](https://www.studyforrest.org/data.html), [emotion annotations](https://pmc.ncbi.nlm.nih.gov/articles/PMC4416536/) | 12 observers; long audiovisual movie fMRI, physiology/eye data, portrayed-emotion annotations | External dynamic-context test | Portrayed emotion is not necessarily felt emotion |
| 2 | [NFED ds005047](https://openneuro.org/datasets/ds005047) | 5 participants; 1,320 natural facial-expression videos | Face/expression positive control | Not a general scene-affect dataset |
| 2 | [BOLD Moments ds005165](https://openneuro.org/datasets/ds005165), [paper](https://doi.org/10.1038/s41467-024-50310-3) | 10 participants; 1,102 three-second natural videos with rich content metadata | Additional short-video benchmark with new affect ratings | Used in TRIBE v2 training, so it cannot be a clean external TRIBE test |
| 2 | [Dense Amygdala ds006947](https://openneuro.org/datasets/ds006947), [paper](https://doi.org/10.1038/s41597-026-07065-x) | 3 extensively sampled participants; about 520 min of movies; dense amygdala/ventral coverage | Subject-specific limbic and temporal-context analysis | Very small N and partial coverage; use as mechanistic replication |
| 3 | [Affective videos ds000205](https://legacy.openfmri.org/dataset/ds000205/) | 11 participants; five-second audiovisual clips; valence/arousal | Small cross-person benchmark | Limited scale |
| Generic | [NSD](https://naturalscenesdataset.org/) | 8 densely sampled participants; 7 T; 9,000–10,000 COCO images/person | Generic visual-neural pretraining and stimulus-rich control | Affect is incidental and no native participant affect labels exist |

The first four datasets are sufficient for the central fMRI work. Start with the Abdel-Ghaffar set because it matches the image task, has an independent validation set, and already has a local project directory: `N:\Experimental_Data\yujunchen\projects\Abdel_Ghaffar_2024_Searchlight`.

### 3.2 EEG datasets

| Priority | Dataset | Scale and paradigm | Role | Critical caveat |
|---|---|---|---|---|
| 1 | [OpenNeuro ds006866](https://doi.org/10.18112/openneuro.ds006866.v1.0.0) | N=148; 64 channels; 240 five-second social/non-social neutral, negative, and reappraisal trials; trial ratings | Main large static-emotion/LPP and regulation training set | Negative/regulation-heavy; IAPS-like stimuli may require separate licensing |
| 1 | [Schindler OSF](https://osf.io/hd78f/), [paper](https://doi.org/10.1111/psyp.13585) | N=104; 64 channels; positive/negative/neutral IAPS; task relevance | Clean EPN/LPP benchmark with positive and negative conditions | Keep exact images together across all folds and external sets |
| 1 | [OpenNeuro ds006861](https://doi.org/10.18112/openneuro.ds006861.v1.0.2) | N=120; active/sham tDCS sessions; 32-channel EEG, ECG, EDA | Untouched external and regulation/perturbation-sensitive test | Session and intervention must be modeled hierarchically |
| 1 | User's real IAPS EEG | Approximately 20 participants, 31 channels, 60 images, long epochs | Direct replication of the current LPP/RSA questions and shared-stimulus bridge to prior fMRI | Too few unique images for a final generalization claim; repetitions must stay grouped by image |
| 2 | [EmoEEG-MC](https://doi.org/10.18112/openneuro.ds005540.v1.0.7), [paper](https://doi.org/10.1038/s41597-025-05349-2) | 59 public participants; 64-channel EEG and peripherals; video and imagery; 7 categories | Cross-context representation and subject adapter training | Contexts differ substantially, which is both strength and confound |
| 2 | [FACED](https://doi.org/10.7303/syn50614194), [paper](https://doi.org/10.1038/s41597-023-02650-w) | N=123; 32 channels; 28 videos; 9 categories and ratings | Category-rich population training | Only 28 stimuli; leave-video/source-out is mandatory |
| 2 | [DENS ds003751](https://doi.org/10.18112/openneuro.ds003751.v1.0.2) | 38 public recordings; 128-channel EEG, ECG, EMG; naturalistic clips and localized emotion reports | Dynamic context and peripheral physiology | Button-press, audio, motion, and uncertain event latency |
| 2 | [THINGS-EEG2](https://doi.org/10.1016/j.neuroimage.2022.119754) | N=10; 82,160 trials/person; 16,740 object-image conditions | Generic early visual EEG pretraining and a baseline for BERG | Rapid viewing, short epochs, only ten participants; not an LPP training set |
| 3 | [DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/), [DREAMER](https://zenodo.org/records/546113), [SEED](https://bcmi.sjtu.edu.cn/home/seed/) | Common video emotion datasets with EEG and some peripherals | Secondary transfer and robustness | Few shared clips, small N or low-density EEG; high leakage risk |

Recommended holdout policy: train/adapt with ds006866 + FACED/EmoEEG-MC; develop on the user's IAPS EEG and Schindler; keep ds006861 or one entire complete dataset untouched for the final external test.

### 3.3 Behavioral stimulus datasets

Behavioral datasets can teach the stimulus encoder affective structure, but they cannot by themselves teach a neural encoding function.

- [OASIS](https://doi.org/10.3758/s13428-016-0715-3): 900 open images, valence/arousal.
- [NAPS](https://pmc.ncbi.nlm.nih.gov/articles/PMC4030128/): 1,356 images, valence/arousal/approach-avoidance.
- [FindingEmo](https://proceedings.neurips.cc/paper_files/paper/2024/file/08a7229eaba3b35cd8f933ac678f2096-Paper-Datasets_and_Benchmarks_Track.pdf): 25,869 complex social images with valence/arousal/emotion annotations.
- [EMOTIC](https://arxiv.org/abs/2003.13401): 23,571 images and 34,320 annotated people, with categories and VAD; valuable for person-in-context factorization.
- [EmoSet](https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_EmoSet_A_Large-scale_Visual_Emotion_Dataset_with_Rich_Attributes_ICCV_2023_paper.pdf): 118,102 human-labeled images with eight categories and interpretable visual attributes.
- [Cowen–Keltner clips](https://doi.org/10.1073/pnas.1702247114): about 2,185 videos with rich category/dimension ratings.
- [LIRIS-ACCEDE](https://www.interdigital.com/data_sets/liris-accede): 9,800 short Creative Commons film clips with valence/arousal ranking.

IAPS remains useful for direct comparison with the LPP/PINES literature but is controlled-access and should not be the only training corpus.

### 3.4 What unlabeled emotional data can and cannot do

There are four distinct data regimes:

1. **Stimulus + measured brain + labels:** strongest; supports neural encoding and explicit affect supervision.
2. **Stimulus + measured brain, no affect labels:** useful for self-supervised stimulus–brain alignment, masked temporal prediction, cross-subject alignment, and later pseudolabeling.
3. **Stimulus + affect labels, no brain:** useful for an affective stimulus adapter; cannot establish a neural mapping.
4. **Stimulus alone:** useful for visual/video pretraining but provides no direct neural or affect constraint.

Unlabeled emotional neural recordings can help if the stimuli and brain signals are time-aligned. Use an ensemble of independently trained affect models only for low-confidence auxiliary pseudolabels; keep all headline tests human-labeled and measured-neural.

## 4. Affective-validity benchmark for BERG and TRIBE v2

### 4.1 Four questions every model must answer

1. **Neural fidelity:** Does predicted activity match real held-out EEG/fMRI, not just labels?
2. **Incremental affect:** Does the predicted neural response carry affect beyond the source backbone and image-only affect model?
3. **Confound resistance:** Does affect survive control of low-level, object, face, action, scene, language, and audio features?
4. **Generalization:** Does it transfer to unseen stimuli, subjects, semantic categories, and datasets?

### 4.2 Model matrix

| Model | Expected strength | Expected failure | Required comparison |
|---|---|---|---|
| BERG fWRF/NSD | Retinotopic early visual and object structure | Global semantics, late context, subject affect | Gabor/low-level and object-feature baselines |
| BERG fMRI ViT | Global visual semantics | Affect-specific and temporal/contextual variance | Raw ViT layer probes versus predicted fMRI |
| BERG THINGS-EEG2 ViT | Early object-related scalp pattern to 600 ms | Sustained LPP, task regulation, trial dynamics | Raw ViT, THINGS-trained linear head, and affect-adapted head |
| TRIBE v2 | Naturalistic multimodal cortical dynamics and semantic localizers | Still-image OOD behavior, group-average affect, subject-specific state, uncertain subcortical API coverage | V-JEPA/video backbone, modality ablations, cortical/volumetric coverage audit |
| EmoNet / affect vision encoder | Stimulus affect labels and evocative-scene schemas | Neural validity and individual felt state | Same images through neural encoder and real neural data |

For each model, probe three points:

- source stimulus backbone;
- predicted neural response;
- real measured neural response.

Interpretation:

- Affect in backbone but not predicted brain: neural readout suppresses affective variance.
- Affect in neither: stimulus representation is inadequate.
- Predicted brain exceeds a matched backbone and matches real neural geometry: neural alignment adds value.
- Predicted brain decodes affect but fails measured-neural prediction: likely semantic copying, not neural fidelity.

### 4.3 Benchmark stimulus conditions

Use a deliberately crossed battery:

1. **Low-level matched:** luminance, contrast, spatial frequency, saturation, color, and saliency matched.
2. **Object/semantic matched:** same object or scene categories across valence/arousal.
3. **Cross-category:** train on people/animals; test on scenes/objects, and reverse.
4. **Face control:** identity-disjoint emotional expressions with FACS/landmark baseline.
5. **People removed/masked:** scene emotion without visible faces/bodies.
6. **Context removed/masked:** person/face crop without the scene.
7. **Cross-dataset:** IAPS → NAPS/OASIS/FindingEmo and vice versa.
8. **Temporal context:** isolated frame versus 4 s, 10 s, and 30 s video context.
9. **Regulation/internal state:** identical negative stimulus under watch versus reappraise/suppress.

### 4.4 Nuisance feature set

Fit nuisance and target models with nested cross-validation or banded ridge/variance partitioning. Include:

- luminance, contrast, color histograms, spatial frequency/GIST;
- saliency, fixation density, center bias;
- face/body/person count, facial landmarks/FACS, nudity;
- object, action, place, and scene embeddings;
- CLIP/DINO/V-JEPA layer features;
- motion energy, optic flow, cuts, camera motion;
- audio energy, pitch, speech, prosody, music;
- captions/text sentiment and topic;
- trial order, repetition, run/session, duration;
- eye movement, blink, EOG/EMG, heart rate, respiration, skin conductance where recorded.

Primary effect: cross-validated incremental neural variance or representation fit uniquely explained by affect/appraisal after these features—not a large uncorrected classification accuracy.

### 4.5 Split design

Use nested outer folds that simultaneously group:

- exact stimulus and all repetitions;
- source video/movie and adjacent temporal blocks;
- actor/identity;
- perceptual near-duplicates, detected with hashes/embeddings;
- semantic/object cluster;
- participant;
- dataset/device/site for the external test.

Report four outer-fold settings separately:

1. seen person, unseen stimulus;
2. unseen person, seen stimulus;
3. unseen person, unseen stimulus — primary population claim;
4. unseen dataset/context/device — strongest claim.

All preprocessing, scaling, PCA, feature selection, hyperparameter tuning, subject alignment, and class thresholding must be learned inside training folds. Select models only on validation; evaluate one locked test once.

### 4.6 Metrics and inference

**Encoding metrics**

- cross-validated sensor × time or voxel/vertex correlation;
- ordinary and noise-ceiling-normalized R²;
- waveform MAE and temporal generalization;
- ROI-balanced and whole-brain scores;
- signed EPN/LPP contrast prediction;
- real-versus-predicted representational geometry.

**Affect metrics**

- continuous: concordance correlation, Spearman r, MAE, calibration;
- categorical: balanced accuracy, macro-F1, AUROC, log loss/Brier score;
- RSA: cross-validated Mahalanobis neural RDMs, partial RSA, variance partitions.

**Inference**

- bootstrap both subjects and stimuli;
- hierarchical models with random subject and stimulus effects;
- grouped label permutation and video-level/circular-shift temporal nulls;
- cluster permutation or TFCE across EEG sensor × time;
- whole-brain correction and predeclared ROIs;
- learning curves and reliability/noise ceilings.

Do not treat model-generated “subjects” as inferential replication. Do not interpret classifier weights as activation maps without a Haufe transformation ([Haufe et al., 2014](https://doi.org/10.1016/j.neuroimage.2013.10.067)).

## 5. Proposed model: Affect-BERG

### 5.1 Minimum viable version

Do not train a full foundation model first. Build a frozen-backbone adapter that makes the key scientific comparison cheap and fair.

```text
stimulus
 ├─ generic visual/object/action stream (DINO/ViT/V-JEPA)
 ├─ face/expression/body stream
 ├─ scene/context affect stream
 ├─ audio/prosody stream for videos
 ├─ speech/text semantic stream for videos
 └─ temporal-context encoder
                 ↓ gated fusion
shared affect/appraisal latent
 ├─ graded category head
 ├─ valence/arousal/dominance/intensity head
 ├─ appraisal/component head
 └─ subject/session residual
                 ↓
neural heads
 ├─ cortical fMRI surface head
 ├─ volumetric subcortical fMRI head
 └─ coordinate-aware time-resolved EEG head
```

Recommended implementation choices:

- freeze the source backbone initially;
- retain spatial patch tokens rather than averaging them away;
- add small LoRA/low-rank adapters and a gated affect latent;
- learn group/canonical response plus small subject and session adapters;
- include sensor/vertex coordinates and montage/reference adapters;
- use an explicit HRF/FIR module for fMRI;
- use a temporal convolution, state-space, or recurrent module for EEG;
- produce EEG to at least 1.5–2 s post-stimulus for LPP work;
- weight ROIs/sensors/time so reliable early visual cortex does not dominate the loss.

### 5.2 Strong recurrent version

After the minimum model establishes an affect-specific deficit and adapter benefit, add a top-down recurrent pathway:

1. spatial visual tokens feed a semantic/appraisal state;
2. the appraisal state sends learned gain back to spatial tokens;
3. recurrent steps are supervised by temporal EEG and context-sensitive fMRI;
4. early output is read before feedback; late output is read after one or more feedback steps;
5. an ablation removes only feedback while preserving feedforward capacity.

This produces a clean prediction: feedback should improve EPN/LPP and context/regulation effects more than P1/N1 or simple object decoding.

### 5.3 Losses

A practical multi-task objective is:

```text
L = λ_fMRI L_noise-ceiling-weighted-fMRI
  + λ_EEG L_sensor×time-waveform
  + λ_RSA L_neural-geometry
  + λ_affect L_category+continuous+appraisal
  + λ_ERP L_EPN/LPP-ROI
  + λ_contrast L_stimulus-neural-alignment
  + λ_regularize L_adapter/subject-residual
```

Important details:

- optimize neural prediction first; affect decoding is downstream validation;
- use measured subject-level data and noise ceilings;
- learn HRF lag/FIR parameters only in training folds;
- use cross-validated ROI weighting;
- keep category outputs soft/multi-label;
- retain participant-specific ratings where available instead of replacing them with normative means.

### 5.4 Training schedule

1. **Generic neural pretraining:** retain BERG/TRIBE/THINGS/NSD abilities.
2. **Stimulus-side affect adaptation:** train the factorized affect latent on large behavioral image/video datasets.
3. **Paired affective neural adaptation:** train low-rank fMRI and EEG heads/adapters using Abdel-Ghaffar, Horikawa, Emo-FilM, BOLD5000-affect, ds006866, FACED, and EmoEEG-MC.
4. **Subject adaptation:** fit a group response plus small subject residual; subject dropout tests unseen-person prediction.
5. **Recurrent extension:** add top-down dynamics only after the static adapter baseline is stable.
6. **Locked external validation:** REELMO/StudyForrest or another complete fMRI set; ds006861 or another complete EEG set.

### 5.5 Required baselines and ablations

Every model comparison must include:

- low-level/Gabor/GIST only;
- object/scene/action/face features only;
- source backbone with a direct affect probe;
- original neural encoder;
- original encoder + equal-parameter non-affective adapter;
- original encoder + affect adapter;
- feedforward affect adapter versus matched recurrent adapter;
- shuffled affect labels;
- no subject adapter versus subject adapter;
- no temporal context versus 4/10/30 s context;
- visual-only, audio-only, text-only, and multimodal video;
- affect latent without neural loss, neural loss without affect heads, and joint model.

If the affect adapter improves label decoding but not measured-neural prediction, it is only a better stimulus classifier. If it improves neural prediction only in V1/LOC and loses after nuisance residualization, it is still likely visual confounding. The compelling pattern is an improvement in late EEG and higher-order/transmodal/limbic fMRI while early visual fidelity remains intact.

## 6. Pre-registered hypotheses

### 6.1 Existing-model benchmark

**H1 — Face versus scene.** Unadapted models will decode facial-expression contrasts better than object-balanced scene affect. Much of the face advantage will be explained by landmarks/FACS/identity/pose and will peak early.

**H2 — Early/late dissociation.** BERG EEG will predict P1/N1 and object-related activity better than EPN/LPP. The late deficit will remain after equalizing signal-to-noise and trial count.

**H3 — Backbone versus brain projection.** Some affect will be decodable from ViT/V-JEPA/CLIP features. A generic predicted-brain representation will not consistently add affective information over its own backbone across datasets.

**H4 — Object/schema regime.** Object features will be competitive or superior for everyday BOLD5000 images, whereas explicit affect-schema features will add more for deliberately evocative IAPS/Cowen–Keltner stimuli.

**H5 — TRIBE modality/context.** TRIBE v2 will benefit from temporal context and audio/language for movie emotion. Still-image repetition into silent video will be less reliable and more sensitive to aggregation choice.

### 6.2 Affect-adapter hypotheses

**H6 — Paired emotional neural training.** Affect-rich paired neural adaptation will improve unseen-person/unseen-stimulus neural encoding more than an equal amount of non-affective or stimulus-label-only adaptation.

**H7 — Factorized supervision.** Joint category + valence/arousal + appraisal supervision will generalize better than any single target family.

**H8 — Arousal first.** Arousal/motivational relevance will transfer across datasets more reliably than signed valence. Participant-specific ratings and subject embeddings will especially help valence.

**H9 — LPP specificity.** Arousal/relevance features will add unique prediction of centroparietal 300–1500 ms activity and posterior 600–1000 ms alpha/low-beta suppression after visual/semantic controls. Ordered positive-versus-negative valence will contribute less to scalar LPP amplitude.

**H10 — Regulation.** Given identical negative-image content, watch versus reappraise/suppress instructions will alter late predictions while leaving early visual predictions relatively stable. This is the cleanest test of internal-state sensitivity.

**H11 — Temporal hierarchy.** Low-level features will peak at 80–180 ms; object/face/semantic features at 180–350 ms; relevance/arousal and appraisal at 300–1500+ ms.

**H12 — Recurrent gain.** A matched recurrent model will selectively improve late EEG, context-dependent fMRI, and regulation effects. Feedback ablation will reduce these improvements while sparing early visual encoding.

### 6.3 fMRI regional predictions

- early visual cortex: low-level and recurrent gain, but affect should survive low-level controls;
- LOC/ventral temporal cortex: object/schema and semantic × affect interactions;
- pSTS: facial expression, action, and social context;
- amygdala/insula/OFC/ACC: appraisal, salience, bodily relevance, and subject state;
- vmPFC: integrated valence/arousal trajectories;
- hippocampus: fine-grained category/context/conjunctive structure;
- transmodal/default-network regions: high-dimensional categories, context, memory, narrative, and appraisal.

Treat these as predeclared hypotheses with appropriate cortical and volumetric analyses. The current local average-surface TRIBE output cannot establish an amygdala or hippocampal result unless the exact checkpoint/API has verified subcortical coverage.

## 7. EEG analysis specification

### 7.1 Primary windows and topographies

Pre-register independently localized or literature-defined windows; do not choose them on the test contrast.

| Target | Suggested window | Topography | Interpretation |
|---|---:|---|---|
| P1/N1 | 80–180 ms | occipital/posterior | low-level sensory fidelity |
| EPN | 180–300 ms | posterior | facilitated selection of emotional content |
| Early LPP | 300–600 ms | centroparietal, Pz/CPz/POz neighborhood | motivated attention and evaluation |
| Middle/late LPP | 600–1500+ ms | centroparietal/sustained | sustained relevance, regulation, memory |
| Posterior alpha/low-beta ERD | about 600–1000 ms | posterior | sustained cortical engagement |

The current BERG output permits only an early-LPP test through about 600 ms. A new model and real EEG pipeline should extend beyond 1 s.

### 7.2 Preprocessing safeguards

- preserve DC or use a low high-pass cutoff around 0.01–0.1 Hz for sustained potentials;
- report sensitivity to baseline correction and model baseline as a covariate;
- avoid high-pass settings around 0.3 Hz or above that can distort sustained ERPs ([Tanner et al., 2015](https://doi.org/10.1111/psyp.12437));
- preregister reference and repeat the key analysis with one alternative;
- retain EOG/EMG/ECG channels and test artifact-only decoding;
- use sensor-space results as primary; source/connectivity analyses are secondary;
- keep all repetitions/windows from one trial/stimulus in one fold;
- use stimulus- and participant-level inference, not synthetic subject counts.

### 7.3 Real-versus-generated validation

For each condition, compare:

- grand-average waveform and scalp topography;
- subject-level waveform correlation and noise-normalized R²;
- emotional-neutral EPN/LPP amplitude difference;
- time-resolved representational geometry;
- temporal generalization matrix;
- posterior alpha/low-beta effects;
- cross-subject and cross-stimulus decoding trained only on real EEG and tested on real versus predicted EEG.

The critical decoder is trained on real neural data. A decoder trained and tested on synthetic EEG only is not evidence of neural affect.

## 8. fMRI analysis specification

### 8.1 Primary endpoint

Use cross-dataset change in noise-ceiling-normalized measured-fMRI prediction uniquely attributable to affect/appraisal features after variance partitioning visual, semantic, face, audio, and language features.

### 8.2 Secondary endpoints

- category/VAD/appraisal decoding by models trained only on real fMRI;
- subject-specific versus group-average ratings;
- object-driven versus emotion-schema stimulus regimes;
- pSTS face/context dissociation;
- cortical versus volumetric subcortical results;
- time-context and modality ablations;
- real-versus-predicted RDM alignment;
- overlap with independent signatures such as PINES, treated as a specificity benchmark rather than ground truth ([Chang et al., 2015](https://doi.org/10.1371/journal.pbio.1002180)).

### 8.3 Temporal safeguards for movies

- split by complete source movie or long temporal blocks;
- model HRF, annotation reaction lag, autocorrelation, and affective carryover;
- use circular-shift or phase-randomized nulls;
- do not let adjacent time points enter different folds;
- compare framewise with 4/10/30-second context;
- regress motion, heart rate, and respiration, but also analyze physiology as a possible mediator.

## 9. Paper strategy

### Paper 1 — the high-probability paper

**Working title:** *Facial expression is not scene affect: an affective-validity benchmark for stimulus-to-brain encoding models*

**Question:** What do general-purpose visual/neural encoders preserve about affect, and what do they miss?

**Core comparisons:**

- BERG fWRF;
- BERG ViT fMRI;
- BERG THINGS-EEG2;
- TRIBE v2;
- source backbones;
- low-level/object/face/semantic affect baselines.

**Primary figures:**

1. Model architectures and the backbone → predicted brain → real brain comparison.
2. Confound-controlled stimulus battery and grouped split design.
3. Face versus scene decoding with time courses and identity/content controls.
4. Real versus predicted EEG: early visual success and EPN/LPP deficit.
5. fMRI variance partitions: visual/object/semantic versus unique affect across ROIs.
6. Cross-dataset generalization and failure map.

**Publishable outcomes:**

- Existing encoders capture early/visual expression cues but not sustained affect.
- Some generic semantic encoders carry affect, but the predicted-neural projection does not add value.
- TRIBE or a BERG model does show unique affect beyond its backbone, if validated against real brain data.
- A well-powered, leakage-resistant null is itself useful if it establishes affective construct validity as a missing benchmark.

Do not headline “reentry is necessary” in Paper 1. The benchmark can motivate that hypothesis.

### Paper 2 — the model paper

**Working title:** *Affect-BERG: factorized emotion-aware neural encoding reproduces sustained motivated attention across EEG and fMRI*

**Question:** Does theory-guided affective neural adaptation improve genuine neural prediction and external generalization?

**Primary contrast:** each frozen original encoder versus the same encoder with an equal-parameter affect adapter, followed by a matched feedforward-versus-recurrent comparison.

**Primary endpoints:**

- measured neural prediction on unseen people and unseen stimuli;
- incremental affective variance after nuisance controls;
- EPN/LPP waveform and topography;
- external dataset generalization;
- preservation of early visual fidelity.

**Mechanistic claim level:**

- If only label decoding improves: stimulus-affect classifier, not neural encoding.
- If measured late EEG and high-level fMRI improve: affect-aware neural encoding.
- If matched recurrence selectively improves late/context/regulation effects: computational support for recurrent gain.
- If neural perturbation or stronger causal evidence is added: cautiously discuss biological reentry.

### A possible single-paper version

One paper can combine benchmark and adapter if the adapter produces a clean external effect. The story becomes:

> Broad encoders reproduce sensory localizers but omit controlled affective variance; a small factorized affect adapter restores late motivated-attention and higher-order affective representations without degrading early sensory prediction.

This is stronger and more manageable than training a completely new foundation model from scratch.

## 10. Decision gates

### Gate 0 — pipeline validity

Proceed only after:

- time axes, sampling rate, baseline correction, and grayscale/color choices are fixed;
- every split is grouped by exact stimulus/source/identity;
- test-set model ranking is removed;
- inference uses real subjects and/or stimulus units appropriately;
- raw, corrected, and alternative-preprocessing sensitivity results agree qualitatively.

### Gate 1 — real affective signal

Require replication in measured data:

- emotional > neutral EPN/LPP or arousal-linked sustained EEG;
- semantic + affect improvement in measured fMRI;
- reliability/noise ceiling sufficient for model comparison.

If the real signal does not replicate, do not interpret a synthetic signal.

### Gate 2 — existing-model deficit

Require a controlled dissociation such as:

- good early visual/object/face neural prediction;
- poor late/regulation/context affect prediction;
- deficit persists after matched SNR/trial count and fair readout.

This converts the current null into a specific model limitation.

### Gate 3 — adapter value

The affect adapter must:

- improve measured neural prediction, not only label decoding;
- beat the same parameter budget with non-affective data/labels;
- survive nuisance residualization;
- generalize to unseen stimuli and unseen participants;
- preserve early visual performance.

### Gate 4 — theory claim

Call the result “recurrent-gain consistent” only if the matched recurrent advantage is late/contextual and feedback ablation is selective. Otherwise use the broader “factorized affective encoding” claim.

## 11. Twelve-month execution plan

### Weeks 1–2: repair and freeze the benchmark

- Correct the personal EEG notebooks' time, baseline, smoothing, and split issues.
- Build a single stimulus manifest with dataset, source, identity, semantic category, affect labels, hashes, and nuisance features.
- Freeze a benchmark version of BERG, BERG-main, and TRIBE v2.
- Pre-register primary ROIs, EEG windows, split rules, metrics, and the locked test policy.
- Reproduce measured IAPS EEG ERP/RSA before evaluating generated EEG.

### Weeks 3–6: run the existing-model benchmark

- Extract source-backbone and predicted-neural features for identical stimuli.
- Run grouped nested CV, grouped permutations, and subject/stimulus bootstrap.
- Produce face/content, object-balanced, cross-category, and cross-dataset results.
- For TRIBE, compare still-image aggregation methods and run true video context/modality ablations.

### Months 2–4: measured-neural reference models

- Ingest Abdel-Ghaffar fMRI and one high-dimensional video fMRI set.
- Ingest ds006866 and Schindler EEG; keep one EEG dataset locked.
- Fit low-level, semantic, object, face, EmoNet, category, VAD, and appraisal encoding/RSA baselines.
- Estimate reliability and noise ceilings.

### Months 4–6: minimum Affect-BERG adapter

- Add the factorized stimulus affect latent and low-rank neural adapters.
- Train separate fMRI and EEG heads with a shared stimulus latent.
- Run label-only, neural-only, joint, equal-parameter, and shuffled-label ablations.
- Select only on validation.

### Months 6–9: recurrence and subject state

- Extend EEG output to at least 1.5–2 s.
- Add recurrent/top-down gain and matched feedforward control.
- Add subject/session adapters and regulation data.
- Test temporal-context lengths and modality ablations.

### Months 9–12: locked external test and paper

- Evaluate once on the untouched fMRI and EEG datasets.
- Complete dual subject/stimulus bootstrap, robustness, and preprocessing sensitivity.
- Release manifests, splits, model cards, and code that do not redistribute restricted stimuli.
- Write Paper 1; include Paper 2 if the adapter passes Gate 3.

## 12. Immediate experiment queue

In exact order:

1. **Repair the BERG EEG analysis** and regenerate face, IAPS-balanced, NAPS-balanced, and LPP plots with correct time/base/smoothing and stimulus-level inference.
2. **Reproduce the real LPP** in the user's IAPS EEG and one public dataset with predeclared Pz/CPz/POz and EPN/LPP windows.
3. **Build the backbone–brain–real triad** for the same images: ViT features, BERG-predicted EEG, real EEG.
4. **Run grouped cross-category decoding** and variance partitioning against luminance/color/GIST, DINO/CLIP, objects, faces, and EmoNet.
5. **Audit the 2026 NSD emotion decoding** for repeated-image and semantic leakage.
6. **Reanalyze local TRIBE results at image/video level** with nuisance covariates, validation-only selection, video-level bootstrap/permutation, and source-backbone comparison.
7. **Benchmark on measured Abdel-Ghaffar fMRI**, preserving its independent test set.
8. **Only then train the minimal affect adapter.**

## 13. Interpretation guide

| Observed result | Defensible conclusion |
|---|---|
| Face decoding only, early peak, landmark baseline explains it | Expression geometry is preserved; no general affect claim |
| Object-balanced scene affect fails in backbone and neural output | Stimulus representation/training objective is insufficient |
| Affect exists in backbone but disappears in predicted brain | Neural mapping suppresses affective variance |
| Predicted brain decodes labels but fails real-neural prediction | Semantic copying; not neural affect validity |
| Affect adapter improves labels only | Better affective computer-vision model |
| Affect adapter improves measured late EEG and higher-order fMRI after controls | Emotion-aware neural encoding |
| Recurrent model selectively improves late/context/regulation effects | Evidence consistent with recurrent motivated-attention gain |
| Only arousal/LPP improves | Model learned salience/motivational relevance, not differentiated emotion |
| Category + VAD + appraisal beats all single heads externally | Hybrid/factorized representation is supported |

## 14. Key risks

- **Leakage:** random windows/repetitions from the same stimulus can produce impressive but invalid accuracy. This is especially severe in long-video EEG datasets.
- **Stimulus emotion versus felt emotion:** intended class, normative rating, portrayed emotion, and participant report are different targets.
- **Restricted stimuli:** IAPS and some movie/image sources cannot be redistributed.
- **Average-subject models:** group predictions can suppress person-specific affect.
- **Synthetic inference:** significance over deterministic predictions is model characterization, not a population neuroscience statistic.
- **Low unique-stimulus count:** FACED-like datasets have many participants but few videos; population N does not replace stimulus generalization.
- **Physiological/artifact confounds:** emotion changes gaze, blink, facial EMG, respiration, heart rate, and motion.
- **Valence instability:** signed valence is less cross-context stable than arousal and may need individual calibration.
- **Temporal mismatch:** fMRI models cannot establish millisecond LPP dynamics; EEG models trained only to 600 ms cannot model a sustained late potential.
- **Overclaiming anatomy:** cortical surface predictions cannot support subcortical claims without verified volumetric outputs.

## 15. Selected evidence library

### Core neural theory and phenomena

- [Cuthbert et al. (2000): emotion, arousal, and sustained positive potential](https://doi.org/10.1016/S0301-0511(99)00044-7)
- [Schupp et al. (2003): EPN and facilitated emotional selection](https://doi.org/10.1111/1467-9280.01411)
- [Sabatinelli et al. (2007): LPP and visual-cortical BOLD](https://doi.org/10.1093/cercor/bhl017)
- [Liu et al. (2012): simultaneous EEG-fMRI substrate of LPP](https://doi.org/10.1523/JNEUROSCI.3109-12.2012)
- [Keil et al. (2003): emotional ssVEP gain](https://doi.org/10.3758/CABN.3.3.195)
- [Bo et al. (2022): time-resolved IAPS EEG/fMRI decoding](https://doi.org/10.1016/j.neuroimage.2022.119532)
- [Freese & Amaral (2006): amygdala projections to TE and V1](https://pmc.ncbi.nlm.nih.gov/articles/PMC2564872/)

### Visual, categorical, and appraisal representations

- [Kragel et al. (2019): EmoNet and visual emotion schemas](https://doi.org/10.1126/sciadv.aaw4358)
- [Horikawa et al. (2020): high-dimensional distributed emotion](https://doi.org/10.1016/j.isci.2020.101060)
- [Abdel-Ghaffar et al. (2024): semantic and affective natural-image tuning](https://doi.org/10.1038/s41467-024-49073-8)
- [Gao et al. (2025): object representations drive everyday emotion schemas](https://doi.org/10.1038/s42003-025-08145-1)
- [Mohammadi et al. (2023): component-process/appraisal fMRI](https://doi.org/10.1093/cercor/bhad093)
- [Ma & Kragel (2026): hippocampal–prefrontal emotion maps](https://doi.org/10.1038/s41467-025-68240-z)
- [Cowen & Keltner (2017): graded high-dimensional emotion reports](https://doi.org/10.1073/pnas.1702247114)

### Current foundation-model context

- [BERG code](https://github.com/gifale95/BERG) and [model documentation](https://brain-encoding-response-generator.readthedocs.io/en/stable/models/overview.html)
- [TRIBE v2 paper](https://arxiv.org/abs/2605.04326) and [code](https://github.com/facebookresearch/tribev2)
- [EmoMind preprint (2026)](https://arxiv.org/abs/2605.16739) — directly relevant, but not peer reviewed
- [Emo-FilM dataset](https://doi.org/10.1038/s41597-025-04803-5)

## Final decision

The project should not be framed as “find a model that decodes emotion.” It should be framed as:

> Determine which components of affective neural processing are preserved by general-purpose stimulus-to-brain encoders, then add the smallest theory-guided and data-supported mechanism necessary to recover the missing measured neural variance.

That framing turns the current negative results into a strong scientific starting point, makes the face result interpretable, provides a fair test of recurrence, and creates a credible route from BERG/TRIBE benchmarking to an emotion-capable encoding model and paper.
