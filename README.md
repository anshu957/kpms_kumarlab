# KeyPoint-MoSeq Behavioral Analysis

Unsupervised behavioral analysis using KeyPoint-MoSeq AR-HMM models on JABS pose estimation data from the Kumar Lab combined Open Field Arena (OFA) dataset.

---

## Pipeline Overview

The pipeline runs in 5 sequential stages:

```
JABS H5 Files  →  CSV Poses  →  Formatted Data  →  PCA  →  AR-HMM  →  Syllables + Visualizations
   (raw)         Stage 1         Stage 2          Stage 3  Stage 4       Stage 5
```

### Stage 1 — Preprocessing: H5 → CSV
**Script:** `scripts/preprocess_poses.py` | **Library:** `src/preprocessing.py`

Reads JABS pose H5 files and converts them to a flat CSV format readable by KPMS. Handles both JABS v2 and v6 formats:

- **v6:** `points` shape `(n_frames, 1, n_keypoints, 2)`, `confidence` shape `(n_frames, 1, n_keypoints)`
- **v2:** `points` shape `(n_frames, n_keypoints, 2)`, `confidence` shape `(n_frames, n_keypoints)`

JABS stores coordinates as `(row, col)` = `(image_y, image_x)`. The converter swaps these into `(x, y)` order and writes an interleaved CSV: `x1, y1, conf1, x2, y2, conf2, …` per frame. Parallel conversion and keypoint subsets (10k, removing noisy tail tip keypoints) are supported.

### Stage 2 — Data Loading & Egocentric Formatting
**Library:** `src/utils.py::load_keypoints_pd` → `src/methods.py::load_and_format_data`

Reads CSVs and reshapes each row into `(n_frames, n_keypoints, 3)`, splitting into `coords` and `confidences`. Then `kpms.format_data()` applies **egocentric normalization**: each frame is centered on the midpoint between `BASE_NECK_INDEX` (anterior) and `BASE_TAIL_INDEX` (posterior), and rotated so the head-to-tail axis points in a canonical direction. This removes absolute position and arena orientation from the representation.

### Stage 3 — PCA Dimensionality Reduction
**Library:** `src/methods.py::perform_pca`

Fits PCA on all egocentric poses across all recordings. Logs the number of components needed to explain ≥90% of variance. Generates and saves a scree plot and per-PC trajectory plots. The PCA object is saved to `pca.p` for model initialization.

### Stage 4 — AR-HMM Model Fitting (Two-Phase)
**Library:** `src/methods.py::fit_and_save_model`

Fits a switching autoregressive Hidden Markov Model over the PCA-compressed pose trajectories. Uses a deliberate two-phase strategy:

**Phase A — AR-only** (`ar_only=True`, `num_iters=arhmm_iters`):
Uses the target `kappa` stickiness. Learns discrete syllable assignments while holding the continuous latent space fixed at the PCA initialization. Establishes stable state boundaries before allowing the continuous space to move.

**Phase B — Full joint model** (`ar_only=False`, `kappa → 0.1 × kappa`):
Reduces kappa by 10× to encourage finer-grained segmentation, then jointly optimizes both the continuous latent trajectories and discrete syllable assignments via Gibbs sampling.

After fitting, syllables are reindexed by usage frequency (syllable 0 = most common) and results saved as `results.h5` and per-recording CSVs.

### Stage 5 — Visualization
**Library:** `src/methods.py::generate_plots_and_movies`

- **Trajectory plots:** Mean egocentric pose trajectory ± window around each syllable's occurrence
- **Grid movies:** Representative video clips for each discovered syllable
- **Similarity dendrogram:** Hierarchical clustering of syllables by cosine distance of their pose trajectory fingerprints

**Downstream analysis** (similarity matrices, heatmaps) is in `notebooks/donwstream.ipynb`.

---

## Key Configuration Parameters

| Parameter | Default (notebook) | Production (OFA) | Effect |
|---|---|---|---|
| `kappa` | `0.1` | `1e6` | **Stickiness** — higher = longer behavioral bouts |
| `arhmm_iters` | `10` | `50` | Phase A Gibbs sampling iterations |
| `full_model_iters` | `10` | `300–400` | Phase B joint model iterations (needs ~300+ to converge) |
| `mixed_map_iters` | `8` | `4–8` | GPU memory management (higher = less memory, slower) |
| `pose_version` | `v6` | `v6` | JABS pose format version |

### Bodypart Skeleton (12-keypoint)

```
NOSE_INDEX, LEFT_EAR_INDEX, RIGHT_EAR_INDEX, BASE_NECK_INDEX,
LEFT_FRONT_PAW_INDEX, RIGHT_FRONT_PAW_INDEX, CENTER_SPINE_INDEX,
LEFT_REAR_PAW_INDEX, RIGHT_REAR_PAW_INDEX, BASE_TAIL_INDEX,
MID_TAIL_INDEX, TIP_TAIL_INDEX
```

- **Egocentric anterior:** `BASE_NECK_INDEX`
- **Egocentric posterior:** `BASE_TAIL_INDEX`
- **10k subset** (`config/config_10k.yml`): Drops `MID_TAIL_INDEX` and `TIP_TAIL_INDEX` to eliminate noisy distal tail tracking

---

## Running the Pipeline

### 0. Environment Setup

```bash
conda create -n kpms python=3.9
conda activate kpms
pip install -r requirements.txt
pip install keypoint-moseq
```

### 1. Data Organization

Store data outside the project and use symlinks:

```bash
cd data/
ln -s /path/to/your/combined_OFA combined_OFA
```

### 2. Preprocessing (one-time, per dataset)

```bash
python scripts/preprocess_poses.py \
    --input data/combined_OFA/raw_h5/ \
    --output data/combined_OFA/poses_csv/ \
    --pose-version v6 \
    --validate \
    --n-jobs 8
```

To also generate the recommended 10-keypoint subset:

```bash
python scripts/preprocess_poses.py \
    --input data/combined_OFA/raw_h5/ \
    --output data/combined_OFA/poses_csv/ \
    --subsets 10k
```

### 3a. Training via HPC (recommended for full OFA dataset)

Edit paths and parameters in `scripts/submit_job.sh`, then:

```bash
sbatch scripts/submit_job.sh
```

SLURM config: partition `gpu_a100`, 1× A100 GPU, 124 GB RAM, 11-hour wall time.

### 3b. Training via CLI

```bash
python scripts/train_kpms.py \
    --pose-dir data/combined_OFA/poses_csv_10k/ \
    --project-path results/combined_OFA_kappa_1e6 \
    --video-dir data/combined_OFA/videos/ \
    --kappa 1e6 \
    --arhmm-iters 50 \
    --full-model-iters 300 \
    --config config/config_10k.yml

# Train without videos: skips the video-based motif grid movies but still
# generates pose-based trajectory plots and the similarity dendrogram
python scripts/train_kpms.py \
    --pose-dir data/combined_OFA/poses_csv_10k/ \
    --project-path results/combined_OFA_kappa_1e6 \
    --kappa 1e6 \
    --arhmm-iters 50 \
    --full-model-iters 300 \
    --config config/config_10k.yml \
    --skip-videos
```

### 3c. Interactive (notebook)

Open `notebooks/main.ipynb`. **JAX environment variables in the first two cells must run before any other cell.**

### 4. Downstream Analysis

Open `notebooks/donwstream.ipynb` — computes syllable similarity matrices and heatmaps from a saved checkpoint.

### 5. Tests

```bash
python -m pytest tests/
```

---

## Output Structure

```
results/<project_name>/
├── config.yml               # KPMS internal config (bodyparts, skeleton, orientation)
├── pca.p                    # Fitted PCA object
├── training_params.yml      # Hyperparameters (for reproducibility)
└── <YYYY_MM_DD-HH_MM_SS>/   # Timestamped model checkpoint directory
    ├── checkpoint.h5
    ├── results.h5            # Full results (syllables per frame)
    ├── results/              # Per-recording syllable CSVs
    ├── trajectory_plots/     # Per-syllable mean pose trajectories
    └── grid_movies/          # Per-syllable representative video clips
```

## Project Structure

```
kpms_kumarlab/
├── config/              # default.yml (12kp), config_10k.yml (10kp subset)
├── data/                # Symlinks to external datasets
├── examples/            # Small JABS600 test data (2 recordings)
├── scripts/
│   ├── preprocess_poses.py   # H5 → CSV conversion
│   ├── train_kpms.py         # CLI training entrypoint
│   └── submit_job.sh         # SLURM HPC submission
├── src/
│   ├── preprocessing.py      # H5 parsing, coordinate conversion, CSV writing
│   ├── utils.py              # CSV loading, config, logging, GPU monitoring
│   └── methods.py            # PCA, AR-HMM fitting, visualization wrappers
├── notebooks/
│   ├── main.ipynb            # End-to-end interactive pipeline
│   └── donwstream.ipynb      # Post-hoc syllable analysis
├── results/                  # Training outputs
└── tests/test_essential.py   # Unit tests (mock data, no external dependencies)
```

## Citation

This work is a custom pipeline (to work with JABS data) on the KeyPoint-MoSeq framework, which is described in the following publication and its associated repository:

Weinreb C, et al. (2023). "Keypoint-MoSeq: parsing behavior by linking point tracking to pose dynamics." *Nature Methods*.

[KeyPoint-MoSeq Repository](https://github.com/dattalab/keypoint-moseq)
