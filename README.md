# KeyPoint-MoSeq Behavioral Analysis

Unsupervised behavioral analysis using KeyPoint-MoSeq AR-HMM models.

## Quick Start

### 1. Installation

#### Method 1: Install from source (recommended)

```bash
# Clone the repository
git clone https://github.com/anshu957/kpms_kumarlab.git
cd kpms_kumarlab

# Create conda environment
conda create -n kpms python=3.9
conda activate kpms

# Install the package and all dependencies
pip install -e .
```

#### Method 2: Install dependencies only

```bash
conda create -n kpms python=3.9
conda activate kpms
pip install -r requirements.txt
```

**Note:** The package is now properly configured with `setup.py` and `pyproject.toml`, making it easy to install and deploy.

### 2. Data Organization

Store data outside project, use symlinks:

```bash
cd data/
ln -s ~/datasets/experiment_1 exp1
ln -s ~/datasets/experiment_2 exp2
```

### 3. Preprocessing (one-time)

Convert H5 pose files to CSV:

```bash
python scripts/preprocess_poses.py \
    --input data/exp1/raw_h5/ \
    --output data/exp1/poses/
```

### 4. Training

```bash
python scripts/train_kpms.py \
    --pose-dir data/exp1/poses/ \
    --video-dir data/exp1/videos/ \
    --project-path results/exp1_run1
```

Override parameters:

```bash
python scripts/train_kpms.py \
    --pose-dir data/exp1/poses/ \
    --video-dir data/exp1/videos/ \
    --project-path results/exp1_kappa05 \
    --kappa 0.5 \
    --arhmm-iters 20
```

### 5. Results

```
results/exp1_run1/
├── config.yml           # KPMS internal config
├── training_params.yml  # Training hyperparameters (reproducibility)
├── results.csv          # Syllable assignments per frame
├── checkpoints/         # Trained model
├── logs/                # Training logs
└── pca/                 # PCA plots
```

## HPC Usage

Edit `scripts/submit_job.sh` and submit:

```bash
sbatch scripts/submit_job.sh
```

## Interactive Analysis

Open `notebooks/main.ipynb` for step-by-step exploration.

## Configuration

Default parameters: `config/default.yml`

CLI arguments override defaults. Training hyperparameters saved to `results/<experiment>/training_params.yml`.

## Project Structure

```
kpms_kumarlab/
├── config/              # Default parameters
├── data/                # Symlinks to external datasets
├── examples/            # Small test data
├── scripts/             # CLI tools (preprocess, train)
├── src/                 # Core library
└── notebooks/           # Interactive analysis
```

## Citation

This work is a custom pipeline (to work with JABS data) on the KeyPoint-MoSeq framework, which is described in the following publication and its associated repository:

Weinreb C, et al. (2023). "Keypoint-MoSeq: parsing behavior by linking point tracking to pose dynamics." *Nature Methods*.

[KeyPoint-MoSeq Repository](https://github.com/dattalab/keypoint-moseq)