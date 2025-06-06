# Unsupervised Behavior Analysis with KeyPoint-MoSeq

A research toolkit for analyzing animal behavior using KeyPoint-MoSeq (KPMS) with JABS pose estimation format.

## Overview

This repository contains a simplified pipeline for discovering behavioral syllables from pose estimation data using the KeyPoint-MoSeq method. The approach models animal behavior as sequences of discrete, stereotyped actions (syllables) using an autoregressive hidden Markov model (AR-HMM).

**Key Features:**
- Process JABS format pose data (H5 to CSV conversion)
- Complete KeyPoint-MoSeq analysis pipeline
- Behavioral syllable discovery and visualization
- Research-focused design for ease of use

## Quick Start

### Installation

1. **Clone this repository:**
```bash
git clone <repository-url>
cd unsupervised_behavior_jax
```

2. **Create conda environment:**
```bash
conda create -n kpms python=3.9
conda activate kpms
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
pip install keypoint-moseq
```
💡 Tip: If installing on HPC, make sure you first run the interactive session with a GPU and then install keypoint-moseq package. 

### Data Setup

1. **Place your pose data:**
   - Copy your H5 pose files to `data/` directory
   - Or place CSV files directly in `examples/jabs600_v2/poses/`

2. **Convert H5 to CSV (if needed):**
   - Use the conversion functions in `src/preprocessing.py`
   - See `notebooks/main.ipynb` for examples

### Running the Analysis

Open and run `notebooks/main.ipynb` which demonstrates the complete pipeline:

1. **Data Loading and Formatting**
2. **Principal Component Analysis (PCA)**
3. **AR-HMM Model Fitting**
4. **Result Visualization**

## Project Structure

```
kpms_kumarlab/
├── data/                    # Place your raw H5 pose files here
├── examples/jabs600_v2/
│   ├── poses/              # CSV pose files (converted or direct)
│   └── videos/             # Corresponding video files
├── notebooks/
│   └── main.ipynb          # Main analysis notebook
├── src/
│   ├── methods.py          # Core KPMS pipeline functions
│   ├── utils.py            # Data loading and utility functions
│   ├── preprocessing.py    # H5 to CSV conversion functions
│   └── __init__.py         # Package initialization
├── results/                # Analysis outputs and visualizations
├── tests/
│   └── test_essential.py   # Basic functionality tests
└── docs/
    └── README.md           # This documentation
```

## Data Format

**Expected Input:** JABS pose estimation format
- **H5 files:** Raw JABS output with pose predictions
- **CSV files:** Converted format with keypoints and confidence scores
- **12 keypoints** representing mouse skeleton
- **Format:** Each row contains x1,y1,conf1,x2,y2,conf2,...,x12,y12,conf12

## Core Functions

### Data Loading (`src/utils.py`)
```python
from src.utils import load_keypoints_pd

# Load CSV pose files from directory
coordinates, confidences = load_keypoints_pd("examples/jabs600_v2/poses/")
```

### H5 Conversion (`src/preprocessing.py`)
```python
from src.preprocessing import h5_to_csv_poses

# Convert H5 files to CSV format
h5_to_csv_poses("data/poses.h5", "examples/jabs600_v2/poses/")
```

### KPMS Pipeline (`src/methods.py`)
```python
from src.methods import load_and_format_data, perform_pca, fit_and_save_model

# Complete pipeline execution
data, metadata, coordinates = load_and_format_data(pose_dir, project_path)
pca = perform_pca(data, config_func, project_path)
model, model_name, results = fit_and_save_model(data, metadata, pca, config_func, project_path)
```

## Configuration

Key parameters you may want to adjust:

- **`kappa`** (default: 0.1): Stickiness parameter controlling syllable duration
- **`ar_only` iterations** (default: 10): AR-HMM fitting iterations
- **`full_model` iterations** (default: 10): Full model iterations after AR-HMM

## Testing

Run basic functionality tests:
```bash
cd tests
python test_essential.py
```

## Expected Outputs

The pipeline generates:
- **Model checkpoints** in `results/`
- **Behavioral syllables** as CSV files
- **Trajectory plots** showing behavioral sequences
- **Grid movies** visualizing each discovered syllable
- **Similarity dendrograms** showing syllable relationships

## Computational Requirements

- **GPU:** Recommended for faster training (CUDA support via JAX)
- **Memory:** 8+ GB RAM recommended
- **Storage:** Several GB for outputs depending on data size
- **Training time:** 30 minutes to several hours depending on data size and parameters

## Troubleshooting

**Common Issues:**
1. **JAX/CUDA setup:** Ensure proper JAX installation for your CUDA version
2. **Memory errors:** Reduce batch size or data chunk size in configuration
3. **Data format errors:** Verify CSV files have correct number of columns (36 for 12 keypoints)
4. **Import errors:** Ensure all dependencies are installed in the correct conda environment

**Getting Help:**
- Check the Jupyter notebook examples
- Review log files in the results directory
- Validate your data format using the utility functions


