# KeyPoint-MoSeq JAX (KPMS-JAX)

A research toolkit for analyzing animal behavior using KeyPoint-MoSeq (KPMS) with JABS pose estimation format.

## Overview

This repository contains a simplified pipeline for discovering behavioral syllables from pose estimation data using the KeyPoint-MoSeq method. The approach models animal behavior as sequences of discrete, stereotyped actions (syllables) using an autoregressive hidden Markov model (AR-HMM).

**Key Features:**
- Process JABS format pose data (H5 to CSV conversion)
- Complete KeyPoint-MoSeq analysis pipeline
- Behavioral syllable discovery and visualization
- Research-focused design for ease of use

## Installation

You can set up this project using either Conda (recommended) or Python virtual environments.

### Option 1: Conda Environment (Recommended)

```bash
# Clone the repository
git clone https://github.com/anshu957/kpms_kumarlab.git
cd kpms_kumarlab

# Create and activate conda environment
conda create -n kpms python=3.9
conda activate kpms

# Install dependencies
pip install -r requirements.txt
pip install keypoint-moseq
```

### Option 2: Python Virtual Environment

```bash
# Clone the repository
git clone https://github.com/anshu957/kpms_kumarlab.git
cd kpms_kumarlab

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install keypoint-moseq
```

### Verify Installation

Test your installation:
```bash
python -c "import keypoint_moseq; import jax; print('✅ Installation successful!')"
```

## Quick Start

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
    └── README.md           # Detailed documentation
```

## Data Format

**Expected Input:** JABS pose estimation format
- **H5 files:** Raw JABS output with pose predictions
- **CSV files:** Converted format with keypoints and confidence scores
- **12 keypoints** representing mouse skeleton
- **Format:** Each row contains x1,y1,conf1,x2,y2,conf2,...,x12,y12,conf12

## Usage Example

```python
from src.utils import load_keypoints_pd
from src.methods import load_and_format_data, perform_pca, fit_and_save_model

# Load CSV pose files from directory
coordinates, confidences = load_keypoints_pd("examples/jabs600_v2/poses/")

# Complete pipeline execution
data, metadata, coordinates = load_and_format_data(pose_dir, project_path)
pca = perform_pca(data, config_func, project_path)
model, model_name, results = fit_and_save_model(data, metadata, pca, config_func, project_path)
```

## Requirements

- Python 3.9+
- JAX with GPU support (recommended)
- CUDA support (for GPU acceleration)

## Documentation

For detailed documentation, examples, and troubleshooting, see [`docs/README.md`](docs/README.md).

## Testing

Run basic functionality tests:
```bash
cd tests
python test_essential.py
```

## Contributing

This is a research toolkit. Please refer to the [Kumar Lab](https://github.com/KumarLabJax) for collaboration guidelines.

## License

See [LICENSE](LICENSE) for details. 
