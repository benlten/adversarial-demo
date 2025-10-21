# Adversarial Robustness Toolkit (ART) Demo

A comprehensive Jupyter notebook demonstrating adversarial attacks and defenses using IBM's Adversarial Robustness Toolbox (ART) with native high-resolution images.

![Python Version](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

This project demonstrates:
- **5 Adversarial Attacks**: FGSM, PGD, BIM, DeepFool, Carlini-Wagner L2
- **4 Defense Methods**: FGSM-AT, PGD-AT, Input Transformation (JPEG), Clean baseline
- **Multi-Defense Comparison**: Comprehensive evaluation across all attack-defense combinations
- **Native 224×224 Images**: ImageNette dataset (no upscaling artifacts)
- **GPU Acceleration**: MPS (Apple Silicon) and CUDA support
- **Complete Visualizations**: Attack effectiveness, defense metrics, pixel modifications, heatmaps

## Features

### Attacks Implemented
1. **FGSM** (Fast Gradient Sign Method) - Single-step gradient attack
2. **PGD** (Projected Gradient Descent) - Multi-step iterative attack (Madry-style)
3. **BIM** (Basic Iterative Method) - Iterative gradient attack
4. **DeepFool** - Minimal perturbation attack
5. **Carlini-Wagner L2** - Optimization-based strong attack

### Defense Methods
1. **FGSM Adversarial Training (FGSM-AT)**: Fast training, moderate defense against FGSM attacks
2. **PGD Adversarial Training (PGD-AT)**: Strong generalization against all attacks (Madry et al. 2018)
3. **Input Transformation**: JPEG compression preprocessing defense (no training required)
4. **Clean Baseline**: No defense (for comparison)

**Multi-Defense Framework**: Automatically trains and compares all defense methods against all attacks with comprehensive visualizations (grouped bar charts, heatmaps, improvement metrics)

### Visualizations
- Attack success rates and comparison charts
- Perturbation heatmaps (per-pixel magnitude)
- L2/L∞ norm distributions
- Binary masks showing modified pixels
- Defense effectiveness bar charts
- Training image modifications (5-column layout)

## Installation

### Prerequisites
- Python 3.12
- Conda (recommended) or virtualenv
- ~5GB disk space for ImageNette dataset

### Quick Start

**Method 1: Using requirements.txt (Recommended)**
```bash
# Clone or navigate to repository
cd DEMO_CODE

# Create conda environment
conda create -n art_demo python=3.12 -y
conda activate art_demo

# Install PyTorch (CPU version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install all dependencies from requirements.txt
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook ART_FGSM_CW_Visualizer.ipynb
```

**Method 2: Manual Installation**
```bash
# Create conda environment
conda create -n art_demo python=3.12 -y
conda activate art_demo

# Install dependencies manually
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install adversarial-robustness-toolbox matplotlib tqdm pandas jupyter

# Launch Jupyter
jupyter notebook ART_FGSM_CW_Visualizer.ipynb
```

### GPU Support

**Apple Silicon (MPS):**
```bash
# PyTorch with MPS support is included in the default installation
# Verify MPS availability:
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

**NVIDIA (CUDA):**
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Running the Notebook

1. **Launch Jupyter**:
   ```bash
   conda activate art_demo
   jupyter notebook ART_FGSM_CW_Visualizer.ipynb
   ```

2. **Execute cells sequentially**:
   - Cell 1-2: Setup and device detection
   - Cell 3-4: Data loading (auto-downloads ImageNette on first run)
   - Cell 5: Model preparation
   - Cell 6: Run adversarial attacks
   - Cell 7-9: Visualizations
   - Cell 10: Adversarial training defense
   - Cell 11: Summary

3. **Expected Runtime**:
   - First run: ~45-60 minutes (includes ImageNette download & training both FGSM-AT and PGD-AT models)
   - Subsequent runs: ~15-20 minutes (uses cached model weights)
   - Quick test mode: ~5-10 minutes (set QUICK_TEST=True in Section 10.1 for 2-epoch training)

### Key Features

#### Progress Bars Everywhere
- ImageNette download progress (1.5GB)
- Training progress with loss values
- Attack-by-attack progress tracking
- Defense comparison progress

#### Model Weight Caching
- `model_weights_resnet18_imagenette.pth` - Clean model (saved after first run)
- `model_weights_resnet18_imagenette_robust.pth` - Robust model (saved after adversarial training)

## Project Structure

```
adversarial-demo/
├── ART_FGSM_CW_Visualizer.ipynb    # Main notebook
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore rules (excludes weights/ and .claude/)
├── data/                            # Auto-created
│   └── imagenette2/                 # ImageNette dataset (auto-downloaded)
│       ├── train/                   # 9,469 training images
│       └── val/                     # 3,925 validation images
└── weights/                         # Model weights directory (excluded from git)
    ├── model_weights_resnet18_imagenette.pth           # Clean model
    ├── model_weights_resnet18_imagenette_robust.pth    # FGSM-trained model
    └── model_weights_resnet18_imagenette_robust_PGD.pth # PGD-trained model
```

## Dataset

**ImageNette** - A subset of ImageNet with 10 classes:
- Native 224×224 resolution (no upscaling!)
- ~1.5GB download size
- Classes: tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute

Auto-downloaded on first run from: https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz

## Results

### Multi-Defense Comparison (Example with Full Training)
| Defense Method   | FGSM    | PGD     | BIM     | Average |
|------------------|---------|---------|---------|---------|
| Clean            | 0.281   | 0.031   | 0.031   | 0.115   |
| FGSM-AT          | 0.844   | 0.781   | 0.813   | 0.813   |
| PGD-AT           | 0.875   | 0.844   | 0.844   | 0.854   |
| InputTransform   | 0.406   | 0.063   | 0.031   | 0.167   |

**Key Findings:**
- PGD-AT provides strongest defense across all attacks (85% average)
- FGSM-AT defends well against FGSM but weaker against PGD/BIM (demonstrates need for strong training)
- Input Transformation provides moderate defense without training
- Clean model highly vulnerable to all attacks (<12% average)

### Pixel Modification Statistics
- **Average pixels modified**: ~99.8%
- **Average L2 perturbation**: ~0.52
- **Max L∞ perturbation**: 0.03 (epsilon constraint)

## Visualizations

The notebook generates several key visualizations:

### 1. Attack Comparison Grid
- Clean image vs 5 adversarial attacks side-by-side
- Perturbation heatmaps showing attack patterns

### 2. Defense Effectiveness Charts
- Bar chart: Clean vs Robust model accuracy under attack
- Improvement percentages (green = better defense)

### 3. Pixel Modification Analysis
5-column visualization for adversarial training:
- Column 1: Clean training image
- Column 2: Adversarial image (with % pixels changed)
- Column 3: **Binary mask** (red = changed pixels)
- Column 4: Perturbation heatmap
- Column 5: Amplified difference (10×)

### 4. Norm Distributions
- L2 norm histograms per attack
- L∞ norm histograms per attack

## Technical Details

### Device Support
- **Apple MPS**: Automatically detected and used on Apple Silicon
- **NVIDIA CUDA**: Automatically detected and used when available
- **CPU Fallback**: Works on any system

### Performance Optimizations
- Carlini-Wagner: 50 iterations (vs 200 default) for 4× speedup
- Subset testing: 64 images for main attacks, 32 for defense comparison
- Model weight caching: Skip retraining on subsequent runs
- Adversarial training: 2,000 images × 20 epochs (~5-10 minutes)

### Image Dataset
ImageNette native ~500×375 downscaled to 224×224 

## Requirements

See `requirements.txt` for complete dependency list.

**Core Requirements:**
- Python >= 3.12
- torch >= 2.9.0
- torchvision >= 0.19.0
- adversarial-robustness-toolbox >= 1.20.1
- matplotlib >= 3.9.0
- pandas >= 2.2.0
- tqdm >= 4.66.0
- jupyter >= 1.0.0

**Installation:**
```bash
pip install -r requirements.txt
```

## References

- **IBM ART**: https://github.com/Trusted-AI/adversarial-robustness-toolbox
- **ImageNette**: https://github.com/fastai/imagenette
- **FGSM**: Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (2015)
- **PGD**: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks" (2018)
- **Carlini-Wagner**: Carlini & Wagner, "Towards Evaluating the Robustness of Neural Networks" (2017)
- **DeepFool**: Moosavi-Dezfooli et al., "DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks" (2016)

## Troubleshooting

### MPS Not Available on macOS
```bash
# Check PyTorch version (needs 2.0+)
python -c "import torch; print(torch.__version__)"

# Reinstall if needed
pip install --upgrade torch torchvision
```

### Out of Memory Errors
- Reduce batch size in data loading cells
- Reduce number of test images (N = 64 → 32)
- Use CPU instead of GPU

### Slow Performance
- Enable GPU acceleration (MPS/CUDA)
- Reduce Carlini-Wagner iterations (50 → 20)
- Use cached model weights (avoid retraining)

## License

MIT License - See LICENSE file for details

## Citation

If you use this notebook in your research, please cite:

```bibtex
@misc{art_demo_2025,
  title={Adversarial Robustness Toolkit Demo with ImageNette},
  author={DSC291 Fall 2025},
  year={2025},
  howpublished={\url{https://github.com/benlten/adversarial-demo}}
}
```

## Acknowledgments

- IBM Research for the Adversarial Robustness Toolbox
- fast.ai for the ImageNette dataset
- PyTorch team for excellent deep learning framework
- UCSD DSC291 Fall 2025 course staff
