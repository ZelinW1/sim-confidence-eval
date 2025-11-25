# SimConfidenceEval - Sim-Real Confidence Evaluation Platform

A professional evaluation framework for measuring the similarity and confidence between simulated and real image data.

## Project Overview

This platform integrates **multiple image similarity evaluation metrics** (classical metrics and deep learning-based metrics) to comprehensively assess paired simulated and real images. It supports flexible configuration systems, batch processing, GPU acceleration, and result visualization.

## Key Features

- 🎯 **Multiple Evaluation Metrics**: Including PSNR, SSIM, FSIM, LPIPS, FID, etc.
- ⚡ **GPU Acceleration**: Full GPU support for batch processing acceleration
- 📊 **Flexible Configuration**: YAML-based configuration system with command-line override support
- 📈 **Result Visualization**: Generates score distribution histograms, worst-case comparisons, etc.
- 🔧 **Modular Design**: Easy to extend with new metrics and data formats
- 📝 **Comprehensive Logging**: Detailed run logs and configuration backups

## Project Structure

```
SimConfidenceEval/
├── main.py                          # Main entry point script
├── requirements.txt                 # Python dependencies
├── configs/                         # Configuration files directory
│   ├── basic_eval.yaml             # Basic evaluation configuration
│   └── dataset_eval.yaml           # Dataset evaluation configuration
├── data/                            # Data files directory
│   ├── sim/                        # Simulated data
│   │   └── warship/               # Example category: warship
│   └── real/                       # Real data
│       └── warship/
├── src/                             # Source code
│   ├── core/                       # Core module
│   │   └── evaluator.py           # Main evaluator class
│   ├── data/                       # Data processing module
│   │   ├── datasets.py            # Dataset definition and loading
│   │   ├── transforms.py          # Data augmentation and transforms
│   │   └── utils.py               # Data processing utilities
│   ├── metrics/                    # Evaluation metrics module
│   │   ├── base.py                # Base class definition
│   │   ├── classical.py           # Classical metrics (PSNR/SSIM/FSIM)
│   │   └── deep.py                # Deep learning metrics (LPIPS/FID)
│   └── utils/                      # Utility module
│       ├── logger.py              # Logging utilities
│       ├── io.py                  # File I/O utilities
│       └── visualizer.py          # Visualization utilities
├── test/                            # Test code
│   └── test_metrics.py            # Metrics unit tests
└── outputs/                         # Output results directory
    └── run_001/                    # Example run results
```

## Supported Evaluation Metrics

| Metric | Type | Description |
|--------|------|-------------|
| **PSNR** | Classical | Peak Signal-to-Noise Ratio, higher is better |
| **SSIM** | Classical | Structural Similarity Index, considers luminance, contrast, structure |
| **FSIM** | Classical | Feature Similarity Index, based on phase consistency |
| **LPIPS** | Deep Learning | Learned Perceptual Image Patch Similarity, using pretrained neural networks |
| **FID** | Deep Learning | Fréchet Inception Distance, evaluates overall distribution similarity |

## Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Or using conda
conda env create -f environment.yml
```

### 2. Prepare Data

Organize simulated and real data in the following structure:

```
data/
├── sim/
│   └── <category>/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── real/
    └── <category>/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

### 3. Configure Evaluation Parameters

Edit `configs/basic_eval.yaml`:

```yaml
experiment_name: "my_experiment"
output_dir: "./outputs/run_001"

data:
  mode: "paired"                    # Paired mode
  sim_path: "./data/sim/"
  real_path: "./data/real/"
  iou_threshold: 0.8               # IoU filtering threshold
  batch_size: 32
  image_size: [640, 480]           # [H, W]

metrics:
  - name: "PSNR"
    params:
      psnr_max: 50.0
  - name: "SSIM"
  - name: "LPIPS"
    params:
      net: "vgg"
      lpips_max: 1.0
  - name: "FID"
    params:
      feature: 2048

visualization:
  plot_hist: True                  # Generate histograms
  save_bad_cases: True             # Save worst-case comparisons
```

### 4. Run Evaluation

```bash
# Use default configuration
python main.py

# Specify configuration file
python main.py --config configs/dataset_eval.yaml

# Override output directory
python main.py --config configs/basic_eval.yaml --output ./outputs/custom_run
```

## Output Results

After evaluation completes, the output directory contains:

```
outputs/run_001/
├── config_backup.yaml                    # Configuration file backup
├── final_report_summary.csv              # Summary report (mean, median, etc. for each metric)
├── final_report_detailed.csv             # Detailed report (scores for each image pair)
├── plots/
│   ├── score_distribution_PSNR.png      # Score distribution histograms
│   ├── score_distribution_SSIM.png
│   ├── ...
│   ├── worst_10_case_comparison.png     # Worst-case comparisons
│   └── ...
└── logs/
    └── eval_*.log                       # Run logs
```

## Configuration Details

### Data Configuration

- `mode`: Data loading mode
  - `paired`: Paired mode, requires identical directory structures for sim and real data
- `iou_threshold`: Box filtering threshold for detection quality control
- `batch_size`: Batch size for processing, adjust based on GPU memory
- `image_size`: Unified image size `[height, width]`

### Metrics Configuration

Each metric can include `name` and optional `params`:

- **PSNR** Parameters:
  - `psnr_max`: Upper limit for normalization (default: 50)

- **LPIPS** Parameters:
  - `net`: Feature network `alex` / `vgg` / `squeeze`
  - `version`: Model version
  - `lpips_max`: Normalization upper limit (default: 1.0)

- **FID** Parameters:
  - `feature`: Feature dimension `64` / `192` / `768` / `2048`

### Visualization Configuration

- `plot_hist`: Whether to generate score distribution histograms
- `save_bad_cases`: Whether to save worst-case image pairs

## Extension Guide

### Adding New Evaluation Metrics

1. **Create a metric class in `src/metrics/`**

```python
from src.metrics.base import BaseMetric
import torch

class MyMetric(BaseMetric):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        # Initialize your metric
        
    def forward(self, preds, target):
        # Implement computation logic
        score = ...  # Compute similarity
        return self._normalize(score)
    
    def _normalize(self, x):
        # Normalize to [0, 1]
        return x / max_value
```

2. **Register in `src/metrics/__init__.py`**

```python
from .classical import PSNR, SSIM, FSIM
from .deep import LPIPS, FID
from .custom import MyMetric  # Import new metric
```

3. **Use in configuration file**

```yaml
metrics:
  - name: "MyMetric"
    params:
      param1: value1
```

### Supporting New Data Formats

Modify the `build_dataloader` function in `src/data/datasets.py` to support other data formats (e.g., detection box annotations, point clouds, etc.).

## Performance Recommendations

- Use GPU: Ensure CUDA is available for 10-100x speedup
- Batch size adjustment: Adjust `batch_size` based on GPU memory (recommended: 32-128)
- Image size: Smaller sizes for faster processing, larger sizes for better accuracy (recommended: 640x480)
- Metric selection: LPIPS/FID are slower; prioritize classical metrics for quick evaluation

## Dependency Notes

Key dependencies include:

- PyTorch: Deep learning framework
- PyYAML: Configuration file parsing
- scikit-image: Image processing algorithms
- Pillow: Image I/O
- matplotlib: Visualization
- pandas: Data processing

See `requirements.txt` for details.

## FAQ

**Q: How to handle images with inconsistent sizes?**  
A: The framework automatically resizes all images to the `image_size` specified in the configuration.

**Q: What image formats are supported?**  
A: All formats supported by PIL are supported (JPEG, PNG, BMP, etc.).

**Q: How to run on CPU?**  
A: The framework automatically detects CUDA availability and falls back to CPU if GPU is not available.

**Q: How to skip certain metrics?**  
A: Simply delete or comment out unwanted metrics in the configuration file.

## License

MIT

## Contact

For issues or suggestions, please submit an Issue or Pull Request.
