# RO1 Heatmap Pipeline

A complete pipeline for processing whole slide images (WSI) to generate pathology density heatmaps for Tau, Iron, and Amyloid stains.

## Package Contents

```
RO1_Heatmap_Pipeline/
├── README.md                    # This file
├── README_REPROCESSING.md       # Detailed pipeline documentation
├── models/                      # Trained EfficientNet-B4 checkpoints
│   ├── tau_efficientnet_v2.ckpt
│   ├── iron_efficientnet_v2.ckpt
│   └── amyloid_efficientnet_v2.ckpt
├── configs/                     # Configuration files
│   ├── pipeline_config.txt          # Tau/Iron pipeline config
│   ├── pipeline_config_amyloid.txt  # Amyloid pipeline config
│   ├── setup-predict_Tau_efficientnet_v2.json
│   ├── setup-predict_Iron_efficientnet_v2.json
│   └── setup-predict_Amyloid_efficientnet_v2.json
├── environment/                 # Conda environment specs
│   └── environment_thios_net.yml
└── scripts/
    ├── python/                  # Prediction scripts
    │   ├── predict_iron_effnet.py
    │   ├── mask_predictions_to_tissue.py
    │   └── unet2D_iron.py
    ├── pipeline/                # Tiling & heatmap scripts
    │   ├── run_pipeline.py
    │   └── run_pipeline_part2.py
    ├── shell/                   # Shell scripts
    │   └── run_pipeline_full.sh
    └── batch/                   # SLURM batch scripts
        ├── single_slide_reprocess_tau.sh
        ├── single_slide_reprocess_iron.sh
        ├── single_slide_reprocess_amyloid.sh
        ├── launch_all_*_reprocess.sh
        └── cleanup_intermediate_files.sh
```

## Quick Start

### 1. Set up environments

You need two conda environments:

**ThioS_net** (for Stage 2 - deep learning prediction):
```bash
conda env create -f environment/environment_thios_net.yml
```

**gdaltest** (for Stage 1 & 3 - tiling and heatmaps):
```bash
# Create gdaltest environment with required packages
conda create -n gdaltest python=3.9
conda activate gdaltest
pip install numpy nibabel gdal pillow scikit-image configparser
```

### 2. Update paths in configuration files

Before running, update the paths in the config files to match your installation:

**configs/setup-predict_*.json files:**
```json
{
  "model_path": "/YOUR/PATH/RO1_Heatmap_Pipeline/models/tau_efficientnet_v2.ckpt",
  ...
}
```

**configs/pipeline_config.txt:**
```ini
[global]
SCRIPT_DIR = /YOUR/PATH/RO1_Heatmap_Pipeline/scripts/shell
```

**scripts/batch/single_slide_reprocess_*.sh:**
- Update `PIPELINE_CONFIG` path
- Update `PREDICTION_CONFIG` path
- Update paths to Python scripts

### 3. Process a single slide

```bash
export SLIDE_NAME="your_slide_name"
export SLIDE_DIR="/path/to/slide/directory"

sbatch --export=SLIDE_NAME,SLIDE_DIR \
    scripts/batch/single_slide_reprocess_tau.sh
```

## Pipeline Stages

| Stage | Description | Environment | Time |
|-------|-------------|-------------|------|
| 1 | Tile WSI & create tissue masks | gdaltest | ~30-60 min |
| 2 | Run EfficientNet prediction | ThioS_net | ~10-30 min |
| 2.5 | Remove background false positives | ThioS_net | ~1-5 min |
| 3 | Generate density heatmaps | gdaltest | ~5-15 min |

## Models

| Stain | Checkpoint | Val Loss |
|-------|------------|----------|
| Tau (CP13) | `tau_efficientnet_v2.ckpt` | 0.14 |
| Iron | `iron_efficientnet_v2.ckpt` | 0.11 |
| Amyloid | `amyloid_efficientnet_v2.ckpt` | 0.10 |

All models use EfficientNet-B4 backbone with:
- Input: 128×128 RGB patches
- Output: 2-class binary segmentation
- Framework: PyTorch + MONAI

## Output Files

After processing, each slide will have:
- `{slide}_files/heatmap/hm_map_0.1/heat_map_0.1_res10.nii` - Density heatmap (NIfTI)
- `{slide}_files/heatmap/color_map_0.1/` - Colorized visualization

## Requirements

- SLURM cluster (scripts use sbatch)
- 400GB RAM per job (for large slides)
- 32 CPUs per job
- ~24 hours per slide (varies by size)

## License

Internal use only - Mayo Clinic / RO1 Project

## Contact

For questions, contact the RO1 project team.
