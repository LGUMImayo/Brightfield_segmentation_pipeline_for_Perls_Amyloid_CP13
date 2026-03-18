# RO1_GCP Pipeline Reprocessing Documentation

## Overview

This document describes the batch reprocessing pipeline for whole slide images (WSI) in the RO1_GCP/Pipeline_merged project. The pipeline processes histopathology slides for three stain types:

- **Tau (CP13)** - Tau protein detection
- **Iron** - Iron deposits detection  
- **Amyloid** - Amyloid plaques detection

## Directory Structure

```
/fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_GCP/
├── Pipeline_merged/
│   ├── RO1_Tau/Cases/          # Tau slide data
│   ├── RO1_Iron/Cases/         # Iron slide data
│   └── RO1_Amyloid/Cases/      # Amyloid slide data
├── success_tau_stage3.txt      # List of slides to reprocess
├── success_iron_stage3.txt
├── success_amyloid_stage3.txt
└── reprocess_logs/             # Processing logs
    ├── success_*_reprocess.txt
    └── failed_*_reprocess.txt
```

## Pipeline Stages

The reprocessing pipeline consists of 4 stages:

### Stage 1: Tiling & Mask Creation (Pipeline 1)

**Purpose:** Convert WSI to tiles and create tissue masks

**Environment:** `gdaltest`

**Script:** `/fslustre/qhs/ext_chen_yuheng_mayo_edu/high-res-3D-tau/scripts/run_pipeline_full.sh`

**Config Files:**
- Tau/Iron: `/fslustre/qhs/ext_chen_yuheng_mayo_edu/high-res-3D-tau/pipeline_config.txt`
- Amyloid: `/fslustre/qhs/ext_chen_yuheng_mayo_edu/high-res-3D-tau/pipeline_config_amyloid.txt`

**Key Outputs:**
- `{slide}_files/heatmap/seg_tiles/` - Tissue-masked image tiles
- `{slide}_files/mask/final_mask/` - Binary tissue masks
- `TileMasker.log` - Processing log in slide directory

**Key Config Parameters:**
| Parameter | Tau/Iron | Amyloid |
|-----------|----------|---------|
| EROSION_PIXELS | 0 | 50 |
| PIX_1MM | 2890 | 2890 |
| HMAP_RES | 0.1 | 0.1 |

---

### Stage 2: EfficientNet Prediction

**Purpose:** Run deep learning model to detect pathology in each tile

**Environment:** `ThioS_net` (numpy 1.26.4 - matches training environment)

**Script:** `/fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/rhizonet/predict_iron_effnet.py`

**Config Files:**
- Tau: `setup-predict_Tau_efficientnet_v2.json`
- Iron: `setup-predict_Iron_efficientnet_v2.json`
- Amyloid: `setup-predict_Amyloid_efficientnet_v2.json`

Located in: `/fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/data/setup_files/`

**Model Architecture:**
- EfficientNet-B4 backbone
- Input: 128×128 RGB patches
- Output: 2-class binary segmentation

**Trained Models:**
| Stain | Checkpoint |
|-------|------------|
| Tau | `checkpoint-epoch=187-val_loss=0.14.ckpt` |
| Iron | `checkpoint-epoch=197-val_loss=0.11.ckpt` |
| Amyloid | `checkpoint-epoch=191-val_loss=0.10.ckpt` |

**Outputs:**
- `{slide}_files/heatmap/{Stain}_seg_tiles/` - Prediction masks (*_mask.tif)

---

### Stage 2.5: Post-Processing (Tissue Masking)

**Purpose:** Remove false positive predictions in background regions

**Script:** `/fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/rhizonet/mask_predictions_to_tissue.py`

**How it works:**
1. Loads prediction mask and corresponding seg_tile
2. Computes RGB sum of seg_tile pixels
3. Creates tissue mask where RGB sum > 0 (non-black regions)
4. Zeros out predictions in background (black) regions

**Usage:**
```bash
python mask_predictions_to_tissue.py \
    --pred_dir "{FILES_DIR}/heatmap/{Stain}_seg_tiles" \
    --seg_dir "{FILES_DIR}/heatmap/seg_tiles"
```

---

### Stage 3: Heatmap Creation (Pipeline 2)

**Purpose:** Aggregate tile predictions into density heatmaps

**Environment:** `gdaltest`

**Script:** `/fslustre/qhs/ext_chen_yuheng_mayo_edu/high-res-3D-tau/pipeline/run_pipeline_part2.py`

**Usage:**
```bash
python run_pipeline_part2.py {FILES_DIR} {CONFIG_FILE} {STAIN_TYPE}
```

**Outputs:**
- `{slide}_files/heatmap/hm_map_0.1/heat_map_0.1_res10.nii` - Final heatmap (NIfTI format)
- `{slide}_files/heatmap/color_map_0.1/` - Colorized visualization

**Heatmap Value Interpretation:**
- Values represent pathology density (scaled by SCALE_FACTOR=1000)
- Max theoretical value: 100,000 (100% coverage)
- Values > 100,000 indicate processing errors

---

## Batch Scripts

### Launcher Scripts

Submit all slides for a stain type:

| Script | Stain |
|--------|-------|
| `launch_all_tau_reprocess.sh` | Tau |
| `launch_all_iron_reprocess.sh` | Iron |
| `launch_all_amyloid_reprocess.sh` | Amyloid |

**Usage:**
```bash
./launch_all_tau_reprocess.sh
# Prompts for confirmation before submitting jobs
```

### Single Slide Scripts

Process one slide (submitted by launcher):

| Script | Stain |
|--------|-------|
| `single_slide_reprocess_tau.sh` | Tau |
| `single_slide_reprocess_iron.sh` | Iron |
| `single_slide_reprocess_amyloid.sh` | Amyloid |

**SLURM Configuration:**
- Partition: `huge-n128-512g`
- Memory: 400G
- CPUs: 32
- Time: 24 hours

**Manual Usage:**
```bash
sbatch --export=SLIDE_NAME="2353_22_#123_MFG_CP13",SLIDE_DIR="/path/to/slide" \
    single_slide_reprocess_tau.sh
```

### Cleanup Script

Remove intermediate files after processing:

```bash
# Dry run (preview what would be deleted)
./cleanup_intermediate_files.sh --dry-run

# Clean specific stain
./cleanup_intermediate_files.sh --stain tau

# Clean all stains
./cleanup_intermediate_files.sh
```

**Files Removed:**
- `heatmap/seg_tiles/` - Input tiles
- `heatmap/{Stain}_seg_tiles/` - Prediction masks
- `heatmap/hm_map_0.1/heat_map_0.1.npy` - Full-res numpy array
- `mask/final_mask/tiles/` - Mask tiles
- `output/RES*/tiles/` - Output tiles

**Files Preserved:**
- `heatmap/hm_map_0.1/*.nii` - Final heatmaps
- `heatmap/color_map_0.1/` - Visualizations
- `min_max.npy`, `scale.npy` - Metadata

---

## Input Files

### Success Lists

Files listing slides to reprocess:

| File | Slide Count |
|------|-------------|
| `success_tau_stage3.txt` | 223 |
| `success_iron_stage3.txt` | 210 |
| `success_amyloid_stage3.txt` | 264 |

**Format:** One slide name per line
```
2353_22_#123_MFG_CP13
3063_22_#10_BS_CP13
...
```

### Slide Naming Convention

Pattern: `{CASE_ID}_#{SLIDE_NUM}_{REGION}_{STAIN}`

Examples:
- `2353_22_#123_MFG_CP13` - Case 2353_22, slide 123, Medial Frontal Gyrus, CP13 stain
- `3063_22_#42_BS_Iron` - Case 3063_22, slide 42, Brainstem, Iron stain
- `3063_22_#59_MFG_Amyloid` - Case 3063_22, slide 59, Medial Frontal Gyrus, Amyloid stain

---

## Monitoring & Logs

### Job Monitoring

```bash
# Check running jobs
squeue -u $USER

# Watch job progress
watch -n 60 'squeue -u $USER | wc -l'
```

### Log Locations

| Log Type | Location |
|----------|----------|
| SLURM stdout | `/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/reprocess_{stain}_*.log` |
| SLURM stderr | `/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/reprocess_{stain}_*.err` |
| Success list | `/fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_GCP/reprocess_logs/success_{stain}_reprocess.txt` |
| Failed list | `/fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_GCP/reprocess_logs/failed_{stain}_reprocess.txt` |

### Failure Codes

| Code | Description |
|------|-------------|
| `DIR_NOT_FOUND` | Slide directory does not exist |
| `STAGE1_FAILED` | No seg_tiles created |
| `STAGE1_EMPTY` | seg_tiles directory empty |
| `TILEMASKER_LOG_MISSING` | TileMasker did not run |
| `STAGE2_FAILED` | No prediction masks created |
| `STAGE2_EMPTY` | Prediction masks empty |
| `STAGE3_FAILED` | Heatmap not created |
| `STAGE3_VERIFY_FAILED` | Heatmap validation failed (empty) |
| `STAGE3_ABNORMAL` | Heatmap values exceed theoretical max |

---

## Troubleshooting

### Common Issues

**1. TileMasker.log missing**
- Stage 1 (Pipeline 1) did not complete properly
- Check if input WSI file exists and is readable
- Verify disk space availability

**2. Heatmap values exceed 100,000**
- Usually indicates masking issues
- Check if tissue mask properly excludes background
- May need to investigate specific slide manually

**3. Empty heatmap**
- No pathology detected in slide
- Could be valid (clean slide) or detection failure
- Verify seg_tiles contain actual tissue

### Manual Reprocessing

To reprocess a single slide manually:

```bash
# Set variables
export SLIDE_NAME="your_slide_name"
export SLIDE_DIR="/path/to/slide/directory"

# Submit job
sbatch --export=SLIDE_NAME,SLIDE_DIR single_slide_reprocess_tau.sh
```

---

## Dependencies

### Conda Environments

| Environment | Purpose | Key Packages |
|-------------|---------|--------------|
| `gdaltest` | Pipeline 1 & 3 (tiling, heatmaps) | GDAL, nibabel, numpy |
| `ThioS_net` | Stage 2 (prediction) | PyTorch 2.4.1, MONAI 1.3.2, numpy 1.26.4 |

### External Scripts

| Script | Location | Purpose |
|--------|----------|---------|
| `run_pipeline.py` | `high-res-3D-tau/pipeline/` | Stage 1 orchestrator |
| `run_pipeline_part2.py` | `high-res-3D-tau/pipeline/` | Stage 3 orchestrator |
| `predict_iron_effnet.py` | `rhizonet/rhizonet/` | EfficientNet inference |
| `mask_predictions_to_tissue.py` | `rhizonet/rhizonet/` | Post-processing |

---

## Results Summary

As of last processing run:

| Stain | Total Slides | Reprocessed | Failed |
|-------|--------------|-------------|--------|
| Tau | 223 | 112 | 7 |
| Iron | 210 | 105 | 39 |
| Amyloid | 264 | 227 | 420* |

*Note: Amyloid failures include many STAGE3_ABNORMAL warnings which may still produce usable heatmaps.

---

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    launch_all_{stain}_reprocess.sh              │
│                         (Batch Launcher)                        │
└──────────────────────────────┬──────────────────────────────────┘
                               │ sbatch for each slide
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              single_slide_reprocess_{stain}.sh                  │
│                    (SLURM Job Script)                           │
└─────────────────────────────────────────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
┌───────────────┐    ┌────────────────┐    ┌─────────────────┐
│   STAGE 1     │    │    STAGE 2     │    │    STAGE 3      │
│  (gdaltest)   │───▶│   (rhizonet)   │───▶│   (gdaltest)    │
│               │    │                │    │                 │
│ run_pipeline  │    │ predict_iron   │    │ run_pipeline    │
│ _full.sh      │    │ _effnet.py     │    │ _part2.py       │
│               │    │       +        │    │                 │
│ Creates:      │    │ mask_pred...py │    │ Creates:        │
│ - seg_tiles   │    │                │    │ - heatmap.nii   │
│ - masks       │    │ Creates:       │    │ - colormap      │
└───────────────┘    │ - *_mask.tif   │    └─────────────────┘
                     └────────────────┘
```

---

## Contact

For questions about this pipeline, contact the RO1 project team.

Last updated: February 2026
