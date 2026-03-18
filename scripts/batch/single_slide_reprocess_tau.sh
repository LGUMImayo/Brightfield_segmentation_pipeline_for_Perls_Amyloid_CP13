#!/bin/bash
#SBATCH --job-name=reproc_tau_%j
#SBATCH --output=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/reprocess_tau_%j.log
#SBATCH --error=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/reprocess_tau_%j.err
#SBATCH --partition=huge-n128-512g
#SBATCH --mem=400G
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00

# ==============================================================================
# Single Slide Full Reprocessing - Tau
# Runs Stage 1 + Stage 2 + Stage 3 from scratch
# ==============================================================================
# Usage: sbatch --export=SLIDE_NAME=<name>,SLIDE_DIR=<path> single_slide_reprocess_tau.sh
# ==============================================================================

echo "============================================================"
echo "FULL REPROCESSING: Tau"
echo "Slide: $SLIDE_NAME"
echo "Directory: $SLIDE_DIR"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "============================================================"

source /home/ext_chen_yuheng_mayo_edu/miniconda3/etc/profile.d/conda.sh

# Paths
PIPELINE_CONFIG="/fslustre/qhs/ext_chen_yuheng_mayo_edu/high-res-3D-tau/pipeline_config.txt"
PREDICTION_CONFIG="/fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/data/setup_files/setup-predict_Tau_efficientnet_v2.json"
FILES_DIR="${SLIDE_DIR}/${SLIDE_NAME}_files"

# Log files
SUCCESS_LOG="/fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_GCP/reprocess_logs/success_tau_reprocess.txt"
FAILED_LOG="/fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_GCP/reprocess_logs/failed_tau_reprocess.txt"

mkdir -p "$(dirname "$SUCCESS_LOG")"

# Validate inputs
if [ -z "$SLIDE_NAME" ] || [ -z "$SLIDE_DIR" ]; then
    echo "ERROR: SLIDE_NAME and SLIDE_DIR must be set"
    exit 1
fi

if [ ! -d "$SLIDE_DIR" ]; then
    echo "ERROR: Slide directory not found: $SLIDE_DIR"
    echo "$SLIDE_NAME|DIR_NOT_FOUND" >> "$FAILED_LOG"
    exit 1
fi

echo "Files directory: $FILES_DIR"

# ==============================================================================
# STAGE 1: Tiling and Mask Creation
# ==============================================================================
echo ""
echo "============================================================"
echo "STAGE 1: Pipeline 1 (Tiling & Masks)"
echo "============================================================"

conda activate gdaltest
export PYTHONPATH="/fslustre/qhs/ext_chen_yuheng_mayo_edu/high-res-3D-tau:$PYTHONPATH"

# Clean up old heatmap data to start fresh
echo "Cleaning up old heatmap data..."
rm -rf "$FILES_DIR/heatmap/seg_tiles" 2>/dev/null
rm -rf "$FILES_DIR/heatmap/Tau_seg_tiles" 2>/dev/null
rm -rf "$FILES_DIR/heatmap/hm_map_0.1" 2>/dev/null
rm -rf "$FILES_DIR/heatmap/color_map_0.1" 2>/dev/null

# Run full Pipeline 1
echo "Running Pipeline 1..."
START_STAGE1=$(date +%s)

bash /fslustre/qhs/ext_chen_yuheng_mayo_edu/high-res-3D-tau/scripts/run_pipeline_full.sh \
    "$SLIDE_DIR" "$PIPELINE_CONFIG" 2>&1

END_STAGE1=$(date +%s)
ELAPSED1=$((END_STAGE1 - START_STAGE1))

# Verify Stage 1 output
if [ ! -d "$FILES_DIR/heatmap/seg_tiles" ]; then
    echo "STAGE 1 FAILED - No seg_tiles created"
    echo "$SLIDE_NAME|STAGE1_FAILED" >> "$FAILED_LOG"
    exit 1
fi

SEG_TILE_COUNT=$(find "$FILES_DIR/heatmap/seg_tiles" -name "*.tif" 2>/dev/null | wc -l)
if [ "$SEG_TILE_COUNT" -eq 0 ]; then
    echo "STAGE 1 FAILED - seg_tiles empty"
    echo "$SLIDE_NAME|STAGE1_EMPTY" >> "$FAILED_LOG"
    exit 1
fi

# Verify TileMasker ran successfully
# TileMasker.log is created in SLIDE_DIR, not FILES_DIR/heatmap
TILEMASKER_LOG="$SLIDE_DIR/TileMasker.log"
if [ ! -f "$TILEMASKER_LOG" ]; then
    echo "WARNING: TileMasker.log not found - TileMasker may not have run"
    echo "$SLIDE_NAME|TILEMASKER_LOG_MISSING" >> "$FAILED_LOG"
    exit 1
fi

echo "TileMasker.log found: $TILEMASKER_LOG"
echo "TileMasker log contents (last 5 lines):"
tail -5 "$TILEMASKER_LOG"

# Verify seg_tiles differ from output tiles (i.e., masking was applied)
OUTPUT_TILES_DIR="$FILES_DIR/heatmap/output/RES/tiles"
if [ -d "$OUTPUT_TILES_DIR" ]; then
    SAMPLE_SEG_TILE=$(find "$FILES_DIR/heatmap/seg_tiles" -name "*.tif" | head -1)
    SAMPLE_OUT_TILE="$OUTPUT_TILES_DIR/$(basename "$SAMPLE_SEG_TILE")"
    if [ -f "$SAMPLE_SEG_TILE" ] && [ -f "$SAMPLE_OUT_TILE" ]; then
        SEG_SIZE=$(stat -c%s "$SAMPLE_SEG_TILE")
        OUT_SIZE=$(stat -c%s "$SAMPLE_OUT_TILE")
        if [ "$SEG_SIZE" -eq "$OUT_SIZE" ]; then
            # Check if files are actually identical
            if cmp -s "$SAMPLE_SEG_TILE" "$SAMPLE_OUT_TILE"; then
                echo "WARNING: seg_tile identical to output tile - TileMasker may not have applied masks"
                echo "Sample: $(basename "$SAMPLE_SEG_TILE")"
            else
                echo "TileMasker verification: seg_tiles differ from output tiles (masking applied)"
            fi
        else
            echo "TileMasker verification: seg_tiles differ from output tiles (masking applied)"
        fi
    fi
fi

echo "STAGE 1 SUCCESS - Created $SEG_TILE_COUNT seg_tiles in ${ELAPSED1}s"

# ==============================================================================
# STAGE 2: EfficientNet Prediction
# ==============================================================================
echo ""
echo "============================================================"
echo "STAGE 2: EfficientNet Prediction"
echo "============================================================"

conda activate ThioS_net
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=16

echo "Running EfficientNet prediction..."
START_STAGE2=$(date +%s)

# Use predict_iron_effnet.py for Tau too (same architecture, different config)
python /fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/rhizonet/predict_iron_effnet.py \
    --config_file "$PREDICTION_CONFIG" --wsi_dir "$FILES_DIR" \
    --output_folder Tau_seg_tiles 2>&1

END_STAGE2=$(date +%s)
ELAPSED2=$((END_STAGE2 - START_STAGE2))

# Verify Stage 2 output
if [ ! -d "$FILES_DIR/heatmap/Tau_seg_tiles" ]; then
    echo "STAGE 2 FAILED - No Tau_seg_tiles created"
    echo "$SLIDE_NAME|STAGE2_FAILED" >> "$FAILED_LOG"
    exit 1
fi

PRED_TILE_COUNT=$(find "$FILES_DIR/heatmap/Tau_seg_tiles" -name "*_mask.tif" 2>/dev/null | wc -l)
if [ "$PRED_TILE_COUNT" -eq 0 ]; then
    echo "STAGE 2 FAILED - Tau_seg_tiles empty"
    echo "$SLIDE_NAME|STAGE2_EMPTY" >> "$FAILED_LOG"
    exit 1
fi

echo "STAGE 2 SUCCESS - Created $PRED_TILE_COUNT prediction masks in ${ELAPSED2}s"

# ==============================================================================
# STAGE 2.5: Post-processing - Mask predictions to tissue regions
# ==============================================================================
echo ""
echo "============================================================"
echo "STAGE 2.5: Mask predictions to tissue regions"
echo "============================================================"

echo "Removing false positive predictions in background regions..."
python /fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/rhizonet/mask_predictions_to_tissue.py \
    --pred_dir "$FILES_DIR/heatmap/Tau_seg_tiles" \
    --seg_dir "$FILES_DIR/heatmap/seg_tiles" 2>&1

echo "STAGE 2.5 COMPLETE"

# ==============================================================================
# STAGE 3: Heatmap Creation
# ==============================================================================
echo ""
echo "============================================================"
echo "STAGE 3: Heatmap Creation"
echo "============================================================"

conda activate gdaltest
export PYTHONPATH="/fslustre/qhs/ext_chen_yuheng_mayo_edu/high-res-3D-tau:$PYTHONPATH"

echo "Running Pipeline 2 (Heatmap)..."
START_STAGE3=$(date +%s)

python /fslustre/qhs/ext_chen_yuheng_mayo_edu/high-res-3D-tau/pipeline/run_pipeline_part2.py \
    "$FILES_DIR" "$PIPELINE_CONFIG" "Tau" 2>&1

END_STAGE3=$(date +%s)
ELAPSED3=$((END_STAGE3 - START_STAGE3))

# Verify Stage 3 output
HEATMAP_FILE="$FILES_DIR/heatmap/hm_map_0.1/heat_map_0.1_res10.nii"
if [ ! -f "$HEATMAP_FILE" ]; then
    echo "STAGE 3 FAILED - Heatmap not created"
    echo "$SLIDE_NAME|STAGE3_FAILED" >> "$FAILED_LOG"
    exit 1
fi

# Verify heatmap values are correct (max should be <= 100000)
echo "Verifying heatmap values..."
VERIFY_RESULT=$(python -c "
import nibabel as nib
import numpy as np
import sys
try:
    img = nib.load('$HEATMAP_FILE')
    data = img.get_fdata()
    max_val = np.nanmax(data)
    nonzero_count = np.count_nonzero(data)
    nonzero_pct = 100.0 * nonzero_count / data.size
    
    print(f'Max: {max_val:.2f}, Nonzero: {nonzero_pct:.2f}%')
    
    # Check if values are valid (max should be <= 100000 for 100%)
    if max_val > 100000:
        print(f'WARNING: Max value {max_val} exceeds theoretical max of 100000!')
        print(f'Ratio: {max_val/100000:.2f}x')
        sys.exit(2)
    elif nonzero_count == 0:
        print('ERROR: Heatmap is empty!')
        sys.exit(1)
    else:
        print('Heatmap values are VALID')
        sys.exit(0)
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
" 2>&1)

VERIFY_CODE=$?
echo "$VERIFY_RESULT"

if [ $VERIFY_CODE -eq 1 ]; then
    echo "STAGE 3 FAILED - Heatmap verification failed"
    echo "$SLIDE_NAME|STAGE3_VERIFY_FAILED|$VERIFY_RESULT" >> "$FAILED_LOG"
    exit 1
elif [ $VERIFY_CODE -eq 2 ]; then
    echo "STAGE 3 WARNING - Heatmap has abnormal values (not fixed)"
    echo "$SLIDE_NAME|STAGE3_ABNORMAL|$VERIFY_RESULT" >> "$FAILED_LOG"
fi

echo "STAGE 3 SUCCESS in ${ELAPSED3}s"

# ==============================================================================
# CLEANUP: Remove intermediate files
# ==============================================================================
echo ""
echo "============================================================"
echo "CLEANUP: Removing intermediate files"
echo "============================================================"

rm -rf "$FILES_DIR/heatmap/seg_tiles/" 2>/dev/null
rm -rf "$FILES_DIR/heatmap/Tau_seg_tiles/" 2>/dev/null
rm -f "$FILES_DIR/heatmap/hm_map_0.1/heat_map_0.1.npy" 2>/dev/null
rm -rf "$FILES_DIR/mask/final_mask/tiles/" 2>/dev/null
rm -rf "$FILES_DIR/output/RES"*/tiles/ 2>/dev/null

REMAINING=$(du -sh "$FILES_DIR" 2>/dev/null | cut -f1)
echo "Size after cleanup: $REMAINING"

# ==============================================================================
# SUMMARY
# ==============================================================================
echo ""
echo "============================================================"
echo "REPROCESSING COMPLETE"
echo "============================================================"
echo "Slide: $SLIDE_NAME"
echo "Stage 1 (Tiling): ${ELAPSED1}s - $SEG_TILE_COUNT tiles"
echo "Stage 2 (Prediction): ${ELAPSED2}s - $PRED_TILE_COUNT masks"
echo "Stage 3 (Heatmap): ${ELAPSED3}s"
TOTAL_TIME=$((ELAPSED1 + ELAPSED2 + ELAPSED3))
echo "Total time: ${TOTAL_TIME}s ($((TOTAL_TIME / 60)) minutes)"
echo "End: $(date)"

echo "$SLIDE_NAME" >> "$SUCCESS_LOG"
exit 0
