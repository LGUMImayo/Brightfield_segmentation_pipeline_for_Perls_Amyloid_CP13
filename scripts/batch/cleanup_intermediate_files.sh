#!/bin/bash
# ==============================================================================
# Cleanup Intermediate Files from Completed Reprocessed Slides
# Removes seg_tiles, prediction masks, and full-res heatmap numpy files
# Keeps: hm_map_0.1/*.nii, color_map_0.1/, min_max.npy, scale.npy
# ==============================================================================

set -e

# Stain configurations
declare -A STAIN_DIRS=(
    ["iron"]="/fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_GCP/Pipeline_merged/RO1_Iron/Cases"
    ["tau"]="/fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_GCP/Pipeline_merged/RO1_Tau/Cases"
    ["amyloid"]="/fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_GCP/Pipeline_merged/RO1_Amyloid/Cases"
)

declare -A STAIN_PRED_DIRS=(
    ["iron"]="Iron_seg_tiles"
    ["tau"]="Tau_seg_tiles"
    ["amyloid"]="Amyloid_seg_tiles"
)

LOG_DIR="/fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_GCP/reprocess_logs"

# Parse arguments
DRY_RUN=false
STAIN=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --stain)
            STAIN="$2"
            shift 2
            ;;
        *)
            echo "Usage: $0 [--dry-run] [--stain iron|tau|amyloid]"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "Intermediate Files Cleanup"
echo "Dry run: $DRY_RUN"
echo "Stain filter: ${STAIN:-all}"
echo "============================================================"

TOTAL_FREED=0
SLIDES_CLEANED=0

for stain in iron tau amyloid; do
    # Skip if stain filter doesn't match
    if [ -n "$STAIN" ] && [ "$stain" != "$STAIN" ]; then
        continue
    fi

    SUCCESS_FILE="$LOG_DIR/success_${stain}_reprocess.txt"
    BASE_DIR="${STAIN_DIRS[$stain]}"
    PRED_DIR="${STAIN_PRED_DIRS[$stain]}"

    if [ ! -f "$SUCCESS_FILE" ]; then
        echo "[$stain] No success file found, skipping"
        continue
    fi

    # Get unique slides
    SLIDES=$(sort -u "$SUCCESS_FILE")
    SLIDE_COUNT=$(echo "$SLIDES" | wc -l)
    
    echo ""
    echo "=== $stain ($SLIDE_COUNT slides) ==="

    while IFS= read -r slide; do
        [ -z "$slide" ] && continue

        # Extract case ID
        CASE_ID=$(echo "$slide" | grep -oP '^\d+_\d+')
        if [ -z "$CASE_ID" ]; then
            echo "  ⚠️ Cannot extract case ID from: $slide"
            continue
        fi

        FILES_DIR="$BASE_DIR/$CASE_ID/$slide/${slide}_files"
        
        if [ ! -d "$FILES_DIR" ]; then
            continue
        fi

        # Check what can be cleaned
        SIZE_BEFORE=0
        DIRS_TO_CLEAN=()
        FILES_TO_CLEAN=()

        # seg_tiles directory
        if [ -d "$FILES_DIR/heatmap/seg_tiles" ]; then
            size=$(du -s "$FILES_DIR/heatmap/seg_tiles" 2>/dev/null | cut -f1)
            SIZE_BEFORE=$((SIZE_BEFORE + size))
            DIRS_TO_CLEAN+=("$FILES_DIR/heatmap/seg_tiles")
        fi

        # Prediction masks directory
        if [ -d "$FILES_DIR/heatmap/$PRED_DIR" ]; then
            size=$(du -s "$FILES_DIR/heatmap/$PRED_DIR" 2>/dev/null | cut -f1)
            SIZE_BEFORE=$((SIZE_BEFORE + size))
            DIRS_TO_CLEAN+=("$FILES_DIR/heatmap/$PRED_DIR")
        fi

        # Full-res heatmap numpy file
        if [ -f "$FILES_DIR/heatmap/hm_map_0.1/heat_map_0.1.npy" ]; then
            size=$(du -s "$FILES_DIR/heatmap/hm_map_0.1/heat_map_0.1.npy" 2>/dev/null | cut -f1)
            SIZE_BEFORE=$((SIZE_BEFORE + size))
            FILES_TO_CLEAN+=("$FILES_DIR/heatmap/hm_map_0.1/heat_map_0.1.npy")
        fi

        # Mask tiles directory
        if [ -d "$FILES_DIR/mask/final_mask/tiles" ]; then
            size=$(du -s "$FILES_DIR/mask/final_mask/tiles" 2>/dev/null | cut -f1)
            SIZE_BEFORE=$((SIZE_BEFORE + size))
            DIRS_TO_CLEAN+=("$FILES_DIR/mask/final_mask/tiles")
        fi

        # Output RES tiles directory
        for res_tiles in "$FILES_DIR"/output/RES*/tiles; do
            if [ -d "$res_tiles" ]; then
                size=$(du -s "$res_tiles" 2>/dev/null | cut -f1)
                SIZE_BEFORE=$((SIZE_BEFORE + size))
                DIRS_TO_CLEAN+=("$res_tiles")
            fi
        done

        if [ $SIZE_BEFORE -eq 0 ]; then
            continue
        fi

        # Convert to MB
        SIZE_MB=$((SIZE_BEFORE / 1024))
        TOTAL_FREED=$((TOTAL_FREED + SIZE_BEFORE))
        SLIDES_CLEANED=$((SLIDES_CLEANED + 1))

        if [ "$DRY_RUN" = true ]; then
            echo "  [DRY] $slide: would free ${SIZE_MB}MB"
        else
            # Actually remove
            for dir in "${DIRS_TO_CLEAN[@]}"; do
                rm -rf "$dir" 2>/dev/null
            done
            for file in "${FILES_TO_CLEAN[@]}"; do
                rm -f "$file" 2>/dev/null
            done
            echo "  ✓ $slide: freed ${SIZE_MB}MB"
        fi

    done <<< "$SLIDES"
done

# Summary
TOTAL_GB=$((TOTAL_FREED / 1024 / 1024))
echo ""
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
echo "Slides processed: $SLIDES_CLEANED"
if [ "$DRY_RUN" = true ]; then
    echo "Space that would be freed: ${TOTAL_GB}GB"
    echo ""
    echo "Run without --dry-run to actually delete files"
else
    echo "Space freed: ${TOTAL_GB}GB"
fi
echo "============================================================"
