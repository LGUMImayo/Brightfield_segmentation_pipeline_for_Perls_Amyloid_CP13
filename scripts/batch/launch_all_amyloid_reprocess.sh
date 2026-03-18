#!/bin/bash
# ==============================================================================
# Launch All Amyloid Slides for Reprocessing
# Submits individual SLURM jobs for each slide
# ==============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SUCCESS_FILE="/fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_GCP/success_amyloid_stage3.txt"
AMYLOID_BASE_DIR="/fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_GCP/Pipeline_merged/RO1_Amyloid/Cases"

# Log directory
LOG_DIR="/fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_GCP/reprocess_logs"
mkdir -p "$LOG_DIR"

# Count slides
TOTAL_SLIDES=$(wc -l < "$SUCCESS_FILE")
echo "============================================================"
echo "Amyloid Reprocessing Launcher"
echo "Total slides to reprocess: $TOTAL_SLIDES"
echo "============================================================"

# Confirmation
read -p "Submit $TOTAL_SLIDES jobs? (y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "Aborted."
    exit 0
fi

# Submit jobs
JOB_COUNT=0
while IFS= read -r SLIDE_NAME; do
    # Skip empty lines
    [ -z "$SLIDE_NAME" ] && continue
    
    # Extract case ID (e.g., 3063_22 from 3063_22_#59_MFG_Amyloid)
    CASE_ID=$(echo "$SLIDE_NAME" | grep -oP '^\d+_\d+')
    
    if [ -z "$CASE_ID" ]; then
        echo "ERROR: Cannot extract case ID from $SLIDE_NAME"
        continue
    fi
    
    SLIDE_DIR="$AMYLOID_BASE_DIR/$CASE_ID/$SLIDE_NAME"
    
    if [ ! -d "$SLIDE_DIR" ]; then
        echo "ERROR: Directory not found: $SLIDE_DIR"
        echo "$SLIDE_NAME|DIR_NOT_FOUND" >> "$LOG_DIR/failed_amyloid_reprocess.txt"
        continue
    fi
    
    echo "Submitting: $SLIDE_NAME"
    sbatch --export=SLIDE_NAME="$SLIDE_NAME",SLIDE_DIR="$SLIDE_DIR" \
        "$SCRIPT_DIR/single_slide_reprocess_amyloid.sh"
    
    JOB_COUNT=$((JOB_COUNT + 1))
    
    # Add small delay to avoid overwhelming the scheduler
    sleep 0.2
    
done < "$SUCCESS_FILE"

echo "============================================================"
echo "Submitted $JOB_COUNT jobs"
echo "Monitor with: squeue -u $USER"
echo "Logs: /fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/reprocess_amyloid_*.log"
echo "============================================================"
