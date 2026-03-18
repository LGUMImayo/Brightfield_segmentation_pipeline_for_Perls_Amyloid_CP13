#!/bin/bash
#SBATCH --job-name=gcs_decompress_pipeline
#SBATCH --output=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/decompress_%j.out
#SBATCH --error=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/decompress_%j.err
#SBATCH --partition=med-n16-64g
#SBATCH --time=99:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8   # Use more if available
#SBATCH --mem=32G

# Source conda.sh to make conda commands available
source /home/ext_chen_yuheng_mayo_edu/miniconda3/etc/profile.d/conda.sh
# Activate the correct conda environment
conda activate gdaltest
export PYTHONPATH=$PYTHONPATH:/fslustre/qhs/ext_chen_yuheng_mayo_edu/high-res-3D-tau

if [ "$#" -ne 2 ]; then
	echo "Usage: run_pipeline_full.sh <ROOT_DIR> <CONFIG_PATH>"
	exit 0  
fi

ROOT_DIR=$1
CONF_FILE=$2

echo $ROOT_DIR
echo $CONF_FILE

python /fslustre/qhs/ext_chen_yuheng_mayo_edu/high-res-3D-tau/pipeline/run_pipeline_part2.py $ROOT_DIR $CONF_FILE

