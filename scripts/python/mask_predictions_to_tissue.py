#!/usr/bin/env python3
"""
Post-processing script to mask prediction masks to tissue regions only.

This script takes prediction masks and the corresponding seg_tiles,
and zeros out any predictions in black (background) regions of the seg_tiles.

This fixes an issue where some models predict positive in background regions.
"""

import os
import sys
import glob
import argparse
import numpy as np
from PIL import Image

# Try to import tqdm, but make it optional
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def mask_prediction_to_tissue(pred_mask, seg_tile, threshold=0):
    """
    Mask prediction to only include tissue regions.
    
    Args:
        pred_mask: 2D numpy array of prediction mask
        seg_tile: 3D numpy array (H, W, 3) of RGB input tile
        threshold: sum of RGB values below which is considered background
    
    Returns:
        Masked prediction array
    """
    # Calculate sum of RGB channels
    seg_sum = seg_tile[:,:,0].astype(np.int32) + seg_tile[:,:,1].astype(np.int32) + seg_tile[:,:,2].astype(np.int32)
    
    # Create tissue mask (non-black regions)
    tissue_mask = (seg_sum > threshold)
    
    # Apply tissue mask to predictions
    masked_pred = pred_mask.copy()
    masked_pred[~tissue_mask] = 0
    
    return masked_pred


def process_directory(pred_dir, seg_dir, output_dir=None, in_place=True):
    """
    Process all prediction masks in a directory.
    
    Args:
        pred_dir: Directory containing prediction masks (*_mask.tif)
        seg_dir: Directory containing corresponding seg_tiles (tile_XXXX.tif)
        output_dir: Output directory for masked predictions (if not in_place)
        in_place: If True, overwrite original prediction masks
    
    Returns:
        Dictionary with statistics
    """
    if output_dir is None:
        output_dir = pred_dir
    
    if not in_place and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Find all prediction masks
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "tile_*_mask.tif")))
    
    stats = {
        'total_files': len(pred_files),
        'processed': 0,
        'skipped': 0,
        'pixels_removed': 0,
        'tiles_with_bg_preds': 0
    }
    
    print(f"Processing {len(pred_files)} prediction masks...")
    print(f"  Prediction dir: {pred_dir}")
    print(f"  Seg tiles dir: {seg_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  In-place: {in_place}")
    
    for pred_file in tqdm(pred_files, desc="Masking predictions"):
        # Get corresponding seg_tile filename
        # tile_0000_mask.tif -> tile_0000.tif
        basename = os.path.basename(pred_file)
        seg_filename = basename.replace("_mask", "")
        seg_file = os.path.join(seg_dir, seg_filename)
        
        if not os.path.exists(seg_file):
            print(f"  Warning: seg_tile not found: {seg_file}")
            stats['skipped'] += 1
            continue
        
        # Load files
        pred_mask = np.array(Image.open(pred_file))
        seg_tile = np.array(Image.open(seg_file))
        
        # Check shapes match
        if pred_mask.shape[:2] != seg_tile.shape[:2]:
            print(f"  Warning: shape mismatch - pred {pred_mask.shape} vs seg {seg_tile.shape}")
            stats['skipped'] += 1
            continue
        
        # Count predictions in background before masking
        seg_sum = seg_tile[:,:,0].astype(np.int32) + seg_tile[:,:,1].astype(np.int32) + seg_tile[:,:,2].astype(np.int32)
        background = (seg_sum == 0)
        pred_nonzero = (pred_mask > 0)
        bg_preds = (pred_nonzero & background).sum()
        
        if bg_preds > 0:
            stats['tiles_with_bg_preds'] += 1
            stats['pixels_removed'] += bg_preds
        
        # Apply tissue masking
        masked_pred = mask_prediction_to_tissue(pred_mask, seg_tile)
        
        # Save result
        output_file = os.path.join(output_dir, basename)
        Image.fromarray(masked_pred).save(output_file)
        
        stats['processed'] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Mask predictions to tissue regions only")
    parser.add_argument("--pred_dir", required=True, help="Directory with prediction masks")
    parser.add_argument("--seg_dir", required=True, help="Directory with seg_tiles")
    parser.add_argument("--output_dir", help="Output directory (default: overwrite in place)")
    parser.add_argument("--no_inplace", action="store_true", help="Don't overwrite original files")
    
    args = parser.parse_args()
    
    in_place = not args.no_inplace
    output_dir = args.output_dir if args.output_dir else args.pred_dir
    
    stats = process_directory(
        pred_dir=args.pred_dir,
        seg_dir=args.seg_dir,
        output_dir=output_dir,
        in_place=in_place
    )
    
    print("\n=== Summary ===")
    print(f"Total files: {stats['total_files']}")
    print(f"Processed: {stats['processed']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Tiles with background predictions: {stats['tiles_with_bg_preds']}")
    print(f"Total pixels removed: {stats['pixels_removed']}")
    
    if stats['pixels_removed'] > 0:
        print(f"\nSuccessfully removed {stats['pixels_removed']} false positive predictions in background regions.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
