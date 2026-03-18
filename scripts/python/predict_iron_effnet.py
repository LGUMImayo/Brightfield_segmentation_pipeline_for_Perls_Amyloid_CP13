#!/usr/bin/env python3
"""
Iron Segmentation Prediction Script for EfficientNet-B4 v2 Model

This script runs inference on WSI tiles using the trained EfficientNet-B4 model.
Designed to work with the iron segmentation pipeline.

Usage:
    python predict_iron_effnet.py --config_file <config.json> --wsi_dir <slide_dir>
"""

import os
import glob
import argparse
import json
from tqdm import tqdm
import numpy as np
import torch
from skimage import io
from PIL import Image

# Import the Iron-specific model
try:
    from unet2D_iron import Unet2D, PredDataset2D
except ImportError:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from unet2D_iron import Unet2D, PredDataset2D

from monai.transforms import (
    Compose,
    ScaleIntensityRange,
    EnsureType
)


def load_model(model_path, model_params, device):
    """Load the EfficientNet Iron model."""
    print(f"Loading model from: {model_path}")
    
    # Initialize model without datasets (for inference)
    unet = Unet2D(None, None, **model_params)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    unet.load_state_dict(checkpoint['state_dict'])
    unet = unet.to(device)
    unet.eval()
    
    print(f"Model loaded successfully on {device}")
    return unet


def preprocess_image(img_path):
    """
    Preprocess a single image for inference.
    Returns tensor in (C, H, W) format normalized to [0, 1].
    """
    # Read image
    img = io.imread(img_path)
    
    # Handle different image formats
    if img.ndim == 2:
        # Grayscale to RGB
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[-1] == 4:
        # RGBA to RGB
        img = img[:, :, :3]
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Convert to (C, H, W)
    img = np.transpose(img, (2, 0, 1))
    
    return torch.from_numpy(img)


def pad_to_divisible(img_tensor, divisor=32):
    """
    Pad image tensor to be divisible by divisor (required for UNet encoder-decoder).
    Returns padded tensor and original size for cropping back.
    
    Args:
        img_tensor: (C, H, W) tensor
        divisor: Size must be divisible by this (32 for EfficientNet-B4)
    """
    _, h, w = img_tensor.shape
    
    # Calculate padding needed
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    
    if pad_h == 0 and pad_w == 0:
        return img_tensor, (h, w)
    
    # Pad on bottom and right
    padded = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    
    return padded, (h, w)


def predict_single_tile(model, img_tensor, device):
    """Run inference on a single tile with padding for arbitrary sizes."""
    with torch.no_grad():
        # Pad to be divisible by 32 (EfficientNet requirement)
        padded_tensor, original_size = pad_to_divisible(img_tensor, divisor=32)
        orig_h, orig_w = original_size
        
        # Add batch dimension
        img_batch = padded_tensor.unsqueeze(0).to(device)
        
        # Get logits
        logits = model(img_batch)
        
        # Get probabilities
        probs = torch.softmax(logits, dim=1)
        
        # Get predictions (class with highest probability)
        preds = torch.argmax(probs, dim=1)
        
        # Crop back to original size
        probs = probs[0, :, :orig_h, :orig_w].cpu()
        preds = preds[0, :orig_h, :orig_w].cpu()
        
        return probs, preds


def save_prediction(pred_mask, output_path, scale_to_255=True):
    """Save prediction mask as TIFF."""
    mask = pred_mask.numpy()
    if scale_to_255:
        mask = (mask * 255).astype(np.uint8)
    io.imsave(output_path, mask, check_contrast=False)


def predict_wsi_tiles(args):
    """
    Main prediction function for WSI tiles.
    Searches for 'seg_tiles' directories and processes all tiles.
    """
    root_dir = args.get('wsi_dir') or args['pred_data_dir']
    model_path = args['model_path']
    
    # Get output folder name from args, default to TAU_seg_tiles for backward compatibility
    output_folder_name = args.get('output_folder_name', 'TAU_seg_tiles')
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get model params from config or use defaults
    model_params = args.get('model_params', {
        'in_channels': 3,
        'num_classes': 2,
        'patch_size': (128, 128),
        'pred_patch_size': (128, 128),
        'model': 'efficientnet',
        'backbone': 'efficientnet-b4',
        'pretrained': True
    })
    
    # Load model
    model = load_model(model_path, model_params, device)
    
    print(f"Searching for 'seg_tiles' directories in: {root_dir}")
    
    # Track statistics
    total_processed = 0
    total_skipped = 0
    errors = []
    
    # Find all seg_tiles directories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'seg_tiles' in dirnames:
            seg_tiles_dir = os.path.join(dirpath, 'seg_tiles')
            output_dir = os.path.join(dirpath, output_folder_name)
            os.makedirs(output_dir, exist_ok=True)
            
            # Find all tile images
            image_files = sorted(glob.glob(os.path.join(seg_tiles_dir, '*.tif')))
            if not image_files:
                continue
            
            slide_name = os.path.basename(dirpath)
            print(f"\n--- Processing: {slide_name} ({len(image_files)} tiles) ---")
            
            # Check for already processed files
            existing_masks = glob.glob(os.path.join(output_dir, '*_mask.tif'))
            if len(existing_masks) >= len(image_files):
                print(f"  Skipping: Already processed ({len(existing_masks)} masks exist)")
                total_skipped += len(image_files)
                continue
            
            # Process each tile
            for img_path in tqdm(image_files, desc=f"  {slide_name}"):
                try:
                    # Get output filename
                    basename = os.path.splitext(os.path.basename(img_path))[0]
                    output_path = os.path.join(output_dir, f"{basename}_mask.tif")
                    
                    # Skip if already exists
                    if os.path.exists(output_path):
                        total_skipped += 1
                        continue
                    
                    # Preprocess
                    img_tensor = preprocess_image(img_path)
                    
                    # Predict
                    probs, pred = predict_single_tile(model, img_tensor, device)
                    
                    # Save mask
                    save_prediction(pred, output_path)
                    
                    total_processed += 1
                    
                except Exception as e:
                    errors.append((img_path, str(e)))
                    continue
    
    # Summary
    print(f"\n{'='*60}")
    print(f"PREDICTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Tiles processed: {total_processed}")
    print(f"  Tiles skipped:   {total_skipped}")
    print(f"  Errors:          {len(errors)}")
    
    if errors:
        print(f"\nErrors encountered:")
        for path, err in errors[:10]:  # Show first 10 errors
            print(f"  - {os.path.basename(path)}: {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    
    return total_processed, errors


def main():
    parser = argparse.ArgumentParser(description="Iron segmentation prediction with EfficientNet-B4")
    parser.add_argument("--config_file", type=str, required=True,
                        help="Path to JSON configuration file")
    parser.add_argument("--wsi_dir", type=str, default=None,
                        help="WSI directory to process (overrides config)")
    parser.add_argument("--output_folder", type=str, default=None,
                        help="Output folder name (e.g., Iron_seg_tiles, TAU_seg_tiles)")
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from: {args.config_file}")
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    
    # Merge command line args
    config['wsi_dir'] = args.wsi_dir
    
    # Set output folder name - command line > config > default
    if args.output_folder:
        config['output_folder_name'] = args.output_folder
    elif 'output_folder_name' not in config:
        config['output_folder_name'] = 'TAU_seg_tiles'
    
    # Run prediction
    predict_wsi_tiles(config)


if __name__ == "__main__":
    main()
