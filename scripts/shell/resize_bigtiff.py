import sys
import os
import tifffile
import numpy as np
from skimage.transform import rescale
import argparse

def resize_image(input_path, output_path, scale=0.1):
    print(f"Reading image: {input_path}")
    try:
        # Read the image
        # tifffile handles BigTIFF and memory mapping efficiently
        img = tifffile.imread(input_path)
        print(f"Original shape: {img.shape}, dtype: {img.dtype}")

        # Rescale
        # preserve_range=True ensures the data range is kept (important for 16-bit images etc)
        # anti_aliasing=True is good for downsampling
        print(f"Resizing with scale {scale}...")
        
        # Handle multi-channel images if necessary, though usually these are 2D or 3D
        # If it's RGB (H, W, 3), we want to resize H and W.
        # If it's (Z, H, W), we might want to resize H and W only or all.
        # Assuming standard 2D or RGB image for this pipeline based on 'convert' usage.
        
        is_multichannel = False
        if img.ndim == 3 and img.shape[2] in [3, 4]:
            is_multichannel = True
            
        resized_img = rescale(img, scale, anti_aliasing=True, preserve_range=True, channel_axis=2 if is_multichannel else None)
        
        # Cast back to original dtype
        resized_img = resized_img.astype(img.dtype)
        print(f"Resized shape: {resized_img.shape}")

        # Save
        print(f"Saving to: {output_path}")
        tifffile.imwrite(output_path, resized_img, bigtiff=True)
        print("Done.")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize a BigTIFF image.")
    parser.add_argument("input_file", help="Path to input TIFF file")
    parser.add_argument("output_file", help="Path to output TIFF file")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)

    resize_image(args.input_file, args.output_file)
