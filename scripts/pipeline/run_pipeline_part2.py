import os
import sys
import fnmatch
import skimage.io as io
import logging
import glob
from PipelineRunner import  PipelineRunner
from ImageTiler import ImageTiler
from MaskTiler import MaskTiler
from TileMasker import TileMasker
from HeatmapCreator import HeatmapCreator
from ColormapCreator import ColormapCreator
import configparser


def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print('Usage: run_pipeline.py <root_dir> <config_file> [stain_type]')
        print('  stain_type: TAU, Amyloid, or Iron (default: TAU)')
        exit()

    root_dir = str(sys.argv[1])  # abs path to where the images are
    conf_file = str(sys.argv[2])
    stain_type = str(sys.argv[3]) if len(sys.argv) == 4 else 'TAU'

    # --- FIX: Normalize heatmap folder name ---
    # Some slides have 'heat_map' instead of 'heatmap'. We need to consolidate them.
    # We look for the *_files directory inside root_dir
    files_dir = None
    for item in os.listdir(root_dir):
        if item.endswith('_files') and os.path.isdir(os.path.join(root_dir, item)):
            files_dir = os.path.join(root_dir, item)
            break
    
    if files_dir:
        heat_map_path = os.path.join(files_dir, 'heat_map')
        heatmap_path = os.path.join(files_dir, 'heatmap')
        
        # If 'heat_map' exists, we need to move its contents to 'heatmap'
        if os.path.exists(heat_map_path):
            print(f"Found 'heat_map' folder in {files_dir}. Consolidating to 'heatmap'...")
            
            # Ensure 'heatmap' exists
            if not os.path.exists(heatmap_path):
                os.makedirs(heatmap_path)
                print(f"Created 'heatmap' folder.")
            
            # Move contents
            import shutil
            for item in os.listdir(heat_map_path):
                src = os.path.join(heat_map_path, item)
                dst = os.path.join(heatmap_path, item)
                # Only move if destination doesn't exist to avoid overwriting
                if not os.path.exists(dst):
                    try:
                        shutil.move(src, dst)
                        print(f"Moved {item} to 'heatmap'")
                    except Exception as e:
                        print(f"Error moving {item}: {e}")
                else:
                    print(f"Skipping {item}, already exists in 'heatmap'")
            
            # Optional: Remove empty heat_map folder
            try:
                os.rmdir(heat_map_path)
                print("Removed empty 'heat_map' folder.")
            except:
                pass
    # ------------------------------------------

    print('### Creating pipeline ###')

    #create the pipeline
    pipeline = PipelineRunner(root_dir,conf_file)
    
    # Use stain type from command line argument
    print(f'Using stain type: {stain_type}')
    
    #img_tiles = ImageTiler('Image Tiling',root_dir)
    #mask_tiles = MaskTiler('Mask Resizing and Tiling',root_dir)
    #apply_mask = TileMasker('Mask Tiles',root_dir)
    heatmap_comp = HeatmapCreator('Compute HeatMap',root_dir,stain_type=stain_type)
    colormap_comp = ColormapCreator('Compute ColorMap', root_dir,stain_type=stain_type)


    #pipeline.add_stage(img_tiles)
    #pipeline.add_stage(mask_tiles)
    #pipeline.add_stage(apply_mask)
    pipeline.add_stage(heatmap_comp)
    pipeline.add_stage(colormap_comp)

    print('__________________________')
    print('Stages:')
    for st in pipeline.get_stages():
        print('    '+st.get_stage_name())
    print('__________________________')
    print('Starting...')

    #run pipeline
    pipeline.execute()

    print('### Pipeline Finished ###')





if __name__ == '__main__':
    main()
