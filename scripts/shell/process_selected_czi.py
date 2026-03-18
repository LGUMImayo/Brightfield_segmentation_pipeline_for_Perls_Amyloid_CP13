import czifile
import tifffile
import argparse
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
from ome_types import from_xml, OME
from ome_types.model import Image, Pixels, Channel
import numpy as np
import os
import re
import subprocess
from datetime import datetime
import gc

# --- Helper Functions (Copied from czi_to_ome_tiff.py) ---

def get_czi_metadata(czi_path):
    with czifile.CziFile(czi_path) as czi:
        metadata_xml = czi.metadata()
        return metadata_xml, czi.shape, czi.axes

def create_ome_metadata(image_shape, image_axes, czi_xml_metadata, image_name):
    try:
        root = ET.fromstring(czi_xml_metadata)
        ns = {'czi': 'http://www.zeiss.com/czi/2.0'}
        scaling_node = root.find('.//czi:Scaling', ns)
        
        phys_size_x, phys_size_y, phys_size_z = (1.0, 1.0, 1.0)

        if scaling_node is not None:
            for scale in scaling_node.findall('.//czi:Value', ns):
                axis_id = scale.get('Id')
                value = float(scale.text)
                if axis_id == 'X':
                    phys_size_x = value * 1e6
                elif axis_id == 'Y':
                    phys_size_y = value * 1e6
                elif axis_id == 'Z':
                    phys_size_z = value * 1e6

        ome = OME()
        image = Image(
            id='Image:0',
            name=image_name,
            pixels=Pixels(
                id='Pixels:0',
                size_c=image_shape[image_axes.find('C')],
                size_t=1,
                size_z=image_shape[image_axes.find('Z')],
                size_y=image_shape[image_axes.find('Y')],
                size_x=image_shape[image_axes.find('X')],
                dimension_order='XYZCT',
                type='uint8',
                physical_size_x=phys_size_x,
                physical_size_y=phys_size_y,
                physical_size_z=phys_size_z,
                physical_size_x_unit='µm',
                physical_size_y_unit='µm',
                physical_size_z_unit='µm',
                channels=[Channel(id=f'Channel:0:{i}') for i in range(image_shape[image_axes.find('C')])]
            )
        )
        ome.images.append(image)

        channel_nodes = root.findall('.//czi:Channel', ns)
        for i, channel_node in enumerate(channel_nodes):
            channel_name = channel_node.get('Name')
            if channel_name and i < len(ome.images[0].pixels.channels):
                 sanitized_channel_name = channel_name.encode('ascii', 'replace').decode('ascii')
                 ome.images[0].pixels.channels[i].name = sanitized_channel_name

        return ome

    except Exception as e:
        print(f"Could not create detailed OME metadata: {e}", file=sys.stderr)
        return None

def convert_czi_to_ome_tiff(input_path: Path, output_path: Path):
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}", file=sys.stderr)
        return

    print(f"Reading CZI file: {input_path}")
    try:
        with czifile.CziFile(input_path) as czi:
            czi_metadata_xml = czi.metadata()
            axes = czi.axes
            root = ET.fromstring(czi_metadata_xml)
            ns = {'czi': 'http://www.zeiss.com/czi/2.0'}

            num_scenes = 1
            if 'S' in axes:
                s_index = axes.find('S')
                num_scenes = czi.shape[s_index]
            print(f"Found {num_scenes} scene(s) in CZI file.")

            scene_nodes = root.findall('.//czi:Scene', ns)
            can_stitch = (len(scene_nodes) == num_scenes and num_scenes > 1)

            if num_scenes > 1 and not can_stitch:
                print("\n!!! WARNING: Multiple scenes found, but position metadata is missing or incomplete.", file=sys.stderr)

            if can_stitch:
                print("  - Scene position metadata found. Proceeding with stitching.")
                full_data = czi.asarray()
                scenes_data = []
                if 'S' in axes:
                    s_index = axes.find('S')
                    for i in range(num_scenes):
                        scenes_data.append(full_data.take(i, axis=s_index))
                else:
                    scenes_data.append(full_data)
                del full_data
                gc.collect()

                scale_x = None
                scale_node = root.find(".//czi:Scaling/czi:Items/czi:Distance[@Id='X']/czi:Value", ns)
                if scale_node is not None and scale_node.text is not None:
                    scale_x = float(scale_node.text)
                
                if scale_x is None:
                    scale_x = 1.0
                    print("!!! WARNING: Could not determine pixel scaling. Physical size metadata will be incorrect.", file=sys.stderr)

                scene_positions_pixels = []
                for i in range(num_scenes):
                    pos_x_node = scene_nodes[i].find('czi:PositionX', ns)
                    pos_y_node = scene_nodes[i].find('czi:PositionY', ns)
                    if pos_x_node is None or pos_y_node is None:
                        raise ValueError(f"Could not find position for Scene {i}.")
                    pos_x_pixels = int(float(pos_x_node.text) / scale_x)
                    pos_y_pixels = int(float(pos_y_node.text) / scale_x)
                    scene_positions_pixels.append((pos_x_pixels, pos_y_pixels))

                min_x = min(p[0] for p in scene_positions_pixels)
                min_y = min(p[1] for p in scene_positions_pixels)
                max_x, max_y = 0, 0
                scene_axes_str = axes.replace('S', '')
                y_idx, x_idx = scene_axes_str.find('Y'), scene_axes_str.find('X')
                for i, pos in enumerate(scene_positions_pixels):
                    scene_shape = scenes_data[i].shape
                    max_x = max(max_x, pos[0] + scene_shape[x_idx])
                    max_y = max(max_y, pos[1] + scene_shape[y_idx])

                stitched_width, stitched_height = max_x - min_x, max_y - min_y
                
                c_axis_index = scene_axes_str.find('C')
                if c_axis_index != -1:
                    num_channels = scenes_data[0].shape[c_axis_index]
                    stitched_shape = (num_channels, stitched_height, stitched_width)
                    stitched_axes = 'CYX'
                else:
                    num_channels = 1
                    stitched_shape = (stitched_height, stitched_width)
                    stitched_axes = 'YX'

                print(f"Creating stitched canvas with dimensions (H, W): ({stitched_height}, {stitched_width})")
                canvas = np.zeros(stitched_shape, dtype=scenes_data[0].dtype)

                for i, scene_data in enumerate(scenes_data):
                    pos_x, pos_y = scene_positions_pixels[i]
                    paste_x, paste_y = pos_x - min_x, pos_y - min_y
                    squeezed_data = np.squeeze(scene_data)
                    if 'C' in stitched_axes:
                        h, w = squeezed_data.shape[1], squeezed_data.shape[2]
                        canvas[:, paste_y:paste_y+h, paste_x:paste_x+w] = squeezed_data
                    else:
                        h, w = squeezed_data.shape[0], squeezed_data.shape[1]
                        canvas[paste_y:paste_y+h, paste_x:paste_x+w] = squeezed_data
                
                final_shape = (num_channels, 1, stitched_height, stitched_width)
                ome_metadata = create_ome_metadata(final_shape, 'CZYX', czi_metadata_xml, output_path.name)
                ome_xml = ome_metadata.to_xml() if ome_metadata else None
                if ome_xml: ome_xml = ome_xml.replace('µm', 'um')
                photometric = 'rgb' if 'C' in stitched_axes and num_channels == 3 else 'minisblack'

                print(f"Writing stitched OME-TIFF file: {output_path}")
                with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
                    tif.write(canvas, photometric=photometric, metadata={'axes': stitched_axes.replace('C', 'S')})
            
            else:
                if num_scenes > 1:
                    print("!!! WARNING: CZI library version requires loading all scenes into memory.", file=sys.stderr)
                    print("!!! Forcing an in-memory mosaic of all scenes (last scene on top).", file=sys.stderr)

                print("  - Loading all scene data into memory...")
                full_data = czi.asarray()
                scenes_data = []
                if 'S' in axes:
                    s_index = axes.find('S')
                    for i in range(num_scenes):
                        scenes_data.append(full_data.take(i, axis=s_index))
                else:
                    scenes_data.append(full_data)
                del full_data
                gc.collect()
                print("  - Scene data loaded.")

                print("  - Inspecting scene dimensions...")
                max_height = 0
                max_width = 0
                max_channels = 1
                scene_axes_str = axes.replace('S', '')
                original_dtype = scenes_data[0].dtype

                for scene_data in scenes_data:
                    squeezed_data = np.squeeze(scene_data)
                    squeezed_axes = "".join(ax for j, ax in enumerate(scene_axes_str) if scene_data.shape[j] > 1)
                    
                    c_idx = squeezed_axes.find('C')
                    if c_idx == -1: c_idx = squeezed_axes.find('0')
                    y_idx = squeezed_axes.find('Y')
                    x_idx = squeezed_axes.find('X')

                    if y_idx != -1: max_height = max(max_height, squeezed_data.shape[y_idx])
                    if x_idx != -1: max_width = max(max_width, squeezed_data.shape[x_idx])
                    if c_idx != -1: max_channels = max(max_channels, squeezed_data.shape[c_idx])
                    
                    del squeezed_data
                    gc.collect()

                if max_channels > 1:
                    num_channels = max_channels
                    canvas_shape = (num_channels, max_height, max_width)
                    canvas_axes = 'CYX'
                else:
                    num_channels = 1
                    canvas_shape = (max_height, max_width)
                    canvas_axes = 'YX'

                print(f"  - Creating in-memory canvas with shape {canvas_shape} and axes '{canvas_axes}'")
                canvas = np.zeros(canvas_shape, dtype=original_dtype)
                
                final_shape = (num_channels, 1, max_height, max_width)
                ome_metadata = create_ome_metadata(final_shape, 'CZYX', czi_metadata_xml, output_path.name)
                ome_xml = ome_metadata.to_xml() if ome_metadata else None
                if ome_xml:
                    ome_xml = ome_xml.encode('ascii', 'replace').decode('ascii')
                photometric = 'rgb' if 'C' in canvas_axes and num_channels == 3 else 'minisblack'

                print("  - Mosaicking scenes to canvas (last on top)...")
                for i, scene_data in enumerate(scenes_data):
                    squeezed_data = np.squeeze(scene_data)
                    
                    if num_channels > 1:
                        squeezed_axes = "".join(ax for j, ax in enumerate(scene_axes_str) if scene_data.shape[j] > 1)
                        c_src_idx = squeezed_axes.find('C')
                        if c_src_idx == -1: c_src_idx = squeezed_axes.find('0')
                        y_src_idx = squeezed_axes.find('Y')
                        x_src_idx = squeezed_axes.find('X')

                        if c_src_idx == -1:
                            h, w = squeezed_data.shape[y_src_idx], squeezed_data.shape[x_src_idx]
                            mask = squeezed_data > 0
                            for c in range(num_channels):
                                canvas[c, :h, :w] = np.where(mask, squeezed_data, canvas[c, :h, :w])
                        else:
                            transposed_data = np.transpose(squeezed_data, (c_src_idx, y_src_idx, x_src_idx))
                            h, w = transposed_data.shape[1], transposed_data.shape[2]
                            mask = np.any(transposed_data > 0, axis=0)
                            for c in range(num_channels):
                                canvas[c, :h, :w] = np.where(mask, transposed_data[c], canvas[c, :h, :w])
                    else:
                        h, w = squeezed_data.shape[0], squeezed_data.shape[1]
                        mask = squeezed_data > 0
                        canvas[:h, :w] = np.where(mask, squeezed_data, canvas[:h, :w])
                    
                    del squeezed_data
                    gc.collect()
                
                del scenes_data
                gc.collect()

                print(f"Writing mosaicked TIFF file: {output_path}")
                tifffile.imwrite(
                    output_path,
                    canvas,
                    bigtiff=True,
                    photometric=photometric,
                    metadata={'axes': canvas_axes.replace('C', 'S')}
                )

        print(f"Conversion successful: {output_path}")

    except Exception as e:
        print(f"An error occurred during conversion of {input_path}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

# --- Main Function with Pattern Filtering ---

def main():
    parser = argparse.ArgumentParser(
        description="Convert selected Zeiss CZI files to OME-TIFF format."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="The root directory to search for CZI files."
    )
    parser.add_argument(
        "--base_output_dir",
        type=str,
        default="/fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_GCP/Pipeline2",
        help="The base output directory for the pipeline."
    )
    parser.add_argument(
        "--patterns",
        nargs='+',
        help="List of filename patterns to process (e.g. #63_MFG_Iron #63_PUT_Iron)"
    )
    args = parser.parse_args()

    input_root = Path(args.input_dir)
    pipeline_base = Path(args.base_output_dir)
    stain_map = {
        "amyloid": "RO1_Amyloid",
        "tau": "RO1_Tau",
        "iron": "RO1_Iron"
    }
    rois = ["mfg", "put", "bs"]
    staining_keywords = {"amyloid": "amyloid", "tau": "cp13", "iron": "iron"}

    if not input_root.is_dir():
        print(f"Error: Input directory not found at {input_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Starting CZI to TIFF conversion process for directory: {input_root}")
    if args.patterns:
        print(f"Filtering for patterns: {args.patterns}")

    for czi_path in input_root.rglob('*.czi'):
        # --- Filter Logic ---
        if args.patterns:
            if not any(p in czi_path.name for p in args.patterns):
                continue
        
        print(f"\nProcessing file: {czi_path}")
        
        stem = czi_path.stem
        parts = stem.split('_')
        
        case_match = re.match(r'(\d+_\d+)', stem)
        if not case_match:
            print(f"  - Skipping: Could not determine case number from '{stem}'")
            continue
        case_number = case_match.group(1)

        staining = "unknown"
        if "cp13" in stem.lower():
            staining = "tau"
        else:
            for key in stain_map:
                if f"_{key.lower()}_" in f"_{stem.lower()}_":
                    staining = key
                    break
        
        if staining == "unknown":
            print(f"  - Skipping: Could not determine staining from '{stem}'")
            continue

        stain_folder = stain_map.get(staining)
        if not stain_folder:
            print(f"  - Skipping: No mapping for staining '{staining}'")
            continue
            
        slice_folder_name = stem
        stain_keyword_in_stem = staining_keywords.get(staining)
        roi_in_stem = next((roi for roi in rois if f"_{roi}_" in f"_{stem.lower()}_"), None)

        if stain_keyword_in_stem and roi_in_stem:
            parts = stem.split('_')
            original_stain_part = next((p for p in parts if p.lower() == stain_keyword_in_stem), None)
            original_roi_part = next((p for p in parts if p.lower() == roi_in_stem), None)

            if original_stain_part and original_roi_part:
                if stem.find(original_stain_part) < stem.find(original_roi_part):
                    print(f"  - Rearranging filename: '{original_stain_part}' and '{original_roi_part}'")
                    temp_name = stem.replace(original_stain_part, "___ROI_PLACEHOLDER___")
                    temp_name = temp_name.replace(original_roi_part, original_stain_part)
                    slice_folder_name = temp_name.replace("___ROI_PLACEHOLDER___", original_roi_part)

        try:
            base_slice_dir = pipeline_base / stain_folder / "Cases" / case_number / slice_folder_name
            files_dir = base_slice_dir / f"{slice_folder_name}_files"
            
            output_dir = files_dir / "output"
            output_res_dir = output_dir / "RES"
            mask_dir = files_dir / "mask" / "final_mask"
            heatmap_dir = files_dir / "heatmap"
            
            output_res_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)
            heatmap_dir.mkdir(parents=True, exist_ok=True)
            
            output_filename = output_res_dir / f"{slice_folder_name}.tiff"
            
            print(f"  - Case: {case_number}, Staining: {staining}")
            print(f"  - Slice Folder: {slice_folder_name}")
            print(f"  - Output: {output_filename}")

            convert_czi_to_ome_tiff(czi_path, output_filename)

            if not output_filename.exists():
                print(f"  - Original TIFF not found after conversion attempt. Cannot run resize script.", file=sys.stderr)
            else:
                print(f"  - Forcing resize task submission to SLURM for directory: {output_dir}")
                resize_script_path = "/fslustre/qhs/ext_chen_yuheng_mayo_edu/high-res-3D-tau/slurm_scripts/run_convert.sh"
                
                if os.path.exists(resize_script_path):
                    try:
                        sbatch_command = [
                            "sbatch",
                            resize_script_path,
                            str(output_dir)
                        ]
                        
                        result = subprocess.run(
                            sbatch_command,
                            capture_output=True,
                            text=True,
                            check=True
                        )
                        
                        print(f"    - SLURM job submitted successfully. Job ID: {result.stdout.strip()}")

                    except subprocess.CalledProcessError as e:
                        print(f"  - Error submitting SLURM job for resize script.", file=sys.stderr)
                        print(f"  - Return Code: {e.returncode}", file=sys.stderr)
                    except Exception as sub_e:
                        print(f"  - An unexpected error occurred while trying to submit the SLURM job: {sub_e}", file=sys.stderr)
                else:
                    print(f"  - Warning: Resize script not found at {resize_script_path}", file=sys.stderr)

        except Exception as e:
            print(f"  - Failed to process or resize for {czi_path}: {e}", file=sys.stderr)
            continue

    print("\nAll files processed.")

if __name__ == "__main__":
    main()
