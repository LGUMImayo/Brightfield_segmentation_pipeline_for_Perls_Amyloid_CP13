import czifile
import tifffile
import argparse
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
from ome_types import from_xml, OME
from ome_types.model import Image, Pixels, Channel # Import necessary models
import numpy as np
import os
import re
import subprocess
from datetime import datetime
import gc

def get_czi_metadata(czi_path):
    """Extracts basic metadata from the CZI file."""
    with czifile.CziFile(czi_path) as czi:
        metadata_xml = czi.metadata()
        return metadata_xml, czi.shape, czi.axes

def create_ome_metadata(image_shape, image_axes, czi_xml_metadata, image_name):
    """Creates an OME object with metadata from the CZI file."""
    try:
        # Parse CZI metadata to find physical sizes
        root = ET.fromstring(czi_xml_metadata)
        ns = {'czi': 'http://www.zeiss.com/czi/2.0'}
        scaling_node = root.find('.//czi:Scaling', ns)
        
        phys_size_x, phys_size_y, phys_size_z = (1.0, 1.0, 1.0)

        if scaling_node is not None:
            for scale in scaling_node.findall('.//czi:Value', ns):
                axis_id = scale.get('Id')
                value = float(scale.text)
                if axis_id == 'X':
                    phys_size_x = value * 1e6  # Convert from m to µm
                elif axis_id == 'Y':
                    phys_size_y = value * 1e6  # Convert from m to µm
                elif axis_id == 'Z':
                    phys_size_z = value * 1e6  # Convert from m to µm

        # Create a new OME object
        ome = OME()
        
        # Correctly create Image and Pixels objects
        image = Image(
            id='Image:0',
            name=image_name,
            pixels=Pixels(
                id='Pixels:0',
                size_c=image_shape[image_axes.find('C')],
                size_t=1, # Assuming T=1 if not present
                size_z=image_shape[image_axes.find('Z')],
                size_y=image_shape[image_axes.find('Y')],
                size_x=image_shape[image_axes.find('X')],
                dimension_order='XYZCT', # OME standard order
                type='uint8',  # Assuming RGB is uint8
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

        # Add channel information if available
        channel_nodes = root.findall('.//czi:Channel', ns)
        for i, channel_node in enumerate(channel_nodes):
            channel_name = channel_node.get('Name')
            if channel_name and i < len(ome.images[0].pixels.channels):
                 # Sanitize channel name to be 7-bit ASCII for TIFF compliance
                 sanitized_channel_name = channel_name.encode('ascii', 'replace').decode('ascii')
                 ome.images[0].pixels.channels[i].name = sanitized_channel_name

        return ome

    except Exception as e:
        print(f"Could not create detailed OME metadata: {e}", file=sys.stderr)
        print("Falling back to basic OME metadata.", file=sys.stderr)
        return None


def normalize_scene_to_canvas(scene_data, scene_axes_str):
    """
    Normalize a scene array to either CYX or YX.
    Handles extra dimensions (e.g., Z/T) by max-projecting non-spatial axes.
    """
    arr = np.squeeze(scene_data)
    axes_present = [ax for idx, ax in enumerate(scene_axes_str) if scene_data.shape[idx] > 1]

    if len(axes_present) != arr.ndim:
        axes_present = axes_present[-arr.ndim:]

    if arr.ndim < 2:
        raise ValueError(f"Cannot infer Y/X axes from scene with shape {arr.shape} and axes {axes_present}")

    if 'Y' in axes_present and 'X' in axes_present:
        y_idx = axes_present.index('Y')
        x_idx = axes_present.index('X')
    else:
        y_idx = arr.ndim - 2
        x_idx = arr.ndim - 1

    c_idx = -1
    if 'C' in axes_present:
        c_idx = axes_present.index('C')
    elif '0' in axes_present:
        c_idx = axes_present.index('0')

    if c_idx in (y_idx, x_idx):
        c_idx = -1

    keep_axes = {y_idx, x_idx}
    if c_idx != -1:
        keep_axes.add(c_idx)

    reduce_axes = tuple(i for i in range(arr.ndim) if i not in keep_axes)
    if reduce_axes:
        arr = np.max(arr, axis=reduce_axes)

    arr = np.squeeze(arr)

    if arr.ndim == 2:
        return arr, 'YX'

    if arr.ndim == 3:
        cyx_order = None

        if len(axes_present) >= 3 and c_idx != -1:
            filtered_axes = [axes_present[i] for i in range(len(axes_present)) if i in keep_axes]
            if len(filtered_axes) == 3 and 'Y' in filtered_axes and 'X' in filtered_axes:
                c_new = filtered_axes.index('C') if 'C' in filtered_axes else (filtered_axes.index('0') if '0' in filtered_axes else -1)
                y_new = filtered_axes.index('Y')
                x_new = filtered_axes.index('X')
                if c_new not in (y_new, x_new) and len({c_new, y_new, x_new}) == 3:
                    cyx_order = (c_new, y_new, x_new)

        if cyx_order is None:
            if arr.shape[0] in (1, 2, 3, 4) and arr.shape[1] > 16 and arr.shape[2] > 16:
                cyx_order = (0, 1, 2)
            elif arr.shape[2] in (1, 2, 3, 4) and arr.shape[0] > 16 and arr.shape[1] > 16:
                cyx_order = (2, 0, 1)

        if cyx_order is not None:
            if len(set(cyx_order)) != 3:
                gray = np.max(arr, axis=tuple(i for i in range(arr.ndim - 2))) if arr.ndim > 2 else arr
                return gray, 'YX'
            return np.transpose(arr, cyx_order), 'CYX'

        gray = np.max(arr, axis=0)
        return gray, 'YX'

    if arr.ndim > 3:
        arr = np.max(arr, axis=tuple(range(arr.ndim - 2)))
        return arr, 'YX'

    raise ValueError(f"Expected 2D/3D scene after normalization, got shape {arr.shape}")


def convert_zstack_max_projection(input_path: Path, output_path: Path):
    """
    Memory-efficient Z-stack max-projection for CZI files with Z > 1.
    Reads subblocks (tiles) one at a time and accumulates a max-projection
    canvas, avoiding loading the entire Z-stack into memory.
    
    For a file with axes SCZYX0 and shape (1,1,18,157374,126639,3):
      - Full load would need ~334 GiB
      - This approach needs only ~5.7 GiB (one YX plane * RGB)
    """
    print(f"  - Using tile-by-tile Z-stack max-projection (memory-efficient)")
    with czifile.CziFile(input_path) as czi:
        czi_metadata_xml = czi.metadata()
        axes = czi.axes
        shape = czi.shape
        dtype = czi.dtype

        z_idx = axes.find('Z')
        y_idx = axes.find('Y')
        x_idx = axes.find('X')
        num_z = shape[z_idx]
        height = shape[y_idx]
        width = shape[x_idx]

        # Detect RGB channel (axis '0')
        has_rgb = '0' in axes
        rgb_idx = axes.find('0') if has_rgb else -1
        num_rgb = shape[rgb_idx] if has_rgb else 1

        # For scene-based files, determine scene count
        has_scenes = 'S' in axes
        s_idx = axes.find('S') if has_scenes else -1
        num_scenes = shape[s_idx] if has_scenes else 1

        print(f"    Z={num_z}, Y={height}, X={width}, RGB={num_rgb}, Scenes={num_scenes}")

        # Calculate canvas memory requirement
        canvas_gb = height * width * num_rgb * np.dtype(dtype).itemsize / (1024**3)
        print(f"    Canvas memory: {canvas_gb:.1f} GiB (vs {np.prod(shape) * np.dtype(dtype).itemsize / (1024**3):.1f} GiB for full load)")

        # Create the output canvas — just one YX(C) plane
        if num_rgb > 1:
            canvas = np.zeros((height, width, num_rgb), dtype=dtype)
            canvas_axes_str = 'YX0'
        else:
            canvas = np.zeros((height, width), dtype=dtype)
            canvas_axes_str = 'YX'

        # Get the global origin offsets from the subblock directory
        subblocks = czi.filtered_subblock_directory
        min_y = min(e.start[y_idx] for e in subblocks)
        min_x = min(e.start[x_idx] for e in subblocks)

        # Process subblocks one at a time — max-project onto canvas
        total_sb = len(subblocks)
        print(f"    Processing {total_sb} subblocks across {num_z} Z-planes...")
        processed = 0
        for entry in subblocks:
            tile_data = entry.data_segment().data()
            tile_data = np.squeeze(tile_data)  # Remove singleton dims → (tile_h, tile_w, 3) or (tile_h, tile_w)

            # Get tile position relative to canvas origin
            tile_y = entry.start[y_idx] - min_y
            tile_x = entry.start[x_idx] - min_x

            if tile_data.ndim == 3 and num_rgb > 1:
                tile_h, tile_w = tile_data.shape[0], tile_data.shape[1]
            elif tile_data.ndim == 2:
                tile_h, tile_w = tile_data.shape
            else:
                # Unexpected shape — try to handle gracefully
                if tile_data.shape[-1] in (1, 2, 3, 4) and tile_data.ndim == 3:
                    tile_h, tile_w = tile_data.shape[0], tile_data.shape[1]
                else:
                    tile_h, tile_w = tile_data.shape[-2], tile_data.shape[-1]
                    tile_data = tile_data.reshape(tile_h, tile_w) if num_rgb == 1 else tile_data.reshape(tile_h, tile_w, num_rgb)

            # Clip to canvas bounds
            end_y = min(tile_y + tile_h, height)
            end_x = min(tile_x + tile_w, width)
            clip_h = end_y - tile_y
            clip_w = end_x - tile_x

            if clip_h <= 0 or clip_w <= 0 or tile_y < 0 or tile_x < 0:
                processed += 1
                continue

            # Max-project this tile onto the canvas
            region = canvas[tile_y:end_y, tile_x:end_x]
            tile_clipped = tile_data[:clip_h, :clip_w]
            np.maximum(region, tile_clipped, out=region)

            del tile_data
            processed += 1
            if processed % 1000 == 0:
                print(f"      {processed}/{total_sb} subblocks processed...")

        print(f"    All {total_sb} subblocks processed.")

        # Convert canvas to CYX format for TIFF writing
        if num_rgb > 1:
            # canvas is (Y, X, C) → transpose to (C, Y, X)
            canvas = np.transpose(canvas, (2, 0, 1))
            final_axes = 'CYX'
            num_channels = num_rgb
        else:
            final_axes = 'YX'
            num_channels = 1

        # Create OME metadata
        final_shape = (num_channels, 1, height, width)
        ome_metadata = create_ome_metadata(final_shape, 'CZYX', czi_metadata_xml, output_path.name)
        ome_xml = ome_metadata.to_xml() if ome_metadata else None
        if ome_xml:
            ome_xml = ome_xml.encode('ascii', 'replace').decode('ascii')
        photometric = 'rgb' if final_axes == 'CYX' and num_channels == 3 else 'minisblack'

        # Write the max-projected TIFF
        print(f"Writing max-projected TIFF: {output_path}")
        tifffile.imwrite(
            output_path,
            canvas,
            bigtiff=True,
            photometric=photometric,
            metadata={'axes': final_axes.replace('C', 'S')}
        )

        del canvas
        gc.collect()
        print(f"  Z-stack max-projection conversion successful: {output_path}")


def convert_czi_to_ome_tiff(input_path: Path, output_path: Path):
    """
    Converts a CZI file to an OME-TIFF. It attempts to stitch scenes into a
    single layer, but falls back to saving them as separate pages if position
    metadata is missing.
    """
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}", file=sys.stderr)
        return

    print(f"Reading CZI file: {input_path}")
    skip_names = {
        "4583_22_204_PUT_Iron.czi",
        "10480_22_63_MFG_Iron.czi",
    }
    if input_path.name in skip_names:
        print(f"Skipping: File explicitly excluded ({input_path.name})")
        return

    # --- Quick pre-check: detect Z-stack before heavy processing ---
    try:
        with czifile.CziFile(input_path) as czi:
            axes = czi.axes
            shape = czi.shape
            file_size_gb = input_path.stat().st_size / (1024**3)
            if file_size_gb > 300.0:
                print(f"Skipping: File too large ({file_size_gb:.1f} GB > 300.0 GB)")
                return
            z_idx = axes.find('Z')
            num_z = shape[z_idx] if z_idx != -1 else 1
    except Exception as e:
        if "can not read ZISRAW segment" in str(e):
            print(f"Skipping corrupted CZI: {input_path.name}", file=sys.stderr)
            return
        raise

    # If Z-stack detected, use memory-efficient tile-by-tile max-projection
    if num_z > 1:
        print(f"  - Detected Z-stack with {num_z} planes (axes: {axes}, shape: {shape})")
        try:
            convert_zstack_max_projection(input_path, output_path)
        except Exception as e:
            print(f"An error occurred during Z-stack projection of {input_path}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        return

    # --- Standard conversion for non-Z-stack files ---
    memmap_path = None
    try:
        with czifile.CziFile(input_path) as czi:
            czi_metadata_xml = czi.metadata()
            axes = czi.axes
            root = ET.fromstring(czi_metadata_xml)
            ns = {'czi': 'http://www.zeiss.com/czi/2.0'}

            estimated_bytes = int(np.prod(czi.shape)) * np.dtype(czi.dtype).itemsize
            memmap_threshold_gb = 8.0
            use_memmap = (estimated_bytes / (1024**3)) >= memmap_threshold_gb
            memmap_path = None
            full_data = None

            def load_full_data():
                nonlocal full_data, memmap_path
                if full_data is not None:
                    return full_data
                if use_memmap:
                    memmap_path = output_path.with_suffix(output_path.suffix + ".memmap")
                    print(f"  - Using memmap for CZI data: {memmap_path}")
                    full_data = czi.asarray(out=str(memmap_path))
                else:
                    full_data = czi.asarray()
                return full_data

            # --- 1. Determine number of scenes ---
            num_scenes = 1
            if 'S' in axes:
                s_index = axes.find('S')
                num_scenes = czi.shape[s_index]
            print(f"Found {num_scenes} scene(s) in CZI file.")

            # --- 2. Check for Stitching Feasibility ---
            scene_nodes = root.findall('.//czi:Scene', ns)
            can_stitch = (len(scene_nodes) == num_scenes and num_scenes > 1)

            if num_scenes > 1 and not can_stitch:
                print("\n!!! WARNING: Multiple scenes found, but position metadata is missing or incomplete.", file=sys.stderr)

            # --- BRANCH: STITCH SCENES (if possible) ---
            if can_stitch:
                # This branch still loads all scenes at once. It assumes if stitching is possible,
                # the total size is manageable. This could be further optimized if needed.
                print("  - Scene position metadata found. Proceeding with stitching.")
                full_data = load_full_data()
                scene_axes_str = axes.replace('S', '')
                if 'S' in axes:
                    s_index = axes.find('S')
                else:
                    s_index = None

                # --- Get Scaling Info ---
                scale_x = None
                scale_node = root.find(".//czi:Scaling/czi:Items/czi:Distance[@Id='X']/czi:Value", ns)
                if scale_node is not None and scale_node.text is not None:
                    scale_x = float(scale_node.text)
                
                if scale_x is None:
                    scale_x = 1.0
                    print("!!! WARNING: Could not determine pixel scaling. Physical size metadata will be incorrect.", file=sys.stderr)

                # --- Get Scene Positions ---
                scene_positions_pixels = []
                for i in range(num_scenes):
                    pos_x_node = scene_nodes[i].find('czi:PositionX', ns)
                    pos_y_node = scene_nodes[i].find('czi:PositionY', ns)
                    if pos_x_node is None or pos_y_node is None:
                        raise ValueError(f"Could not find position for Scene {i}.")
                    pos_x_pixels = int(float(pos_x_node.text) / scale_x)
                    pos_y_pixels = int(float(pos_y_node.text) / scale_x)
                    scene_positions_pixels.append((pos_x_pixels, pos_y_pixels))

                # --- Calculate Canvas and Stitch ---
                min_x = min(p[0] for p in scene_positions_pixels)
                min_y = min(p[1] for p in scene_positions_pixels)
                max_x, max_y = 0, 0
                y_idx, x_idx = scene_axes_str.find('Y'), scene_axes_str.find('X')
                scene_shape = [full_data.shape[j] for j, ax in enumerate(axes) if ax != 'S']
                for pos in scene_positions_pixels:
                    max_x = max(max_x, pos[0] + scene_shape[x_idx])
                    max_y = max(max_y, pos[1] + scene_shape[y_idx])

                stitched_width, stitched_height = max_x - min_x, max_y - min_y
                
                c_axis_index = scene_axes_str.find('C')
                if c_axis_index != -1:
                    num_channels = scene_shape[c_axis_index]
                    stitched_shape = (num_channels, stitched_height, stitched_width)
                    stitched_axes = 'CYX'
                else:
                    num_channels = 1
                    stitched_shape = (stitched_height, stitched_width)
                    stitched_axes = 'YX'

                print(f"Creating stitched canvas with dimensions (H, W): ({stitched_height}, {stitched_width})")
                canvas = np.zeros(stitched_shape, dtype=full_data.dtype)

                for i in range(num_scenes):
                    scene_data = full_data.take(i, axis=s_index) if s_index is not None else full_data
                    pos_x, pos_y = scene_positions_pixels[i]
                    paste_x, paste_y = pos_x - min_x, pos_y - min_y
                    squeezed_data = np.squeeze(scene_data)
                    if 'C' in stitched_axes:
                        h, w = squeezed_data.shape[1], squeezed_data.shape[2]
                        canvas[:, paste_y:paste_y+h, paste_x:paste_x+w] = squeezed_data
                    else:
                        h, w = squeezed_data.shape[0], squeezed_data.shape[1]
                        canvas[paste_y:paste_y+h, paste_x:paste_x+w] = squeezed_data
                    del scene_data
                    del squeezed_data
                    gc.collect()
                
                # --- Save Stitched Image ---
                final_shape = (num_channels, 1, stitched_height, stitched_width)
                ome_metadata = create_ome_metadata(final_shape, 'CZYX', czi_metadata_xml, output_path.name)
                ome_xml = ome_metadata.to_xml() if ome_metadata else None
                if ome_xml: ome_xml = ome_xml.replace('µm', 'um')
                photometric = 'rgb' if 'C' in stitched_axes and num_channels == 3 else 'minisblack'

                print(f"Writing stitched OME-TIFF file: {output_path}")
                # Remove the complex 'description' tag to ensure TIFF compliance
                with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
                    tif.write(canvas, photometric=photometric, metadata={'axes': stitched_axes.replace('C', 'S')})
            
            # --- BRANCH: IN-MEMORY MOSAIC (fallback) ---
            else:
                if num_scenes > 1:
                    print("!!! WARNING: CZI library version requires loading all scenes into memory.", file=sys.stderr)
                    print("!!! Forcing an in-memory mosaic of all scenes (last scene on top).", file=sys.stderr)

                full_data = load_full_data()
                if 'S' in axes:
                    s_index = axes.find('S')
                else:
                    s_index = None
                print("  - Scene data loaded.")

                # --- Special case: single scene, write directly without extra canvas ---
                if num_scenes == 1:
                    scene_axes_str = axes.replace('S', '')
                    canvas, canvas_axes = normalize_scene_to_canvas(full_data, scene_axes_str)
                    num_channels = canvas.shape[0] if canvas_axes == 'CYX' else 1

                    final_shape = (num_channels, 1, canvas.shape[-2], canvas.shape[-1])
                    ome_metadata = create_ome_metadata(final_shape, 'CZYX', czi_metadata_xml, output_path.name)
                    ome_xml = ome_metadata.to_xml() if ome_metadata else None
                    if ome_xml:
                        ome_xml = ome_xml.encode('ascii', 'replace').decode('ascii')
                    photometric = 'rgb' if canvas_axes == 'CYX' and num_channels == 3 else 'minisblack'

                    print(f"Writing mosaicked TIFF file: {output_path}")
                    tifffile.imwrite(
                        output_path,
                        canvas,
                        bigtiff=True,
                        photometric=photometric,
                        metadata={'axes': canvas_axes.replace('C', 'S')}
                    )

                    del canvas
                    gc.collect()
                    return

                # --- PASS 1: Inspect all scenes (from memory) to determine final canvas dimensions ---
                print("  - Inspecting scene dimensions...")
                max_height = 0
                max_width = 0
                max_channels = 1
                scene_axes_str = axes.replace('S', '')
                original_dtype = full_data.dtype

                for i in range(num_scenes):
                    scene_data = full_data.take(i, axis=s_index) if s_index is not None else full_data
                    normalized_scene, normalized_axes = normalize_scene_to_canvas(scene_data, scene_axes_str)

                    if normalized_axes == 'CYX':
                        max_channels = max(max_channels, normalized_scene.shape[0])
                        max_height = max(max_height, normalized_scene.shape[1])
                        max_width = max(max_width, normalized_scene.shape[2])
                    else:
                        max_height = max(max_height, normalized_scene.shape[0])
                        max_width = max(max_width, normalized_scene.shape[1])

                    del scene_data
                    del normalized_scene
                    gc.collect()

                # --- Create the master canvas shape ---
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
                
                # --- Create OME-XML for the final file ---
                final_shape = (num_channels, 1, max_height, max_width)
                ome_metadata = create_ome_metadata(final_shape, 'CZYX', czi_metadata_xml, output_path.name)
                ome_xml = ome_metadata.to_xml() if ome_metadata else None
                if ome_xml:
                    # Sanitize the entire XML string to prevent tag sorting errors
                    ome_xml = ome_xml.encode('ascii', 'replace').decode('ascii')
                photometric = 'rgb' if 'C' in canvas_axes and num_channels == 3 else 'minisblack'

                # --- PASS 2: Write all scenes (from memory) directly to the in-memory canvas ---
                print("  - Mosaicking scenes to canvas (last on top)...")
                for i in range(num_scenes):
                    scene_data = full_data.take(i, axis=s_index) if s_index is not None else full_data
                    normalized_scene, normalized_axes = normalize_scene_to_canvas(scene_data, scene_axes_str)

                    if num_channels > 1:
                        if normalized_axes == 'CYX':
                            h, w = normalized_scene.shape[1], normalized_scene.shape[2]
                            mask = np.any(normalized_scene > 0, axis=0)
                            for c in range(min(num_channels, normalized_scene.shape[0])):
                                canvas[c, :h, :w] = np.where(mask, normalized_scene[c], canvas[c, :h, :w])
                        else:
                            h, w = normalized_scene.shape[0], normalized_scene.shape[1]
                            mask = normalized_scene > 0
                            for c in range(num_channels):
                                canvas[c, :h, :w] = np.where(mask, normalized_scene, canvas[c, :h, :w])
                    else:
                        if normalized_axes == 'CYX':
                            grayscale = np.max(normalized_scene, axis=0)
                            h, w = grayscale.shape
                            mask = grayscale > 0
                            canvas[:h, :w] = np.where(mask, grayscale, canvas[:h, :w])
                        else:
                            h, w = normalized_scene.shape
                            mask = normalized_scene > 0
                            canvas[:h, :w] = np.where(mask, normalized_scene, canvas[:h, :w])

                    del scene_data
                    del normalized_scene
                    gc.collect()

                # --- Write the final canvas to disk in a single, robust operation ---
                print(f"Writing mosaicked TIFF file: {output_path}")
                # Remove the complex 'description' tag to ensure TIFF compliance
                tifffile.imwrite(
                    output_path,
                    canvas,
                    bigtiff=True,
                    photometric=photometric,
                    metadata={'axes': canvas_axes.replace('C', 'S')}
                )

        print(f"Conversion successful: {output_path}")

        if memmap_path and memmap_path.exists():
            try:
                memmap_path.unlink()
            except OSError:
                pass
    except Exception as e:
        if "can not read ZISRAW segment" in str(e):
            print(f"Skipping corrupted CZI: {input_path.name}", file=sys.stderr)
            return
        print(f"An error occurred during conversion of {input_path}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        if memmap_path and memmap_path.exists():
            try:
                memmap_path.unlink()
            except OSError:
                pass


def main():
    """Main function to parse arguments and run the conversion."""
    parser = argparse.ArgumentParser(
        description="Convert Zeiss CZI files in a directory to OME-TIFF format, organizing them into a specific folder structure."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        default=None,
        nargs='?',
        help="The root directory to search for CZI files. Defaults to INPUT_DATA_DIR env var or a specific hardcoded path."
    )
    parser.add_argument(
        "--base_output_dir",
        type=str,
        default="/fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_GCP/Pipeline2",
        help="The base output directory for the pipeline. Defaults to /fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_GCP/Pipeline2"
    )
    args = parser.parse_args()

    # Determine input directory: Argument > Env Var > Default
    if args.input_dir:
        input_root = Path(args.input_dir)
    elif os.environ.get("INPUT_DATA_DIR"):
        input_root = Path(os.environ["INPUT_DATA_DIR"])
    else:
        input_root = Path("/fslustre/qhs/ext_chen_yuheng_mayo_edu/decompress_files/temp_decompress_czi_194542/RAW")

    pipeline_base = Path(args.base_output_dir)
    stain_map = {
        "amyloid": "RO1_Amyloid",
        "tau": "RO1_Tau",
        "iron": "RO1_Iron"
    }
    # Define ROIs and staining keywords for rearrangement
    rois = ["mfg", "put", "bs"]
    staining_keywords = {"amyloid": "amyloid", "tau": "cp13", "iron": "iron"}


    if not input_root.is_dir():
        print(f"Error: Input directory not found at {input_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Starting CZI to TIFF conversion process for directory: {input_root}")

    for czi_path in input_root.rglob('*.czi'):
        print(f"\nProcessing file: {czi_path}")
        
        # 1. Parse filename
        stem = czi_path.stem
        parts = stem.split('_')
        
        # Regex to find case number (e.g., 2318_22) and staining
        case_match = re.match(r'(\d+_\d+)', stem)
        if not case_match:
            print(f"  - Skipping: Could not determine case number from '{stem}'")
            continue
        case_number = case_match.group(1)

        staining = "unknown"
        # Normalize the stem for matching: replace '-' and other separators with '_'
        stem_normalized = re.sub(r'[-\s]+', '_', stem.lower())
        # Check for CP13 first and assign to Tau group
        if "cp13" in stem_normalized:
            staining = "tau"
        else:
            for key in stain_map:
                if f"_{key.lower()}_" in f"_{stem_normalized}_":
                    staining = key
                    break
        
        if staining == "unknown":
            print(f"  - Skipping: Could not determine staining (Amyloid, Tau, Iron, or CP13) from '{stem}'")
            continue

        # 2. Determine output directory and slice folder name
        stain_folder = stain_map.get(staining)
        if not stain_folder:
            print(f"  - Skipping: No mapping for staining '{staining}'")
            continue
            
        # Rearrange filename if staining comes before ROI
        slice_folder_name = stem
        stain_keyword_in_stem = staining_keywords.get(staining)
        roi_in_stem = next((roi for roi in rois if f"_{roi}_" in f"_{stem_normalized}_"), None)

        if stain_keyword_in_stem and roi_in_stem:
            # Find the original case-sensitive parts from the filename
            parts = stem.split('_')
            original_stain_part = next((p for p in parts if p.lower() == stain_keyword_in_stem), None)
            original_roi_part = next((p for p in parts if p.lower() == roi_in_stem), None)

            if original_stain_part and original_roi_part:
                # Check if the staining part appears before the ROI part
                if stem.find(original_stain_part) < stem.find(original_roi_part):
                    print(f"  - Rearranging filename: '{original_stain_part}' and '{original_roi_part}'")
                    # Swap the parts to create the new folder name
                    temp_name = stem.replace(original_stain_part, "___ROI_PLACEHOLDER___")
                    temp_name = temp_name.replace(original_roi_part, original_stain_part)
                    slice_folder_name = temp_name.replace("___ROI_PLACEHOLDER___", original_roi_part)

        # 3. Construct final output path and create directories
        try:
            base_slice_dir = pipeline_base / stain_folder / "Cases" / case_number / slice_folder_name
            files_dir = base_slice_dir / f"{slice_folder_name}_files"
            
            # Create all required directories
            output_dir = files_dir / "output"
            output_res_dir = output_dir / "RES"
            mask_dir = files_dir / "mask" / "final_mask"
            heatmap_dir = files_dir / "heatmap"
            
            output_res_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)
            heatmap_dir.mkdir(parents=True, exist_ok=True)
            
            # Define log file for the resize script
            resize_log_path = base_slice_dir / "resize_script.log"

            output_filename = output_res_dir / f"{slice_folder_name}.tiff"
            resized_filename = output_res_dir / f"res10_{slice_folder_name}.tiff"
            
            print(f"  - Case: {case_number}, Staining: {staining}")
            print(f"  - Slice Folder: {slice_folder_name}")

            # 4. Check if already converted (both full-res and resized exist)
            if output_filename.exists() and resized_filename.exists():
                print(f"  - Skipping: Already converted (TIFF exists at {output_filename})")
                continue
            elif output_filename.exists():
                print(f"  - Resuming: Full-res TIFF exists, but resized version missing")
                # Skip conversion, but still submit resize job below
            else:
                # Run conversion
                print(f"  - Converting CZI to TIFF: {output_filename}")
                convert_czi_to_ome_tiff(czi_path, output_filename)

            # 5. Run resize script if the full-resolution TIFF exists and resized doesn't.
            if not output_filename.exists():
                print(f"  - Original TIFF not found after conversion attempt. Cannot run resize script.", file=sys.stderr)
            elif resized_filename.exists():
                print(f"  - Resized TIFF already exists, skipping resize job")
            else:
                print(f"  - Submitting resize task to SLURM for directory: {output_dir}")
                resize_script_path = "/fslustre/qhs/ext_chen_yuheng_mayo_edu/high-res-3D-tau/slurm_scripts/run_convert.sh"
                
                if os.path.exists(resize_script_path):
                    try:
                        # Submit the resize script as a new SLURM job
                        # The output and error files will be determined by the #SBATCH directives in the script
                        sbatch_command = [
                            "sbatch",
                            resize_script_path,
                            str(output_dir) # Pass the directory as an argument
                        ]
                        
                        result = subprocess.run(
                            sbatch_command,
                            capture_output=True,
                            text=True,
                            check=True
                        )
                        
                        print(f"    - SLURM job submitted successfully. Job ID: {result.stdout.strip()}")
                        print(f"    - Monitor job with 'squeue -u $USER' or check logs in the specified output directory.")

                    except subprocess.CalledProcessError as e:
                        print(f"  - Error submitting SLURM job for resize script.", file=sys.stderr)
                        print(f"  - Return Code: {e.returncode}", file=sys.stderr)
                        print(f"  - sbatch stdout: {e.stdout}", file=sys.stderr)
                        print(f"  - sbatch stderr: {e.stderr}", file=sys.stderr)
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