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


def convert_czi_to_ome_tiff(input_path: Path, output_path: Path):
    """
    Converts a CZI file to an OME-TIFF file.
    """
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}", file=sys.stderr)
        return

    print(f"Reading CZI file: {input_path}")
    try:
        with czifile.CziFile(input_path) as czi:
            img_array = czi.asarray()
            czi_metadata_xml = czi.metadata()
            shape = czi.shape
            axes = czi.axes

        print(f"Image shape: {shape}")
        print(f"Image axes: {axes}")

        # Handle cases where RGB is the last dimension (e.g., '...X0')
        if axes.endswith('0'):
            print("Detected RGB data in the last dimension. Reordering axes for TIFF.")
            # Find the index of the '0' dimension
            rgb_axis_index = axes.find('0')
            # Move the RGB axis to be the channel axis 'C'
            img_array = np.moveaxis(img_array, rgb_axis_index, axes.find('C'))
            # Update axes string to reflect the change for metadata
            axes = axes.replace('0', '') # Remove the '0'
            print(f"New shape: {img_array.shape}")

        # Find the scene dimension
        scene_axis_index = axes.find('S')
        if scene_axis_index == -1:
             # If no scene dimension, treat as a single scene
            num_scenes = 1
            img_array = np.expand_dims(img_array, axis=0) # Add a dummy scene axis
        else:
            num_scenes = img_array.shape[scene_axis_index]

        print(f"Found {num_scenes} scene(s). Writing to OME-TIFF.")

        # Create OME-XML metadata string
        ome_metadata = create_ome_metadata(img_array.shape, axes, czi_metadata_xml, output_path.name)
        ome_xml = ome_metadata.to_xml() if ome_metadata else None

        # Sanitize OME-XML for TIFF ASCII compatibility AFTER generation
        if ome_xml:
            ome_xml = ome_xml.replace('µm', 'um')

        print(f"Writing OME-TIFF file: {output_path}")
        # Write the array to an OME-TIFF file
        with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
            for i in range(num_scenes):
                # Get the data for the current scene
                scene_data = img_array.take(i, axis=scene_axis_index)
                scene_axes = axes.replace('S', '')
                
                # Squeeze out singleton dimensions and update axes string accordingly
                squeezed_data = scene_data
                squeezed_axes = scene_axes
                # Iterate backwards to not mess up indices during removal
                for axis_idx in range(len(scene_axes) - 1, -1, -1):
                    if scene_data.shape[axis_idx] == 1:
                        squeezed_data = np.squeeze(squeezed_data, axis=axis_idx)
                        squeezed_axes = squeezed_axes[:axis_idx] + squeezed_axes[axis_idx+1:]

                # Determine photometric interpretation
                photometric = 'rgb' if 'C' in squeezed_axes and squeezed_data.shape[squeezed_axes.find('C')] == 3 else 'minisblack'

                # For the first image, write the full OME-XML metadata
                # For subsequent images, append them without metadata
                if i == 0:
                    tif.write(
                        squeezed_data,
                        photometric=photometric,
                        description=ome_xml,
                        metadata={'axes': squeezed_axes}
                    )
                else:
                    tif.write(
                        squeezed_data,
                        photometric=photometric,
                        subfiletype=0
                    )

        print(f"Conversion successful: {output_path}")

    except Exception as e:
        print(f"An error occurred during conversion of {input_path}: {e}", file=sys.stderr)


def main():
    """Main function to parse arguments and run the conversion."""
    parser = argparse.ArgumentParser(
        description="Convert Zeiss CZI files in a directory to OME-TIFF format, organizing them into a specific folder structure."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        default="/fslustre/qhs/ext_chen_yuheng_mayo_edu/decompress_files/temp_decompress_czi_112640",
        nargs='?',
        help="The root directory to search for CZI files. Defaults to /fslustre/qhs/ext_chen_yuheng_mayo_edu/decompress_files/temp_decompress_czi_112640"
    )
    parser.add_argument(
        "--base_output_dir",
        type=str,
        default="/fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_GCP/Pipeline",
        help="The base output directory for the pipeline. Defaults to /fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_GCP/Pipeline"
    )
    args = parser.parse_args()

    input_root = Path(args.input_dir)
    pipeline_base = Path(args.base_output_dir)
    stain_map = {
        "amyloid": "RO1_Amyloid",
        "tau": "RO1_Tau",
        "iron": "RO1_Iron"
    }

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
        for key in stain_map:
            if f"_{key.lower()}_" in f"_{stem.lower()}_":
                staining = key
                break
        
        if staining == "unknown":
            print(f"  - Skipping: Could not determine staining (Amyloid, Tau, Iron) from '{stem}'")
            continue

        # 2. Determine output directory
        stain_folder = stain_map.get(staining)
        if not stain_folder:
            print(f"  - Skipping: No mapping for staining '{staining}'")
            continue
            
        # Create slice folder name by removing staining
        slice_folder_name = stem.replace(f'_{staining.capitalize()}_', '_').replace(f'_{staining.upper()}_', '_').replace(f'_{staining.lower()}_', '_')

        # 3. Construct final output path and create directories
        try:
            output_dir = pipeline_base / stain_folder / "Cases" / case_number / slice_folder_name / f"{slice_folder_name}_files" / "output" / "RES"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_filename = output_dir / f"{slice_folder_name}.tiff"
            
            print(f"  - Case: {case_number}, Staining: {staining}")
            print(f"  - Output will be saved to: {output_filename}")

            # 4. Run conversion
            convert_czi_to_ome_tiff(czi_path, output_filename)