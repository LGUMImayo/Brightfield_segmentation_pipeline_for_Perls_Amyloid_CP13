import sys
import tifffile

def inspect_tiff(file_path):
    print(f"Inspecting: {file_path}")
    try:
        with tifffile.TiffFile(file_path) as tif:
            print(f"Is BigTIFF: {tif.is_bigtiff}")
            print(f"Byte order: {tif.byteorder}")
            print(f"Number of pages: {len(tif.pages)}")
            
            for i, page in enumerate(tif.pages):
                print(f"Page {i}:")
                print(f"  Shape: {page.shape}")
                print(f"  Dtype: {page.dtype}")
                print(f"  Compression: {page.compression}")
                print(f"  Tags: {page.tags.keys()}")
                if i > 2:
                    print("  ... (stopping after first few pages)")
                    break
            
            if len(tif.series) > 0:
                print(f"Number of series: {len(tif.series)}")
                for i, series in enumerate(tif.series):
                    print(f"Series {i}:")
                    print(f"  Shape: {series.shape}")
                    print(f"  Dtype: {series.dtype}")
                    print(f"  Axes: {series.axes}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_tiff.py <file_path>")
        sys.exit(1)
    inspect_tiff(sys.argv[1])
