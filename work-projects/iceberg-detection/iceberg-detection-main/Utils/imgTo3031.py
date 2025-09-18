import rasterio
import numpy as np


def imgTo3031(original_img, processed_img, output_path):
    """Transforms the processed image (an array) to a .tif file,
    with the same metadata as the original image, usually in EPSG:3031,
    and saves it to the output path"""
    output = processed_img.copy()
    output = np.float32(output)
    with rasterio.open(original_img) as src:
        out_meta = src.meta.copy()
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(output, 1)
        dest.close()
    src.close()
