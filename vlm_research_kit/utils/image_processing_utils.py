import pydicom
from pydicom.pixels import apply_voi_lut
import numpy as np
from PIL import Image
import os

# Adapted from: https://www.kaggle.com/code/mrutyunjaybiswal/vbd-chest-x-ray-abnormalities-detection-eda
def dicom_to_jpeg_high_quality(
    dicom_path,
    output_path=None,
    voi_lut=True,
    fix_monochrome=True,
    quality=95,
    verbose=True,
):
    """
    Converts a DICOM file to a high-quality JPEG image, with optional
    detailed print-based debugging at each step.
    """

    def log(msg):
        if verbose:
            print(msg)

    log("\n=== Starting DICOM to JPEG Conversion ===")
    log(f"Input DICOM: {dicom_path}")

    # Step 1: Read DICOM
    dcm_data = pydicom.dcmread(dicom_path)
    pixel_array = dcm_data.pixel_array
    log(f"[1] DICOM read. Pixel array shape: {pixel_array.shape}, dtype: {pixel_array.dtype}")
    log(f"    PhotometricInterpretation: {getattr(dcm_data, 'PhotometricInterpretation', 'N/A')}")

    # Step 2: Apply VOI LUT if available
    if voi_lut:
        try:
            data = apply_voi_lut(pixel_array, dcm_data)
            log("[2] Applied VOI LUT successfully.")
            log(f"    dtype: {data.dtype}, min={np.min(data)}, max={np.max(data)}")
        except Exception as e:
            log(f"[2] Could not apply VOI LUT: {e}")
            data = pixel_array
    else:
        data = pixel_array
        log("[2] VOI LUT skipped by configuration.")

    # Step 3: Handle MONOCHROME1 inversion
    if (
        fix_monochrome
        and hasattr(dcm_data, "PhotometricInterpretation")
        and dcm_data.PhotometricInterpretation == "MONOCHROME1"
    ):
        log("[3] Correcting MONOCHROME1 inversion.")
        max_val = np.max(data)
        data = max_val - data
        log(f"    After inversion: min={np.min(data)}, max={np.max(data)}")
    else:
        log("[3] No inversion applied.")

    # Step 4: Normalize pixel data to 0–255
    data = data.astype(np.float32)
    min_val, max_val = np.min(data), np.max(data)
    log(f"[4] Before normalization: min={min_val}, max={max_val}")
    if max_val > min_val:
        data = (data - min_val) / (max_val - min_val)
    else:
        data = np.zeros_like(data)
        log("    Warning: uniform pixel array detected (min == max).")
    data_uint8 = (data * 255).astype(np.uint8)
    log(f"    After normalization: dtype={data_uint8.dtype}, min={data_uint8.min()}, max={data_uint8.max()}")

    # Step 5: Convert to grayscale PIL image
    im = Image.fromarray(data_uint8, mode="L")
    log(f"[5] Created PIL image: size={im.size}, mode={im.mode}")

    # Step 6: Save as JPEG or return
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        im.save(
            output_path,
            format="JPEG",
            quality=quality,
            optimize=True,
            subsampling=0,
        )
        log(f"[6] Saved to {output_path} with quality={quality}")
    else:
        log("[6] Output path not provided, returning PIL Image.")

    log("=== Conversion complete ===\n")
    return im