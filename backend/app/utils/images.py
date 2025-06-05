from io import BytesIO
from base64 import b64encode
from typing import IO
import numpy as np
from PIL import Image  # <--- исправлено

def open_image(image_fp: IO[bytes]) -> Image.Image:
    image_fp.seek(0)
    return Image.open(image_fp).convert("RGB")

def _image_b64encode(image: Image.Image) -> str:
    with BytesIO() as io:
        image.save(io, format="png", quality=100)
        io.seek(0)
        return b64encode(io.read()).decode()

def image_to_img_src(image: Image.Image) -> str:
    return f'data:image/png;base64,{_image_b64encode(image)}'

def open_mask(file) -> Image.Image:
    """
    Opens a mask image and ensures it's in the correct format:
    - Converts to grayscale
    - Inverts if necessary (since we're using inverted masks in segmentation)
    - Returns a PIL Image
    """
    mask = Image.open(file).convert('L')  # Convert to grayscale
    # Invert the mask since we're using inverted masks in segmentation
    mask = Image.eval(mask, lambda x: 255 - x)
    return mask

def apply_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    image_np = np.array(image)
    mask_np = np.array(mask)

    mask_np = (mask_np > 128).astype(np.uint8)

    masked_image = image_np * mask_np[:, :, None]

    return Image.fromarray(masked_image.astype(np.uint8))
