from io import BytesIO
from base64 import b64encode
from typing import IO, Union, Tuple
import numpy as np
from PIL import Image


def open_image(image_fp: IO[bytes]) -> Image.Image:
    image_fp.seek(0)
    return Image.open(image_fp).convert("RGB")

def _image_b64encode(image: Image.Image) -> str:
    with BytesIO() as io:
        image.save(io, format="png", quality=100)
        io.seek(0)
        return b64encode(io.read()).decode()


def image_to_img_src(image: Image.Image) -> str:
    return f"data:image/png;base64,{_image_b64encode(image)}"


def open_mask(file: IO[bytes]) -> Image.Image:
    mask = Image.open(file).convert("L")
    mask = Image.eval(mask, lambda x: 255 - x)
    return mask


def apply_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    image_np = np.array(image)
    mask_np = np.array(mask)
    mask_np = (mask_np > 128).astype(np.uint8)
    masked_image = image_np * mask_np[:, :, None]
    return Image.fromarray(masked_image.astype(np.uint8))


def crop_image_by_mask(
    image: Image.Image,
    mask: Image.Image
) -> Image.Image:
    mask_np = np.array(mask)
    non_zero = np.argwhere(mask_np)

    if non_zero.size == 0:
        raise ValueError("Маска не содержит ненулевых пикселей")

    top_left = non_zero.min(axis=0)
    bottom_right = non_zero.max(axis=0) + 1

    y1, x1 = top_left
    y2, x2 = bottom_right

    return image.crop((x1, y1, x2, y2))