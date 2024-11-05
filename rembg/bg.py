import io
from enum import Enum
from typing import Any, List, Optional, Tuple, Union, cast, Dict
import base64
import numpy as np
import onnxruntime as ort
from cv2 import (
    BORDER_DEFAULT,
    MORPH_ELLIPSE,
    MORPH_OPEN,
    GaussianBlur,
    getStructuringElement,
    morphologyEx,
)
from PIL import Image, ImageOps
from PIL.Image import Image as PILImage
from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml
from pymatting.util.util import stack_images
from scipy.ndimage import binary_erosion

from .session_factory import new_session
from .sessions import sessions_class
from .sessions.base import BaseSession

ort.set_default_logger_severity(3)

kernel = getStructuringElement(MORPH_ELLIPSE, (3, 3))


class ReturnType(Enum):
    BYTES = 0
    PILLOW = 1
    NDARRAY = 2


def alpha_matting_cutout(
    img: PILImage,
    mask: PILImage,
    foreground_threshold: int,
    background_threshold: int,
    erode_structure_size: int,
) -> PILImage:
    """
    Perform alpha matting on an image using a given mask and threshold values.

    This function takes a PIL image `img` and a PIL image `mask` as input, along with
    the `foreground_threshold` and `background_threshold` values used to determine
    foreground and background pixels. The `erode_structure_size` parameter specifies
    the size of the erosion structure to be applied to the mask.

    The function returns a PIL image representing the cutout of the foreground object
    from the original image.
    """
    if img.mode == "RGBA" or img.mode == "CMYK":
        img = img.convert("RGB")

    img_array = np.asarray(img)
    mask_array = np.asarray(mask)

    is_foreground = mask_array > foreground_threshold
    is_background = mask_array < background_threshold

    structure = None
    if erode_structure_size > 0:
        structure = np.ones(
            (erode_structure_size, erode_structure_size), dtype=np.uint8
        )

    is_foreground = binary_erosion(is_foreground, structure=structure)
    is_background = binary_erosion(is_background, structure=structure, border_value=1)

    trimap = np.full(mask_array.shape, dtype=np.uint8, fill_value=128)
    trimap[is_foreground] = 255
    trimap[is_background] = 0

    img_normalized = img_array / 255.0
    trimap_normalized = trimap / 255.0

    alpha = estimate_alpha_cf(img_normalized, trimap_normalized)
    foreground = estimate_foreground_ml(img_normalized, alpha)
    cutout = stack_images(foreground, alpha)

    cutout = np.clip(cutout * 255, 0, 255).astype(np.uint8)
    cutout = Image.fromarray(cutout)

    return cutout


def naive_cutout(img: PILImage, mask: PILImage) -> PILImage:
    """
    Perform a simple cutout operation on an image using a mask.

    This function takes a PIL image `img` and a PIL image `mask` as input.
    It uses the mask to create a new image where the pixels from `img` are
    cut out based on the mask.

    The function returns a PIL image representing the cutout of the original
    image using the mask.
    """
    empty = Image.new("RGBA", (img.size), 0)
    cutout = Image.composite(img, empty, mask)
    return cutout


def putalpha_cutout(img: PILImage, mask: PILImage) -> PILImage:
    """
    Apply the specified mask to the image as an alpha cutout.

    Args:
        img (PILImage): The image to be modified.
        mask (PILImage): The mask to be applied.

    Returns:
        PILImage: The modified image with the alpha cutout applied.
    """
    img.putalpha(mask)
    return img


def get_concat_v_multi(imgs: List[PILImage]) -> PILImage:
    """
    Concatenate multiple images vertically.

    Args:
        imgs (List[PILImage]): The list of images to be concatenated.

    Returns:
        PILImage: The concatenated image.
    """
    pivot = imgs.pop(0)
    for im in imgs:
        pivot = get_concat_v(pivot, im)
    return pivot


def get_concat_v(img1: PILImage, img2: PILImage) -> PILImage:
    """
    Concatenate two images vertically.

    Args:
        img1 (PILImage): The first image.
        img2 (PILImage): The second image to be concatenated below the first image.

    Returns:
        PILImage: The concatenated image.
    """
    dst = Image.new("RGBA", (img1.width, img1.height + img2.height))
    dst.paste(img1, (0, 0))
    dst.paste(img2, (0, img1.height))
    return dst


def post_process(mask: np.ndarray) -> np.ndarray:
    """
    Post Process the mask for a smooth boundary by applying Morphological Operations
    Research based on paper: https://www.sciencedirect.com/science/article/pii/S2352914821000757
    args:
        mask: Binary Numpy Mask
    """
    mask = morphologyEx(mask, MORPH_OPEN, kernel)
    mask = GaussianBlur(mask, (5, 5), sigmaX=2, sigmaY=2, borderType=BORDER_DEFAULT)
    mask = np.where(mask < 127, 0, 255).astype(np.uint8)  # type: ignore
    return mask


def apply_background_color(img: PILImage, color: Tuple[int, int, int, int]) -> PILImage:
    """
    Apply the specified background color to the image.

    Args:
        img (PILImage): The image to be modified.
        color (Tuple[int, int, int, int]): The RGBA color to be applied.

    Returns:
        PILImage: The modified image with the background color applied.
    """
    r, g, b, a = color
    colored_image = Image.new("RGBA", img.size, (r, g, b, a))
    colored_image.paste(img, mask=img)

    return colored_image


def fix_image_orientation(img: PILImage) -> PILImage:
    """
    Fix the orientation of the image based on its EXIF data.

    Args:
        img (PILImage): The image to be fixed.

    Returns:
        PILImage: The fixed image.
    """
    return cast(PILImage, ImageOps.exif_transpose(img))


def download_models() -> None:
    """
    Download models for image processing.
    """
    for session in sessions_class:
        session.download_models()


def remove(
    data: Union[bytes, PILImage, np.ndarray],
    alpha_matting: bool = False,
    alpha_matting_foreground_threshold: int = 240,
    alpha_matting_background_threshold: int = 10,
    alpha_matting_erode_size: int = 10,
    session: Optional[BaseSession] = None,
    only_mask: bool = False,
    post_process_mask: bool = False,
    bgcolor: Optional[Tuple[int, int, int, int]] = None,
    force_return_bytes: bool = False,
    *args: Optional[Any],
    **kwargs: Optional[Any]
) -> Dict[str, Any]:
    """
    Remove the background from an input image and return the masks, their centers, and the base64-encoded image.

    Parameters:
        data (Union[bytes, PILImage, np.ndarray]): The input image data.
        alpha_matting (bool, optional): Flag indicating whether to use alpha matting. Defaults to False.
        alpha_matting_foreground_threshold (int, optional): Foreground threshold for alpha matting. Defaults to 240.
        alpha_matting_background_threshold (int, optional): Background threshold for alpha matting. Defaults to 10.
        alpha_matting_erode_size (int, optional): Erosion size for alpha matting. Defaults to 10.
        session (Optional[BaseSession], optional): A session object for the 'u2net' model. Defaults to None.
        only_mask (bool, optional): Flag indicating whether to return only the binary masks. Defaults to False.
        post_process_mask (bool, optional): Flag indicating whether to post-process the masks. Defaults to False.
        bgcolor (Optional[Tuple[int, int, int, int]], optional): Background color for the cutout image. Defaults to None.
        force_return_bytes (bool, optional): Flag indicating whether to return the cutout image as bytes. Defaults to False.
        *args (Optional[Any]): Additional positional arguments.
        **kwargs (Optional[Any]): Additional keyword arguments.

    Returns:
        Dict[str, Any]: A dictionary containing the list of masks, their center coordinates, and the base64-encoded image.
    """
    if isinstance(data, bytes) or force_return_bytes:
        return_type = ReturnType.BYTES
        img = cast(PILImage, Image.open(io.BytesIO(cast(bytes, data))))
    elif isinstance(data, PILImage):
        return_type = ReturnType.PILLOW
        img = cast(PILImage, data)
    elif isinstance(data, np.ndarray):
        return_type = ReturnType.NDARRAY
        img = cast(PILImage, Image.fromarray(data))
    else:
        raise ValueError(
            "Input type {} is not supported. Try using force_return_bytes=True to force python bytes output".format(
                type(data)
            )
        )

    putalpha = kwargs.pop("putalpha", False)
    img = fix_image_orientation(img)

    if session is None:
        session = new_session("u2net", *args, **kwargs)

    masks = session.predict(img, *args, **kwargs)
    cutouts = []
    mask_images = []  # To store the individual mask images
    mask_centers = []  # To store the center coordinates of each mask

    for mask in masks:
        if post_process_mask:
            mask = Image.fromarray(post_process(np.array(mask)))

        mask_images.append(mask)  # Collect each mask

        # Get bounding box and center of the mask
        mask_array = np.array(mask)
        y_indices, x_indices = np.where(mask_array > 0)  # Get non-zero points
        if y_indices.size > 0 and x_indices.size > 0:
            min_x, max_x = x_indices.min(), x_indices.max()
            min_y, max_y = y_indices.min(), y_indices.max()
            center_x = (min_x + max_x) // 2
            center_y = (min_y + max_y) // 2
            mask_centers.append({"x": center_x, "y": center_y})
        else:
            mask_centers.append({"x": None, "y": None})

        if only_mask:
            cutout = mask
        elif alpha_matting:
            try:
                cutout = alpha_matting_cutout(
                    img,
                    mask,
                    alpha_matting_foreground_threshold,
                    alpha_matting_background_threshold,
                    alpha_matting_erode_size,
                )
            except ValueError:
                if putalpha:
                    cutout = putalpha_cutout(img, mask)
                else:
                    cutout = naive_cutout(img, mask)
        else:
            if putalpha:
                cutout = putalpha_cutout(img, mask)
            else:
                cutout = naive_cutout(img, mask)

        cutouts.append(cutout)

    cutout = img
    if len(cutouts) > 0:
        cutout = get_concat_v_multi(cutouts)

    if bgcolor is not None and not only_mask:
        cutout = apply_background_color(cutout, bgcolor)

    # Convert final cutout image to base64
    bio = io.BytesIO()
    cutout.save(bio, "PNG")
    bio.seek(0)
    base64_image = base64.b64encode(bio.read()).decode('utf-8')

    # Return the masks, centers, and base64-encoded cutout image
    return {
        "masks": mask_images,
        "mask_centers": mask_centers,
        "image_base64": base64_image
    }
