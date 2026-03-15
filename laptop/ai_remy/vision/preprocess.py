"""
Resize and compress image for Gemini API (no OpenCV).
Accepts: image bytes (JPEG/PNG) or numpy array (H, W, 3) BGR or RGB.
Returns: JPEG bytes suitable for API.
"""
from io import BytesIO
from typing import Union

import numpy as np
from PIL import Image

from ai_remy import config


def preprocess_frame(
    image_input: Union[bytes, np.ndarray],
    max_size: int = None,
    quality: int = None,
) -> bytes:
    """Convert frame to resized, compressed JPEG bytes."""
    max_size = max_size or config.MAX_IMAGE_PX
    quality = quality or config.JPEG_QUALITY

    if isinstance(image_input, bytes):
        img = Image.open(BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        # Assume (H, W, 3); OpenCV is BGR, PIL is RGB
        if image_input.shape[2] == 3:
            img = Image.fromarray(image_input[:, :, ::-1], "RGB")
        else:
            img = Image.fromarray(image_input, "RGB")
    else:
        raise TypeError("image_input must be bytes or numpy array")

    # Resize by longest edge
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()
