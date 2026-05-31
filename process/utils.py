from typing import List

import cv2
import numpy as np
from pathlib import Path


def get_image_files(directory: Path, extensions: tuple = ('.jpg', '.jpeg', '.png')) -> List[Path]:
    if not directory.exists():
        return []
    return [
        f for f in directory.rglob('*')
        if f.is_file() and f.suffix.lower() in extensions
    ]


def create_binary_mask(image_path: Path, output_path: Path, threshold: int = 5) -> bool:
    img = cv2.imread(str(image_path))
    if img is None:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if np.sum(mask) == 0:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), mask)
    return True


def blend_images(leaf_path: Path, mask_path: Path, bg_path: Path, output_path: Path) -> bool:
    leaf = cv2.imread(str(leaf_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    bg = cv2.imread(str(bg_path))

    if leaf is None or mask is None or bg is None:
        return False

    h, w = leaf.shape[:2]
    bg_resized = cv2.resize(bg, (w, h), interpolation=cv2.INTER_CUBIC)

    mask_norm = mask.astype(np.float32) / 255.0
    if len(mask_norm.shape) == 2:
        mask_norm = np.stack([mask_norm] * 3, axis=-1)

    composite = leaf.astype(np.float32) * mask_norm + bg_resized.astype(np.float32) * (1 - mask_norm)
    composite = np.clip(composite, 0, 255).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), composite)
    return True


def fill_holes_in_mask(input_path: Path, output_path: Path, kernel_size: int = 9) -> bool:
    mask = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    _, binary_mask = cv2.threshold(closed_mask, 127, 255, cv2.THRESH_BINARY)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), binary_mask)
    return True