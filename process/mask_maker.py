import cv2
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm

def create_binary_mask_from_black_bg(image_path: Path, mask_output_path: Path, threshold: int = 5) -> bool:
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

    cv2.imwrite(str(mask_output_path), mask)
    return True

"""
Обрабатывает все изображения в input_dir и сохраняет маски в output_mask_dir.
Поддерживает вложенные директории (сохраняет структуру).
"""
def process_dataset(input_dir: Path, output_mask_dir: Path):

    files = sorted(
        [f for f in input_dir.iterdir() if f.is_file()],
        key=lambda x: x.name
    )

    for idx, old_path in enumerate(files, start=1):
        new_name = f"{idx}{old_path.suffix.lower()}"
        new_path = old_path.parent / new_name
        old_path.rename(new_path)

        rel_path = new_path.relative_to(input_dir)
        mask_path = output_mask_dir / rel_path.with_suffix(".png")
        mask_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            create_binary_mask_from_black_bg(new_path, mask_path)
            print(f"✅ {new_path} → {mask_path}")
        except Exception as e:
            print(f"❌ Ошибка при обработке {new_path}: {e}")


def blend_leaf_on_background(leaf_path: Path, mask_path: Path, bg_path: Path, output_path: Path):
    leaf = cv2.imread(str(leaf_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    bg = cv2.imread(str(bg_path))

    if leaf is None or mask is None or bg is None:
        raise ValueError(f"Ошибка загрузки: {leaf_path}, {mask_path}, {bg_path}")

    h, w = leaf.shape[:2]

    bg_resized = cv2.resize(bg, (w, h), interpolation=cv2.INTER_CUBIC)

    mask_norm = mask.astype(np.float32) / 255.0
    mask_norm = np.stack([mask_norm] * 3, axis=-1)

    composite = leaf * mask_norm + bg_resized * (1 - mask_norm)
    composite = composite.astype(np.uint8)

    cv2.imwrite(str(output_path), composite)


def generate_composites(
        leaves_dir: Path,
        masks_dir: Path,
        backgrounds_dir: Path,
        output_dir: Path
):
    output_dir.mkdir(parents=True, exist_ok=True)

    bg_files = [f for f in backgrounds_dir.iterdir() if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    if not bg_files:
        raise ValueError("Нет фоновых изображений!")

    leaf_files = sorted(
        [f for f in leaves_dir.iterdir() if f.is_file()],
        key=lambda x: int(x.stem)
    )
    mask_files = sorted(
        [f for f in masks_dir.iterdir() if f.is_file()],
        key=lambda x: int(x.stem)
    )

    if len(leaf_files) != len(mask_files):
        raise ValueError("Число листьев и масок не совпадает!")

    print(f"Найдено: {len(leaf_files)} листьев, {len(bg_files)} фонов")

    for i, (leaf_path, mask_path) in enumerate(tqdm(zip(leaf_files, mask_files), total=len(leaf_files)), start=1):
        bg_path = random.choice(bg_files)  # случайный фон
        out_path = output_dir / f"{i}.jpg"
        blend_leaf_on_background(leaf_path, mask_path, bg_path, out_path)