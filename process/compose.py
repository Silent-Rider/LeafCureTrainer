import random
from pathlib import Path

from tqdm import tqdm

from utils import create_binary_mask, blend_images, fill_holes_in_mask, get_image_files


def generate_masks_tree(iso_dir: Path, mask_dir: Path, threshold: int = 5) -> int:
    image_files = get_image_files(iso_dir)
    success_count = 0

    for img_path in tqdm(image_files, desc="Генерация масок"):
        rel_path = img_path.relative_to(iso_dir)
        target_mask_path = mask_dir / rel_path.with_suffix(".png")

        if create_binary_mask(img_path, target_mask_path, threshold):
            success_count += 1

    return success_count


def generate_composites_tree(iso_dir: Path,
                             mask_dir: Path,
                             bg_dir: Path,
                             comp_dir: Path,
                             images_per_leaf: int = 1
                             ) -> int:
    if not mask_dir.exists():
        raise FileNotFoundError("Папка с масками не найдена.")

    bg_files = get_image_files(bg_dir)
    if not bg_files:
        raise ValueError("Нет фоновых изображений.")

    image_files = get_image_files(iso_dir)
    total_generated = 0

    for img_path in tqdm(image_files, desc="Создание композитов"):
        rel_path = img_path.relative_to(iso_dir)
        mask_path = mask_dir / rel_path.with_suffix(".png")

        if not mask_path.exists():
            continue

        target_class_dir = comp_dir / rel_path.parent
        target_class_dir.mkdir(parents=True, exist_ok=True)

        for i in range(images_per_leaf):
            bg_path = random.choice(bg_files)

            if images_per_leaf > 1:
                out_name = f"{img_path.stem}_{i}{img_path.suffix}"
            else:
                out_name = img_path.name

            out_path = target_class_dir / out_name

            if blend_images(img_path, mask_path, bg_path, out_path):
                total_generated += 1

    return total_generated


def process_masks_holes(input_dir: Path, output_dir: Path, kernel_size: int = 9) -> int:
    files = get_image_files(input_dir, extensions=('.png', '.jpg', '.jpeg', '.bmp'))
    processed = 0

    for src_path in tqdm(files, desc="Заполнение отверстий"):
        rel_path = src_path.relative_to(input_dir)
        dst_path = output_dir / rel_path

        if fill_holes_in_mask(src_path, dst_path, kernel_size):
            processed += 1

    return processed