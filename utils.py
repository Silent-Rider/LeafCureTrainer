import random
from pathlib import Path

import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm

from process.mask_maker import blend_leaf_on_background, create_binary_mask_from_black_bg


def generate_masks_tree(iso_dir: Path, mask_dir: Path, threshold: int):
    if not iso_dir.exists():
        raise FileNotFoundError(f"Папка {iso_dir} не найдена!")

    print("🌳 Шаг 1: Генерация дерева масок...")

    # Находим все изображения рекурсивно
    image_files = list(iso_dir.rglob("*"))
    image_files = [f for f in image_files if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]

    print(f"   Найдено изображений: {len(image_files)}")

    for img_path in tqdm(image_files, desc="Создание масок"):
        rel_path = img_path.relative_to(iso_dir)
        target_mask_path = mask_dir / rel_path.with_suffix(".png")
        target_mask_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            create_binary_mask_from_black_bg(img_path, target_mask_path, threshold=threshold)
        except Exception as e:
            print(f"❌ Ошибка при создании маски для {img_path}: {e}")

    print(f"✅ Маски сохранены в: {mask_dir}")


def generate_composites_tree(iso_dir: Path,
                             mask_dir: Path,
                             bg_dir: Path,
                             comp_dir: Path,
                             images_per_leaf: int = 1):
    if not mask_dir.exists():
        raise FileNotFoundError("Папка с масками не найдена! Сначала запусти generate_masks_tree().")
    if not bg_dir.exists():
        raise FileNotFoundError(f"Папка с фонами не найдена: {bg_dir}")

    # Собираем все фоны
    bg_files = [f for f in bg_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    if not bg_files:
        raise ValueError("Нет фоновых изображений в папке background!")

    print(f"🎨 Шаг 2: Генерация композитов...")
    print(f"   Найдено фонов: {len(bg_files)}")

    # Находим все изображения исходников рекурсивно
    image_files = list(iso_dir.rglob("*"))
    image_files = [f for f in image_files if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]

    total_generated = 0

    for img_path in tqdm(image_files, desc="Создание композитов"):
        rel_path = img_path.relative_to(iso_dir)

        mask_path = mask_dir / rel_path.with_suffix(".png")

        if not mask_path.exists():
            print(f"⚠️ Маска не найдена для {rel_path}, пропускаем.")
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

            success = blend_leaf_on_background(img_path, mask_path, bg_path, out_path)
            if success:
                total_generated += 1

    print(f"✅ ГОТОВО! Сгенерировано {total_generated} композитов в: {comp_dir}")


def fill_mask_holes(input_dir: Path, output_dir: Path, kernel_size: int = 9):
    output_dir.mkdir(parents=True, exist_ok=True)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    files = []
    for root, _, fnames in os.walk(input_dir):
        rel_path = Path(root).relative_to(input_dir)
        for fname in fnames:
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                files.append((Path(root) / fname, output_dir / rel_path / fname))

    print(f"Обработка {len(files)} масок (kernel={kernel_size})...")

    for src, dst in tqdm(files, desc="Заполнение отверстий"):
        dst.parent.mkdir(parents=True, exist_ok=True)

        mask = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        _, binary_mask = cv2.threshold(closed_mask, 127, 255, cv2.THRESH_BINARY)

        cv2.imwrite(str(dst), binary_mask)

    print("Готово.")


def rename_files_sequentially(directory: str):
    dir_path = Path(directory)

    if not dir_path.exists():
        print(f"Ошибка: Папка {directory} не найдена.")
        return

    files = sorted([f for f in dir_path.iterdir() if f.is_file()])

    if not files:
        print("Папка пуста.")
        return

    print(f"Найдено {len(files)} файлов. Начинаю безопасное переименование...")

    # ЭТАП 1: Переименовываем во временные имена (tmp_1, tmp_2...)
    temp_files_map = {}  # Старый путь -> Временный путь
    print("   Этап 1/2: Создание временных имен...")
    for index, file_path in enumerate(files, start=1):
        temp_name = f"tmp_{index}{file_path.suffix}"
        temp_path = dir_path / temp_name

        try:
            file_path.rename(temp_path)
            temp_files_map[file_path] = temp_path
        except Exception as e:
            print(f"❌ Ошибка при создании временного имени для {file_path.name}: {e}")
            return

    print("   Этап 2/2: Присвоение финальных имен...")
    final_count = 0
    for index, old_path in enumerate(files, start=1):
        temp_path = temp_files_map.get(old_path)
        if not temp_path or not temp_path.exists():
            continue

        final_name = f"{index}{old_path.suffix}"
        final_path = dir_path / final_name

        try:
            temp_path.rename(final_path)
            final_count += 1
        except Exception as e:
            print(f"❌ Ошибка при финальном переименовании {temp_path.name}: {e}")

    print(f"✅ Готово! Переименовано {final_count} файлов.")


def copy_corresponding_images(mask_dir: str, source_img_dir: str, target_img_dir: str):
    if not os.path.exists(target_img_dir):
        os.makedirs(target_img_dir)
        print(f"Создана директория: {target_img_dir}")

    mask_files = os.listdir(mask_dir)

    copied_count = 0
    not_found_count = 0

    print("Начало обработки...")

    for mask_filename in mask_files:
        if not mask_filename.lower().endswith('.png'):
            continue

        name_without_ext = os.path.splitext(mask_filename)[0]

        possible_extensions = ['.jpg', '.jpeg']
        found = False

        for ext in possible_extensions:
            source_filename = name_without_ext + ext
            source_path = os.path.join(source_img_dir, source_filename)

            # Проверяем существование файла
            if os.path.isfile(source_path):
                target_path = os.path.join(target_img_dir, source_filename)

                shutil.copy2(source_path, target_path)
                copied_count += 1
                found = True
                break

        if not found:
            not_found_count += 1

    print(f"Обработка завершена.")
    print(f"Скопировано файлов: {copied_count}")
    print(f"Не найдено соответствующих изображений: {not_found_count}")


# copy_corresponding_images(
#     mask_dir=r"dataset\segment\leaf\fine-tuning\mask",
#     source_img_dir=r"dataset\segment\spot\images",
#     target_img_dir=r"dataset\segment\leaf\fine-tuning\images"
# )

# rename_files_sequentially("dataset/segment/leaf/new/background")

# generate_masks_tree(
#     Path("dataset/segment/leaf/new/segmented"),
#     Path(f"dataset/segment/leaf/new/mask"),
#     5)

# generate_composites_tree(
#     iso_dir = Path("dataset/segment/leaf/fine-tuning/correcting/segmented"),
#     mask_dir = Path("dataset/segment/leaf/fine-tuning/correcting/mask"),
#     bg_dir = Path("dataset/segment/leaf/fine-tuning/correcting/background"),
#     comp_dir = Path("dataset/segment/leaf/fine-tuning/correcting/composite")
# )