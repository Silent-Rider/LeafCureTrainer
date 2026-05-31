from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from keras.src.applications.mobilenet_v3 import preprocess_input
from tqdm import tqdm

from config import IMAGE_SIZE
from model.load import load_fitted_model


def segment_and_apply_masks(base_input_dir: Path,
                            base_output_dir: Path,
                            mask_output_dir: Path,
                            model_name: str):
    model = load_fitted_model(model_name, "segment")
    generate_masks_batch(model, base_input_dir, mask_output_dir)
    apply_masks_batch(base_input_dir, mask_output_dir, base_output_dir)


def generate_masks_batch(model, input_dir: Path, output_dir: Path):
    print(f"Генерация масок")
    output_dir.mkdir(parents=True, exist_ok=True)

    src_files_map = get_image_files_map(input_dir)

    if not src_files_map:
        print("Изображения не найдены.")
        return 0

    tasks = []
    for stem, src_path in src_files_map.items():
        rel_path = src_path.relative_to(input_dir).parent
        dst_file = output_dir / rel_path / f"{stem}.png"
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        tasks.append((src_path, dst_file))

    success_count = 0
    error_count = 0

    for src_file, dst_file in tqdm(tasks, desc="Генерация масок", unit="file"):
        try:
            mask_arr = predict_mask(src_file, model)
            mask_img = Image.fromarray((mask_arr * 255).astype(np.uint8), mode='L')
            mask_img.save(dst_file)
            success_count += 1
        except Exception as e:
            error_count += 1
            tqdm.write(f"Ошибка [{src_file.name}]: {e}")

    print(f"Сегментация завершена. Маски сгенерированы. Успешно: {success_count}, Ошибок: {error_count}")
    return success_count


def apply_masks_batch(original_dir: Path, mask_dir: Path, result_dir: Path):
    print(f"Применение масок")
    result_dir.mkdir(parents=True, exist_ok=True)

    orig_files_map = get_image_files_map(original_dir)
    mask_files_map = get_image_files_map(mask_dir)

    tasks = []
    missing_origins = 0

    for stem, mask_path in mask_files_map.items():
        if stem not in orig_files_map:
            missing_origins += 1
            continue

        orig_path = orig_files_map[stem]

        rel_path = mask_path.relative_to(mask_dir).parent
        dst_file = result_dir / rel_path / f"{stem}.jpg"
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        tasks.append((orig_path, mask_path, dst_file))

    if missing_origins:
        print(f"Предупреждение: Не найдено оригиналов для {missing_origins} масок.")

    if not tasks:
        print("Нет пар для обработки.")
        return 0

    success_count = 0
    error_count = 0

    for orig_path, mask_path, dst_path in tqdm(tasks, desc="Наложение масок", unit="file"):
        try:
            with Image.open(orig_path) as img:
                original_img = img.convert('RGB') if img.mode != 'RGB' else img
                orig_arr = np.array(original_img)

            with Image.open(mask_path) as m_img:
                mask_img = m_img.convert('L')
                if mask_img.size != original_img.size:
                    mask_img = mask_img.resize(original_img.size, resample=Image.Resampling.NEAREST)
                mask_arr = np.array(mask_img)

            mask_normalized = mask_arr.astype(np.float32) / 255.0
            mask_3ch = mask_normalized[:, :, np.newaxis]

            result_arr = orig_arr.astype(np.float32) * mask_3ch
            result_arr = np.clip(result_arr, 0, 255).astype(np.uint8)

            result_img = Image.fromarray(result_arr)
            result_img.save(dst_path, format="JPEG")
            success_count += 1

        except Exception as e:
            error_count += 1
            tqdm.write(f"Ошибка [{orig_path.name}]: {e}")

    print(f"Применение масок завершено. Успешно: {success_count}, Ошибок: {error_count}")
    return success_count


def get_image_files_map(directory: Path) -> dict[str, Path]:
    files_map = {}
    for p in directory.rglob('*'):
        if p.is_file() and p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
            files_map[p.stem] = p
    return files_map


def load_image(image_path: str, image_size: tuple, preprocess_input_function):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32)
    image = preprocess_input_function(image)
    return image


def predict_mask(image_path: Path, model) -> np.ndarray:
    original_img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = original_img.size

    input_tensor = load_image(str(image_path), IMAGE_SIZE, preprocess_input)
    input_tensor = tf.expand_dims(input_tensor, 0)
    pred = model.predict(input_tensor, verbose=0)[0]

    if len(pred.shape) == 3:
        mask_pred = pred[:, :, 0]
    else:
        mask_pred = pred

    binary_mask = (mask_pred > 0.5).astype(np.float32)

    mask_pil = Image.fromarray((binary_mask * 255).astype(np.uint8))
    full_size_mask = mask_pil.resize((orig_w, orig_h), resample=Image.Resampling.NEAREST)

    return np.array(full_size_mask) / 255.0