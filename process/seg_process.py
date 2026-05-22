import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from keras.src.applications.mobilenet_v3 import preprocess_input
from tqdm import tqdm

from model.load import load_fitted_model

IMG_SIZE = (256, 256)

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

    input_tensor = load_image(str(image_path), IMG_SIZE, preprocess_input)
    input_tensor = tf.expand_dims(input_tensor, 0)
    pred = model.predict(input_tensor, verbose=0)[0]

    if len(pred.shape) == 3:
        mask_pred = pred[:, :, 0]
    else:
        mask_pred = pred

    binary_mask = (mask_pred > 0.5).astype(np.float32)

    mask_pil = Image.fromarray((binary_mask * 255).astype(np.uint8))
    full_size_mask = mask_pil.resize((orig_w, orig_h), resample=Image.Resampling.BILINEAR)

    return np.array(full_size_mask) / 255.0


def generate_masks_batch(model, input_dir: Path, output_dir: Path):
    print(f"[Этап 1] Генерация масок: {input_dir} -> {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = []
    for root, _, files in os.walk(input_dir):
        root_path = Path(root)
        rel_path = root_path.relative_to(input_dir)

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                src_file = root_path / file

                file_obj = Path(file)
                new_filename = file_obj.stem + '.png'

                dst_file = output_dir / rel_path / new_filename
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                image_files.append((src_file, dst_file))

    if not image_files:
        print("Изображения не найдены.")
        return 0

    success_count = 0
    error_count = 0

    for src_file, dst_file in tqdm(image_files, unit="файл", desc="Генерация масок", colour="cyan"):
        try:
            mask_arr = predict_mask(src_file, model)
            mask_img = Image.fromarray((mask_arr * 255).astype(np.uint8), mode='L')
            mask_img.save(dst_file)
            success_count += 1
        except Exception as e:
            error_count += 1
            tqdm.write(f"Ошибка генерации маски {src_file.name}: {e}")

    print(f"-> Маски готовы. Успешно: {success_count}, Ошибок: {error_count}")
    return success_count


def apply_masks_batch(original_dir: Path, mask_dir: Path, result_dir: Path):
    print(f"[Этап 2] Применение масок: {original_dir} + {mask_dir} -> {result_dir}")
    result_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for root, _, files in os.walk(mask_dir):
        root_path = Path(root)
        rel_path = root_path.relative_to(mask_dir)

        for file in files:
            if not file.lower().endswith('.png'):
                continue

            mask_file = root_path / file
            file_obj = Path(file)
            base_name = file_obj.stem

            orig_root = original_dir / rel_path
            original_file = None

            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                candidate = orig_root / f"{base_name}{ext}"
                if candidate.exists():
                    original_file = candidate
                    break

            if original_file:
                out_filename = f"{base_name}.jpg"
                dst_file = result_dir / rel_path / out_filename
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                tasks.append((original_file, mask_file, dst_file))
            else:
                tqdm.write(f"Предупреждение: Не найден оригинал для маски {file}")

    if not tasks:
        print("Пары файлов для применения масок не найдены.")
        return 0

    success_count = 0
    error_count = 0

    for orig_path, mask_path, dst_path in tqdm(tasks, unit="файл", desc="Наложение масок", colour="magenta"):
        try:
            with Image.open(orig_path) as img:
                if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[3])
                    original_img = background
                else:
                    original_img = img.convert('RGB')

                orig_arr = np.array(original_img)

            with Image.open(mask_path) as m_img:
                mask_img = m_img.convert('L')
                if mask_img.size != original_img.size:
                    mask_img = mask_img.resize(original_img.size, resample=Image.Resampling.BILINEAR)

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
            tqdm.write(f"Ошибка наложения маски {orig_path.name}: {e}")

    print(f"-> Результат готов. Успешно: {success_count}, Ошибок: {error_count}")
    return success_count


def run(base_input_dir: Path, base_output_dir: Path, mask_output_dir: Path, model_name: str):
    print("=" * 40)
    print("ЗАПУСК ПЛАЙНЛАЙНА ОБРАБОТКИ (2 ЭТАПА)")
    print("=" * 40)

    print("Инициализация модели...")
    try:
        model = load_fitted_model(model_name, "segment")
        print("Модель загружена успешно.\n")
    except Exception as e:
        print(f"Критическая ошибка загрузки модели: {e}")
        return

    generate_masks_batch(model, base_input_dir, mask_output_dir)
    apply_masks_batch(base_input_dir, mask_output_dir, base_output_dir)

    print("\n" + "=" * 40)
    print("ВСЕ ЭТАПЫ ЗАВЕРШЕНЫ")
    print(f"Маски сохранены в: {mask_output_dir.resolve()}")
    print(f"Финальные результаты в: {base_output_dir.resolve()}")
    print("=" * 40)