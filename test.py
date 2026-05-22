import csv
from pathlib import Path

import tensorflow as tf
from keras.src.applications.mobilenet_v3 import preprocess_input as mobilenet_preprocess_input

from model.load import load_fitted_model

# ==================== КОНФИГУРАЦИЯ ====================
BINARY_MODELS = {
    'apple': 'apple_binary',
    'corn': 'corn_binary',
    'grape': 'grape_binary',
    'potato': 'potato_binary',
    'tomato': 'tomato_binary'
}

LEAF_SEG_MODEL_NAME = 'leaf_seg_final'
SPOT_SEG_MODEL_NAME = 'spot_seg_final'

BASE_DATA_DIR = 'data'
RESULTS_CSV = 'results/detailed_comparison.csv'
THRESHOLD_SEVERITY = 0.03

# ==================== ЗАГРУЗКА МОДЕЛЕЙ ====================
print("Загрузка моделей...")
try:
    leaf_seg_model = load_fitted_model(LEAF_SEG_MODEL_NAME, 'segment')
    spot_seg_model = load_fitted_model(SPOT_SEG_MODEL_NAME, 'segment')
    print(f"✓ Сегментаторы загружены")
except Exception as e:
    print(f"✗ Ошибка загрузки сегментаторов: {e}")
    exit(1)

binary_models = {}
for crop, model_name in BINARY_MODELS.items():
    try:
        binary_models[crop] = load_fitted_model(model_name, 'classify')
        print(f"✓ Загружен {crop}_binary")
    except Exception as e:
        print(f"✗ Ошибка загрузки {model_name}: {e}")

print("Все модели загружены.\n")


# ==================== ФУНКЦИИ ====================
def preprocess_image(image_path: str, img_size=(256, 256)):
    image = tf.io.read_file(image_path)
    if image_path.lower().endswith('.png'):
        image = tf.image.decode_png(image, channels=3)
    else:
        image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32)
    image = mobilenet_preprocess_input(image)
    return image


def get_leaf_mask(image, model):
    pred = model(tf.expand_dims(image, 0))[0]
    return tf.cast(pred > 0.5, tf.float32)


def get_spot_mask(masked_image, model):
    pred = model(tf.expand_dims(masked_image, 0))[0]
    # Используем настраиваемый порог для маски
    return tf.cast(pred > 0.5, tf.float32)


def compute_severity(image_raw, leaf_seg_model, spot_seg_model):
    leaf_mask = get_leaf_mask(image_raw, leaf_seg_model)
    leaf_area = tf.reduce_sum(leaf_mask)

    h, w = tf.shape(image_raw)[0], tf.shape(image_raw)[1]
    total_pixels = tf.cast(h * w, tf.float32)

    if leaf_area < 100 or (leaf_area / total_pixels) < 0.05:
        return None, None

    masked_image = image_raw * leaf_mask
    spot_mask = get_spot_mask(masked_image, spot_seg_model)

    # Логическое И: пятна только внутри листа
    final_spot_mask = spot_mask * leaf_mask
    spot_area = tf.reduce_sum(final_spot_mask)

    severity = spot_area / (leaf_area + 1e-7)
    return float(severity), float(leaf_area / total_pixels)


def predict_binary(image, binary_model):
    pred = binary_model(tf.expand_dims(image, 0))[0][0]
    return int(pred > 0.5), float(pred)


# ==================== ТЕСТ ====================
def run_detailed_comparison():
    results = []
    stats = {
        'total': 0,
        'tp': 0,  # Оба сказали Sick
        'tn': 0,  # Оба сказали Healthy
        'fp_binary': 0,  # Binary=Sick, Seg=Healthy
        'fn_binary': 0,  # Binary=Healthy, Seg=Sick
        'missed_by_both': 0,  # GT=Sick, но оба сказали Healthy
        'false_alarm_both': 0  # GT=Healthy, но оба сказали Sick
    }

    print(f"Тест: Mask Threshold={MASK_THRESHOLD}, Severity Threshold={THRESHOLD_SEVERITY}")
    print("-" * 80)

    base_path = Path(BASE_DATA_DIR)

    for crop in BINARY_MODELS.keys():
        crop_dir = base_path / crop
        if not crop_dir.exists(): continue

        binary_model = binary_models.get(crop)
        if not binary_model: continue

        for class_name in ['sick', 'healthy']:
            class_dir = crop_dir / class_name
            if not class_dir.exists(): continue

            image_files = [f for f in class_dir.iterdir() if
                           f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]

            for img_path in image_files:
                try:
                    image = preprocess_image(str(img_path))
                    severity, leaf_ratio = compute_severity(image, leaf_seg_model, spot_seg_model)

                    if severity is None: continue

                    bin_pred, bin_conf = predict_binary(image, binary_model)
                    seg_pred = 1 if severity >= THRESHOLD_SEVERITY else 0

                    gt_sick = 1 if class_name == 'sick' else 0
                    stats['total'] += 1

                    # Логика подсчета
                    if gt_sick == 1:
                        if bin_pred == 1 and seg_pred == 1:
                            stats['tp'] += 1
                        elif bin_pred == 0 and seg_pred == 0:
                            stats['missed_by_both'] += 1
                        elif bin_pred == 1 and seg_pred == 0:
                            stats['fp_binary'] += 1  # Binary видит, Seg нет
                        elif bin_pred == 0 and seg_pred == 1:
                            stats['fn_binary'] += 1  # Seg видит, Binary нет
                    else:  # Healthy
                        if bin_pred == 0 and seg_pred == 0:
                            stats['tn'] += 1
                        elif bin_pred == 1 and seg_pred == 1:
                            stats['false_alarm_both'] += 1
                        elif bin_pred == 1 and seg_pred == 0:
                            stats['fp_binary'] += 1  # Binary ложно сработал
                        elif bin_pred == 0 and seg_pred == 1:
                            stats['fn_binary'] += 1  # Seg ложно сработал

                    results.append({
                        'path': str(img_path.relative_to(base_path)),
                        'gt': class_name,
                        'bin_pred': 'Sick' if bin_pred else 'Healthy',
                        'seg_pred': 'Sick' if seg_pred else 'Healthy',
                        'severity': f"{severity:.4f}",
                        'bin_conf': f"{bin_conf:.4f}"
                    })
                except Exception as e:
                    continue

    # Вывод подробной статистики
    print(f"\n{'МЕТРИКА':<30} {'КОЛИЧЕСТВО':<10} {'ПОЯСНЕНИЕ'}")
    print("-" * 80)
    print(f"{'Всего фото':<30} {stats['total']:<10}")
    print(f"{'Совпадения (Sick/Sick)':<30} {stats['tp']:<10} Оба верно нашли болезнь")
    print(f"{'Совпадения (Healthy/Healthy)':<30} {stats['tn']:<10} Оба верно нашли здоровье")
    print(f"{'Пропуск обоими (GT=Sick)':<30} {stats['missed_by_both']:<10} ❌ БОЛЬЗЬ ЕСТЬ, но никто не увидел")
    print(f"{'Ложная тревога обоими (GT=H)':<30} {stats['false_alarm_both']:<10} ❌ ЗДОРОВ, но оба испугались")
    print("-" * 80)
    print(f"{'Binary FP (Seg прав)':<30} {stats['fp_binary']:<10} Бинарник ошибся (шум/текстура)")
    print(f"{'Binary FN (Seg прав)':<30} {stats['fn_binary']:<10} Бинарник пропустил (Seg увидел)")

    # Сохранение CSV
    Path(RESULTS_CSV).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nПодробные результаты в {RESULTS_CSV}")


if __name__ == "__main__":
    run_detailed_comparison()