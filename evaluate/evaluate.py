from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from keras.src.applications.efficientnet_v2 import preprocess_input as efficientnet_preprocess
from keras.src.applications.mobilenet_v3 import preprocess_input as mobilenet_preprocess
from sklearn.metrics import classification_report

from config import TASK_NAME, BATCH_SIZE, IMAGE_SIZE, PLANTS
from model.load import load_fitted_model
from process.data_gen import get_image_paths_and_labels, get_classification_test_dataset, CLASS_INDICES_FOLDER


def evaluate_binary_models(plants: set[str] = PLANTS):
    for plant in plants:
        if plant not in PLANTS:
            print(f'Указано неподдерживаемое растение: {plant}')
            return
    models = {p + "_binary" for p in plants}
    for model in models:
        print(f'\n{"=" * 60}')
        print(f'Оценка: {model.upper()}')
        print(f'{"=" * 60}\n')

        result = prepare_evaluation_data(model, 'binary_test',is_categorical=False)
        if not result:
            continue

        model, image_paths, true_labels_np, test_dataset, _ = result

        predictions = model.predict(test_dataset, verbose=0)

        if predictions.ndim > 1 and predictions.shape[1] == 2:
            pred_probs = predictions[:, 1]
        else:
            pred_probs = predictions.flatten()

        pred_classes = (pred_probs > 0.5).astype(int)

        total = len(true_labels_np)
        correct = np.sum(pred_classes == true_labels_np)
        accuracy = correct / total * 100

        healthy_total = np.sum(true_labels_np == 1)
        healthy_correct = np.sum((pred_classes == 1) & (true_labels_np == 1))

        disease_total = np.sum(true_labels_np == 0)
        disease_correct = np.sum((pred_classes == 0) & (true_labels_np == 0))

        print('Сводка')
        print(f'Точность: {correct}/{total} ({accuracy:.2f}%)')
        print(f'Здоровые (Class 1): {healthy_correct}/{healthy_total} верно предсказаны')
        print(f'Больные (Class 0): {disease_correct}/{disease_total} верно предсказаны')
        print(f'\nДетальные предсказания:\n')

        errors = []
        for fpath, true_label, pred_class, prob in zip(image_paths, true_labels_np, pred_classes, pred_probs):
            file_name = Path(fpath).name
            true_str = 'ЗДОРОВ' if true_label == 1 else 'БОЛЕН'
            pred_str = 'ЗДОРОВ' if pred_class == 1 else 'БОЛЕН'
            status = 'ОК' if true_label == pred_class else 'ОШИБКА'

            print(f'{status} {file_name}')
            print(f'Истинное: {true_str} | Предсказанное: {pred_str} | Вероятность(здоров): {prob:.4f}')

            if true_label != pred_class:
                print(f'Неправильно классифицировано!')
                errors.append((fpath, true_label, pred_class, prob))
            print()

        if errors:
            print(f'\nОбщее количество ошибок: {len(errors)}/{total}\n')
            false_healthy = sum(1 for _, t, p, _ in errors if t == 1 and p == 0)
            false_diseased = sum(1 for _, t, p, _ in errors if t == 0 and p == 1)
            print(f'  Здоровые ложно классифицированные как Больные: {false_healthy}')
            print(f'  Больные ложно классифицированные как Здоровые: {false_diseased}')
        else:
            print('\nОшибок нет. Идеальная классификация\n')

        print(f'{"=" * 60}\n')


def evaluate_categorical_models(plants: set[str] = PLANTS):
    for plant in plants:
        if plant not in PLANTS:
            print(f'Указано неподдерживаемое растение: {plant}')
            return
    models = {p + "_categorical" for p in plants}
    for model_name in models:
        print(f'\n{"=" * 60}')
        print(f'Оценка: {model_name.upper()}')
        print(f'{"=" * 60}\n')

        result = prepare_evaluation_data(model_name, 'categorical_test', is_categorical=True)
        if not result:
            continue

        model, image_paths, true_labels_np, test_dataset, target_names = result

        predictions = model.predict(test_dataset, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)
        pred_probs_max = np.max(predictions, axis=1)

        accuracy = np.sum(pred_classes == true_labels_np) / len(true_labels_np) * 100

        print('Сводка')
        print(f'Глобальная точность: {accuracy:.2f}%')
        print(f'Общее количество примеров: {len(true_labels_np)}')

        print(f'\nОтчет о классификации:\n')
        report = classification_report(
            true_labels_np,
            pred_classes,
            target_names=target_names,
            digits=4
        )
        print(report)

        errors_idx = np.where(pred_classes != true_labels_np)[0]

        if len(errors_idx) > 0:
            print(f'\nОбщее количество ошибок: {len(errors_idx)}/{len(true_labels_np)}\n')
            print('Примеры неправильно классифицированных изображений:')

            for i in errors_idx[:10]:
                fpath = image_paths[i]
                true_lbl = true_labels_np[i]
                pred_lbl = pred_classes[i]
                conf = pred_probs_max[i]

                true_name = target_names[true_lbl] if true_lbl < len(target_names) else f'Class_{true_lbl}'
                pred_name = target_names[pred_lbl] if pred_lbl < len(target_names) else f'Class_{pred_lbl}'

                print(f'Ошибка {Path(fpath).name}')
                print(f'Истинное: {true_name} | Предсказанное: {pred_name} | Уверенность: {conf:.4f}')
        else:
            print('\nОшибок нет. Идеальная классификация\n')

        print(f'{"=" * 60}\n')


def load_class_names(model_name: str, test_dir: str) -> list[str]:
    indices_file = Path(CLASS_INDICES_FOLDER) / f'{model_name}.txt'
    class_map = {}

    if indices_file.exists():
        with open(indices_file, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    idx_str, name = line.split(':', 1)
                    class_map[int(idx_str)] = name.strip()
    else:
        classes = sorted([d.name for d in Path(test_dir).iterdir() if d.is_dir()])
        class_map = {i: c for i, c in enumerate(classes)}

    num_classes = len(class_map)
    return [class_map[i] for i in range(num_classes)]


def prepare_evaluation_data(model_name: str,
                            test_dir_suffix: str,
                            is_categorical: bool = False) -> Optional[Tuple]:
    preprocess_function = efficientnet_preprocess if model_name == 'tomato_categorical' else mobilenet_preprocess
    model = load_fitted_model(model_name, 'classify')
    test_dir = f'dataset/{TASK_NAME}/{test_dir_suffix}/{model_name}'

    try:
        image_paths, true_labels_list = get_image_paths_and_labels(test_dir)
    except Exception as e:
        print(f"Ошибка во время чтения директории {test_dir}: {e}")
        return None

    if not image_paths:
        print(f"Не найдено изображений в {test_dir}")
        return None

    true_labels = np.array(true_labels_list)
    test_dataset = get_classification_test_dataset(
        image_dir=test_dir,
        image_size=IMAGE_SIZE,
        preprocess_input_function=preprocess_function,
        batch_size=BATCH_SIZE
    )

    target_names = None
    if is_categorical:
        target_names = load_class_names(model_name, test_dir)

    return model, image_paths, true_labels, test_dataset, target_names