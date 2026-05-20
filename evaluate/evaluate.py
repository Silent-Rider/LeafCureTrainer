from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report

from config import TASK_NAME
from model.build import create_mobile_net_v3_large, create_efficient_net_v2_b2
from model.load import load_fitted_model
from process.data_gen import get_image_paths_and_labels, get_classification_test_dataset, CLASS_INDICES_FOLDER


def evaluate_binary_models():
    image_size = (256, 256)

    model_names = [
        'apple_binary',
        'corn_binary',
        'grape_binary',
        'potato_binary',
        'tomato_binary'
    ]

    _, preprocess_input_function = create_mobile_net_v3_large(image_size)

    for plant in model_names:
        print(f'\n{"=" * 60}')
        print(f'🍏 EVALUATION: {plant.upper()}')
        print(f'{"=" * 60}\n')
        try:
            model = load_fitted_model(plant)
            print(f'{plant} model is loaded')
        except Exception as e:
            print(f"ERROR loading model {plant}: {e}")
            continue

        test_dir = f'dataset/{TASK_NAME}/binary_test/{plant}'

        try:
            image_paths, true_labels_list = get_image_paths_and_labels(test_dir)
        except Exception as e:
            print(f"ERROR reading directory {test_dir}: {e}")
            continue

        if not image_paths:
            print(f"No images found in {test_dir}")
            continue

        true_labels_np = np.array(true_labels_list)

        test_dataset = get_classification_test_dataset(
            image_dir=test_dir,
            image_size=image_size,
            preprocess_input_function=preprocess_input_function,
            batch_size=32
        )
        predictions = model.predict(test_dataset, verbose=0)

        if predictions.ndim > 1 and predictions.shape[1] == 1:
            pred_probs = predictions.flatten()
        elif predictions.ndim > 1 and predictions.shape[1] == 2:
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

        print(f'📊 SUMMARY')
        print(f'Accuracy: {correct}/{total} ({accuracy:.2f}%)')
        print(f'Healthy (Class 1): {healthy_correct}/{healthy_total} correctly predicted')
        print(f'Diseased (Class 0): {disease_correct}/{disease_total} correctly predicted')
        print(f'\n📋 DETAILED PREDICTIONS:\n')

        for i, (fpath, true_label, pred_class, prob) in enumerate(
                zip(image_paths, true_labels_np, pred_classes, pred_probs)
        ):
            file_name = Path(fpath).name

            true_str = 'HEALTHY' if true_label == 1 else 'DISEASED'
            pred_str = 'HEALTHY' if pred_class == 1 else 'DISEASED'
            status = '✅' if true_label == pred_class else '❌'

            print(f'{status} {file_name}')
            print(f'   True: {true_str} | Pred: {pred_str} | Prob(healthy): {prob:.4f}')

            if true_label != pred_class:
                print(f'   ⚠️  MISCLASSIFIED!')
            print()

        errors = [(f, t, p, pr) for f, t, p, pr
                  in zip(image_paths, true_labels_np, pred_classes, pred_probs)
                  if t != p]

        if errors:
            print(f'\n⚠️  TOTAL ERRORS: {len(errors)}/{total}\n')
            false_healthy = sum(1 for _, t, p, _ in errors if t == 1 and p == 0)
            false_diseased = sum(1 for _, t, p, _ in errors if t == 0 and p == 1)

            print(f'  Healthy misclassified as Diseased: {false_healthy}')
            print(f'  Diseased misclassified as Healthy: {false_diseased}')
        else:
            print(f'\n✅ NO ERRORS! Perfect classification!\n')

        print(f'{"=" * 60}\n')


def evaluate_categorical_models():
    image_size = (256, 256)
    batch_size = 32

    model_names = [
        'apple_categorical',
        'corn_categorical',
        'grape_categorical',
        'potato_categorical',
        'tomato_categorical'
    ]

    _, preprocess_input_function = create_efficient_net_v2_b2(image_size)

    for model_name in model_names:
        print(f'\n{"=" * 60}')
        print(f'🍏 EVALUATION: {model_name.upper()}')
        print(f'{"=" * 60}\n')

        try:
            model = load_fitted_model(model_name)
            print(f'{model_name} model is loaded')
        except Exception as e:
            print(f"ERROR loading model {model_name}: {e}")
            continue

        test_dir = f'dataset/{TASK_NAME}/categorical_test/{model_name}'
        try:
            image_paths, true_labels_list = get_image_paths_and_labels(test_dir)
        except Exception as e:
            print(f"ERROR reading directory {test_dir}: {e}")
            continue

        if not image_paths:
            print(f"No images found in {test_dir}")
            continue

        true_labels_np = np.array(true_labels_list)
        indices_file = Path(CLASS_INDICES_FOLDER) / f'{model_name}.txt'
        class_names_map = {}

        if indices_file.exists():
            with open(indices_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if ':' in line:
                        idx_str, name = line.split(':', 1)
                        class_names_map[int(idx_str)] = name.strip()
        else:
            print("WARNING: Class indices file not found. Using folder order.")
            classes = sorted([d.name for d in Path(test_dir).iterdir() if d.is_dir()])
            class_names_map = {i: c for i, c in enumerate(classes)}

        num_classes = len(class_names_map)
        target_names = [class_names_map[i] for i in range(num_classes)]

        test_dataset = get_classification_test_dataset(
            image_dir=test_dir,
            image_size=image_size,
            preprocess_input_function=preprocess_input_function,
            batch_size=batch_size
        )

        predictions = model.predict(test_dataset, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)
        pred_probs_max = np.max(predictions, axis=1)

        accuracy = np.sum(pred_classes == true_labels_np) / len(true_labels_np) * 100

        print(f'📊 SUMMARY')
        print(f'Global Accuracy: {accuracy:.2f}%')
        print(f'Total samples: {len(true_labels_np)}')

        print(f'\n📋 CLASSIFICATION REPORT:\n')
        report = classification_report(
            true_labels_np,
            pred_classes,
            target_names=target_names,
            digits=4
        )
        print(report)

        errors_idx = np.where(pred_classes != true_labels_np)[0]

        if len(errors_idx) > 0:
            print(f'\n⚠️  TOTAL ERRORS: {len(errors_idx)}/{len(true_labels_np)}\n')
            print('Sample of misclassified images:')

            for i in errors_idx[:10]:
                fpath = image_paths[i]
                true_lbl = true_labels_np[i]
                pred_lbl = pred_classes[i]
                conf = pred_probs_max[i]

                true_name = class_names_map.get(true_lbl, f'Class_{true_lbl}')
                pred_name = class_names_map.get(pred_lbl, f'Class_{pred_lbl}')

                print(f'❌ {Path(fpath).name}')
                print(f'   True: {true_name} | Pred: {pred_name} | Conf: {conf:.4f}')
        else:
            print(f'\n✅ NO ERRORS! Perfect classification!\n')

        print(f'{"=" * 60}\n')