import random
from pathlib import Path
from typing import Callable

import tensorflow as tf
from tensorflow.data import Dataset

from config import CLASSIFY_TYPE

CLASS_INDICES_FOLDER = 'metadata/class_indices'


def get_classification_train_val_datasets(image_dir: str,
                                          preprocess_input_function: Callable,
                                          image_size: tuple,
                                          batch_size: int,
                                          validation_split: float,
                                          model_name: str = None):

    image_paths, labels = get_image_paths_and_labels(image_dir, model_name)
    num_classes = len(set(labels))

    return get_train_val_datasets(
        image_paths=image_paths,
        annotations=labels,
        annotation_function=lambda label:
        tf.cast(label, tf.float32) if CLASSIFY_TYPE == 'binary' else tf.one_hot(label, depth=num_classes),
        preprocess_input_function=preprocess_input_function,
        image_size=image_size,
        batch_size=batch_size,
        validation_split=validation_split
    )


def get_segmentation_train_val_datasets(image_dir: str,
                                        mask_dir: str,
                                        preprocess_input_function: Callable,
                                        image_size: tuple,
                                        batch_size: int,
                                        validation_split: float):
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    image_paths = sorted([str(p) for p in image_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
    mask_paths = sorted([str(p) for p in mask_dir.iterdir() if p.suffix.lower() in {'.jpg', '.png'}])

    assert len(image_paths) == len(mask_paths), "Число изображений и масок не совпадает!"

    return get_train_val_datasets(
        image_paths=image_paths,
        annotations=mask_paths,
        annotation_function=lambda mask_path: load_mask(mask_path, image_size),
        preprocess_input_function=preprocess_input_function,
        image_size=image_size,
        batch_size=batch_size,
        validation_split=validation_split
    )


def load_image(image_path: str, image_size: tuple, preprocess_input_function):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32)
    image = preprocess_input_function(image)
    return image


def load_mask(mask_path: str, image_size: tuple):
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)

    mask = tf.image.resize(mask, image_size, method='nearest')
    mask = tf.cast(mask, tf.float32) / 255.0
    return mask


def get_train_val_datasets(image_paths: list[str],
                           annotations: list,
                           annotation_function: Callable,
                           preprocess_input_function: Callable,
                           image_size: tuple,
                           batch_size: int,
                           validation_split: float):

    combined = list(zip(image_paths, annotations))
    random.seed(42)
    random.shuffle(combined)
    image_paths, annotations = zip(*combined)

    image_paths = list(image_paths)
    annotations = list(annotations)

    split_idx = int(len(image_paths) * (1 - validation_split))
    train_img, val_img = image_paths[:split_idx], image_paths[split_idx:]
    train_ann, val_ann = annotations[:split_idx], annotations[split_idx:]

    train_dataset = Dataset.from_tensor_slices((train_img, train_ann))
    val_dataset = Dataset.from_tensor_slices((val_img, val_ann))

    train_dataset = (train_dataset.map(
        lambda image_path, annotation: (
            load_image(image_path, image_size, preprocess_input_function), annotation_function(annotation)
        ), num_parallel_calls=tf.data.AUTOTUNE)
                     .shuffle(buffer_size=1000)
                     .batch(batch_size)
                     .prefetch(tf.data.AUTOTUNE))

    val_dataset = (val_dataset.map(
        lambda image_path, annotation: (
            load_image(image_path, image_size, preprocess_input_function), annotation_function(annotation)
        ), num_parallel_calls=tf.data.AUTOTUNE)
                   .batch(batch_size)
                   .prefetch(tf.data.AUTOTUNE))

    return train_dataset, val_dataset


def get_image_paths_and_labels(image_dir: str, model_name: str = None):
    image_dir = Path(image_dir)

    class_names = sorted([d.name for d in image_dir.iterdir() if d.is_dir()])
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

    if model_name is not None:
        save_class_indices(model_name, class_names)

    image_paths, labels = [], []
    for cls in class_names:
        for p in (image_dir / cls).iterdir():
            if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                image_paths.append(str(p))
                labels.append(class_to_idx[cls])

    return image_paths, labels


def save_class_indices(model_name: str, class_names: list[str]):
    with open(f'{CLASS_INDICES_FOLDER}/{model_name}.txt', 'w') as f:
        for idx, class_name in enumerate(class_names):
            f.write(f'{idx}: {class_name}\n')


def get_classification_test_dataset(image_dir: str,
                                    image_size: tuple,
                                    preprocess_input_function: Callable,
                                    batch_size: int,
                                    model_name: str = None):

    image_paths, labels = get_image_paths_and_labels(image_dir, model_name)
    test_dataset = Dataset.from_tensor_slices((image_paths, labels))
    return (test_dataset.map(
        lambda image_path, annotation: (
            load_image(image_path, image_size, preprocess_input_function), annotation
        ), num_parallel_calls=tf.data.AUTOTUNE)
                     .batch(batch_size)
                     .prefetch(tf.data.AUTOTUNE))