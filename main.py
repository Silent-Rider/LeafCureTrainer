from pathlib import Path

from analysis.csv_utils import load_history_from_csv
from analysis.plot import draw_subplots
from config import CLASSIFY_TYPE, TASK_NAME
from model.build import create_mobile_net_v3_large, create_classification_model, create_efficient_net_b0, \
    create_resnet50
from process.data_gen import get_classification_train_val_datasets
from train.train import fit_and_save_model


def main():
    image_size = (256, 256)
    epochs = 30
    initial_epoch = 0

    model_name = f"tomato_{CLASSIFY_TYPE}"
    image_dir = f"dataset/{TASK_NAME}/{CLASSIFY_TYPE}/tomato_{CLASSIFY_TYPE}"

    base_model, preprocess_input_function = create_resnet50(image_size)

    train_dataset, val_dataset = get_classification_train_val_datasets(
        image_dir,
        preprocess_input_function,
        image_size,
        32,
        0.15,
        model_name
    )

    num_classes = 1 if CLASSIFY_TYPE == 'binary' else len([d for d in Path(image_dir).iterdir() if d.is_dir()])

    model = create_classification_model(base_model, num_classes=num_classes, learning_rate=1e-3)

    history = fit_and_save_model(
        model,
        train_dataset,
        val_dataset,
        epochs,
        model_name=model_name,
        export_format='keras',
        checkpoints=False,
        logging=True,
        reduce_on_plateau=True,
        early_stopping=True
    )

    if initial_epoch != 0:
        history = load_history_from_csv(model_name)

    draw_subplots(history, filename=f'{model_name}.png')


if __name__ == "__main__":
    main()