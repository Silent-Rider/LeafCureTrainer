from pathlib import Path

from analysis.plot import draw_subplots
from config import CLASSIFY_TYPE, TASK_NAME, IMAGE_SIZE, EPOCHS
from model.build import create_classification_model, create_efficient_net_v2_b2
from process.data_gen import get_classification_train_val_datasets
from train.train import fit_and_save_model


def main():
    model_name = f"tomato_{CLASSIFY_TYPE}"
    image_dir = f"dataset/{TASK_NAME}/{CLASSIFY_TYPE}/tomato_{CLASSIFY_TYPE}"

    base_model, preprocess_input_function = create_efficient_net_v2_b2(IMAGE_SIZE)

    train_dataset, val_dataset = get_classification_train_val_datasets(
        image_dir,
        preprocess_input_function,
        IMAGE_SIZE,
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
        EPOCHS,
        model_name=model_name,
        export_format='keras',
        checkpoints=False,
        logging=True,
        reduce_on_plateau=True,
        early_stopping=True
    )

    draw_subplots(history, filename=f'{model_name}.png')


if __name__ == "__main__":
    main()