from keras.src.metrics import BinaryIoU
from keras.src.optimizers import Adam

from analysis.plot import draw_subplots
from config import IMAGE_SIZE, BATCH_SIZE, TASK_NAME, EPOCHS
from model.build import create_mobile_net_v3_large, tversky_loss, dice_coef
from model.load import load_fitted_model
from process.data_gen import get_segmentation_train_val_datasets
from train.train import fit_and_save_model


def main():
    model_name = 'spot_seg_v2'
    base_model, preprocess_input_function = create_mobile_net_v3_large(IMAGE_SIZE)
    train_dataset, val_dataset = get_segmentation_train_val_datasets(
        image_dir='dataset/segment/spot/fine-tuning/images',
        mask_dir='dataset/segment/spot/fine-tuning/spot_masks',
        preprocess_input_function=preprocess_input_function,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.15)

    model = load_fitted_model('spot_seg_final', TASK_NAME)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=tversky_loss,
        metrics=[
            BinaryIoU(target_class_ids=[1], threshold=0.5, name='iou'),
            dice_coef
        ]
    )
    history = fit_and_save_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=EPOCHS,
        model_name=model_name,
        checkpoints=False,
        logging=True,
        reduce_on_plateau=True,
        early_stopping=True
    )
    draw_subplots(history, model_name)

if __name__ == "__main__":
    main()