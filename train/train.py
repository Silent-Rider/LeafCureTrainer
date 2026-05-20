from pathlib import Path

from keras import Model
from keras.src.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping

from config import TASK_NAME
from model.save import save_model
from train.log_callback import LogCallback


CHECKPOINTS_FOLDER = f'artifacts/{TASK_NAME}/checkpoints'

def fit_and_save_model(model: Model,
                       train_dataset,
                       val_dataset,
                       epochs: int,
                       model_name: str,
                       initial_epoch: int = 0,
                       export_format: str = 'keras',
                       checkpoints: bool = False,
                       logging: bool = False,
                       reduce_on_plateau: bool = False,
                       early_stopping: bool = False):

    callbacks = get_callbacks(checkpoints, logging, reduce_on_plateau, early_stopping, model_name)
    history = model.fit(train_dataset,
                        validation_data=val_dataset,
                        epochs=epochs,
                        callbacks=callbacks,
                        initial_epoch=initial_epoch)

    save_model(model=model,
               model_name=model_name,
               export_format=export_format,
               task_name=TASK_NAME)

    return history.history


def get_callbacks(checkpoints: bool,
                  logging: bool,
                  reduce_on_plateau: bool,
                  early_stopping: bool,
                  model_name: str) -> list:
    callbacks = []
    if checkpoints:
        checkpoints_path = Path(CHECKPOINTS_FOLDER) / f'{model_name}'
        checkpoints_path.mkdir(exist_ok=True)
        checkpoint_name = 'epoch{epoch:02d}.keras'
        filepath = str(checkpoints_path / checkpoint_name)
        checkpoint = ModelCheckpoint(filepath=filepath)
        callbacks.append(checkpoint)
    if logging:
        text_logger = LogCallback(f'{model_name}.txt')
        csv_logger = CSVLogger(f'{LogCallback.LOG_FOLDER}/{model_name}.csv', append=True)
        callbacks.append(text_logger)
        callbacks.append(csv_logger)

    monitor = 'val_loss' if TASK_NAME == 'classify' else 'val_iou'
    mode = 'min' if TASK_NAME == 'classify' else 'max'

    if reduce_on_plateau:
        callbacks.append(
            ReduceLROnPlateau(monitor=monitor,
                              factor=0.5,
                              verbose=1,
                              mode=mode,
                              patience=2,
                              min_delta=1e-4,
                              min_lr=1e-7)
        )
    if early_stopping:
        callbacks.append(
            EarlyStopping(monitor=monitor,
                          verbose=1,
                          mode=mode,
                          patience=5,
                          restore_best_weights=True)
        )

    return callbacks
