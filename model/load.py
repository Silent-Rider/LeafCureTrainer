from keras import Model
from keras.src.metrics import BinaryIoU
from keras.src.saving import load_model

from model.build import bce_dice_loss, dice_coef
from train.train import MODELS_FOLDER, CHECKPOINTS_FOLDER


def load_fitted_model(model_name: str, custom_task_name: str = None) -> Model:
    if custom_task_name is not None:
        task_models_folder = f'artifacts/{custom_task_name}/models'
        model = load_model(filepath=f'{task_models_folder}/{model_name}.keras')
    else:
        # custom_objects_dict = {
        #     'bce_dice_loss': bce_dice_loss,
        #     'dice_coef': dice_coef
        # }
        model = load_model(filepath=f'{MODELS_FOLDER}/{model_name}.keras'
                           # custom_objects=custom_objects_dict, compile=False
                           )
    return model


def load_model_checkpoint(model_name: str, epoch: int, custom_task_name: str = None) -> Model:
    if custom_task_name is not None:
        task_checkpoints_folder = f'artifacts/{custom_task_name}/checkpoints'
        model = load_model(f'{task_checkpoints_folder}/{model_name}/epoch{epoch:02d}.keras')
    else:
        model = load_model(f'{CHECKPOINTS_FOLDER}/{model_name}/epoch{epoch:02d}.keras')
    return model