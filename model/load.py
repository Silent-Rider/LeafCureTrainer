from keras import Model
from keras.src.saving import load_model


def load_fitted_model(model_name: str, task_name: str, need_compile: bool = False) -> Model:
    models_folder = f'artifacts/{task_name}/models'
    model = load_model(filepath=f'{models_folder}/{model_name}.keras', compile=need_compile)
    return model


def load_model_checkpoint(model_name: str, epoch: int, task_name: str) -> Model:
    checkpoints_folder = f'artifacts/{task_name}/checkpoints'
    model = load_model(f'{checkpoints_folder}/{model_name}/epoch{epoch:02d}.keras')
    return model