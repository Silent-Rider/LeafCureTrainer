from keras import Model
import tensorflow as tf


def save_model(model: Model,
               model_name: str,
               export_format: str,
               task_name: str):
    if not export_format:
        return
    match export_format:
        case 'keras':
            folder = f'artifacts/{task_name}/models'
            model.save(f'{folder}/{model_name}.keras')
        case 'tflite':
            folder = f'artifacts/{task_name}/export'
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            with open(f'{folder}/{model_name}.tflite', 'wb') as f:
                f.write(tflite_model)