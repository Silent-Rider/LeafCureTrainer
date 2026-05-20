from model.load import load_fitted_model
from model.save import save_model


def main():
    task_name = 'classify'
    classify_type = 'categorical'
    plants = [
        'apple', 'corn', 'grape', 'potato', 'tomato'
    ]
    for plant in plants:
        model_name = f"{plant}_{classify_type}"
        model = load_fitted_model(model_name, task_name, False)
        save_model(model=model,
                   model_name=model_name,
                   export_format='tflite',
                   task_name=task_name)

if __name__ == "__main__":
    main()