from pathlib import Path

from evaluate.evaluate import evaluate_binary_models, evaluate_categorical_models
from process.seg_process import segment_and_apply_masks


def main():
    segment_and_apply_masks(base_input_dir=Path(r"dataset\classify\binary_test\grape_binary\Grape___diseases"),
                            base_output_dir=Path(r"C:\Users\Silent Rider\Desktop\test\mask"),
                            mask_output_dir=Path(r"C:\Users\Silent Rider\Desktop\test\masked"),
                            model_name='spot_seg_final')

if __name__ == "__main__":
    main()