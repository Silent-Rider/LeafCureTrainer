import time
from tensorflow.keras.callbacks import Callback

from config import TASK_NAME


class LogCallback(Callback):
    LOG_FOLDER = f'artifacts/{TASK_NAME}/logs'
    train_start_time = 0

    def __init__(self, file_name='model_log.txt'):
        super().__init__()
        self.file_name = file_name

    def on_train_begin(self, logs=None):
        self.train_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed_total = time.time() - self.train_start_time
        metrics = ''

        if TASK_NAME == 'classify':
            acc = logs.get('accuracy', 0.0)
            val_acc = logs.get('val_accuracy', 0.0)

            f1 = logs.get('f1_score')
            val_f1 = logs.get('val_f1_score')
            precision = logs.get('precision')
            val_precision = logs.get('val_precision')
            recall = logs.get('recall')
            val_recall = logs.get('val_recall')

            optional_metrics = ''

            optional_metrics += (f"f1_score: {f1:.4f} - val_f1_score: {val_f1:.4f}\n"
                                if f1 is not None else "")
            optional_metrics += (f"precision: {precision:.4f} - val_precision: {val_precision:.4f}\n"
                                 if precision is not None else "")
            optional_metrics += (f"recall: {recall:.4f} - val_recall: {val_recall:.4f}\n"
                                 if recall is not None else "")

            metrics = f"accuracy: {acc:.4f} - val_accuracy: {val_acc:.4f}\n{optional_metrics}"
        elif TASK_NAME == 'segment':
            iou = logs.get('iou', 0.0)
            val_iou = logs.get('val_iou', 0.0)
            metrics = f"iou: {iou:.4f} - val_iou: {val_iou:.4f}\n"

            dice_coef = logs.get('dice_coef')
            val_dice_coef = logs.get('val_dice_coef')
            metrics += (f"dice_coef: {dice_coef:.4f} - val_dice_coef: {val_dice_coef:.4f}\n"
                        if dice_coef is not None else "")

        loss = logs.get('loss', 0.0)
        val_loss = logs.get('val_loss', 0.0)
        metrics += f"loss: {loss:.4f} - val_loss: {val_loss:.4f}\n"

        log_msg = (
            f"\tEpoch {epoch + 1}/{self.params['epochs']}\n"
            f'{metrics}'
            f"total time: {int(elapsed_total // 60)} minutes {elapsed_total % 60:.2f} seconds\n\n"
        )

        with open(f'{self.LOG_FOLDER}/{self.file_name}', mode='a') as log_file:
            log_file.write(log_msg)
