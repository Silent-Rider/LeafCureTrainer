import matplotlib.pyplot as plt

from config import TASK_NAME

PLOTS_FOLDER = f'artifacts/{TASK_NAME}/plots'
METRICS_MAP = {
    'loss': ('функции потерь', 'Функция потерь'),
    'accuracy': ('точности (Accuracy)', 'Точность'),
    'f1_score': ('F1-меры', 'F1-мера'),
    'iou': ('метрики IoU', 'Метрика IoU'),
    'precision': ('прецизионности (Precision)', 'Прецизионность'),
    'recall': ('полноты (Recall)', 'Полнота '),
    'dice_coef': ('коэффициента Дайса (Dice Coefficient)', 'Коэффициент Дайса')
}

def draw_single_plot(metric: str, values: tuple, filename:str=None, is_colored:bool=True):
    if metric not in METRICS_MAP:
        print(f'Метрика {metric} не существует')
        return
    plt.figure(figsize=(10, 6))
    metric_title, metric_ylabel = METRICS_MAP[metric]
    if is_colored:
        prepare_plot_colored(*values, metric_title, metric_ylabel)
    else:
        prepare_plot_black_and_white(*values, metric_title, metric_ylabel)
    if filename:
        plt.savefig(f'{PLOTS_FOLDER}/{metric}_{filename}', dpi=300, bbox_inches='tight')
    plt.show()


def draw_subplots(history:dict, filename:str=None, title:str=None, is_colored:bool=True):
    metrics: list[tuple[str, tuple]] = []
    for metric in METRICS_MAP.keys():
        train_values = history.get(metric)
        val_values = history.get(f"val_{metric}")
        if train_values and val_values:
            metrics.append((metric, (train_values, val_values)))
    if len(metrics) == 0:
        return

    plt.figure(figsize=(15, 5))
    if title:
        plt.suptitle(title)
    for i, (metric_name, metric_values) in enumerate(metrics):
        metric_title, metric_ylabel = METRICS_MAP[metric_name]
        plt.subplot(1, len(metrics), i + 1)
        if is_colored:
            prepare_plot_colored(*metric_values, metric_title, metric_ylabel)
        else:
            prepare_plot_black_and_white(*metric_values, metric_title, metric_ylabel)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    if filename:
        plt.savefig(f'{PLOTS_FOLDER}/{filename}', dpi=300, bbox_inches='tight')
    plt.show()


def prepare_plot_colored(train_data, val_data, title:str, ylabel:str):
    epochs_iter = list(range(1, len(train_data) + 1))
    plt.plot(epochs_iter, train_data, label='Обучающая выборка', marker='o', color='tab:blue')
    plt.plot(epochs_iter, val_data, label='Валидационная выборка', marker='s', color='tab:orange')
    plt.title('Динамика ' + title)
    plt.xlabel('Эпоха')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)


def prepare_plot_black_and_white(train_data, val_data, title: str, ylabel: str):
    epochs_iter = list(range(1, len(train_data) + 1))
    plt.plot(epochs_iter, train_data, label='Обучающая выборка', linestyle='-', color='black', linewidth=2)
    plt.plot(epochs_iter, val_data, label='Валидационная выборка', linestyle='--', color='black', linewidth=2)
    plt.xlabel('Эпоха', fontsize=14, fontfamily='Times New Roman')
    plt.ylabel(ylabel, fontsize=14, fontfamily='Times New Roman')
    plt.grid(True, linestyle=':', alpha=0.7)