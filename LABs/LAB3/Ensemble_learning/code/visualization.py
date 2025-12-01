import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# -------------------------------------------------------
# 1. 模型性能对比（柱状图）
# -------------------------------------------------------
def plot_model_comparison(results, save_path=None):
    """
    results: dict, key=model_name, value=accuracy or f1
    """
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(results.keys()), y=list(results.values()))
    plt.xticks(rotation=45)
    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


# -------------------------------------------------------
# 2. AdaBoost 训练误差下降曲线
# -------------------------------------------------------
def plot_adaboost_training_curve(errors, save_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(errors, marker='o', linestyle='-', color="red")
    plt.title("AdaBoost Training Error per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Training Error")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


# -------------------------------------------------------
# 3. 混淆矩阵
# -------------------------------------------------------
def plot_confusion(y_true, y_pred, title="Confusion Matrix", save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()
