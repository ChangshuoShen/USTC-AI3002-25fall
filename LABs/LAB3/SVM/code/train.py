# train.py
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from arguments import get_arguments
from dataloader import load_data
from evaluator import evaluate
from visualization import (
    plot_decision_boundary_2D,
    plot_roc_curve,
    plot_grid_heatmap
)

from submission import get_student_model as get_model


# ============================================================== 
# 工具函数：创建保存目录
# ==============================================================

def create_save_dir(base_dir, method):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(base_dir, f"{timestamp}_{method}")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


# ============================================================== 
# 主训练流程
# ==============================================================

def train_and_evaluate(method, args):
    print(f"\n========== Training: {method.upper()} ==========\n")

    # 加载数据
    X_train, X_test, y_train, y_test, scaler = load_data()

    # 模型构建
    model = get_model(method, args)

    # 训练
    print("Fitting model...")
    model.fit(X_train, y_train)

    # 评估
    acc, f1, auc = evaluate(model, X_test, y_test)

    print(f"[{method}] Accuracy = {acc:.4f}")
    print(f"[{method}] F1 Score = {f1:.4f}")
    if auc is not None:
        print(f"[{method}] AUC = {auc:.4f}")
    else:
        print(f"[{method}] AUC = None")

    # 保存目录
    save_dir = create_save_dir(args.save_dir, method)

    # 保存评估结果
    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.6f}\n")
        f.write(f"F1_score: {f1:.6f}\n")
        if auc:
            f.write(f"AUC: {auc:.6f}\n")

    # 可视化决策边界（2D PCA）
    print("Plotting 2D decision boundary...")
    try:
        plot_decision_boundary_2D(model, X_train, y_train, save_dir, method)
    except Exception as e:
        print(f"[Warning] Decision boundary plot failed: {e}")

    # ROC 曲线
    print("Plotting ROC curve...")
    try:
        plot_roc_curve(model, X_test, y_test, save_dir, method)
    except Exception as e:
        print(f"[Warning] ROC plot failed: {e}")

    # Grid Search 专属可视化
    if method == "grid":
        print("Plotting Grid Search heatmaps...")
        try:
            plot_grid_heatmap(model, save_dir)
        except Exception as e:
            print(f"[Warning] Grid heatmap failed: {e}")

    print(f"\nResults saved in: {save_dir}\n")

    return acc, f1, auc



# ============================================================== 
# MAIN
# ==============================================================

def main():

    args = get_arguments()

    if args.method == "all":
        methods = ["linear", "rbf", "poly", "kernel_custom", "grid"]

        for m in methods:
            train_and_evaluate(m, args)

    else:
        train_and_evaluate(args.method, args)


if __name__ == "__main__":
    main()
