# visualization.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc


# ================================================================
# 工具函数：自动保存图片
# ================================================================

def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


# ================================================================
# 1. PCA + 决策边界可视化（最核心模块）
# ================================================================

def plot_decision_boundary_2D(model, X, y, save_dir, method_name):
    """
    将高维数据 PCA 到 2D，再可视化 SVM 决策边界。
    """

    print("[Visual] Running PCA for decision boundary...")

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # 创建网格用于绘图
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    # 网格点逆 PCA 回高维
    grid_2d = np.c_[xx.ravel(), yy.ravel()]
    grid_hd = pca.inverse_transform(grid_2d)

    # 模型预测
    try:
        Z = model.predict(grid_hd)
    except:
        # 针对 precomputed kernel（KernelSVM）
        Z = model.predict(grid_hd)

    Z = Z.reshape(xx.shape)

    # 绘制结果
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")

    scatter = plt.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=y, cmap="coolwarm", s=20, edgecolors="k"
    )

    plt.title(f"Decision Boundary (PCA 2D) - {method_name.upper()}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")

    plt.colorbar(scatter)

    save_fig(os.path.join(save_dir, f"decision_boundary_{method_name}.png"))


# ================================================================
# 2. ROC 曲线可视化
# ================================================================

def plot_roc_curve(model, X_test, y_test, save_dir, method_name):
    print("[Visual] Plotting ROC curve...")

    try:
        proba = model.predict_proba(X_test)[:, 1]
    except:
        print("[Warning] ROC curve skipped (no probability output).")
        return

    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], '--', color='gray')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {method_name.upper()}")
    plt.legend(loc="lower right")

    save_fig(os.path.join(save_dir, f"roc_curve_{method_name}.png"))


# ================================================================
# 3. Grid Search 可视化（heatmap）
# ================================================================

def plot_grid_heatmap(grid_model, save_dir):
    """
    输入 GridSearchSVM_TA 对象（已 fit）,
    自动从 best_params_ / cv_results_ 中绘制 heatmap。
    """
    if not hasattr(grid_model, "best_params"):
        print("[Warning] No grid search results available.")
        return

    # 从 cv_results 中获取要画热图的数据
    results = grid_model.best_model.cv_results_

    params = results["params"]
    mean_test_score = results["mean_test_score"]

    # 提取 C 和 gamma 维度（仅适用于 rbf/poly）
    C_values = sorted({p["C"] for p in params})
    gamma_values = sorted({str(p["gamma"]) for p in params})

    # 构建 score 矩阵
    score_matrix = np.zeros((len(C_values), len(gamma_values)))

    for i, C in enumerate(C_values):
        for j, gamma in enumerate(gamma_values):
            for k, p in enumerate(params):
                if p["C"] == C and str(p["gamma"]) == gamma:
                    score_matrix[i, j] = mean_test_score[k]

    plt.figure(figsize=(8, 6))
    sns.heatmap(score_matrix, annot=True, xticklabels=gamma_values, yticklabels=C_values,
                cmap="YlGnBu", fmt=".3f")

    plt.xlabel("Gamma")
    plt.ylabel("C")
    plt.title("Grid Search F1 Score Heatmap")

    save_fig(os.path.join(save_dir, "grid_heatmap_f1.png"))

    print("[Visual] Grid Search Heatmap saved.")
