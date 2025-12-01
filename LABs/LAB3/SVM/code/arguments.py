# arguments.py

import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description="SVM Lab Experiment Arguments")

    # 主任务选择
    parser.add_argument(
        "--method",
        type=str,
        default="linear",
        choices=["linear", "rbf", "poly", "grid", "all"],
        help="Choose SVM model to train"
    )

    # SVM 超参数
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--gamma", type=str, default="scale")
    parser.add_argument("--degree", type=int, default=3)

    # Grid Search
    parser.add_argument("--use_gridsearch", action="store_true",
                        help="Enable grid search")

    # K-Fold
    parser.add_argument("--kfold", type=int, default=5)

    # PCA 降维可视化
    parser.add_argument("--pca_components", type=int, default=2)

    # 结果保存路径
    parser.add_argument("--save_dir", type=str, default="../results")

    return parser.parse_args()
