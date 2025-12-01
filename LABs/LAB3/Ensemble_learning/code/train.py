import os
import joblib
from datetime import datetime

from arguments import get_args
from dataloader import load_data
from evaluator import evaluate
from submission import VotingModel, BaggingModel, AdaBoostModel, GBDTModel, StackingModel
from visualization import plot_confusion, plot_adaboost_training_curve

def run_one(method, X_train, X_test, y_train, y_test, args):
    print(f"\n===== Running {method} =====")

    # 选择模型
    if method == "voting":
        model = VotingModel()
    elif method == "bagging":
        model = BaggingModel(n_estimators=args.n_estimators)
    elif method == "adaboost":
        model = AdaBoostModel(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate
        )
    elif method == "gbdt":
        model = GBDTModel(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate
        )
    elif method == "stacking":
        model = StackingModel()
    else:
        raise ValueError(f"Unknown method: {method}")

    # 训练
    model.fit(X_train, y_train)

    # 评估
    pred = model.predict(X_test)
    acc, f1 = evaluate(model, X_test, y_test)
    print(f"{method} → Acc={acc:.4f}, F1={f1:.4f}")

    # 创建保存目录
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(args.save_dir, stamp + f"_{method}")
    os.makedirs(save_dir, exist_ok=True)

    # 保存模型
    joblib.dump(model, os.path.join(save_dir, f"{method}.pkl"))

    # 保存混淆矩阵
    plot_confusion(
        y_test, pred,
        title=f"{method} Confusion Matrix",
        save_path=os.path.join(save_dir, f"{method}_cm.png")
    )

    # AdaBoost 特殊：保存训练误差曲线
    if method == "adaboost" and hasattr(model, "train_errors"):
        plot_adaboost_training_curve(
            model.train_errors,
            save_path=os.path.join(save_dir, "adaboost_training_curve.png")
        )

def main():
    args = get_args()
    X_train, X_test, y_train, y_test = load_data()

    # 不指定方法时，默认跑完全部
    if args.method == "all":
        methods = ["voting", "bagging", "adaboost", "gbdt", "stacking"]
        for m in methods:
            run_one(m, X_train, X_test, y_train, y_test, args)
    else:
        run_one(args.method, X_train, X_test, y_train, y_test, args)


if __name__ == "__main__":
    main()
