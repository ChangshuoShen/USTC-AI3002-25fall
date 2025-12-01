import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------
# 功能1：正常加载已经保存的数据（train.py调用）
# --------------------------------------------------------

def load_data(path="../data/breast_cancer.csv"):
    df = pd.read_csv(path)

    X = df.drop("target_class", axis=1).values
    y = df["target_class"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# --------------------------------------------------------
# 功能2：自动生成 breast_cancer.csv
# 运行方式： python dataloader.py
# --------------------------------------------------------
def generate_dataset(save_path="../data/breast_cancer.csv"):
    from sklearn.datasets import load_breast_cancer
    import os

    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target_class"] = data.target

    # 创建 data/ 目录（如果不存在）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df.to_csv(save_path, index=False)
    print(f"[OK] 乳腺癌数据集已生成: {save_path}")
    print(f"数据形状: {df.shape}")
    print(f"标签类别: {set(df['target_class'])}")


# --------------------------------------------------------
# 主入口：允许直接运行 python dataloader.py
# --------------------------------------------------------
if __name__ == "__main__":
    print("开始生成 Breast Cancer Wisconsin Dataset (乳腺癌分类)…")
    generate_dataset()

    # 可选：生成后立即测试 load_data 函数
    print("\n测试 load_data()...")
    X_train, X_test, y_train, y_test = load_data()
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("y unique:", set(y_train))
    print("[OK] dataloader 正常工作")
