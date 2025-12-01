# dataloader.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

DATA_PATH = "../data/pulsar.csv"

def load_raw_data():
    """加载 pulsar 数据集，如果不存在则自动下载。"""

    if not os.path.exists(DATA_PATH):
        print("Downloading Pulsar dataset...")
        url = "https://raw.githubusercontent.com/uwdata/pulsar-dataset/master/pulsar.csv"

        try:
            df = pd.read_csv(url)
        except:
            raise FileNotFoundError(
                "远程 pulsar.csv 下载失败；请检查网络或更换新的镜像链接。"
            )

        os.makedirs("../data", exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH)

    return df


def load_data(test_size=0.2, random_state=42):
    """加载数据 + 缺失值填补 + 标准化"""

    df = load_raw_data()

    # 由于该数据集存在缺失值，我们使用中位数进行填补（初步，如果你认为对你的实验结果造成干扰，可以修改为其他方法）
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(df.drop("target_class", axis=1))
    y = df["target_class"].values

    # 训练/测试划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler
