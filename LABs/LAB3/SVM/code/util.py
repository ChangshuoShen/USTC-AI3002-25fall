# util.py

import os
import time
import numpy as np


def ensure_dir(path):
    """创建目录（如果不存在）"""
    os.makedirs(path, exist_ok=True)
    return path


def get_timestamp():
    """返回格式化时间戳"""
    return time.strftime("%Y%m%d-%H%M%S")


def minmax_scale(X):
    """简单 min-max scaling（某些可视化可能用到）"""
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    return (X - X_min) / (X_max - X_min + 1e-10)
