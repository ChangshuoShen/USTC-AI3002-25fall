# feature_engineering.py

from sklearn.decomposition import PCA
import numpy as np

class PCAProjector:
    """用于将数据降到2维用于可视化。"""

    def __init__(self, n_components=2):
        self.pca = PCA(n_components=n_components)

    def fit(self, X):
        self.pca.fit(X)
        return self

    def transform(self, X):
        return self.pca.transform(X)

    def fit_transform(self, X):
        return self.pca.fit_transform(X)

    def inverse_transform(self, X_2d):
        """仅在可视化决策边界时用到"""
        return self.pca.inverse_transform(X_2d)
