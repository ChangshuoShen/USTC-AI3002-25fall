# 无监督学习实验项目

## 项目结构

```
code/
├── arguments.py      # 命令行参数定义
├── dataloader.py     # 数据加载器（MNIST）
├── models.py         # 无监督学习模型（PCA, KMeans, GMM）
├── autoencoder.py    # 自编码器模型
├── visulation.py     # 可视化工具
├── util.py          # 工具函数
└── train.py         # 主训练脚本
```

## 功能特性

### 1. 数据加载
- 支持 Hugging Face MNIST 数据集
- 自动下载和缓存
- 图像预处理（灰度化、归一化、展平）

### 2. 无监督学习算法
- **PCA**: 主成分分析，使用SVD实现
- **KMeans**: K均值聚类，KMeans++初始化
- **GMM**: 高斯混合模型，EM算法
- **Autoencoder**: 深度自编码器，PyTorch实现
- **t-SNE**: 非线性降维可视化

### 3. 可视化功能
- 2D散点图和文本标签图
- 聚类结果对比
- 重构效果展示
- 训练曲线绘制

## 使用方法

### 基本用法
```bash
cd code
python train.py
```

### 自定义参数
```bash
python train.py \
    --dataset_name "ylecun/mnist" \
    --limit 5000 \
    --pca_components 100 \
    --k 10 \
    --latent_dim 32 \
    --epochs 50 \
    --output_dir "./outputs" \
    --device "cpu"
```

### 主要参数说明

#### 数据设置
- `--dataset_name`: 数据集名称（默认：ylecun/mnist）
- `--limit`: 限制样本数量（默认：None，使用全部）
- `--cache_dir`: 数据缓存目录（默认：./data）

#### 模型参数
- `--pca_components`: PCA主成分数量（默认：100）
- `--k`: KMeans聚类数量（默认：10）
- `--gmm_components`: GMM混合成分数量（默认：10）
- `--latent_dim`: 自编码器潜在维度（默认：32）

#### 训练参数
- `--epochs`: 自编码器训练轮数（默认：50）
- `--batch_size`: 批次大小（默认：128）
- `--lr`: 学习率（默认：1e-3）

#### 可视化参数
- `--plot_samples`: 绘图样本数量（默认：1000）
- `--plot_text_size`: 文本标签大小（默认：8）

## 输出结果

训练完成后，结果将保存在 `outputs/` 目录下：

```
outputs/
├── pca_kmeans/          # PCA+KMeans结果
│   ├── pca_model.npz
│   ├── kmeans_results.npz
│   ├── pca_true_labels.png
│   ├── pca_kmeans_clusters.png
│   └── pca_kmeans_comparison.png
├── gmm/                 # GMM结果
│   ├── pca_model.npz
│   ├── gmm_results.npz
│   ├── gmm_true_labels.png
│   ├── gmm_clusters.png
│   └── gmm_comparison.png
├── autoencoder/         # 自编码器结果
│   ├── autoencoder_model.pth
│   ├── training_curve.png
│   ├── reconstruction.png
│   └── latent_space.png
└── tsne/               # t-SNE结果
    ├── pca_model.npz
    ├── tsne_results.npz
    └── tsne_visualization.png
```

## 依赖包

```bash
pip install torch numpy matplotlib scikit-learn datasets pillow
```

## 注意事项

1. **内存使用**: 大数据集可能需要较多内存，建议使用 `--limit` 参数限制样本数量
2. **计算时间**: t-SNE计算较慢，大数据集建议先用PCA降维
3. **设备选择**: 自编码器训练支持GPU加速，设置 `--device cuda`
4. **可视化**: 文本标签模式在样本较多时可能重叠，可调整 `--plot_text_size` 和 `--plot_samples`

## 扩展功能

### 添加新数据集
在 `dataloader.py` 中添加新的数据加载器类，参考 `MNISTLoader` 的实现。

### 添加新模型
在 `models.py` 中添加新的无监督学习模型，实现 `fit` 和 `predict` 方法。

### 自定义可视化
在 `visulation.py` 中添加新的可视化函数。
