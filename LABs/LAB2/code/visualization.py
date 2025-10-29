# -*- coding: utf-8 -*-
"""
可视化工具函数
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 使用镜像站
import argparse
import yaml
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as PCA_sklearn
from sklearn.manifold import TSNE

from dataloader import MNISTLoader
from submission import PCA, GMM
from autoencoder import AE
from util import ae_encode,ensure_dir


def plot_clustering_result(data_2d: np.ndarray, 
                           cluster_labels: np.ndarray, 
                           true_labels: np.ndarray,
                           title: str,
                           method: str,
                           output_path: Path):
    """
    绘制聚类结果可视化图
    
    Args:
        data_2d: 2D降维后的数据 (N, 2)
        cluster_labels: 聚类标签 (N,)
        true_labels: 真实标签 (N,)
        title: 图标题
        method: 降维方法名称 ('ae', 'pca', 'tsne')
        output_path: 输出文件路径
    """
    # 定义cluster的颜色（最多10种颜色）
    cluster_colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525",
                      "#A83683", "#4E655E", "#853541", "#3A3120", "#535D8E"]
    
    plt.figure(figsize=(10, 10))
    
    # 设置坐标轴范围
    plt.xlim(data_2d[:, 0].min(), data_2d[:, 0].max())
    plt.ylim(data_2d[:, 1].min(), data_2d[:, 1].max())
    
    # 为每个点绘制文本，颜色表示cluster，数字表示真实标签
    for i in range(len(data_2d)):
        # 获取该点所属的cluster的颜色
        cluster_idx = int(cluster_labels[i] % len(cluster_colors))
        color = cluster_colors[cluster_idx]
        
        # 绘制文本：数字表示真实标签，颜色表示cluster
        plt.text(data_2d[i, 0], data_2d[i, 1], str(true_labels[i]),
                color=color,
                fontdict={'weight': 'bold', 'size': 7})
    
    plt.xlabel("Component 1", fontsize=12)
    plt.ylabel("Component 2", fontsize=12)
    plt.title(f'{title} - Color: Cluster, Number: True Label', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_comparison_result(data_2d: np.ndarray, 
                           cluster_labels: np.ndarray, 
                           true_labels: np.ndarray,
                           title: str,
                           method: str,
                           output_path: Path):
    """
    绘制对比可视化图 - 左右两张子图分别显示true_label和cluster_label
    
    Args:
        data_2d: 2D降维后的数据 (N, 2)
        cluster_labels: 聚类标签 (N,)
        true_labels: 真实标签 (N,)
        title: 图标题
        method: 降维方法名称 ('ae', 'pca', 'tsne')
        output_path: 输出文件路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # 左图：使用true_label作为颜色
    ax_left = axes[0]
    scatter1 = ax_left.scatter(data_2d[:, 0], data_2d[:, 1], c=true_labels, cmap="tab10", alpha=0.7)
    ax_left.set_xlabel("Component 1", fontsize=12)
    ax_left.set_ylabel("Component 2", fontsize=12)
    ax_left.set_title('True Label', fontsize=14, fontweight='bold')
    plt.colorbar(scatter1, ax=ax_left)
    
    # 右图：使用cluster_label作为颜色
    ax_right = axes[1]
    scatter2 = ax_right.scatter(data_2d[:, 0], data_2d[:, 1], c=cluster_labels, cmap="tab10", alpha=0.7)
    ax_right.set_xlabel("Component 1", fontsize=12)
    ax_right.set_ylabel("Component 2", fontsize=12)
    ax_right.set_title('Cluster Label', fontsize=14, fontweight='bold')
    plt.colorbar(scatter2, ax=ax_right)
    
    # 设置整体标题
    fig.suptitle(f'{title} - Comparison', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """主可视化函数"""
    parser = argparse.ArgumentParser(description="无监督学习结果可视化")
    
    parser.add_argument("--results_path", type=str, default=None, help="结果路径")
    parser.add_argument("--sample_size", type=int, default=60000, help="可视化样本数量")
    parser.add_argument("--plot_mode", type=str, default='comparison',choices=['clustering', 'comparison'], help="绘图样本数量")
    args = parser.parse_args()
    
    # 加载数据
    print("加载MNIST数据集...")
    loader = MNISTLoader()
    dataset = loader.load()
    trainset = dataset["train"]
    
    # 准备数据
    traindata_raw2d = np.stack(trainset["image2D"])
    traindata_raw1d = np.stack(trainset["image1D"])
    true_labels = trainset["label"]
    
    # 限制样本数量
    if args.sample_size > 0 and len(traindata_raw2d) > args.sample_size:
        indices = np.random.choice(len(traindata_raw2d), args.sample_size, replace=False)
        traindata_raw2d = traindata_raw2d[indices]
        traindata_raw1d = traindata_raw1d[indices]
        true_labels = true_labels[indices]
    
    print(f"使用 {len(traindata_raw2d)} 个样本进行可视化")
    
    # 自编码器编码
    print("自编码器编码...")
    ae = AE.from_pretrained("H2O123h2o/mnist-autoencoder")
    traindata_ae = ae_encode(ae, traindata_raw2d)
    
    # t-SNE降维
    print("t-SNE降维...")
    tsne = TSNE(n_components=2)
    traindata_tsne = tsne.fit_transform(traindata_raw1d)
    
    # PCA降维
    print("PCA降维...")
    pca_sklearn = PCA_sklearn(n_components=2)
    traindata_pca_sklearn = pca_sklearn.fit_transform(traindata_raw1d)
    
    print("使用聚类标签和真实标签进行可视化")
        
    if args.results_path is None:
        print("错误：使用聚类标签时必须指定 --results_path")
        return
            
    # 加载配置
    with open(Path(args.results_path) / "config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    # 根据配置加载相应的模型和预测标签
    dr_method = cfg.get('dr_method', 'pca')
    if dr_method == 'pca':
        print("从PCA结果预测聚类标签...")
        pca = PCA.from_pretrained(os.path.join(args.results_path, "pca.npz"))
        traindata_pca = pca.transform(traindata_raw1d)
        gmm = GMM.from_pretrained(args.results_path + "/gmm")
        labels = gmm.predict(traindata_pca)
    else:
        print("从自编码器结果预测聚类标签...")
        gmm = GMM.from_pretrained(args.results_path + "/gmm")
        labels = gmm.predict(traindata_ae)
    
    # 创建输出目录
    output_dir = args.results_path if args.results_path else "./outputs"
    ensure_dir(output_dir)
    
    # 可视化
    print("生成可视化图像...")
    
    # 使用新的绘图函数
    if args.plot_mode == 'clustering':
        # 自编码器结果
        plot_clustering_result(
            data_2d=traindata_ae,
            cluster_labels=labels,
            true_labels=true_labels,
            title="Autoencoder",
            method="ae",
            output_path=Path(output_dir) / "cluster_ae.png"
        )
        
        # PCA结果
        plot_clustering_result(
            data_2d=traindata_pca_sklearn,
            cluster_labels=labels,
            true_labels=true_labels,
            title="PCA",
            method="pca",
            output_path=Path(output_dir) / "cluster_pca.png"
        )
        
        # t-SNE结果
        plot_clustering_result(
            data_2d=traindata_tsne,
            cluster_labels=labels,
            true_labels=true_labels,
            title="t-SNE",
            method="tsne",
            output_path=Path(output_dir) / "cluster_tsne.png"
        )
    
    elif args.plot_mode == 'comparison':
        # 自编码器对比结果
        plot_comparison_result(
            data_2d=traindata_ae,
            cluster_labels=labels,
            true_labels=true_labels,
            title="Autoencoder",
            method="ae",
            output_path=Path(output_dir) / "comparison_ae.png"
        )
        
        # PCA对比结果
        plot_comparison_result(
            data_2d=traindata_pca_sklearn,
            cluster_labels=labels,
            true_labels=true_labels,
            title="PCA",
            method="pca",
            output_path=Path(output_dir) / "comparison_pca.png"
        )
        
        # t-SNE对比结果
        plot_comparison_result(
            data_2d=traindata_tsne,
            cluster_labels=labels,
            true_labels=true_labels,
            title="t-SNE",
            method="tsne",
            output_path=Path(output_dir) / "comparison_tsne.png"
        )
    
    print(f"可视化完成！结果保存在: {output_dir}")


if __name__ == "__main__":
    main()