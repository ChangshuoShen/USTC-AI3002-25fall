# -*- coding: utf-8 -*-
"""
命令行参数定义
"""
import argparse


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="无监督学习实验Lab2")
    
    # 通用设置
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--results_path", type=str, default=None, help="结果路径")
    parser.add_argument("--cache_dir", type=str, default="./data", help="数据缓存目录")
    
    # 数据设置
    parser.add_argument("--dr_method", type=str, default="autoencoder", choices=["pca", "autoencoder"],help="降维方法")
    
    # PCA设置
    parser.add_argument("--pca_components", type=int, default=100, help="PCA主成分数量")
    
    # GMM设置
    parser.add_argument("--gmm_components", type=int, default=10, help="GMM混合成分数量")
    parser.add_argument("--gmm_max_iter", type=int, default=300, help="GMM最大迭代次数")
    parser.add_argument("--gmm_tol", type=float, default=1e-5, help="GMM收敛阈值")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    print("参数设置:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
