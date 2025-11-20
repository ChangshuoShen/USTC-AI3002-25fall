# -*- coding: utf-8 -*-
"""
主训练脚本 - 整合所有无监督学习方法
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 使用镜像站
import yaml
from autoencoder import AE
import numpy as np

from util import set_seed, handle_results_path,ensure_dir,ae_encode
from arguments import get_args
from dataloader import MNISTLoader
from submission import GMM, PCA

def main():
    args = get_args()
    set_seed(args.seed)
    results_path = handle_results_path(args.results_path)
    ensure_dir(results_path)

    with open(results_path / "config.yaml", "w") as f:
        yaml.dump(vars(args), f)

    dataloader = MNISTLoader()
    dataset = dataloader.load()

    if args.dr_method=="pca":
        trainset = dataset["train"].to_pandas()
        traindata_raw = np.vstack(trainset["image1D"].to_numpy())
        pca = PCA(n_components=args.pca_components)
        pca.fit(traindata_raw)
        traindata = pca.transform(traindata_raw)
        pca.save_pretrained(results_path / "pca")
        print(f"Succesfully saved PCA model to {results_path}/pca")
    elif args.dr_method=="autoencoder":
        trainset = dataset["train"]
        traindata_raw = np.stack(trainset["image2D"])
        ae = AE.from_pretrained("H2O123h2o/mnist-autoencoder")
        traindata = ae_encode(ae, traindata_raw)

    gmm = GMM(n_components=args.gmm_components,max_iter=args.gmm_max_iter,tol=args.gmm_tol)
    gmm.fit(traindata)
    gmm.save_pretrained(results_path / "gmm")
    print(f"Succesfully saved GMM model to {results_path}/gmm")

if __name__ == "__main__":
    main()

