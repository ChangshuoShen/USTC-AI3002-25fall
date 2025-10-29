# dataloader.py 加载相关数据集
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 使用镜像站

import numpy as np
from datasets import load_dataset,load_from_disk
from datasets import DatasetDict
from submission import data_preprocess


dataset_name="ylecun/mnist"

class MNISTLoader:
    def __init__(self) -> None:
        self.dataset_name = dataset_name
    
    def _load(self):
        # Load the dataset
        raw_dataset = load_dataset("ylecun/mnist")

        trainset = raw_dataset["train"]
        testset = raw_dataset["test"]
        return {"train": trainset, "test": testset}

    def _process(self, example):
        return data_preprocess(example)

    def encode_and_save(self):
        raw_dataset = self._load()
        trainset = raw_dataset["train"]
        testset = raw_dataset["test"]
        trainset = trainset.map(self._process)
        testset = testset.map(self._process)

        trainset.set_format(type="numpy", columns=["image2D", "image1D", "label"])
        testset.set_format(type="numpy", columns=["image2D", "image1D", "label"])

        dataset = DatasetDict({"train": trainset, "test": testset})
        dataset.save_to_disk("./data/mnist_encoded")
        print("Encodeddataset saved to disk")

    def load(self):
        if not os.path.exists("./data/mnist_encoded"):
            self.encode_and_save()
        dataset = load_from_disk("./data/mnist_encoded")
        return dataset