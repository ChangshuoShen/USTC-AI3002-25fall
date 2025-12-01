# LAB3_Part1：支持向量机（Support Vector Machine, SVM）

> **课程**：人工智能与机器学习基础
> **作者**：TA：王珏
> **日期**：2025年11月

本次实验将从 **理论、算法推导、工程实践、调参可视化、模型比较、手写核函数** 多角度全面帮助理解 **支持向量机（Support Vector Machine, SVM）** 的原理与应用。

学生将基于脉冲星候选人分类数据（Pulsar Dataset）完成以下任务：

* 线性核 SVM 分类
* 非线性核 SVM（RBF / Polynomial）
* 手写核函数（RBF / Polynomial）
* 使用预计算核矩阵实现自定义 Kernel SVM
* 使用 PCA 进行二维决策边界可视化
* 网格搜索 Grid Search 进行超参数调优（C / γ / kernel）
* 模型评估（Accuracy / F1 / ROC-AUC）
* Margin 分析、支持向量数量分析
* 完成 `submission.py` 中的关键核心算法

最终你应该对 **间隔（margin）、核技巧（kernel trick）、支持向量（support vectors）** 的本质有更深刻的理解。

---

## 截止日期

**12.31 23:59 (UTC+8)**

迟交 **1 / 2 / 4 / 7 天** 分别扣除
**10% / 20% / 40% / 60%** 分数。

---

## 实验评分

| 模块               | 分数  | 说明                                                |
| ------------------ | ----- | --------------------------------------------------- |
| **代码补全** | 40 pt | 完成 submission.py（核函数、KernelSVM、GridSearch） |
| **实验效果** | 20 pt | Accuracy / F1，模型表现、可视化效果                 |
| **现场验收** | 10 pt | 主要是对实验细节和相关知识的简单提问             |
| **报告**     | 30 pt | 完整可视化、结果分析、思考题回答                    |

---

## 环境配置

本课程所有实验使用同一 Conda 环境。

### 实验1已经……

```bash
conda create -n ai25 python=3.10
pip install numpy pandas scikit-learn matplotlib
```

### 实验2已经……

```bash
conda activate ai25
pip install -r requirements.txt
```

### 实验3环境（本次 SVM 实验）

```bash
conda activate ai25
pip install -r requirements.txt
```

---

## 数据集介绍：Pulsar 星脉冲星分类

数据来自：**HTRU2 Pulsar Candidate Dataset**
是一个典型的不平衡二分类数据集。

* 样本数：17,898
* 正样本（真实脉冲星）：约 10%
* 特征数：8
  包括均值、方差、偏度、峰度等统计特征
* 标签：

  * **1**：脉冲星
  * **0**：非脉冲星

适用于：

* 不平衡分类实验
* 高维数据的 SVM 决策边界可视化
* kernel trick 演示

---

## 实验任务（6 大部分）

### 1：数据加载与可视化

* 对原始数据做统计分析
* 画特征直方图 / 相关性热力图
* PCA 到 2 维并绘制散点图
* 为决策边界可视化做准备

### 2：线性核 SVM

要求实现：

* 训练线性 SVC
* 可视化 2D 决策边界（PCA）
* 分析超参数 C 对决策边界的影响

### 3：非线性 SVM 核（RBF / Poly）

要求：

* 训练 RBF 核 SVM
* 训练 Polynomial 核 SVM
* 可视化边界随 gamma、degree 的变化
* 分析非线性模型与线性模型的区别

### 4：手写核函数

在 `submission.py` 中补全：

```python
def rbf_kernel(X1, X2, gamma):
    # TODO
```

```python
def poly_kernel(X1, X2, degree, coef0):
    # TODO
```

并在 KernelSVM 类中完成：

* precomputed kernel 的 fit
* precomputed kernel 的 predict / predict_proba

---

### 5：Grid Search

网格参数范围：

* $C \in {0.1, 1, 10, 100}$
* $gamma \in {"scale", 0.1, 0.01}$
* $kernel \in {"linear", "rbf", "poly"}$

要求：

1. 5-fold cross validation
2. 输出最佳参数
3. 绘制 F1 / ACC 热力图
4. 分析支持向量数量变化趋势（可选）

### 6：可视化总结

你可以尽可能选择更多可以便于你分析的可视化图像，例如：

* PCA 决策边界图
* ROC 曲线
* GridSearch 热力图（若 method=grid）
* 支持向量可视化（可选）

所有图像会自动保存至：

```
results/[timestamp]_[method]/
```

---

## 4. 如何运行代码

训练不同的模型：

```
python train.py --method linear
python train.py --method rbf
python train.py --method poly
```

运行网格搜索：

```
python train.py --method grid
```

一次性训练全部：

```
python train.py --method all
```

---

## 5. 文件结构

你提交的SVM部分的代码文件结构应如下所示：

```
SVM/
└── code/         
    ├── submission.py       ← 补全部分
    ├── 其他任何你修改过的代码文件
└── results/              ← 生成的图像和模型
```

> 注意：你对其他任意文件的修改不会被扣分。
