# LAB3_Part1：支持向量机（Support Vector Machine, SVM）

> **课程**：人工智能与机器学习基础
> **作者**：TA：王珏
> **日期**：2025年11月

[toc]

## 0. 实验简介

本实验将从 **理论、算法推导、调参可视化、模型比较、手写核函数** 多个角度全面帮助理解支持向量机（Support Vector Machine）。

要求在完整的框架下完成：

- 线性核 SVM 分类
- 非线性核（RBF / Polynomial）SVM 分类
- 手写核函数（RBF / Poly）
- 决策边界可视化（PCA 2D）
- 超参数调参（Grid Search）
- 模型评估（Accuracy / F1 / ROC-AUC）
- Margin 分析、Support Vector 分析
- 完成 submission.py 中的关键部分代码

## 1. 数据集介绍：Pulsar 星脉冲星分类

数据来源：*HTRU2 Pulsar Candidate*, 常见机器学习分类任务数据集。

- 样本数：17,898
- 正类（真实脉冲星）：约 10%
- 特征：8 个数值型统计特征（均值/方差/skewness/kurtosis等）
- 标签：
  - 1：脉冲星
  - 0：非脉冲星候选

该数据集 **高度不平衡（imbalanced）**，非常适合展示：

- class_weight 的作用
- RBF 核与高维非线性分隔能力
- 调参如何影响分类边界

## 2. SVM 理论基础

我们简要回顾 SVM 的核心思想

### 2.1 线性可分 SVM（Hard Margin）

目标：在所有能正确分类的超平面中，找到“**间隔最大**”的那个。

超平面：

$$
w^\top x + b = 0
$$

分类要求：

$$
y_i (w^\top x_i + b) \ge 1
$$

几何间隔：

$$
\gamma = \frac{1}{||w||}
$$

最大化间隔等价于最小化：

$$
\min_{w,b} \frac{1}{2}||w||^2
$$

### 2.2 软间隔 SVM（Soft Margin）

现实中数据不可完全线性分隔，于是引入松弛变量 ξ：

$$
y_i(w^\top x_i + b) \ge 1 - \xi_i,\quad \xi_i \ge 0
$$

目标：

$$
\min_{w,b,\xi} \frac12||w||^2 + C\sum_{i=1}^N \xi_i
$$

其中：

- **C 越大 → 惩罚越大 → 越不容忍误差 → 越可能过拟合**
- **C 越小 → 间隔越大 → 容忍更多错误 → 更可能欠拟合**

### 2.3 核技巧（Kernel Trick）

原始特征空间中：

$$
f(x) = w^\top x + b
$$

通过核函数 K：

$$
K(x_i, x_j) = \phi(x_i)^\top \phi(x_j)
$$

可以在高维空间进行线性分类，而无需显式计算 φ(x)。

常用核：

- 线性核：
  $$
  K(x,z)=x^\top z
  $$
- 多项式核：
  $$
  K(x,z) = (\gamma x^\top z + r)^d
  $$
- RBF 核（高斯核）：
  $$
  K(x,z) = e^{-\gamma ||x-z||^2}
  $$

关键思想：

> **核技巧允许模型在高维空间中构造复杂决策边界，而训练仍只依赖内积 $K(x_i, x_j)$。**

### 2.4 Margin / Support Vector

- Support Vectors 是训练集中“最关键的一小部分点”
- 它们决定最终模型的分类边界
- 决策边界的形状（尤其是 RBF 核）高度依赖于支持向量

可以在报告中通过 PCA 可视化展示支持向量。

## 3. 实验内容概述

本实验由 6 大核心任务构成：

### 1：数据加载与可视化

- 加载 pulsar.csv
- 数据统计（均值/方差）
- 热力图（correlation matrix）
- PCA 降维到 2D 用于可视化 SVM 决策边界

### 2：线性核 SVM

学生需完成：

- 训练线性核 SVC
- 输出支持向量个数
- 可视化决策边界（2D PCA）

分析参数 C 对边界的影响。

### 3：非线性核 SVM

支持核：

- RBF
- Polynomial

需要：

- 比较三者表现（linear / rbf / poly）
- 可视化 RBF 决策边界变化（随 gamma）

### 4：手写 Kernel

在 submission.py 中完成：

```python
def rbf_kernel(X1, X2, gamma):
    # TODO
```

```python
def poly_kernel(X1, X2, degree, coef0):
    # TODO
```

并用它们训练 KernelSVM（sklearn 的 SVC 的自定义 kernel 模式）。

### 5：Grid Search

学生需要实现：

* 网格参数：

  * C: [0.1, 1, 10, 100]
  * gamma: ["scale", 0.1, 0.01]
  * kernel: ["rbf", "poly", "linear"]
* 5-fold cross validation
* 生成：

  * accuracy heatmap
  * f1 heatmap
  * support vector count heatmap

### 6：综合可视化

生成所有可视化图：

* PCA 边界图（linear / rbf / poly）
* ROC 曲线
* 调参热图
* 支持向量分布图

所有图像自动保存至：

```
results/[timestamp]/[method]/
```

---


## 5. 报告要求

报告中需包含：

* 实验原理简单介绍
* 核心代码展示
* 模型训练结果
* 三种核决策边界的对比
* 根据助教提供的以及你自己设置的可视化方法分析各种超参数对模型的影响

> 总而言之，你的实验报告只需真实记录你的过程和心得，条理清晰即可。该有的分析和分析的依据都在即可。分析的不对或者结果很反常也没关系，把你的思考和猜想展示一下也可以。
> 非常明显的gpt语言和没有语病但是在无意义重复的报告将不会给很高分数。

### 思考题

1. 为什么 SVM 要最大化间隔？
2. C 越大越容易过拟合，为什么？
3. RBF 核中的 gamma 太大有什么问题？
4. 为什么支持向量数量少，模型更容易泛化？
5. Polynomial 核的 degree 过大有什么影响？

# 6. 如何运行代码

训练指定模型：

```
python train.py --method linear
python train.py --method rbf
python train.py --method poly
python train.py --method grid
```

训练全部模型：

```
python train.py --method all
```

# 7. SVM实验代码结构

```
SVM/
├── README.md
├── requirements.txt
├── data/
│   └── pulsar.csv
│
└── code/
    ├── arguments.py
    ├── dataloader.py
    ├── evaluator.py
    ├── feature_engineering.py
    ├── models.py
    ├── submission.py
    ├── train.py
    ├── visualization.py
    ├── util.py
```
