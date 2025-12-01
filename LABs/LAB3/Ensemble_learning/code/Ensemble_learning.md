# LAB3_Part3：集成学习（Ensemble Learning）

> **课程：** 人工智能与机器学习基础 Lab3
> **作者：** TA：王珏
> **日期：** 2025年11月

[toc]

---

## 0. 实验简介

本实验旨在帮助你系统理解 **集成学习（Ensemble Learning）** 的核心思想与主流方法，并通过在真实数据集上的实验练习加深对学习算法泛化性能提升机制的掌握。

本实验将聚焦于三类最经典的集成策略：

1. **Voting（投票法）**：多模型输出融合，提升稳定性
2. **Bagging（装袋法）**：Bootstrap + 多基学习器，降低方差
3. **Boosting（提升法）**：通过不断关注困难样本逐步提升性能（本实验实现 AdaBoost）

通过对乳腺癌二分类数据集（Breast Cancer Wisconsin Dataset）的实验，我们将观察到：

- Voting 如何融合多个基模型的预测概率
- Bagging 如何通过自助采样降低模型方差
- AdaBoost 如何通过迭代提升分类器性能
- 各方法在真实数据上的准确率、F1 值差异

### 0.1 数据集介绍

本实验使用 **Breast Cancer Wisconsin Dataset（乳腺癌诊断数据集）**，这是机器学习领域最常用的医学二分类数据集之一：

- **样本数**：569
- **特征数**：30（均为连续数值特征）
- **标签含义**：
  - 0：恶性（Malignant）
  - 1：良性（Benign）
- **特点**：数据清洁、易收敛、模型性能可解释，适合作为分类实验数据。

数据集将在 `dataloader.py` 中自动生成：

> (助教也已经提前放置好了数据集文件，你也可以直接使用。)

```bash
python dataloader.py
```

文件将保存在：

```
data/breast_cancer.csv
```

### 0.2 文件组织结构

```
Ensemble_learning/
├── README.md                # 实验要求
├── requirements.txt         # 依赖包列表
│
├── data/
│   └── breast_cancer.csv    # 乳腺癌数据集（自动生成）
│
└── code/
    ├── arguments.py         # 命令行参数定义
    ├── dataloader.py        # 数据加载 & 数据集生成
    ├── models.py            # 基学习器（LR / SVM / DT等等，你也可以自行添加其他的模型）
    ├── submission.py        # TODO
    ├── train.py             # 训练入口（支持 voting / bagging / adaboost / all）
    ├── evaluator.py         # 评估函数（Acc / F1）
    ├── visualization.py     # 各类可视化工具
    ├── Ensemble_learning.md # 实验文档（本文件）
    └── util.py              # Bootstrap等通用工具函数
```

`submission.py`是你主要需要修改的文件，助教将依据其中实现的 Voting / Bagging / AdaBoost 等算法的正确性进行评分。

## 1. 实验流程

### 1.1 环境准备

先激活环境

```bash
conda activate ai25
```

进入实验目录

```bash
cd Ensemble_learning
pip install -r requirements.txt
```

为自动生成数据集，运行：

```bash
python code/dataloader.py
```

### 1.2 数据预处理

数据预处理由 `dataloader.py` 自动执行：

* 读入原始 CSV 文件
* 划分训练/测试集（Stratified 分层采样）
* 标准化（StandardScaler）
* 返回训练集与测试集：

```python
X_train, X_test, y_train, y_test
```

## 2. 核心任务实现

你需要在 `submission.py` 中完成至少三类集成算法的代码实现。

### 2.1 Voting(5pt)

Voting 是最直观的集成策略：

#### • Hard Voting

选择票数最多的类别（本实验不要求），但你可以尝试实现。

#### • Soft Voting（本实验实现）

对所有基模型的 `predict_proba(X)` 输出取平均：

$$
p(x)=\frac{1}{M}\sum_{m=1}^M p_m(x)
$$

最终预测：

$$
\hat{y} = \arg\max_k p_k(x)
$$

### 2.2 Bagging(15pt)

Bagging 通过 Bootstrap 自助采样 + 多个弱学习器降低模型方差。

#### • Bootstrap 样本生成

每次从训练集随机抽取 n 个样本（可重复）：

$$
D_t = {(x_i,y_i)}_{i=1}^n
$$

#### • 多基学习器训练

使用决策树（深度=3）作为基分类器：

```
for t in 1..T:
    X_s, y_s = bootstrap(X, y)
    train decision tree on (X_s, y_s)
```

#### • 多数投票

$$
\hat{y} = mode{h_t(x)}_{t=1}^T
$$

### 2.3 AdaBoost（SAMME）(20pt)

AdaBoost 是 Boosting 策略中最经典的算法，通过提升困难样本权重逐步构建强分类器。

#### 2.3.1 算法思想

初始化样本权重：

$$
w_i = \frac{1}{N}
$$

第 t 轮训练 stump（深度=1 决策树）：

* 得到预测 (h_t(x))
* 计算带权错误率：

$$
\epsilon_t = \sum_i w_i \cdot \mathbb{I}(h_t(x_i)\neq y_i)
$$

* 计算弱分类器权重：

$$
\alpha_t = \frac{1}{2}\ln \frac{1-\epsilon_t}{\epsilon_t}
$$

* 更新样本权重：

$$
w_i \leftarrow w_i \cdot e^{-\alpha_t y_i h_t(x_i)}
$$

并归一化。

最终预测：

$$
H(x)=\text{sign}\left(\sum_{t=1}^T \alpha_t h_t(x)\right)
$$

#### 2.3.2 伪代码

```
initialize w_i = 1/N

for t in 1..T:
    train stump with sample_weight = w
    predict h_t(x)
    compute err_t
    compute alpha_t
    store stump and alpha_t

    update w_i ← w_i * exp(-alpha_t * y_i * h_t(x_i))
    normalize w

final prediction = sign(sum_t alpha_t * h_t(x))
```

### 2.4 选做：XGBoost 与 Stacking（选做部分，15pt）

> 无论这部分是否完成，本次实验分数上限仍为100分。

#### • XGBoost（10pt）

可以尝试手搓 XGBoost，或者直接使用 `from sklearn.ensemble import GradientBoostingClassifier`来直接调试参数。

##### 直接调用 Sklearn 实现

如果选择直接调用，一个可能的实现框架如下：

```python
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

'''
调参：
loss：损失函数。有deviance和exponential两种。deviance是采用对数似然，exponential是指数损失，后者相当于AdaBoost。
n_estimators:最大弱学习器个数，默认是100，调参时要注意过拟合或欠拟合，一般和learning_rate一起考虑。
learning_rate:步长，即每个弱学习器的权重缩减系数，默认为0.1，取值范围0-1，当取值为1时，相当于权重不缩减。较小的learning_rate相当于更多的迭代次数。
subsample:子采样，默认为1，取值范围(0,1]，当取值为1时，相当于没有采样。小于1时，即进行采样，按比例采样得到的样本去构建弱学习器。这样做可以防止过拟合，但是值不能太低，会造成高方差。
init：初始化弱学习器。不使用的话就是第一轮迭代构建的弱学习器.如果没有先验的话就可以不用管

由于GBDT使用CART回归决策树。以下参数用于调优弱学习器，主要都是为了防止过拟合
max_feature：树分裂时考虑的最大特征数，默认为None，也就是考虑所有特征。可以取值有：log2,auto,sqrt
max_depth：CART最大深度，默认为None
min_sample_split：划分节点时需要保留的样本数。当某节点的样本数小于某个值时，就当做叶子节点，不允许再分裂。默认是2
min_sample_leaf：叶子节点最少样本数。如果某个叶子节点数量少于某个值，会同它的兄弟节点一起被剪枝。默认是1
min_weight_fraction_leaf：叶子节点最小的样本权重和。如果小于某个值，会同它的兄弟节点一起被剪枝。一般用于权重变化的样本。默认是0
min_leaf_nodes：最大叶子节点数
'''

gbdt = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=5, subsample=1
                                  , min_samples_split=2, min_samples_leaf=1, max_depth=3
                                  , init=None, random_state=None, max_features=None
                                  , verbose=0, max_leaf_nodes=None, warm_start=False
                                  )
gbdt.fit(X_train, y_train)
y_pred = gbdt.predict(X_test)
```

然后在 `train.py` 中替换相关代码即可。

##### 手写简化版

如果是手写简化版算法，你可以在 `submission.py` 中实现 `GBDTModel` 类。

###### 原理简述

###### 1. GBDT 的基本思想

GBDT（Gradient Boosting Decision Tree）是一类基于 **梯度提升（Gradient Boosting）思想** 的集成算法。
其核心为：

> 利用前一轮模型的损失函数的 **负梯度方向** 作为新的拟合目标，让每一棵树去拟合前一轮模型的“误差”。

对于二分类任务：

* 模型最终形式是一个加法模型
  $$
  F(x) = F_0(x) + \sum_{m=1}^M \eta f_m(x)
  $$
* (F(x)) 是 logit，也就是未经过 sigmoid 的原始分数
* 概率输出为
  $$
  p(x) = \sigma(F(x)) = \frac{1}{1 + e^{-F(x)}}
  $$

GBDT 的每棵树拟合的并不是标签本身，而是**损失函数对 F 的负梯度**，这是一种“函数空间的梯度下降”。

###### 2. 损失函数与梯度推导

对于二分类问题，我们使用 Logistic Loss：

$$
L(y,p) = -\big[ y\log p + (1-y)\log(1-p) \big]
$$

其中：

$$
p = \sigma(F)
$$

对 F 求导：

$$
\frac{\partial L}{\partial F} = p - y
$$

于是：

$$
-\frac{\partial L}{\partial F} = y - p
$$

这正是我们在每轮中要拟合的“伪残差”：

$$
r_{m,i} = y_i - p_i
$$

在标准教材中，每轮训练包括：

1. **计算负梯度（伪残差）**
   $$
   r_{m,i} = y_i - \sigma(F_{m-1}(x_i))
   $$
2. **用 CART 回归树拟合 ((x_i,r_{m,i}))**
   得到叶子节点区域 (R_{m,j})。
3. **对每个叶子节点求最佳输出常数（闭式解）**
   $$
   c_{m,j} =
   \arg\min_c \sum_{x_i \in R_{m,j}}
   L(y_i, F_{m-1}(x_i)+c)
   $$
4. **更新模型**
   $$
   F_m(x) = F_{m-1}(x) + \sum_j c_{m,j} I(x \in R_{m,j})
   $$
5. 最终模型：
   $$
   F_M(x) = F_0(x) + \sum_{m=1}^M \sum_j c_{m,j} I(x \in R_{m,j})
   $$

###### 3. 简化实现

为了降低数学复杂度、提高可操作性，我们可以对上述算法进行简化：

* **不手动求 c 的闭式解**
* **直接使用 CART 回归树输出作为每个叶子的 c**
* **更新方式保持不变**

即：

$$
F_m(x) = F_{m-1}(x) + \eta f_m(x)
$$

可供参考的伪代码如下：

```
输入：
    数据集 {(x_i, y_i)}, y ∈ {0,1}
    学习率 η
    迭代次数 M
输出：
    分类器 F(x)

1. 初始化（先验概率）
    p = mean(y)
    F0 = log(p / (1 - p))
    F(x) = F0

2. 对 m = 1, 2, ..., M 执行：
  
    # (a) 计算当前预测概率 p_i
    p_i = sigmoid(F(x_i))
  
    # (b) 计算负梯度（伪残差）
    r_i = y_i - p_i
  
    # (c) 用 CART 回归树拟合残差
    f_m(x) = Tree.fit( X, r )
  
    # (d) 更新模型（加法模型）
    F(x) ← F(x) + η * f_m(x)
  
    # (e) 记录训练误差（可选）
    loss = mean( logistic_loss(y, sigmoid(F)) )

3. 最终分类器：
    p(x) = sigmoid(F(x))
    y_hat = 1 if p(x) ≥ 0.5 else 0
```

#### • Stacking（5pt）

实现 Stacking 框架，使用 Logistic Regression 作为元学习器。

> 对于以上选做部分，助教并未事先实现对应的可视化分析，学生可参考 `visualization.py` 中已有的可视化函数自行设计。

## 3. 实验训练与测试

运行全部模型：

```bash
python code/train.py
```

指定单个方法：

```bash
python code/train.py --method voting
python code/train.py --method bagging
python code/train.py --method adaboost
python code/train.py --method gbdt （选做）
python code/train.py --method stacking （选做）
```

输出内容包括：

* 各种集成模型的 Accuracy、F1
* 训练过程打印信息
* 完整模型保存到 `results/` 目录

示例：

```
voting → Acc=0.9649, F1=0.9695
bagging → Acc=0.9561, F1=0.9624
adaboost → Acc=0.9736, F1=0.9771
```

## 4. 实验结果和分析（20pt）

本次实验不限制任何可视化分析工具的使用，鼓励同学们发挥创造力，设计多样化的图表来展示实验结果。以下是一些推荐的可视化内容：

助教准备的可视化函数位于 `visualization.py`，包括：

### (1) 混淆矩阵

```bash
Vote Confusion Matrix
Bagging Confusion Matrix
AdaBoost Confusion Matrix
```

### (2) AdaBoost 训练误差下降曲线

展示弱学习器数量增加时模型性能提升的趋势。

### (3) 模型整体性能对比图

便于总结哪个方法更优。

所有图像将自动保存到 `results/[timestamp]_[method]/` 中。

## 5. 思考题

1. **Voting 与 Bagging 在模型融合机制上的本质区别是什么？**
2. **为什么 AdaBoost 能在弱分类器性能有限的情况下构建强分类器？请结合样本权重更新机制解释。**
3. **Bagging 通常能显著降低模型方差，而 Boosting 通常能降低偏差，你如何理解这种差异？**
4. **本实验中 Voting、Bagging、AdaBoost 的结果有何不同？结合数据分布与算法特点解释原因。**
5. 如果你完成了选做，请回答：
   1. **GBDT 相较于 AdaBoost 有何优势？在本实验中表现如何？**
   2. **Stacking 的元学习器为何选择 Logistic Regression？如果改为其他模型会有何影响？**
   3. **Stacking表现如何？与其他集成方法相比有何优劣？**
6. **请结合乳腺癌数据集的特征分布，讨论集成学习是否比单一模型更适合作为医学诊断模型。**
7. **这次实验给你的任何启发。**

> 叠一个很厚的甲： 本次实验是助教同学第一次设计与实现的机器学习实验，其中一定会有很多不足之处，欢迎同学们在报告中提出宝贵意见与建议！让我们一起，把机器学习变得更简单！
> 感谢大家的支持捏！
