# LAB3_PART3：集成学习（Ensemble Learning）

本次实验旨在让学生掌握集成学习的核心方法，包括：

* Voting（Soft Voting）
* Bagging（Bootstrap + 多模型集成）
* AdaBoost（弱学习器叠加，提高分类精度）
* 模型性能可视化与分析（Confusion Matrix、AdaBoost训练曲线、模型比较图）

本实验将基于乳腺癌数据集（Breast Cancer Wisconsin Dataset）完成分类任务，通过集成学习方法提升模型泛化能力，并理解各方法的性能差异和适用场景。

---

## 截止日期

**12.31 23:59 (UTC+8)**

迟交 **1 / 2 / 4 / 7 天** 分别扣除 **10% / 20% / 40% / 60%** 分数。

---

## 实验评分

本实验满分 100 分，具体构成为：

* **代码补全 (40pt)**
  至少Voting / Bagging / AdaBoost 三部分代码补全情况，逻辑是否正确，是否能够运行训练程序。
* **实验效果 (20pt) + 现场验收 (10pt)**
  包括模型的实际表现（Acc / F1），是否正确生成可视化图像，运行流程是否清晰，可现场解释。
* **报告 (30pt)**
  内容完整性、可视化结果呈现、思考题回答质量。

---

## 环境配置

本课程所有实验使用同一个 Conda 环境。在前两次实验中你已经创建了名为 `ai25` 的环境。本实验继续沿用。

### 实验1环境

```bash
conda create -n ai25 python=3.10
pip install numpy pandas scikit-learn matplotlib
```

### 实验2环境

```bash
conda activate ai25
pip install -r requirements.txt
```

### 实验3环境（本次实验）

```bash
conda activate ai25
pip install -r requirements.txt
```

> ⚠️ 如果 `data/breast_cancer.csv` 不存在，可通过以下命令自动生成：
>
> ```bash
> python code/dataloader.py
> ```

---

## 报告要求（30pt）

你的报告应包含以下内容：

### 实验流程描述（5pt）

* 数据加载、预处理流程
* Voting / Bagging / AdaBoost 的总体流程
* 训练与测试步骤

### 模型设计与参数调优（10pt）

* Voting 中使用的基学习器
* Bagging 的弱学习器数量、深度等参数
* AdaBoost 的迭代次数、学习率
* 你尝试的不同参数与结果变化

### 可视化结果展示（5pt）

请至少展示以下图像，并解释你观察到的效果：

* Voting、Bagging、AdaBoost 的混淆矩阵
* AdaBoost 训练误差曲线
* 三模型性能比较图（accuracy 或 F1）

### 思考题总结回答（5pt）

文末“思考题”部分需回答。

> 推荐使用 **LaTeX** 编写报告。

---

## 实验效果

### Code

见 *code/Ensemble_learning.md* 中的代码规范和补全要求。

### Performance

根据你完成的 Voting / Bagging / AdaBoost 模型在测试集上计算 **Accuracy / F1 Score** 进行综合评估。

你需要运行：

```bash
python code/train.py
```

程序会自动训练三种集成方法并输出结果。

### Visualization

要求至少生成以下图像：

* 混淆矩阵 (Confusion Matrix)
* AdaBoost 训练误差曲线（Training Error per Iteration）
* 三种模型性能比较柱状图（accuracy 或 F1）

---

## 提交格式

在本次实验中，原则上只需修改 `submission.py` 文件完成实验要求。但为了便于调试和测试，允许你修改 `code`目录下的所有文件，但请确保最终提交时保留原有目录结构和文件命名。同时，如果你对 `submission.py`以外的代码文件进行了修改，请在报告中注明修改内容和理由。

你的提交应包含：

```
├── submission.py          # 你实现的核心代码（Voting/Bagging/AdaBoost）
├── report.pdf            # 实验报告
└── results/              # 调试时生成的模型和图像（可选）
    └── [时间戳]_[method]/
        ├── voting_cm.png
        ├── bagging_cm.png
        ├── adaboost_cm.png
        ├── adaboost_training_curve.png
        └── comparison.png
```

> 注意：如果你对其他文件做了修改，也请一并提交。

## 提交方式

1. 将**代码文件**打包为 zip，命名：`<学号>-<姓名>-LAB3.zip`
2. 将报告命名为`PB24000001-张三-LAB3.pdf，和代码**分开**提交至 BB 系统。
