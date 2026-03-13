# 🚀 UCPOF: 不确定性校准的提示优化框架

![Paper-ESWA](https://img.shields.io/badge/Paper-ESWA-blue) ![Python](https://img.shields.io/badge/python-3.9+-blue.svg) ![License](https://img.shields.io/badge/License-MIT-yellow.svg)

> [[English]](README.md) | **[中文文档]**

论文 **"How Confident Is the First Token? An Uncertainty-Calibrated Prompt Optimization Framework for Large Language Model Classification and Understanding"** 的官方实现。

## 💡 关于项目

虽然检索增强生成（RAG）显著提升了 LLM 的性能，但在高并发工业环境中不加区分地使用 "始终开启" 的 RAG 会带来严重的数据库 QPS 压力、高推理延迟和过高的 token 成本。

**UCPOF** 是一个自适应、不确定性感知的框架，旨在解决这一问题。通过引入一种新颖的指标——**对数尺度焦点不确定性 (LSFU)**，UCPOF 能够准确识别 LLM 何时真正困惑，何时只是表现出由预训练先验驱动的 "虚假置信度"。它充当智能门控，**仅对高风险样本**触发 RAG，与始终开启的 RAG 相比，减少了 **50.66%** 的检索开销，同时将准确率提高了 **5.75%**。

<p align="center">
  <!-- 建议在这里放一张你论文的 Figure 1 框架图 -->
  <img src="docs/architecture.png" alt="UCPOF 架构" width="80%">
</p>

## ✨ 核心贡献

- **对数尺度焦点不确定性 (LSFU):** 一种无需训练、基于首个 token 的置信度指标，通过标签先验进行校准。
- **Gold-Shot 选择:** 一种静态提示优化策略，选择最稳定和可靠的示例。
- **自适应 RAG 门控:** 智能平衡成本和准确性，实现帕累托最优效率。
- **即插即用:** 易于适应多个开源 LLM（Qwen、LLaMA、ChatGLM）和自定义数据集。

---

## 🛠️ 安装

```bash
# 1. 克隆仓库
git clone https://github.com/ju-sir/UCPOF.git
cd UCPOF

# 2. 创建虚拟环境（可选但推荐）
conda create -n ucpof python=3.9
conda activate ucpof

# 3. 安装依赖
pip install -r requirements.txt
```

---

## 🚀 快速开始（将 UCPOF 应用到你的数据）

我们将配置与代码解耦。你可以通过修改 YAML 配置轻松将 UCPOF 适应到你自己的任务。

### 1. 配置你的数据集和模型
检查 `configs/dataset/ace.yaml` 和 `configs/model/qwen_7b.yaml` 来设置你的数据路径、标签空间和模型路径。

### 2. 运行完整的 UCPOF 管道
```bash
python scripts/run_ucpof.py \
    --dataset_config configs/dataset/ace.yaml \
    --model_config configs/model/qwen_7b.yaml \
    --output_dir ./outputs
```
*此脚本会自动运行离线准备（计算先验、寻找 gold-shots、设置动态阈值）和在线推理（自适应 RAG）。*

---

## 📊 复现论文实验

为了方便审稿人和研究人员，我们提供了现成的脚本，可以完美复现论文中的表格和图表。

### 主要结果和效率（表 2，图 6 和 7）
要复现帕累托效率曲线以及 Baseline、Gold-Shot 和 Full RAG 之间的性能比较：
```bash
python scripts/run_ucpof.py --dataset_config configs/dataset/ace.yaml --run_analysis
python analysis/plot_pareto_efficiency.py --csv_path outputs/results.csv
```

### 消融实验（表 4，5，6，7）
要复现消融实验（例如，检查 $P_{prior}$ 的必要性以及 LSFU 与标准熵的比较）：
```bash
python scripts/run_ablation.py --config configs/experiment/ablation.yaml
```

### 指标验证（图 3 和 4）
要绘制风险-覆盖率曲线和 LSFU 分数的 KDE 分布：
```bash
python analysis/plot_risk_coverage.py --csv_path outputs/results.csv
python analysis/plot_kde_distribution.py --csv_path outputs/results.csv
```

---

## 📁 仓库结构

```text
UCPOF/
├── configs/            # ⚙️ 数据集和模型的 YAML 配置
├── data/               # 📂 数据目录（下载 ACE、AGNews 等的说明）
├── src/                # 🧠 核心源代码（指标、检索器、提示管理器、管道）
├── analysis/           # 📈 绘制图表的脚本（帕累托、KDE、风险-覆盖率）
├── scripts/            # 🏃‍♂️ 执行入口脚本
└── README.md           
```

## 📖 引用

如果你发现这个框架对你的研究有用，请考虑引用我们的论文：

```bibtex
@article{chen2024confident,
  title={How Confident Is the First Token? An Uncertainty-Calibrated Prompt Optimization Framework for Large Language Model Classification and Understanding},
  author={Chen, Wei and Ju, Guoyang and Qi, Yuanyuan},
  journal={Expert Systems with Applications (Under Review)},
  year={2024}
}
```

## 📄 许可证
本项目采用 MIT 许可证 - 详情请参阅 [LICENSE](LICENSE) 文件。