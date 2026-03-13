# 🚀 UCPOF: Uncertainty-Calibrated Prompt Optimization Framework

![Python](https://img.shields.io/badge/python-3.9+-blue.svg) ![License](https://img.shields.io/badge/License-MIT-yellow.svg)

> **[English]** | [[中文文档](README_zh-CN.md)]

## 💡 About The Project

While Retrieval-Augmented Generation (RAG) significantly enhances LLM performance, the indiscriminate use of "always-on" RAG in high-concurrency industrial settings introduces severe Database QPS pressure, high inference latency, and excessive token costs.

**UCPOF** is an adaptive, uncertainty-aware framework designed to solve this. By introducing a novel metric—**Log-Scale Focal Uncertainty (LSFU)**—UCPOF accurately identifies when an LLM is truly confused versus when it is just exhibiting "spurious confidence" driven by pre-training priors. It acts as an intelligent gate, triggering RAG **only for high-risk samples**, reducing retrieval overhead by **50.66%** while boosting accuracy by **5.75%** over always-on RAG.

<p align="center">
  <!-- 建议在这里放一张你论文的 Figure 1 框架图 -->
  <img src="docs/architecture.png" alt="UCPOF Architecture" width="80%">
</p>

## ✨ Core Contributions

- **Log-Scale Focal Uncertainty (LSFU):** A training-free, first-token-based confidence metric calibrated with label priors.
- **Gold-Shot Selection:** A static prompt optimization strategy that selects the most stable and reliable exemplars.
- **Adaptive RAG Gating:** Intelligently balances cost and accuracy, achieving Pareto-optimal efficiency.
- **Plug-and-Play:** Easily adaptable to multiple open-source LLMs (Qwen, LLaMA, ChatGLM) and custom datasets.

---

## 🛠️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/ju-sir/UCPOF.git
cd UCPOF

# 2. Create a virtual environment (optional but recommended)
conda create -n ucpof python=3.9
conda activate ucpof

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start (Applying UCPOF to Your Data)

We decouple configurations from the code. You can easily adapt UCPOF to your own tasks by modifying the YAML configs.

### 1. Configure your Dataset & Model
Check `configs/dataset/ace.yaml` and `configs/model/qwen_7b.yaml` to set your data paths, label spaces, and model paths.

### 2. Run the full UCPOF Pipeline
```bash
python scripts/run_ucpof.py \
    --dataset_config configs/dataset/ace.yaml \
    --model_config configs/model/qwen_7b.yaml \
    --output_dir ./outputs
```
*This script automatically runs the offline preparation (calculating priors, finding gold-shots, setting dynamic thresholds) and online inference (adaptive RAG).*

---

## 📊 Reproducing Paper Experiments

For reviewers and researchers, we provide ready-to-use scripts to perfectly reproduce the tables and figures in our paper.

### Main Results & Efficiency (Table 2, Fig 6 & 7)
To reproduce the Pareto efficiency curves and the performance comparison between Baseline, Gold-Shot, and Full RAG:
```bash
python scripts/run_ucpof.py --dataset_config configs/dataset/ace.yaml --run_analysis
python analysis/plot_pareto_efficiency.py --csv_path outputs/results.csv
```

### Ablation Studies (Table 4, 5, 6, 7)
To reproduce the ablation studies (e.g., examining the necessity of $P_{prior}$ and LSFU vs. standard entropy):
```bash
python scripts/run_ablation.py --config configs/experiment/ablation.yaml
```

### Metric Validation (Fig 3 & 4)
To plot the Risk-Coverage curves and the KDE distribution of LSFU scores:
```bash
python analysis/plot_risk_coverage.py --csv_path outputs/results.csv
python analysis/plot_kde_distribution.py --csv_path outputs/results.csv
```

---

## 📁 Repository Structure

```text
UCPOF/
├── configs/            # ⚙️ YAML configs for datasets and models
├── data/               # 📂 Data directory (instructions to download ACE, AGNews, etc.)
├── src/                # 🧠 Core source code (Metric, Retriever, Prompt Manager, Pipeline)
├── analysis/           # 📈 Scripts for plotting figures (Pareto, KDE, Risk-Coverage)
├── scripts/            # 🏃‍♂️ Entry scripts for execution
└── README.md           
```

## 📖 Acknowledgments

We would like to thank the open-source community for their valuable contributions to the tools and libraries used in this project.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.