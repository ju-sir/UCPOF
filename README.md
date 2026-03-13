# UCPOF: Unified Context Prompt Optimization Framework

## 项目简介

UCPOF (Unified Context Prompt Optimization Framework) 是一个先进的提示工程框架，旨在通过动态 RAG (Retrieval-Augmented Generation) 选择和多维度指标评估来优化大型语言模型 (LLM) 的提示策略。该框架结合了熵计算、logit margin 分析和先验概率评估，实现了智能的上下文选择机制，从而提高模型的推理准确性和可靠性。

## 核心特性

- **动态 RAG 选择**：基于模型输出的不确定性指标自动决定是否使用 RAG 增强
- **多维度指标评估**：计算熵、margin、准确率、ECE、Brier Score 等多种评估指标
- **模块化架构**：清晰的代码结构，便于扩展和维护
- **多数据集支持**：支持 ACE、CASIE、AG News 等多个数据集
- **离线+在线流程**：支持离线特征提取和在线实时推理
- **LSFU 评分机制**：基于熵、margin 和先验概率的综合评估指标
- **类型提示**：全面的 Python 类型提示，代码更加现代化
- **严谨的错误处理**：替代冗余的 try/except 块，使用结构化的错误处理

## 目录结构

```
UCPOF/
├── configs/                     # 🚀 配置文件目录 (核心抽象点) 
│   ├── dataset/ 
│   │   ├── ace.yaml             # ACE 数据集的特定配置 (包含 prompt 模板、类别等) 
│   │   ├── casie.yaml           # CASIE 数据集配置
│   │   └── agnews.yaml          # AG News 数据集配置
│   ├── model/ 
│   │   ├── qwen_7b.yaml         # Qwen 7B 模型的特定配置 
│   │   └── llama3_8b.yaml       # Llama 3 8B 模型配置
│   └── experiment/ 
│       └── ablation.yaml        # 消融实验配置
│
├── data/                        # 数据集存放目录 (提供一个示例或下载脚本) 
│   └── README.md                # 说明如何准备数据格式 
│
├── src/                         # 🧠 核心源代码目录 (面向对象设计) 
│   ├── __init__.py
│   ├── pipeline.py              # 整合框架：执行 Offline + Online 流程 
│   ├── llm_engine.py            # LLM 推理、Logits 获取、熵计算封装 
│   ├── rag_retriever.py         # 向量数据库构建、检索逻辑 (SentenceTransformer) 
│   ├── metric.py                # LSFU 核心指标计算及动态阈值拟合逻辑 
│   ├── prompt_manager.py        # 动态 Prompt 组装 (融合 System prompt, Few-shot, RAG context) 
│   └── utils.py                 # 结果解析 (parse_prediction)、指标保存等杂项 
│
├── analysis/                    # 📊 分析与可视化脚本 (原先的画图代码全放这里) 
│   ├── plot_pareto_efficiency.py  # 对应 pareto_analysis2.py 
│   ├── plot_risk_coverage.py      # 对应 plot_risk_coverage.py 
│   ├── plot_kde_distribution.py   # 对应 shanfeng.py 
│   ├── threshold_analysis.py      # 对应 threshold_justification.py 
│   └── metric_validation.py       # 对应 AUC.py 和 long_tail.py 
│
├── scripts/                     # 🏃‍♂️ 执行入口脚本 (用户直接运行这些) 
│   ├── extract_features.py        # 对应原来的 ace-v1.py (跑第一遍存 CSV) 
│   ├── run_ucpof.py               # 对应 comprehensive_experiment_ET.py 
│   └── run_ablation.py            # 运行消融实验
│
├── requirements.txt             # 依赖环境 
├── README.md                    # 项目说明 
└── LICENSE                      # 开源协议 (MIT) 
```

## 安装说明

### 1. 克隆仓库

```bash
git clone <repository-url>
cd UCPOF
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 准备数据

请按照 `data/README.md` 中的说明准备数据集。默认支持以下数据集：

- **ACE**：事件抽取数据集
- **AG News**：新闻分类数据集
- **CASIE**：情感分析数据集

## 快速开始

### 1. 提取特征

运行 `extract_features.py` 脚本，提取模型输出的特征并保存为 CSV 文件：

```bash
python scripts/extract_features.py --dataset-config configs/dataset/ace.yaml --model-config configs/model/llama3_8b.yaml
```

### 2. 运行完整实验

运行 `run_ucpof.py` 脚本，执行完整的 UCPOF 实验，包括特征提取和分析：

```bash
python scripts/run_ucpof.py --dataset-config configs/dataset/ace.yaml --model-config configs/model/llama3_8b.yaml --run-analysis
```

### 3. 运行消融实验

运行 `run_ablation.py` 脚本，执行消融实验，计算 Acc、Trigger Rate、ECE、Brier Score 和 NLL 等指标：

```bash
python scripts/run_ablation.py --config configs/experiment/ablation.yaml
```

## 配置说明

### 数据集配置

在 `configs/dataset/` 目录下，每个数据集有一个单独的 YAML 配置文件，包含：

- **数据集路径**：使用相对路径，如 `data/ACE/train.json`
- **类别列表**：任务的所有可能类别
- **Prompt 模板**：包含示例和任务说明

示例配置 (`configs/dataset/ace.yaml`)：

```yaml
# ACE 数据集配置

# 数据集路径
paths:
  prior_dataset: "data/ACE/train.json"  # 用于计算先验概率的数据集
  dataset: "data/ACE/train.json"         # 用于评估的数据集

# 类别列表
all_types:
  - "declare bankruptcy"
  - "transfer ownership"
  # ... 其他类别

# Prompt 模板
templates:
  good_examples: |
    --- EXAMPLES ---
    Example 1:
    Input: "I visited all their families ."
    Output: "meet"
  
  task: |
    --- Task ---
    Please select the most appropriate type of the following sentence from the type list.
```

### 模型配置

在 `configs/model/` 目录下，每个模型有一个单独的 YAML 配置文件，包含：

- **模型路径**：Hugging Face 模型 ID 或本地路径
- **模型参数**：如 `torch_dtype`、`device_map` 等
- **生成参数**：如 `max_new_tokens`、`temperature` 等

示例配置 (`configs/model/llama3_8b.yaml`)：

```yaml
model_path: "meta-llama/Meta-Llama-3-8B"
params:
  torch_dtype: "auto"
  device_map: "auto"
generation:
  max_new_tokens: 50
  temperature: 0.7
```

### 实验配置

在 `configs/experiment/` 目录下，创建实验配置文件，包含：

- **数据集配置路径**：指向数据集配置文件
- **模型配置路径**：指向模型配置文件
- **输出目录**：实验结果保存目录
- **采样数量**：用于评估的样本数量

示例配置 (`configs/experiment/ablation.yaml`)：

```yaml
dataset_config: "configs/dataset/ace.yaml"
model_config: "configs/model/llama3_8b.yaml"
output_dir: "output/ablation"
num_samples: 1000
```

## 核心功能详解

### 1. 动态 RAG 选择

UCPOF 通过计算以下指标来决定是否使用 RAG 增强：

- **熵**：衡量模型预测的不确定性
- **Logit Margin**：Top1-Top2 概率差，衡量模型的自信程度
- **先验概率**：利用训练数据计算的类别先验概率

### 2. 多维度指标评估

框架实现了多种指标的计算，包括：

| 指标名称 | 描述 | 计算方法 |
|---------|------|----------|
| **label_entropy** | 标签熵，衡量模型对标签预测的不确定性 | 基于候选标签概率分布计算 |
| **first_token_entropy_top50** | 前 50 个 token 的熵 | 基于 Top-50 token 概率计算 |
| **first_token_entropy_top500** | 前 500 个 token 的熵 | 基于 Top-500 token 概率计算 |
| **first_token_entropy_full** | 完整词表的熵 | 基于完整词表概率计算 |
| **logit_margin** | Top1-Top2 概率差 | 计算最高概率与次高概率的差值 |
| **accuracy** | 预测准确率 | 预测标签与真实标签的匹配程度 |
| **ECE** | 期望校准误差 | 衡量模型预测概率与实际准确率的校准程度 |
| **Brier Score** | 概率预测的准确性 | 均方误差的一种变体 |
| **NLL** | 负对数似然 | 衡量模型对真实标签的预测质量 |
| **LSFU Score** | 综合评估指标 | (1 - 熵) * margin * 先验概率 |

### 3. 离线+在线流程

- **离线流程**：提取特征、计算指标、拟合阈值
- **在线流程**：实时推理、动态 RAG 选择、结果返回

## 代码示例

### 在线推理示例

```python
from src.pipeline import UCPOFPipeline

# 初始化管道
pipeline = UCPOFPipeline(
    dataset_config_path="configs/dataset/ace.yaml",
    model_config_path="configs/model/llama3_8b.yaml"
)

# 构建RAG索引（只需执行一次）
dataset = pipeline.load_dataset("data/ACE/train.json")
pipeline.build_rag_index(dataset)

# 在线推理
result = pipeline.run_online("The company declared bankruptcy last week.")
print(result)
# 输出示例:
# {
#     "input": "The company declared bankruptcy last week.",
#     "prediction": "declare bankruptcy",
#     "label_entropy": 0.123,
#     "logit_margin": 0.876,
#     "rag_used": False
# }
```

### 批量评估示例

```python
from src.pipeline import UCPOFPipeline

# 初始化管道
pipeline = UCPOFPipeline(
    dataset_config_path="configs/dataset/ace.yaml",
    model_config_path="configs/model/llama3_8b.yaml"
)

# 运行离线评估
df = pipeline.run_offline(num_samples=1000)
print(df.describe())
```

## 分析与可视化

UCPOF 提供了丰富的分析和可视化工具：

- **帕累托效率图**：分析不同阈值下的性能权衡
- **风险覆盖率图**：评估模型在不同风险水平下的覆盖情况
- **KDE 分布图**：可视化指标的概率密度分布
- **阈值分析**：确定最优决策阈值

## 性能评估

### 消融实验

通过 `scripts/run_ablation.py` 可以运行消融实验，评估不同组件对性能的影响：

- **Baseline**：无 RAG 增强
- **Static RAG**：固定使用 RAG 增强
- **Dynamic RAG**：基于 LSFU 评分动态选择 RAG

### 评估指标

消融实验会计算以下指标：

- **准确率 (Acc)**：模型预测的正确性
- **触发率 (Trigger Rate)**：使用 RAG 增强的比例
- **ECE**：期望校准误差
- **Brier Score**：概率预测的准确性
- **NLL**：负对数似然

## 扩展到其他数据集

要扩展到其他数据集，只需在 `configs/dataset/` 目录中添加新的 YAML 配置文件，例如：

```yaml
# 新数据集配置
paths:
  prior_dataset: "data/new_dataset/train.json"
  dataset: "data/new_dataset/train.json"

all_types:
  - "Class1"
  - "Class2"
  - "Class3"

templates:
  good_examples: |
    --- EXAMPLES ---
    Example 1:
    Input: "Example input 1"
    Output: "Class1"
    Example 2:
    Input: "Example input 2"
    Output: "Class2"

  task: |
    --- Task ---
    Please classify the following input into one of the following classes: Class1, Class2, Class3. Do not include any additional text, explanations, or notes - only output the selected class.
```

然后使用新的配置文件运行实验：

```bash
python scripts/run_ucpof.py --dataset-config configs/dataset/new_dataset.yaml --model-config configs/model/llama3_8b.yaml --run-analysis
```

## 输出结果

### 常规实验输出

运行完成后，您将在 `output` 目录中看到以下文件：

- **{model_name}_offline_metrics.csv**：详细的评估指标
- **pareto_efficiency.png**：帕累托效率图
- **risk_coverage.png**：风险覆盖率图
- **{metric}_kde.png**：各指标的 KDE 分布图
- **{metric}_threshold_analysis.png**：各指标的阈值分析图
- **metric_auc.png**：各指标的 AUC 评分图
- **long_tail_analysis.png**：长尾分布图

### 消融实验输出

运行消融实验后，您将在 `final_comprehensive_study` 目录中看到以下文件：

- **ablation_results.csv**：详细的消融实验结果
- **ablation_metrics.csv**：计算的评估指标，包括 Acc、Trigger Rate、ECE、Brier Score 和 NLL
- 控制台输出：显示 5 组模型设置的核心指标对比表

## 技术原理

UCPOF 框架通过以下步骤优化 prompt 和动态选择 RAG：

1. **数据预处理**：加载数据集并计算标签先验概率
2. **模型评估**：使用不同的 prompt 模板评估模型性能
3. **指标计算**：计算熵、margin、LSFU 分数等指标
4. **动态 RAG 选择**：基于评估指标动态选择是否使用 RAG
5. **结果分析**：生成可视化分析结果

## 技术亮点

1. **类型提示**：全面的 Python 类型提示，代码更加现代化和易于理解
2. **错误处理**：结构化的错误处理，替代冗余的 try/except 块
3. **相对路径**：使用相对路径，避免硬编码的绝对路径
4. **模块化设计**：清晰的代码结构，便于扩展和维护
5. **配置化**：所有参数和数据集配置都集中在配置文件中，便于管理

## 贡献指南

欢迎贡献代码和建议！请按照以下步骤进行：

1. Fork 仓库
2. 创建分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详情请参阅 LICENSE 文件

## 引用

如果您在研究中使用了 UCPOF 框架，请引用以下论文：

```
@article{ucpof2024,
  title={UCPOF: Unified Context Prompt Optimization Framework with Dynamic RAG Selection},
  author={Your Name and Co-authors},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2024}
}
```

## 联系我们

如有问题或建议，请通过以下方式联系我们：

- GitHub Issues：https://github.com/yourusername/UCPOF/issues
- Email：your.email@example.com
