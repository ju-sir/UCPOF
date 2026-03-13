# UCPOF
<<<<<<< HEAD

UCPOF (Unified Context Prompt Optimization Framework) 是一套用于 prompt 优化和动态 RAG 选择的框架，旨在提高大语言模型在不同任务和数据集上的性能。

## 项目结构

```
UCPOF/
├── configs/                     # 🚀 配置文件目录 (核心抽象点) 
│   ├── dataset/ 
│   │   ├── ace.yaml             # ACE 数据集的特定配置 (包含 prompt 模板、类别等) 
│   │   ├── casie.yaml           # CASIE 数据集配置
│   │   └── agnews.yaml          # AG News 数据集配置
│   └── model/ 
│       ├── qwen_7b.yaml         # Qwen 7B 模型的特定配置 
│       └── llama3_8b.yaml       # Llama 3 8B 模型配置
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
│   └── run_ucpof.py               # 对应 comprehensive_experiment_ET.py 
│
├── requirements.txt             # 依赖环境 
├── README.md                    # 极其重要：项目门面 
└── LICENSE                      # 开源协议 (推荐 MIT) 
```

## 功能特性

- **配置化设计**：所有参数和数据集配置都集中在 `configs` 目录中，便于管理和扩展
- **模块化结构**：代码分为核心引擎、RAG 检索、指标计算、Prompt 管理等模块，结构清晰
- **多数据集支持**：通过配置文件支持多个数据集，默认包含 ACE、CASIE 和 AG News 数据集
- **动态 RAG 选择**：基于模型输出的熵、margin 等指标进行动态 RAG 选择
- **全面的指标分析**：提供多种评估指标，包括熵、margin、准确率等
- **可视化分析**：生成帕累托效率图、风险覆盖率图、KDE 分布图等
- **离线+在线流程**：支持离线特征提取和在线推理

## 安装说明

1. 克隆仓库

```bash
git clone <repository-url>
cd UCPOF
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

## 配置说明

### 数据集配置

在 `configs/dataset/` 目录下，每个数据集有一个单独的 YAML 配置文件，包含：

- 数据集路径
- 类别列表
- Prompt 模板（示例和任务说明）

### 模型配置

在 `configs/model/` 目录下，每个模型有一个单独的 YAML 配置文件，包含：

- 模型路径
- 模型参数
- 生成参数

## 使用方法

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

### 3. 在线推理

可以使用 `UCPOFPipeline` 类进行在线推理：

```python
from src.pipeline import UCPOFPipeline

# 初始化管道
pipeline = UCPOFPipeline(
    dataset_config_path='configs/dataset/ace.yaml',
    model_config_path='configs/model/llama3_8b.yaml'
)

# 构建RAG索引（只需执行一次）
dataset = pipeline.load_dataset('path/to/dataset.json')
pipeline.build_rag_index(dataset)

# 在线推理
result = pipeline.run_online("I visited all their families .", use_rag=True)
print(result)
```

## 扩展到其他数据集

要扩展到其他数据集，只需在 `configs/dataset/` 目录中添加新的 YAML 配置文件，例如：

```yaml
# 新数据集配置
paths:
  prior_dataset: "path/to/new/dataset/train.json"
  dataset: "path/to/new/dataset/train.json"

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

运行完成后，您将在 `output` 目录中看到以下文件：

- **{model_name}_offline_metrics.csv**：详细的评估指标
- **pareto_efficiency.png**：帕累托效率图
- **risk_coverage.png**：风险覆盖率图
- **{metric}_kde.png**：各指标的 KDE 分布图
- **{metric}_threshold_analysis.png**：各指标的阈值分析图
- **metric_auc.png**：各指标的 AUC 评分图
- **long_tail_analysis.png**：长尾分布图

## 技术原理

UCPOF 框架通过以下步骤优化 prompt 和动态选择 RAG：

1. **数据预处理**：加载数据集并计算标签先验概率
2. **模型评估**：使用不同的 prompt 模板评估模型性能
3. **指标计算**：计算熵、margin、LSFU 分数等指标
4. **动态 RAG 选择**：基于评估指标动态选择是否使用 RAG
5. **结果分析**：生成可视化分析结果

## 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。
=======
>>>>>>> b1a9c8e9141d7ecdd865430ab9e3d8c6345f37bf
