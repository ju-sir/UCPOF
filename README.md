UCPOF/
│
├── configs/                     # 🚀 配置文件目录 (核心抽象点)
│   ├── dataset/
│   │   ├── ace.yaml             # ACE 数据集的特定配置 (包含 prompt 模板、类别等)
│   │   ├── casie.yaml
│   │   └── agnews.yaml
│   └── model/
│       ├── qwen_7b.yaml         # 模型的特定配置
│       └── llama3_8b.yaml
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
