# 配置文件

# 通用配置
GENERAL_CONFIG = {
    "NUM_EVALUATION_SENTENCES": 5000,
    "OUTPUT_DIR": "./output",
}

# 模型配置
MODEL_CONFIG = {
    "MODEL_PATH_LIST": [
        # "/data/models/Qwen2.5-3B-Instruct",
        # "/data/models/Qwen2.5-7B-Instruct", 
        # "/data/models/Qwen2.5-14B-Instruct",
        "/data/models/Llama-3-8B-Instruct",
    ],
}

# 数据集配置
DATASET_CONFIG = {
    "ACE": {
        "PRIOR_DATASET_PATH": "/home/jgy/paper-prompt/ACE/train.json",
        "DATASET_PATH": "/home/jgy/paper-prompt/ACE/train.json",
        "FIXED_GOOD_EXAMPLE_TEXT": """--- EXAMPLES ---
Example 1:
Input: "I visited all their families ."
Output: "meet"
Example 2:
Input: "He claimed Iraqi troops had destroyed five tanks ."
Output: "attack"
Example 3:
Input: "Another appeal is now pending in the Federal Court ."
Output: "appeal"
""",
        "FIXED_TASK_TEXT": """--- Task ---
Please select the most appropriate type of the following sentence from the type list.The type list is : ['declare bankruptcy', 'transfer ownership', 'transfer money', 'marry', 'transport', 'die', 'phone write', 'arrest jail', 'convict', 'sentence', 'sue', 'end organization', 'start organization', 'end position', 'start position', 'meet', 'elect', 'attack', 'injure', 'born', 'fine', 'release parole', 'charge indict', 'extradite', 'trial hearing', 'demonstrate', 'divorce', 'nominate', 'appeal', 'pardon', 'execute', 'acquit', 'merge organization'],You must choose one type from the type list and follow the examples output. Do not include any additional text, explanations, or notes - only output the selected type.""",
        "ALL_TYPES": ['declare bankruptcy', 'transfer ownership', 'transfer money', 'marry', 'transport', 'die', 'phone write', 'arrest jail', 'convict', 'sentence', 'sue', 'end organization', 'start organization', 'end position', 'start position', 'meet', 'elect', 'attack', 'injure', 'born', 'fine', 'release parole', 'charge indict', 'extradite', 'trial hearing', 'demonstrate', 'divorce', 'nominate', 'appeal', 'pardon', 'execute', 'acquit', 'merge organization'],
    },
    # 可以在这里添加其他数据集的配置
    # "OTHER_DATASET": {
    #     "PRIOR_DATASET_PATH": "path/to/other/dataset.json",
    #     "DATASET_PATH": "path/to/other/dataset.json",
    #     "FIXED_GOOD_EXAMPLE_TEXT": "...",
    #     "FIXED_TASK_TEXT": "...",
    #     "ALL_TYPES": [...],
    # },
}
