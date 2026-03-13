import json
from collections import Counter

def load_evaluation_data(dataset_path):
    try:
        with open(dataset_path, "r", encoding="utf-8") as fh:
            raw_data = json.load(fh)
    except FileNotFoundError:
        return None
    eval_data = []
    for item in raw_data:
        try:
            sentence = item["messages"][1]["content"]
            label = item["messages"][2]["content"].split('#')[0].strip()
            eval_data.append({"sentence": sentence, "label": label})
        except (IndexError, KeyError, TypeError):
            continue
    return eval_data

def calculate_dataset_priors(dataset, all_types):
    """计算数据集中每个类别的先验概率"""
    total_count = len(dataset)
    label_counts = Counter([item['label'] for item in dataset])
    
    prior_map = {}
    for label_type in all_types:
        # 简单归一化，未出现的类别给0
        count = label_counts.get(label_type, 0)
        # 增加极小值防止log报错（如果后续需要），这里直接存概率
        prior_map[label_type] = count / total_count if total_count > 0 else 0.0
        
    print(f"Dataset Priors calculated on {total_count} samples.")
    return prior_map
