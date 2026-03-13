import json
import pandas as pd
import os
from typing import List, Dict, Any, Optional
from collections import Counter

def parse_prediction_robust(prediction_str: str, all_types: List[str]) -> str:
    """鲁棒解析预测结果"""
    text = prediction_str.lower().strip()
    # 1. 精确匹配引号
    for et in all_types:
        if f'"{et.lower()}"' in text or f"'{et.lower()}'" in text: return et
    # 2. 前缀匹配
    prefixes = ["output:", "answer:", "type:"]
    for prefix in prefixes:
        if text.startswith(prefix):
            area = text[len(prefix):].strip()
            for et in all_types:
                if area.startswith(et.lower()): return et
    # 3. 模糊匹配
    for et in all_types:
        if et.lower() in text: return et
    return "unclassified"

def save_metrics_to_csv(metrics: List[Dict[str, Any]], output_path: str) -> None:
    """保存指标到CSV文件"""
    df = pd.DataFrame(metrics)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Metrics saved to {output_path}")

def load_json_file(file_path: str) -> Optional[Any]:
    """加载JSON文件"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Invalid JSON file: {file_path}")
        return None

def calculate_dataset_priors(dataset: List[Dict[str, str]], all_types: List[str]) -> Dict[str, float]:
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
