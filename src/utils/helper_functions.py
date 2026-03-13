import numpy as np
from scipy.stats import entropy

def safe_entropy(prob_vector):
    pv = np.clip(np.asarray(prob_vector), 1e-12, 1.0)
    s = pv.sum()
    if s == 0: return 0.0
    pv = pv / s # 归一化，确保和为1
    return float(entropy(pv, base=2))

def parse_prediction_robust(prediction_str: str, all_types: list) -> str:
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
