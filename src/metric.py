import numpy as np
from scipy.stats import entropy
import torch
import torch.nn.functional as F
from typing import List, Tuple

class MetricCalculator:
    def __init__(self):
        """初始化指标计算器"""
        pass
    
    def safe_entropy(self, prob_vector: List[float]) -> float:
        """安全计算熵"""
        pv = np.clip(np.asarray(prob_vector), 1e-12, 1.0)
        s = pv.sum()
        if s == 0: return 0.0
        pv = pv / s # 归一化，确保和为1
        return float(entropy(pv, base=2))
    
    def calculate_label_entropy(self, candidate_probs: List[float]) -> float:
        """计算标签熵"""
        return self.safe_entropy(candidate_probs)
    
    def calculate_nll(self, logits: torch.Tensor, target_ids: List[int]) -> float:
        """计算负对数似然"""
        log_probs = F.log_softmax(logits, dim=-1)
        nll = 0.0
        for i, tid in enumerate(target_ids):
            nll -= log_probs[i, tid].item()
        
        return nll
    
    def fit_threshold(self, metrics: List[float], accuracies: List[int]) -> Tuple[float, float]:
        """拟合动态阈值"""
        # 这里实现一个简单的阈值拟合逻辑
        # 实际应用中可能需要更复杂的方法
        best_threshold = 0.0
        best_accuracy = 0.0
        
        # 尝试不同的阈值
        for threshold in np.linspace(0, 1, 100):
            predictions = [1 if metric < threshold else 0 for metric, acc in zip(metrics, accuracies)]
            accuracy = sum([1 for p, a in zip(predictions, accuracies) if p == a]) / len(accuracies)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        return best_threshold, best_accuracy
    
    def calculate_lsfu_score(self, label_entropy: float, logit_margin: float, pred_prior_prob: float) -> float:
        """计算LSFU分数"""
        # 这里实现LSFU分数的计算逻辑
        # 实际应用中可能需要根据具体情况调整
        lsfu_score = (1 - label_entropy) * logit_margin * pred_prior_prob
        return lsfu_score
