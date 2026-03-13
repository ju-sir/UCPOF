import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from scipy.stats import entropy
from typing import Dict, List, Optional, Tuple, Any

class LLMEngine:
    def __init__(self, model_config: Dict[str, Any]):
        self.model_path: str = model_config['model_path']
        self.params: Dict[str, Any] = model_config.get('params', {})
        self.generation_params: Dict[str, Any] = model_config.get('generation', {})
        
        # 加载模型和分词器
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            **self.params
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model.eval()
        
        # 设置默认生成参数
        self.generation_params['pad_token_id'] = self.tokenizer.eos_token_id
        self.generation_params['eos_token_id'] = self.tokenizer.eos_token_id
    
    def get_logits(self, input_text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取模型的logits"""
        inputs = self.tokenizer(input_text, return_tensors="pt", add_special_tokens=False)
        input_ids_tensor = inputs.input_ids.to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids_tensor, output_hidden_states=False, output_attentions=False)
            # 获取最后一个token的logits
            next_token_logits = outputs.logits[:, -1, :]
            next_token_probs = F.softmax(next_token_logits, dim=-1)
        
        return next_token_logits, next_token_probs
    
    def generate(self, input_text: str) -> str:
        """生成模型输出"""
        inputs = self.tokenizer(input_text, return_tensors="pt", add_special_tokens=False)
        input_ids_tensor = inputs.input_ids.to(self.model.device)
        
        with torch.no_grad():
            gen_output = self.model.generate(
                input_ids_tensor, 
                **self.generation_params
            )
            generated_sequence = gen_output[0, input_ids_tensor.shape[1]:]
            prediction_text = self.tokenizer.decode(generated_sequence, skip_special_tokens=True)
        
        return prediction_text
    
    def calculate_entropy(self, probs: torch.Tensor, top_k: Optional[int] = None) -> float:
        """计算熵"""
        if top_k:
            # 计算Top-K熵
            top_k_probs, _ = torch.topk(probs, top_k)
            top_k_probs = top_k_probs / top_k_probs.sum()  # 归一化
            entropy_value = -torch.sum(top_k_probs * torch.log(top_k_probs + 1e-12)).item()
        else:
            # 计算全词表熵
            entropy_value = -torch.sum(probs * torch.log(probs + 1e-12)).item()
        
        return entropy_value
    
    def calculate_margin(self, probs: torch.Tensor) -> float:
        """计算Top1-Top2的margin"""
        top2_values, _ = torch.topk(probs, 2)
        if len(top2_values) >= 2:
            margin = (top2_values[0] - top2_values[1]).item()
        else:
            margin = 1.0
        
        return margin
    
    def get_candidate_probs(self, logits: torch.Tensor, candidate_token_ids: List[int]) -> np.ndarray:
        """获取候选标签的概率"""
        candidate_logits = logits[0, candidate_token_ids]
        candidate_probs = F.softmax(candidate_logits, dim=0).to(torch.float32).cpu().numpy()
        return candidate_probs
    
    def apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """应用聊天模板"""
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
