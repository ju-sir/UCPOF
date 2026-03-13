import os
import random
import pandas as pd
from tqdm import tqdm
import yaml
from src.llm_engine import LLMEngine
from src.rag_retriever import RAGRetriever
from src.metric import MetricCalculator
from src.prompt_manager import PromptManager
from src.utils import parse_prediction_robust, save_metrics_to_csv, calculate_dataset_priors

class UCPOFPipeline:
    def __init__(self, dataset_config_path, model_config_path, output_dir="./output"):
        """初始化UCPOF管道"""
        # 加载配置
        with open(dataset_config_path, 'r') as f:
            self.dataset_config = yaml.safe_load(f)
        
        with open(model_config_path, 'r') as f:
            self.model_config = yaml.safe_load(f)
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化组件
        self.llm_engine = LLMEngine(self.model_config)
        self.rag_retriever = RAGRetriever()
        self.metric_calculator = MetricCalculator()
        self.prompt_manager = PromptManager(self.dataset_config)
        
        # 获取数据集配置
        self.all_types = self.dataset_config.get('all_types', [])
        self.dataset_path = self.dataset_config['paths']['dataset']
        self.prior_dataset_path = self.dataset_config['paths']['prior_dataset']
    
    def load_dataset(self, dataset_path):
        """加载数据集"""
        import json
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
    
    def build_rag_index(self, dataset):
        """构建RAG索引"""
        corpus = [item['sentence'] for item in dataset]
        self.rag_retriever.build_index(corpus)
    
    def run_offline(self, num_samples=5000):
        """运行离线流程，提取特征"""
        # 加载数据集
        full_dataset = self.load_dataset(self.dataset_path)
        if not full_dataset:
            print("Error: Could not load dataset.")
            return None
        
        # 采样部分数据用于评估
        evaluation_data = random.sample(full_dataset, min(num_samples, len(full_dataset)))
        
        # 计算先验概率
        prior_full_dataset = self.load_dataset(self.prior_dataset_path)
        if not prior_full_dataset:
            print("Error: Could not load prior dataset.")
            return None
        prior_map = calculate_dataset_priors(prior_full_dataset, self.all_types)
        
        # 构建RAG索引
        self.build_rag_index(full_dataset)
        
        # 提取特征
        results = []
        for item in tqdm(evaluation_data, desc="Running offline extraction"):
            sentence = item["sentence"]
            true_label = item["label"]
            
            # 尝试两种Prompt顺序
            for prompt_type in ["examples_task", "task_examples"]:
                # 构建Prompt
                messages = self.prompt_manager.build_prompt(sentence, prompt_type=prompt_type)
                full_prompt_text = self.prompt_manager.get_chat_template(messages, self.llm_engine.tokenizer)
                
                # 获取logits和概率
                next_token_logits, next_token_probs = self.llm_engine.get_logits(full_prompt_text)
                
                # 预处理：获取所有候选Label的首个Token ID
                candidate_token_ids = []
                for label_type in self.all_types:
                    ids = self.llm_engine.tokenizer.encode(f' {label_type}', add_special_tokens=False)
                    if len(ids) > 0:
                        candidate_token_ids.append(ids[0])
                
                # 计算候选标签的概率
                candidate_probs = self.llm_engine.get_candidate_probs(next_token_logits, candidate_token_ids)
                
                # 计算指标
                label_entropy = self.metric_calculator.calculate_label_entropy(candidate_probs)
                first_token_entropy_top50 = self.llm_engine.calculate_entropy(next_token_probs[0], top_k=50)
                first_token_entropy_top500 = self.llm_engine.calculate_entropy(next_token_probs[0], top_k=500)
                first_token_entropy_full = self.llm_engine.calculate_entropy(next_token_probs[0])
                logit_margin = self.llm_engine.calculate_margin(next_token_probs[0])
                
                # 生成预测
                prediction_text = self.llm_engine.generate(full_prompt_text)
                pred_label = parse_prediction_robust(prediction_text, self.all_types)
                
                # 获取预测Label的先验概率
                pred_label_prior = prior_map.get(pred_label, 0.0)
                
                # 计算准确率
                if pred_label == "unclassified" or pred_label not in self.all_types:
                    accuracy = 0
                else:
                    accuracy = 1 if pred_label.lower() == true_label.lower() else 0
                
                # 计算LSFU分数
                lsfu_score = self.metric_calculator.calculate_lsfu_score(label_entropy, logit_margin, pred_label_prior)
                
                # 收集结果
                results.append({
                    "order": prompt_type,
                    "accuracy": accuracy,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "label_entropy": label_entropy,
                    "first_token_entropy_top50": first_token_entropy_top50,
                    "first_token_entropy_top500": first_token_entropy_top500,
                    "first_token_entropy_full": first_token_entropy_full,
                    "logit_margin": logit_margin,
                    "pred_prior_prob": pred_label_prior,
                    "lsfu_score": lsfu_score,
                    "sentence": sentence
                })
        
        # 保存结果
        df = pd.DataFrame(results)
        model_name = os.path.basename(os.path.normpath(self.model_config['model_path']))
        csv_path = os.path.join(self.output_dir, f"{model_name}_offline_metrics.csv")
        save_metrics_to_csv(results, csv_path)
        
        return df
    
    def run_online(self, input_text, use_rag=True):
        """运行在线流程，处理单个输入"""
        # 构建RAG上下文
        rag_context = None
        if use_rag:
            rag_context = self.rag_retriever.get_context(input_text)
        
        # 构建Prompt
        messages = self.prompt_manager.build_prompt(input_text, rag_context=rag_context)
        full_prompt_text = self.prompt_manager.get_chat_template(messages, self.llm_engine.tokenizer)
        
        # 获取logits和概率
        next_token_logits, next_token_probs = self.llm_engine.get_logits(full_prompt_text)
        
        # 预处理：获取所有候选Label的首个Token ID
        candidate_token_ids = []
        for label_type in self.all_types:
            ids = self.llm_engine.tokenizer.encode(f' {label_type}', add_special_tokens=False)
            if len(ids) > 0:
                candidate_token_ids.append(ids[0])
        
        # 计算候选标签的概率
        candidate_probs = self.llm_engine.get_candidate_probs(next_token_logits, candidate_token_ids)
        
        # 计算指标
        label_entropy = self.metric_calculator.calculate_label_entropy(candidate_probs)
        logit_margin = self.llm_engine.calculate_margin(next_token_probs[0])
        
        # 生成预测
        prediction_text = self.llm_engine.generate(full_prompt_text)
        pred_label = parse_prediction_robust(prediction_text, self.all_types)
        
        return {
            "input": input_text,
            "prediction": pred_label,
            "label_entropy": label_entropy,
            "logit_margin": logit_margin,
            "rag_used": use_rag
        }
