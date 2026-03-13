import torch
try:
    import torch_gcu
    from torch_gcu import transfer_to_gcu
except Exception as e:
    pass # 忽略非 GCU 环境的错误
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer,logging
import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
logging.set_verbosity_error()
# ==========================================================
# 1. 配置区
# ==========================================================
# 你的评估结果 CSV (作为待修正的数据源)
CSV_PATH = "/home/jgy/paper-prompt/prompt-order/last8/v1(no-prototype)/Qwen2.5-7B-Instruct_full_metrics.csv"
# 你的模型路径
MODEL_PATH = "/data/models/Qwen2.5-7B-Instruct"

# 你的知识库数据路径 (通常是训练集或全量数据集 ace.json)
# 这里的逻辑是：我们从这里面检索"相似且正确"的例子给模型看
KNOWLEDGE_BASE_PATH = "/home/jgy/paper-prompt/ACE/train.json"

OUTPUT_DIR = "./robust_correction_framework"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 阈值策略: 'percentile' (百分位) 或 'absolute' (绝对值)
THRESHOLD_MODE = 'percentile' 
RETHINK_PERCENTILE = 0.20  # 重想最不确定的 20%

# 检索配置
RETRIEVAL_TOP_K = 3  # 每次给模型看 3 个类似例子

# 文本定义 (保持和之前一致)
FIXED_TASK_TEXT = """--- Task ---
Please select the most appropriate type of the following sentence from the type list.The type list is : ['declare bankruptcy', 'transfer ownership', 'transfer money', 'marry', 'transport', 'die', 'phone write', 'arrest jail', 'convict', 'sentence', 'sue', 'end organization', 'start organization', 'end position', 'start position', 'meet', 'elect', 'attack', 'injure', 'born', 'fine', 'release parole', 'charge indict', 'extradite', 'trial hearing', 'demonstrate', 'divorce', 'nominate', 'appeal', 'pardon', 'execute', 'acquit', 'merge organization'],You must choose one type from the type list and follow the examples output. Do not include any additional text, explanations, or notes - only output the selected type."""


ALL_TYPES = ['declare bankruptcy', 'transfer ownership', 'transfer money', 'marry', 'transport', 'die', 'phone write', 'arrest jail', 'convict', 'sentence', 'sue', 'end organization', 'start organization', 'end position', 'start position', 'meet', 'elect', 'attack', 'injure', 'born', 'fine', 'release parole', 'charge indict', 'extradite', 'trial hearing', 'demonstrate', 'divorce', 'nominate', 'appeal', 'pardon', 'execute', 'acquit', 'merge organization']

# ==========================================================
# 2. 核心模块类定义
# ==========================================================

class MetricCalculator:
    """负责计算指标和确定阈值"""
    @staticmethod
    def calc_metric(df):
        # Log-Scale Focal Uncertainty
        df['Focal_Raw'] = df['first_token_entropy_top50'] * ((1 - df['pred_prior_prob']) ** 2)
        df['Metric_Score'] = np.log10(df['Focal_Raw'] + 1e-9)
        return df

    @staticmethod
    def get_threshold(df, mode='percentile', value=0.20):
        # 分数越高 = 越不确定
        if mode == 'percentile':
            # 找到 Top 20% 大的分数作为阈值
            threshold = df['Metric_Score'].quantile(1.0 - value)
            return threshold
        return value

class KnowledgeRetriever:
    """负责构建向量索引和检索相似样本"""
    def __init__(self, dataset_path):
        print("正在构建知识库索引 (Embedding Knowledge Base)...")
        self.encoder = SentenceTransformer('/home/jgy/paper-prompt/prompt-order/last8/all-MiniLM-L6-v2') # 轻量级高效模型
        self.examples = self._load_data(dataset_path)
        
        # 提取所有句子进行编码
        sentences = [ex['sentence'] for ex in self.examples]
        self.embeddings = self.encoder.encode(sentences, show_progress_bar=True, convert_to_numpy=True)
        print(f"知识库构建完成，共 {len(self.examples)} 条样本。")

    def _load_data(self, path):
        # 加载 json 数据，提取 sentence 和 label
        with open(path, 'r') as f:
            data = json.load(f)
        processed = []
        for item in data:
            try:
                sent = item["messages"][1]["content"]
                # 假设 label 在 messages[2]
                lbl = item["messages"][2]["content"].split('#')[0].strip()
                processed.append({'sentence': sent, 'label': lbl})
            except:
                continue
        return processed

    def retrieve(self, query_sentence, k=3, exclude_sentence=None):
        """检索 Top-K 相似样本"""
        query_emb = self.encoder.encode([query_sentence])
        
        # 计算余弦相似度
        similarities = cosine_similarity(query_emb, self.embeddings)[0]
        
        # 排序 (从大到小)
        top_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in top_indices:
            candidate = self.examples[idx]
            # 排除掉自己 (防止泄漏答案)
            if exclude_sentence and candidate['sentence'] == exclude_sentence:
                continue
            
            # 添加到结果
            results.append(candidate)
            if len(results) >= k:
                break
        return results

class CorrectionAgent:
    """负责执行 LLM 推理和修正"""
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map='auto'
        )
        self.model.eval()

    def parse_prediction(self, text):
        # 简单的后处理
        text = text.lower()
        for t in ALL_TYPES:
            if t.lower() in text:
                return t
        return "unclassified"

    def rethink(self, row, similar_examples, original_pred):
        sentence = row['sentence']
        
        # === 关键步骤：构造 "Reference-Augmented" Prompt ===
        # 不只是让它重想，而是给它看类似的正确例子
        
        examples_text = ""
        for i, ex in enumerate(similar_examples):
            examples_text += f"Ref Example {i+1}:\nInput: \"{ex['sentence']}\"\nType: \"{ex['label']}\"\n\n"
            
        rethink_prompt = (
            f"You previously predicted: \"{original_pred}\".\n"
            f"However, this prediction has high uncertainty.\n\n"
            f"Here are {len(similar_examples)} semantically similar examples with their CORRECT types:\n"
            f"{examples_text}"
            f"Based on these references, re-evaluate the input sentence: \"{sentence}\"\n"
            f"What is the most appropriate type? Output ONLY the type."
        )

        messages = [
            {"role": "system", "content": FIXED_TASK_TEXT},
            {"role": "user", "content": f"Input: {sentence}"},
            {"role": "assistant", "content": original_pred},
            {"role": "user", "content": rethink_prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=30, pad_token_id=self.tokenizer.eos_token_id)
            
        gen_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return self.parse_prediction(gen_text)

# ==========================================================
# 3. 主流程执行
# ==========================================================
def main():
    # 1. 加载与计算指标
    print("Step 1: 计算 Log-Scale Focal Uncertainty...")
    df = pd.read_csv(CSV_PATH)
    
    # 筛选正确的数据行 (确保列存在)
    df_filtered = df[df['order'].isin(['Examples->Task'])].copy()
    required_cols = ['first_token_entropy_top50', 'pred_prior_prob', 'sentence', 'pred_label', 'accuracy']
    df = df_filtered.dropna(subset=required_cols).reset_index(drop=True)
    
    # 计算
    df = MetricCalculator.calc_metric(df)
    initial_acc = df['accuracy'].mean()
    print(f"原始准确率: {initial_acc:.2%}")

    # 2. 确定阈值与目标样本
    threshold = MetricCalculator.get_threshold(df, mode=THRESHOLD_MODE, value=RETHINK_PERCENTILE)
    print(f"不确定性阈值 (Top {RETHINK_PERCENTILE*100}%): {threshold:.4f}")
    
    df['needs_rethink'] = df['Metric_Score'] > threshold
    rethink_df = df[df['needs_rethink']]
    print(f"需要重想的样本: {len(rethink_df)} / {len(df)}")

    # 3. 初始化模块
    retriever = KnowledgeRetriever(KNOWLEDGE_BASE_PATH)
    agent = CorrectionAgent(MODEL_PATH)

    # 4. 执行修正循环
    print("\nStep 4: 开始检索增强修正 (RAG-Correction)...")
    improved = 0
    worsened = 0
    unchanged = 0
    
    results = []

    for idx, row in tqdm(rethink_df.iterrows(), total=len(rethink_df)):
        # A. 检索相似样本 (排除自己)
        similar_exs = retriever.retrieve(row['sentence'], k=RETRIEVAL_TOP_K, exclude_sentence=row['sentence'])
        
        # B. 让模型参考这些样本重想
        new_label = agent.rethink(row, similar_exs, row['pred_label'])
        
        # C. 评估
        true_label = row['true_label']
        orig_correct = (str(row['pred_label']).lower() == str(true_label).lower())
        new_correct = (str(new_label).lower() == str(true_label).lower())
        
        # 统计动态
        if not orig_correct and new_correct:
            improved += 1
        elif orig_correct and not new_correct:
            worsened += 1
        else:
            unchanged += 1
            
        # 更新记录
        df.at[idx, 'final_label'] = new_label
        df.at[idx, 'final_accuracy'] = 1 if new_correct else 0
        df.at[idx, 'is_rethought'] = True

    # 填充未重想的样本
    df.loc[~df['needs_rethink'], 'final_label'] = df.loc[~df['needs_rethink'], 'pred_label']
    df.loc[~df['needs_rethink'], 'final_accuracy'] = df.loc[~df['needs_rethink'], 'accuracy']
    df['is_rethought'] = df['is_rethought'].fillna(False)

    # 5. 最终报告
    final_acc = df['final_accuracy'].mean()
    
    print("\n" + "="*50)
    print("🚀 最终优化报告 (Robust RAG-Correction)")
    print("="*50)
    print(f"Baseline 准确率: {initial_acc:.2%}")
    print(f"Refined  准确率: {final_acc:.2%}")
    print(f"净提升: {final_acc - initial_acc:+.2%}")
    print("-" * 30)
    print(f"触发重想机制样本数: {len(rethink_df)}")
    print(f"  [+] 成功纠错 (Wrong->Right): {improved}")
    print(f"  [-] 误伤友军 (Right->Wrong): {worsened}")
    print(f"  [=] 结果不变: {unchanged}")
    print(f"  >>> 净收益 (纠错 - 误伤): {improved - worsened}")
    
    # 保存
    df.to_csv(os.path.join(OUTPUT_DIR, "rag_refined_results.csv"), index=False)
    print(f"结果已保存至 {OUTPUT_DIR}")

if __name__ == "__main__":
    main()