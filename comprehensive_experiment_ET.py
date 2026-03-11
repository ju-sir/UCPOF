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
from scipy.stats import entropy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
logging.set_verbosity_error()
# ==============================================================================
# 1. 全局实验配置 (Experimental Configuration)
# ==============================================================================
# [Input] 训练集指标 CSV (用于计算阈值、挑选Few-shot)
TRAIN_CSV_PATH = "/home/jgy/paper-prompt/prompt-order/last8-ace/v1(no-prototype)/Qwen2.5-14B-Instruct_full_metrics.csv"

# [输入] 测试集原始数据 (ACE test set)
TEST_DATASET_PATH = "/home/jgy/paper-prompt/ACE/dev.json" # 请修改为你的测试集路径

# [Input] 知识库源数据 (通常是训练集)
KB_DATA_PATH = "/home/jgy/paper-prompt/ACE/train.json"

MODEL_PATH = "/data/models/Qwen2.5-14B-Instruct"
OUTPUT_DIR = "./final_comprehensive_study"

# TARGET_LABEL = "Examples->Task"
TARGET_LABEL = "Task->Examples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 核心超参数 ---
# 1. 动态阈值策略: 覆盖训练集中 90% 的错误样本
TARGET_ERROR_COVERAGE = 0.90 

# 2. RAG 参数
STATIC_SHOT_NUM = 3    # 黄金 Few-shot 数量
RAG_RETRIEVAL_NUM = 3  # 反思时检索的参考样本数

# 3. 实验模式
# True: 对所有样本都跑一遍 RAG (为了画完整的 Pareto 曲线，耗时较长)
# False: 只对超过阈值的样本跑 RAG (模拟真实生产环境，速度快)
# 建议先设为 False 跑通，写论文作图时设为 True
RUN_FULL_RAG_ANALYSIS = True 

ALL_TYPES = ['declare bankruptcy', 'transfer ownership', 'transfer money', 'marry', 'transport', 'die', 'phone write', 'arrest jail', 'convict', 'sentence', 'sue', 'end organization', 'start organization', 'end position', 'start position', 'meet', 'elect', 'attack', 'injure', 'born', 'fine', 'release parole', 'charge indict', 'extradite', 'trial hearing', 'demonstrate', 'divorce', 'nominate', 'appeal', 'pardon', 'execute', 'acquit', 'merge organization']

# 原始 Baseline Prompt (用于 Stage 1 对比)
FIXED_GOOD_EXAMPLE_TEXT = """--- EXAMPLES ---
Example 1:
Input: "I visited all their families ."
Output: "meet"
Example 2:
Input: "He claimed Iraqi troops had destroyed five tanks ."
Output: "attack"
Example 3:
Input: "Another appeal is now pending in the Federal Court ."
Output: "appeal"
"""
FIXED_TASK_TEXT = """--- Task ---
Please select the most appropriate type of the following sentence from the type list.The type list is : ['declare bankruptcy', 'transfer ownership', 'transfer money', 'marry', 'transport', 'die', 'phone write', 'arrest jail', 'convict', 'sentence', 'sue', 'end organization', 'start organization', 'end position', 'start position', 'meet', 'elect', 'attack', 'injure', 'born', 'fine', 'release parole', 'charge indict', 'extradite', 'trial hearing', 'demonstrate', 'divorce', 'nominate', 'appeal', 'pardon', 'execute', 'acquit', 'merge organization'],You must choose one type from the type list and follow the examples output. Do not include any additional text, explanations, or notes - only output the selected type."""

# ==============================================================================
# 2. 准备阶段 (Offline Preparation)
# ==============================================================================
class DataDrivenPreparer:
    def __init__(self, train_csv_path, kb_data_path):
        print(f">>> [Init] Loading Training Metrics from {train_csv_path}...")
        self.df = pd.read_csv(train_csv_path)
        
        # 数据清洗
        cols_check = ['accuracy', 'first_token_entropy_top50', 'pred_prior_prob', 'sentence', 'true_label']
        self.df = self.df.dropna(subset=cols_check)

        target_label = TARGET_LABEL
        self.df = self.df[self.df['order'] == target_label].copy()
        
        # 兼容性检查：确保有 NLL
        if 'nll_ground_truth' not in self.df.columns:
            print("Warning: 'nll_ground_truth' missing. Fallback to Entropy for sorting.")
            self.df['nll_ground_truth'] = self.df['first_token_entropy_top50']

        # 计算 Log-Focal Uncertainty
        self.df['Focal_Raw'] = self.df['first_token_entropy_top50'] * ((1 - self.df['pred_prior_prob']) ** 2)
        self.df['Metric_Score'] = np.log10(self.df['Focal_Raw'] + 1e-9)
        
        self.kb_data_path = kb_data_path

    def get_dynamic_threshold(self):
        """策略：Error Coverage Thresholding"""
        error_df = self.df[self.df['accuracy'] == 0]
        if len(error_df) == 0: return -1.0 # Fallback
        
        # 找到能覆盖 X% 错误的阈值
        threshold = error_df['Metric_Score'].quantile(1.0 - TARGET_ERROR_COVERAGE)
        
        # 统计信息
        trigger_rate = (self.df['Metric_Score'] > threshold).mean()
        print(f"[Prep] Dynamic Threshold: {threshold:.4f}")
        print(f"       (Target Error Coverage: {TARGET_ERROR_COVERAGE:.0%})")
        print(f"       (Expected Trigger Rate: {trigger_rate:.2%})")
        return threshold

    def get_prior_map(self):
        total = len(self.df)
        counts = self.df['true_label'].value_counts().to_dict()
        prior_map = {k: v/total for k, v in counts.items()}
        for t in ALL_TYPES:
            if t not in prior_map: prior_map[t] = 1e-6
        return prior_map

    def select_gold_few_shots(self):
        """策略：Dual-Key Sorting (Score + NLL)"""
        print(f"[Prep] Selecting Top-{STATIC_SHOT_NUM} Gold Examples...")
        candidates = self.df[self.df['accuracy'] == 1].copy()
        
        # 双重排序：越小越好
        candidates = candidates.sort_values(by=['Metric_Score', 'nll_ground_truth'],
         ascending=[True, True])
        
        selected_shots = []
        seen_labels = set()
        
        for _, row in candidates.iterrows():
            lbl = row['true_label']
            if lbl not in seen_labels:
                selected_shots.append({
                    'sentence': row['sentence'],
                    'label': lbl
                })
                seen_labels.add(lbl)
            if len(selected_shots) >= STATIC_SHOT_NUM:
                break
        return selected_shots

    def build_knowledge_base(self):
        print("[Prep] Building Vector Knowledge Base...")
        sentences = self.df['sentence'].tolist()
        labels = self.df['true_label'].tolist()
        encoder = SentenceTransformer('/home/jgy/paper-prompt/prompt-order/last8-ace/all-MiniLM-L6-v2')
        embeddings = encoder.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
        print(f"[准备] 知识库构建完成，索引了 {len(sentences)} 条样本。")
        return {'sentences': sentences, 'labels': labels, 'embeddings': embeddings, 'encoder': encoder}

# ==============================================================================
# 3. 综合推理引擎 (Comprehensive Inference Engine)
# ==============================================================================
class ComprehensiveInferenceEngine:
    def __init__(self, model_path, prior_map, threshold, gold_shots, kb):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto')
        self.model.eval()
        
        self.prior_map = prior_map
        self.threshold = threshold
        self.kb = kb
        
        # 预计算 Candidate IDs
        self.candidate_ids = []
        for t in ALL_TYPES:
            ids = self.tokenizer.encode(f' {t}', add_special_tokens=False)
            if ids: self.candidate_ids.append(ids[0])
        self.candidate_tensor = torch.tensor(self.candidate_ids, device=self.model.device)

        # Prompt 1: Baseline
        if TARGET_LABEL == "Examples->Task" :
            self.prompt_baseline = FIXED_GOOD_EXAMPLE_TEXT + "\n" + FIXED_TASK_TEXT
        else: self.prompt_baseline = FIXED_TASK_TEXT + "\n" + FIXED_GOOD_EXAMPLE_TEXT


        # Prompt 2: Optimized Static
        self.prompt_optimized = self._build_prompt(gold_shots)

    def _build_prompt(self, shots):
        text = "--- EXAMPLES ---\n"
        for i, ex in enumerate(shots):
            text += f"Example {i+1}:\nInput: \"{ex['sentence']}\"\nOutput: \"{ex['label']}\"\n"
        return text + "\n" + FIXED_TASK_TEXT

    def parse_pred(self, text, debug: bool = False):
        """
        适配 ["neutral", "contradiction", "entailment"] 的鲁棒解析函数
        核心优化：
        1. 移除标点干扰，统一文本格式
        2. 保留精确引号匹配 + 模糊匹配核心逻辑
        3. 增加调试日志，方便定位问题
        4. 兼容带前缀的输出格式（可选）
        """
        # 保存原始文本（用于调试）
        original_text = text
        # 步骤1：预处理 - 小写、去空格、移除所有标点/特殊字符
        text = text.lower().strip()
        clean_text = (text.replace('"', '')
                    .replace("'", '')
                    .replace(",", '')
                    .replace(".", '')
                    .replace(":", '')
                    .replace(";", '')
                    .replace("!", '')
                    .replace("?", ''))
        
        if debug:
            print(f"原始文本：{original_text} | 清洗后：{clean_text}")
        
        # 步骤2：优先精确匹配引号（核心逻辑不变）
        for t in ALL_TYPES:
            t_lower = t.lower()
            if f'"{t_lower}"' in text or f"'{t_lower}'" in text:
                if debug:
                    print(f"✅ 精确引号匹配：{t}")
                return t
        
        # 步骤3：适配带前缀的输出（新增，增强通用性）
        prefixes = ["answer:", "output:", "type:", "result:"]
        for prefix in prefixes:
            if clean_text.startswith(prefix):
                area = clean_text[len(prefix):].strip()
                for t in ALL_TYPES:
                    t_lower = t.lower()
                    if area.startswith(t_lower):
                        if debug:
                            print(f"✅ 前缀匹配（{prefix}）：{t}")
                        return t
        
        # 步骤4：模糊匹配（核心逻辑不变，基于清洗后的文本）
        for t in ALL_TYPES:
            t_lower = t.lower()
            if t_lower in clean_text:
                if debug:
                    print(f"✅ 模糊匹配：{t}")
                return t
        
        # 所有匹配失败
        if debug:
            print(f"❌ 无匹配结果，返回 unclassified")
        return "unclassified"

    def calc_score(self, logits, label):
        logits = logits[:, -1, :]
        cand_logits = logits[0, self.candidate_tensor]
        probs = F.softmax(cand_logits, dim=0).to(torch.float32).cpu().numpy()
        ent = float(entropy(probs, base=2))
        prior = self.prior_map.get(label, 1e-6)
        return np.log10(ent * ((1 - prior) ** 2) + 1e-9)

    def retrieve(self, query, k=3):
        emb = self.kb['encoder'].encode([query])
        sims = cosine_similarity(emb, self.kb['embeddings'])[0]
        idxs = np.argsort(sims)[::-1][:k]
        return [{'sentence': self.kb['sentences'][i], 'label': self.kb['labels'][i]} for i in idxs]

    def run_pipeline(self, sample):
        """
        一次性运行 Stage 1 -> Stage 2 -> Check -> Stage 3
        """
        sentence = sample['sentence']
        true_label = sample['label']
        
        # --- Stage 1: Baseline (Fixed Prompt) ---
        msgs_1 = [{"role": "system", "content": self.prompt_baseline}, {"role": "user", "content": f"Input: {sentence}\nOutput: "}]
        enc_1 = self.tokenizer.apply_chat_template(msgs_1, tokenize=False, add_generation_prompt=True)
        inp_1 = self.tokenizer(enc_1, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        
        with torch.no_grad():
            gen_1 = self.model.generate(**inp_1, max_new_tokens=20, pad_token_id=self.tokenizer.eos_token_id, do_sample=False)
            txt_1 = self.tokenizer.decode(gen_1[0][inp_1.input_ids.shape[1]:], skip_special_tokens=True)
        
        pred_1 = self.parse_pred(txt_1)
        acc_1 = 1 if pred_1.lower() == true_label.lower() else 0

        # --- Stage 2: Optimized Static (Gold Shots) ---
        msgs_2 = [{"role": "system", "content": self.prompt_optimized}, {"role": "user", "content": f"Input: {sentence}\nOutput: "}]
        enc_2 = self.tokenizer.apply_chat_template(msgs_2, tokenize=False, add_generation_prompt=True)
        inp_2 = self.tokenizer(enc_2, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        
        with torch.no_grad():
            out_2 = self.model(input_ids=inp_2.input_ids) # 需要 Logits
            logits_2 = out_2.logits
            gen_2 = self.model.generate(**inp_2, max_new_tokens=20, pad_token_id=self.tokenizer.eos_token_id, do_sample=False)
            txt_2 = self.tokenizer.decode(gen_2[0][inp_2.input_ids.shape[1]:], skip_special_tokens=True)
        
        pred_2 = self.parse_pred(txt_2)
        acc_2 = 1 if pred_2.lower() == true_label.lower() else 0
        
        # --- Metric Check ---
        score = self.calc_score(logits_2, pred_2)
        
        # --- Stage 3: Dynamic RAG ---
        # 逻辑：如果 score > threshold (或者强制跑全量分析)，则执行 RAG
        # 否则：Dynamic 结果 = Stage 2 结果
        
        pred_3 = pred_2
        acc_3 = acc_2
        is_triggered = False
        
        should_run_rag = (score > self.threshold) or RUN_FULL_RAG_ANALYSIS
        
        if should_run_rag:
            is_triggered = (score > self.threshold) # 只有真正超过阈值才算 Triggered
            
            # Retrieval
            sim_exs = self.retrieve(sentence, k=RAG_RETRIEVAL_NUM)
            ref_text = ""
            for i, ex in enumerate(sim_exs):
                ref_text += f"Ref {i+1}:\nInput: \"{ex['sentence']}\"\nType: \"{ex['label']}\"\n\n"
            
            rethink_prompt = (
                f"You previously predicted: \"{pred_2}\". This prediction is uncertain.\n"
                f"Here are similar examples with CORRECT types:\n{ref_text}"
                f"Based on these, re-evaluate: \"{sentence}\"\nOutput ONLY the type."
            )
            
            msgs_3 = msgs_2 + [{"role": "assistant", "content": pred_2}, {"role": "user", "content": rethink_prompt}]
            enc_3 = self.tokenizer.apply_chat_template(msgs_3, tokenize=False, add_generation_prompt=True)
            inp_3 = self.tokenizer(enc_3, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                gen_3 = self.model.generate(**inp_3, max_new_tokens=20, pad_token_id=self.tokenizer.eos_token_id, do_sample=False)
                txt_3 = self.tokenizer.decode(gen_3[0][inp_3.input_ids.shape[1]:], skip_special_tokens=True)
            
            pred_rag_result = self.parse_pred(txt_3)
            
            # 如果是真正触发，更新 pred_3
            if is_triggered:
                pred_3 = pred_rag_result
                acc_3 = 1 if pred_3.lower() == true_label.lower() else 0
            
            # 如果是强制跑全量但未触发阈值，我们记录下 "Potential RAG Result" 供分析，但不改变主流程的 acc_3
            potential_rag_pred = pred_rag_result
        else:
            potential_rag_pred = pred_2 # 没跑RAG，就认为和Stage2一样

        return {
            "sentence": sentence,
            "true_label": true_label,
            "acc_baseline": acc_1,
            "acc_static": acc_2,
            "acc_dynamic": acc_3,
            "focal_score": score,
            "triggered": is_triggered,
            "pred_rag_potential": potential_rag_pred # 用于后续画 Pareto 曲线
        }

# ==============================================================================
# 4. 主程序
# ==============================================================================
def load_test_data(path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except: return []
    test_samples = []
    for item in data:
        try:
            sent = item["messages"][1]["content"]
            lbl = item["messages"][2]["content"].split('#')[0].strip()
            test_samples.append({'sentence': sent, 'label': lbl})
        except: continue
    return test_samples

def main():
    print(">>> 1. Offline Preparation (Data-Driven)...")
    preparer = DataDrivenPreparer(TRAIN_CSV_PATH, KB_DATA_PATH)
    threshold = preparer.get_dynamic_threshold()
    prior_map = preparer.get_prior_map()
    gold_shots = preparer.select_gold_few_shots()
    kb = preparer.build_knowledge_base()
    
    print("\n>>> 2. Initializing Comprehensive Inference Engine...")
    engine = ComprehensiveInferenceEngine(MODEL_PATH, prior_map, threshold, gold_shots, kb)
    
    test_samples = load_test_data(TEST_DATASET_PATH)
    print(f"\n>>> 3. Running Inference on {len(test_samples)} Test Samples...")
    
    results = []
    for sample in tqdm(test_samples):
        res = engine.run_pipeline(sample)
        results.append(res)
        
    df = pd.DataFrame(results)
    
    # --- 生成报告 ---
    print("\n" + "="*60)
    print("🏆 FINAL PAPER EXPERIMENT REPORT")
    print("="*60)
    print(f"1. Baseline (Fixed Prompt) Acc : {df['acc_baseline'].mean():.2%}")
    print(f"2. Static Optimized Acc        : {df['acc_static'].mean():.2%}")
    print(f"3. Dynamic RAG Acc             : {df['acc_dynamic'].mean():.2%}")
    print("-" * 60)
    print(f"Improvement (Base -> Static)   : {df['acc_static'].mean() - df['acc_baseline'].mean():+.2%}")
    print(f"Improvement (Static -> Dynamic): {df['acc_dynamic'].mean() - df['acc_static'].mean():+.2%}")
    print(f"Total Gain                     : {df['acc_dynamic'].mean() - df['acc_baseline'].mean():+.2%}")
    print("-" * 60)
    print(f"RAG Trigger Rate               : {df['triggered'].mean():.2%}")
    
    # 计算全量 RAG 的潜在准确率 (用于 Efficiency Analysis)
    # 逻辑：如果 Trigger Rate = 100%，准确率是多少？
    # 这需要比较 'pred_rag_potential' 和 'true_label'
    if RUN_FULL_RAG_ANALYSIS:
        df['acc_full_rag'] = df.apply(lambda r: 1 if r['pred_rag_potential'].lower() == r['true_label'].lower() else 0, axis=1)
        print(f"Potential Full RAG Acc         : {df['acc_full_rag'].mean():.2%}")
        print("(Use this to plot the Pareto Efficiency Curve in paper)")
    else:
        print("(Set RUN_FULL_RAG_ANALYSIS=True to get full Pareto data)")

    df.to_csv(os.path.join(OUTPUT_DIR, "final_experiment_results.csv"), index=False)
    print(f"\nResults saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()