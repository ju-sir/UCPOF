import torch
try:
    import torch_gcu
    from torch_gcu import transfer_to_gcu
except Exception as e:
    pass # 忽略非 GCU 环境的错误
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
import json
import numpy as np
import os
from scipy.stats import entropy
from tqdm import tqdm
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from collections import Counter

logging.set_verbosity_error()

# ==========================================================
# 0. 可配置参数
# ==========================================================
NUM_EVALUATION_SENTENCES = 5000
MODEL_PATH_LIST = [
    # "/data/models/Qwen2.5-3B-Instruct",
    # "/data/models/Qwen2.5-7B-Instruct", 
    # "/data/models/Qwen2.5-14B-Instruct",
    "/data/models/Llama-3-8B-Instruct",
]
PRIOR_DATASET_PATH = "/home/jgy/paper-prompt/ACE/train.json"
DATASET_PATH = "/home/jgy/paper-prompt/ACE/train.json"
OUTPUT_DIR = "./v1(no-prototype)"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 文本定义
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

ALL_TYPES = ['declare bankruptcy', 'transfer ownership', 'transfer money', 'marry', 'transport', 'die', 'phone write', 'arrest jail', 'convict', 'sentence', 'sue', 'end organization', 'start organization', 'end position', 'start position', 'meet', 'elect', 'attack', 'injure', 'born', 'fine', 'release parole', 'charge indict', 'extradite', 'trial hearing', 'demonstrate', 'divorce', 'nominate', 'appeal', 'pardon', 'execute', 'acquit', 'merge organization']

# ==========================================================
# 1. 工具函数
# ==========================================================
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

# ==========================================================
# Step 1: 收集真实世界数据集 Label 分布
# ==========================================================
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

# ==========================================================
# Step 2: 实验核心逻辑
# ==========================================================
def run_validation_experiment(model, tokenizer, system_content, evaluation_data, experiment_name, prior_map, task_text_for_loc=None):
    results = []
    
    # --- 预处理：获取所有候选 Label 的首个 Token ID ---
    # 注意：我们假设每个 label 前面有个空格，这符合 Chat 模板的一般拼接习惯
    candidate_token_ids = []
    valid_indices = []
    for i, label_type in enumerate(ALL_TYPES):
        # 加上空格编码，取第一个 token
        ids = tokenizer.encode(f' {label_type}', add_special_tokens=False)
        if len(ids) > 0:
            candidate_token_ids.append(ids[0])
            valid_indices.append(i)
        else:
            print(f"Warning: Label '{label_type}' tokenizes to empty!")
    
    candidate_token_ids_tensor = torch.tensor(candidate_token_ids, device=model.device)

    for item in tqdm(evaluation_data, desc=f"Running: {experiment_name}"):
        sentence = item["sentence"]
        true_label = item["label"]
        
        # 构建 Prompt
        input_sentence_block = f"""Input: {sentence}\nOutput: """
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": input_sentence_block}
        ]
        full_prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(full_prompt_text, return_tensors="pt", add_special_tokens=False)
        input_ids_tensor = inputs.input_ids.to(model.device)

        with torch.no_grad():
            # -------------------------------------------------------
            # A. Forward Pass (计算 Logits 相关指标: Entropy, Margin)
            # -------------------------------------------------------
            outputs = model(input_ids=input_ids_tensor, output_hidden_states=False, output_attentions=False)
            # 获取最后一个 token 的 logits
            next_token_logits = outputs.logits[:, -1, :] # shape: [1, vocab_size]
            next_token_probs = F.softmax(next_token_logits, dim=-1)

            # 1. Label Entropy (受限搜索空间熵)
            # 只提取 ALL_TYPES 对应的 token logits
            candidate_logits = next_token_logits[0, candidate_token_ids_tensor]
            candidate_probs = F.softmax(candidate_logits, dim=0).to(torch.float32).cpu().numpy()
            label_entropy = safe_entropy(candidate_probs)

            # 2. First Token Entropy (全词表熵) & Logit Margin
            # 计算全词表的 Top-K 熵 (K=50)
            top_50_probs, _ = torch.topk(next_token_probs[0], 50)
            top_50_probs = top_50_probs / top_50_probs.sum() # Normalize
            first_token_entropy_top50 = -torch.sum(top_50_probs * torch.log(top_50_probs + 1e-12)).item()
            # 计算全词表的 Top-K 熵 (K=500)
            top_500_probs, _ = torch.topk(next_token_probs[0], 500)
            top_500_probs = top_500_probs / top_500_probs.sum() # Normalize
            first_token_entropy_top500 = -torch.sum(top_500_probs * torch.log(top_500_probs + 1e-12)).item()
            # 计算全词表的熵
            full_vocab_probs = next_token_probs[0]
            first_token_entropy_full = -torch.sum(full_vocab_probs * torch.log(full_vocab_probs + 1e-12)).item()

            # 计算 Margin (Top1 - Top2)
            top2_values, _ = torch.topk(next_token_probs[0], 2)
            if len(top2_values) >= 2:
                logit_margin = (top2_values[0] - top2_values[1]).item()
            else:
                logit_margin = 1.0

            # -------------------------------------------------------
            # B. Generation Pass (获取预测结果)
            # -------------------------------------------------------
            gen_output = model.generate(
                input_ids_tensor, 
                max_new_tokens=20, 
                pad_token_id=tokenizer.eos_token_id, 
                eos_token_id=tokenizer.eos_token_id, 
                do_sample=False 
            )
            generated_sequence = gen_output[0, input_ids_tensor.shape[1]:]
            prediction_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
            print(f"model-out:{prediction_text}")
            
            # 解析预测 Label
            pred_label = parse_prediction_robust(prediction_text, ALL_TYPES)
            print(f"jieguo:{pred_label}")
            
            # 获取预测 Label 的先验概率 (Prior Probability)
            # 如果 pred_label 是 'unclassified' 或者不在列表里，概率为 0
            if pred_label in prior_map:
    # 情况 A: 预测结果在有效列表中，查表获取
                pred_label_prior = prior_map[pred_label]
            else:
                # 情况 B: 预测结果是非法字符/幻觉/格式错误
                # 视为极罕见事件，Prior 设为 0 (或者一个极小值如 1e-6)
                pred_label_prior = 0.0
                
                # 【可选】如果你想在后续分析中区分“答错类”和“格式错”，可以给 pred_label 标记一下
                # pred_label = "INVALID_OUTPUT" 

            # 3. 计算 Accuracy (非法输出肯定算错)
            if pred_label == "unclassified" or pred_label not in ALL_TYPES:
                accuracy = 0
            else:
                # 只有在有效列表里才比较文本
                accuracy = 1 if pred_label.lower() == true_label.lower() else 0

            # -------------------------------------------------------
            # C. Ground Truth NLL (需要额外一次 Forward，加上 True Label)
            # -------------------------------------------------------
            true_label_ids = tokenizer.encode(f" {true_label}", add_special_tokens=False)
            full_seq_tensor = torch.cat([
                input_ids_tensor.squeeze(0), 
                torch.tensor(true_label_ids, device=model.device)
            ], dim=0).unsqueeze(0)
            
            full_out = model(full_seq_tensor)
            log_probs_gt = F.log_softmax(full_out.logits.squeeze(0), dim=-1)
            start_idx = input_ids_tensor.shape[1] - 1
            nll_gt = 0.0
            for i, tid in enumerate(true_label_ids):
                if start_idx + i < log_probs_gt.shape[0]:
                    nll_gt -= log_probs_gt[start_idx + i, tid].item()

        # 收集结果
        results.append({
            "order": experiment_name,
            "accuracy": accuracy,
            "true_label": true_label,
            "pred_label": pred_label,
            
            # 要求的指标
            "label_entropy": label_entropy,       # 仅在候选词上的熵
            "first_token_entropy_top50": first_token_entropy_top50, # 全词表Top50熵
            "first_token_entropy_top500": first_token_entropy_top500, # 全词表Top500熵
            "first_token_entropy_full": first_token_entropy_full, # 全词表Top500熵
            "logit_margin": logit_margin,         # Top1-Top2 Margin
            "nll_ground_truth": nll_gt,           # GT NLL
            "pred_prior_prob": pred_label_prior,   # 预测类别的先验概率

            #事件内容
            "sentence":sentence
        })

    return pd.DataFrame(results)

# ==========================================================
# Step 3: 绘图与分析
# ==========================================================
def analyze_and_plot(df, model_name):
    df['Prediction'] = df['accuracy'].map({1: 'Correct', 0: 'Incorrect'})
    
    # 需要分析的指标列表
    metrics = [
        'label_entropy', 
        'first_token_entropy_top50', 
        'first_token_entropy_top500', 
        'first_token_entropy_full',
        'logit_margin', 
        'nll_ground_truth', 
        'pred_prior_prob'
    ]
    
    # 1. 关系矩阵 (Correlation Matrix)
    plt.figure(figsize=(10, 8))
    # 计算包含 accuracy 的相关性
    corr_cols = ['accuracy'] + metrics
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title(f'Metric Correlation Matrix ({model_name})')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_correlation_matrix.png"))
    plt.close()
    
    # 2. 箱线图 (Boxplots) - 预测正确与否和各指标的关系
    # 创建 2行3列 的子图
    fig, axes = plt.subplots(3, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        sns.boxplot(data=df, x='Prediction', y=metric, order=['Correct', 'Incorrect'], palette="Set2", ax=ax)
        
        # 增加标题说明预期的趋势
        trend = ""
        if metric in ['logit_margin', 'pred_prior_prob']:
            trend = "(Higher is usually Better)"
        else:
            trend = "(Lower is usually Better)"
            
        ax.set_title(f"{metric}\n{trend}")
        ax.set_xlabel("")
    
    # 隐藏多余的子图
    for j in range(len(metrics), len(axes)):
        axes[j].axis('off')

    plt.suptitle(f"Metric Distributions by Correctness ({model_name})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_boxplots.png"))
    plt.close()
    
    print(f"Plots saved to {OUTPUT_DIR}")

# ==========================================================
# 4. 主流程
# ==========================================================
def main():
    # A. 加载全部数据计算先验概率
    prior_full_dataset = load_evaluation_data(PRIOR_DATASET_PATH)
    if not prior_full_dataset:
        print("Error: Could not load dataset.")
        return
    # 计算 Prior Map
    prior_map = calculate_dataset_priors(prior_full_dataset, ALL_TYPES)
    print(f"prior_map = {prior_map}")
    

    # B. 加载测试集
    full_dataset = load_evaluation_data(DATASET_PATH)
    if not full_dataset:
        print("Error: Could not load dataset.")
        return
    
    for MODEL_PATH in MODEL_PATH_LIST:
        model_short_name = os.path.basename(os.path.normpath(MODEL_PATH))
        
        # 采样部分数据用于评估 (为了速度)
        evaluation_data = random.sample(full_dataset, min(NUM_EVALUATION_SENTENCES, len(full_dataset)))
        
        print(f"\n######## Testing Features for: {model_short_name} ########")
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16, device_map='auto'
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model.eval()
        
        # 1. Examples -> Task
        system_content_1 = FIXED_GOOD_EXAMPLE_TEXT + "\n" + FIXED_TASK_TEXT
        df1 = run_validation_experiment(
            model, tokenizer, system_content_1, evaluation_data, "Examples->Task", 
            prior_map, task_text_for_loc=FIXED_TASK_TEXT
        )

        # 2. Task -> Examples
        system_content_2 = FIXED_TASK_TEXT + "\n" + FIXED_GOOD_EXAMPLE_TEXT
        df2 = run_validation_experiment(
            model, tokenizer, system_content_2, evaluation_data, "Task->Examples", 
            prior_map, task_text_for_loc=FIXED_TASK_TEXT
        )
        
        full_df = pd.concat([df1, df2], ignore_index=True)
        
        # 数据统计
        print("\n" + "="*50)
        print(f" Summary for {model_short_name}")
        print("="*50)
        print(full_df.groupby(['order', 'accuracy'])[['label_entropy', 'logit_margin', 'pred_prior_prob']].mean().to_string())
        
        # 绘图
        analyze_and_plot(full_df, model_short_name)
        
        # 保存详细CSV
        csv_path = os.path.join(OUTPUT_DIR, f"{model_short_name}_full_metrics.csv")
        full_df.to_csv(csv_path, index=False)
        print(f"Data saved to {csv_path}")

if __name__ == '__main__':
    main()