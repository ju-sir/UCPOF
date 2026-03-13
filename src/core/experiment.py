import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from src.utils.helper_functions import safe_entropy, parse_prediction_robust

def run_validation_experiment(model, tokenizer, system_content, evaluation_data, experiment_name, prior_map, all_types):
    results = []
    
    # --- 预处理：获取所有候选 Label 的首个 Token ID ---
    # 注意：我们假设每个 label 前面有个空格，这符合 Chat 模板的一般拼接习惯
    candidate_token_ids = []
    valid_indices = []
    for i, label_type in enumerate(all_types):
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
            pred_label = parse_prediction_robust(prediction_text, all_types)
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
            if pred_label == "unclassified" or pred_label not in all_types:
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
