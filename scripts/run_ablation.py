import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import numpy as np
import os
import json
import yaml
from tqdm import tqdm
from scipy.stats import entropy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.utils import parse_prediction_robust, save_metrics_to_csv
from analysis.ablation_analysis import analyze_ablation_results

class DataDrivenPreparer:
    def __init__(self, train_csv_path, kb_data_path, target_label):
        self.df = pd.read_csv(train_csv_path)
        cols_check = ['accuracy', 'first_token_entropy_top50', 'pred_prior_prob', 'sentence', 'true_label']
        self.df = self.df.dropna(subset=cols_check)
        self.df = self.df[self.df['order'] == target_label].copy()
        
        if 'nll_ground_truth' not in self.df.columns:
            self.df['nll_ground_truth'] = self.df['first_token_entropy_top50']

        self.df['Focal_Raw'] = self.df['first_token_entropy_top50'] * ((1 - self.df['pred_prior_prob']) ** 2)
        self.df['Metric_Score_WithPrior'] = np.log10(self.df['Focal_Raw'] + 1e-9)
        self.df['Metric_Score_NoPrior'] = np.log10(self.df['first_token_entropy_top50'] + 1e-9)

    def get_dynamic_threshold_with_prior(self, target_error_coverage):
        error_df = self.df[self.df['accuracy'] == 0]
        if len(error_df) == 0: return -1.0
        return error_df['Metric_Score_WithPrior'].quantile(1.0 - target_error_coverage)

    def get_dynamic_threshold_no_prior(self, target_error_coverage):
        error_df = self.df[self.df['accuracy'] == 0]
        if len(error_df) == 0: return -1.0
        return error_df['Metric_Score_NoPrior'].quantile(1.0 - target_error_coverage)

    def get_prior_map(self, all_types):
        total = len(self.df)
        counts = self.df['true_label'].value_counts().to_dict()
        prior_map = {k: v/total for k, v in counts.items()}
        for t in all_types:
            if t not in prior_map: prior_map[t] = 1e-6
        return prior_map

    def select_gold_few_shots(self, static_shot_num):
        candidates = self.df[self.df['accuracy'] == 1].copy()
        candidates = candidates.sort_values(by=['Metric_Score_WithPrior', 'nll_ground_truth'], ascending=[True, True])
        selected_shots = []
        seen_labels = set()
        for _, row in candidates.iterrows():
            lbl = row['true_label']
            if lbl not in seen_labels:
                selected_shots.append({'sentence': row['sentence'], 'label': lbl})
                seen_labels.add(lbl)
            if len(selected_shots) >= static_shot_num:
                break
        return selected_shots

    def build_knowledge_base(self):
        sentences = self.df['sentence'].tolist()
        labels = self.df['true_label'].tolist()
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = encoder.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
        return {'sentences': sentences, 'labels': labels, 'embeddings': embeddings, 'encoder': encoder}

class AblationInferenceEngine:
    def __init__(self, model_path, prior_map, threshold_with_prior, threshold_no_prior, gold_shots, kb, all_types, templates):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True
        )
        self.model.eval()
        
        self.prior_map = prior_map
        self.threshold_with_prior = threshold_with_prior
        self.threshold_no_prior = threshold_no_prior
        self.kb = kb
        self.all_types = all_types
        self.templates = templates
        
        self.candidate_ids = []
        for t in all_types:
            ids = self.tokenizer.encode(f' {t}', add_special_tokens=False)
            if ids: self.candidate_ids.append(ids[0])
        self.candidate_tensor = torch.tensor(self.candidate_ids, device=self.model.device)

        self.prompt_baseline = templates['good_examples'] + "\n" + templates['task']
        self.prompt_optimized = self._build_prompt(gold_shots)

    def _build_prompt(self, shots):
        text = "--- EXAMPLES --\n"
        for i, ex in enumerate(shots):
            text += f"Example {i+1}:\nInput: \"{ex['sentence']}\"\nOutput: \"{ex['label']}\"\n"
        return text + "\n" + self.templates['task']

    def parse_pred(self, text, debug: bool = False):
        """
        鲁棒解析函数
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
                    .replace(";  ", '')
                    .replace("!", '')
                    .replace("?", ''))
        
        if debug:
            print(f"原始文本：{original_text} | 清洗后：{clean_text}")
        
        # 步骤2：优先精确匹配引号（核心逻辑不变）
        for t in self.all_types:
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
                for t in self.all_types:
                    t_lower = t.lower()
                    if area.startswith(t_lower):
                        if debug:
                            print(f"✅ 前缀匹配（{prefix}）：{t}")
                        return t
        
        # 步骤4：模糊匹配（核心逻辑不变，基于清洗后的文本）
        for t in self.all_types:
            t_lower = t.lower()
            if t_lower in clean_text:
                if debug:
                    print(f"✅ 模糊匹配：{t}")
                return t
        
        # 所有匹配失败
        if debug:
            print(f"❌ 无匹配结果，返回 unclassified")
        return "unclassified"

    def calc_score_with_prior(self, logits, label):
        logits = logits[:, -1, :]
        cand_logits = logits[0, self.candidate_tensor]
        probs = F.softmax(cand_logits, dim=0).to(torch.float32).cpu().numpy()
        ent = float(entropy(probs, base=2))
        prior = self.prior_map.get(label, 1e-6)
        return np.log10(ent * ((1 - prior) ** 2) + 1e-9)

    def calc_score_no_prior(self, logits):
        logits = logits[:, -1, :]
        cand_logits = logits[0, self.candidate_tensor]
        probs = F.softmax(cand_logits, dim=0).to(torch.float32).cpu().numpy()
        ent = float(entropy(probs, base=2))
        return np.log10(ent + 1e-9)

    def retrieve(self, query, k=3):
        emb = self.kb['encoder'].encode([query])
        sims = cosine_similarity(emb, self.kb['embeddings'])[0]
        idxs = np.argsort(sims)[::-1][:k]
        return [{'sentence': self.kb['sentences'][i], 'label': self.kb['labels'][i]} for i in idxs]

    def get_label_prob_and_id(self, pred_label, logits):
        last_logits = logits[:, -1, :]
        cand_logits = last_logits[0, self.candidate_tensor]
        cand_probs = F.softmax(cand_logits, dim=0).to(torch.float32).cpu().numpy()
        label2idx = {t: i for i, t in enumerate(self.all_types)}
        pred_idx = label2idx.get(pred_label, 0)
        prob = cand_probs[pred_idx]
        label_id = self.tokenizer.encode(f' {pred_label}', add_special_tokens=False)[0]
        return prob, label_id

    def calculate_nll(self, logits, target_label_id, input_ids):
        target_ids = torch.tensor([[target_label_id]], device=self.model.device)
        full_input_ids = torch.cat([input_ids, target_ids], dim=1)
        labels = full_input_ids.clone()
        labels[:, :-1] = -100
        with torch.no_grad():
            loss = self.model(full_input_ids, labels=labels).loss.item()
        return loss

    def run_pipeline(self, sample, run_full_rag_analysis, rag_retrieval_num):
        sentence = sample['sentence']
        true_label = sample['label']
        
        # Stage 1 Baseline
        msgs_1 = [{"role": "system", "content": self.prompt_baseline}, {"role": "user", "content": f"Input: {sentence}\nOutput: "}]
        enc_1 = self.tokenizer.apply_chat_template(msgs_1, tokenize=False, add_generation_prompt=True)
        inp_1 = self.tokenizer(enc_1, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        
        with torch.no_grad():
            out_1 = self.model(input_ids=inp_1.input_ids)
            gen_1 = self.model.generate(**inp_1, max_new_tokens=20, pad_token_id=self.tokenizer.eos_token_id, do_sample=False)
            txt_1 = self.tokenizer.decode(gen_1[0][inp_1.input_ids.shape[1]:], skip_special_tokens=True)
        
        pred_1 = self.parse_pred(txt_1)
        acc_1 = 1 if pred_1.lower() == true_label.lower() else 0
        conf_1, label_id_1 = self.get_label_prob_and_id(pred_1, out_1.logits)
        true_label_id = self.tokenizer.encode(f' {true_label}', add_special_tokens=False)[0]
        nll_1 = self.calculate_nll(out_1.logits, true_label_id, inp_1.input_ids)

        # Stage 2 Static
        msgs_2 = [{"role": "system", "content": self.prompt_optimized}, {"role": "user", "content": f"Input: {sentence}\nOutput: "}]
        enc_2 = self.tokenizer.apply_chat_template(msgs_2, tokenize=False, add_generation_prompt=True)
        inp_2 = self.tokenizer(enc_2, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        
        with torch.no_grad():
            out_2 = self.model(input_ids=inp_2.input_ids)
            logits_2 = out_2.logits
            gen_2 = self.model.generate(**inp_2, max_new_tokens=20, pad_token_id=self.tokenizer.eos_token_id, do_sample=False)
            txt_2 = self.tokenizer.decode(gen_2[0][inp_2.input_ids.shape[1]:], skip_special_tokens=True)
        
        pred_2 = self.parse_pred(txt_2)
        acc_2 = 1 if pred_2.lower() == true_label.lower() else 0
        conf_2, label_id_2 = self.get_label_prob_and_id(pred_2, out_2.logits)
        nll_2 = self.calculate_nll(out_2.logits, true_label_id, inp_2.input_ids)
        
        score_with_prior = self.calc_score_with_prior(logits_2, pred_2)
        score_no_prior = self.calc_score_no_prior(logits_2)
        
        # Stage 3 Dynamic
        pred_with_prior = pred_2
        acc_with_prior = acc_2
        conf_with_prior = conf_2
        nll_with_prior = nll_2
        is_triggered_with_prior = False
        
        pred_no_prior = pred_2
        acc_no_prior = acc_2
        conf_no_prior = conf_2
        nll_no_prior = nll_2
        is_triggered_no_prior = False
        
        potential_rag_pred = pred_2
        potential_rag_conf = conf_2
        potential_rag_nll = nll_2
        
        should_run_rag = (score_with_prior > self.threshold_with_prior) or (score_no_prior > self.threshold_no_prior) or run_full_rag_analysis
        if should_run_rag:
            sim_exs = self.retrieve(sentence, k=rag_retrieval_num)
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
            inp_3 = self.tokenizer(enc_3, return_tensors="pt", add_special_tokens=False).to(self.model.device)
            
            with torch.no_grad():
                out_3 = self.model(input_ids=inp_3.input_ids)
                gen_3 = self.model.generate(**inp_3, max_new_tokens=20, pad_token_id=self.tokenizer.eos_token_id, do_sample=False)
                txt_3 = self.tokenizer.decode(gen_3[0][inp_3.input_ids.shape[1]:], skip_special_tokens=True)
            
            pred_rag_result = self.parse_pred(txt_3)
            potential_rag_conf, _ = self.get_label_prob_and_id(pred_rag_result, out_3.logits)
            potential_rag_nll = self.calculate_nll(out_3.logits, true_label_id, inp_3.input_ids)
            potential_rag_pred = pred_rag_result
            
            if score_with_prior > self.threshold_with_prior:
                pred_with_prior = pred_rag_result
                acc_with_prior = 1 if pred_with_prior.lower() == true_label.lower() else 0
                conf_with_prior = potential_rag_conf
                nll_with_prior = potential_rag_nll
                is_triggered_with_prior = True
            
            if score_no_prior > self.threshold_no_prior:
                pred_no_prior = pred_rag_result
                acc_no_prior = 1 if pred_no_prior.lower() == true_label.lower() else 0
                conf_no_prior = potential_rag_conf
                nll_no_prior = potential_rag_nll
                is_triggered_no_prior = True

        return {
            "sentence": sentence,
            "true_label": true_label,
            "acc_baseline": acc_1, "conf_baseline": conf_1, "nll_baseline": nll_1,
            "acc_static": acc_2, "conf_static": conf_2, "nll_static": nll_2,
            "acc_dynamic_with_prior": acc_with_prior, "conf_dynamic_with_prior": conf_with_prior, "nll_dynamic_with_prior": nll_with_prior, "triggered_with_prior": is_triggered_with_prior,
            "acc_dynamic_no_prior": acc_no_prior, "conf_dynamic_no_prior": conf_no_prior, "nll_dynamic_no_prior": nll_no_prior, "triggered_no_prior": is_triggered_no_prior,
            "pred_rag_potential": potential_rag_pred, "conf_rag_potential": potential_rag_conf, "nll_rag_potential": potential_rag_nll
        }

def load_test_data(path):
    with open(path, 'r') as f: data = json.load(f)
    test_samples = []
    for item in data:
        try:
            sent = item["messages"][1]["content"]
            lbl = item["messages"][2]["content"].split('#')[0].strip()
            test_samples.append({'sentence': sent, 'label': lbl})
        except: continue
    return test_samples

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run ablation experiment')
    parser.add_argument('--config', type=str, default='configs/experiment/ablation.yaml', help='Path to ablation experiment configuration file')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建输出目录
    output_dir = config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化数据准备器
    preparer = DataDrivenPreparer(
        train_csv_path=config['paths']['train_csv'],
        kb_data_path=config['paths']['kb_data'],
        target_label=config['target_label']
    )
    
    # 获取阈值和先验概率
    threshold_with_prior = preparer.get_dynamic_threshold_with_prior(config['hyperparams']['target_error_coverage'])
    threshold_no_prior = preparer.get_dynamic_threshold_no_prior(config['hyperparams']['target_error_coverage'])
    prior_map = preparer.get_prior_map(config['all_types'])
    gold_shots = preparer.select_gold_few_shots(config['hyperparams']['static_shot_num'])
    kb = preparer.build_knowledge_base()
    
    # 初始化推理引擎
    engine = AblationInferenceEngine(
        model_path=config['model']['path'],
        prior_map=prior_map,
        threshold_with_prior=threshold_with_prior,
        threshold_no_prior=threshold_no_prior,
        gold_shots=gold_shots,
        kb=kb,
        all_types=config['all_types'],
        templates=config['templates']
    )
    
    # 加载测试数据
    test_samples = load_test_data(config['paths']['test_dataset'])
    
    # 运行实验
    results = []
    for sample in tqdm(test_samples):
        results.append(engine.run_pipeline(
            sample,
            run_full_rag_analysis=config['hyperparams']['run_full_rag_analysis'],
            rag_retrieval_num=config['hyperparams']['rag_retrieval_num']
        ))
    
    # 保存结果
    df = pd.DataFrame(results)
    output_path = os.path.join(output_dir, 'ablation_results.csv')
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    # 分析结果
    analyze_ablation_results(df, output_dir)

if __name__ == "__main__":
    main()
