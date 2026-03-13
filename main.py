import torch
import os
import random
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from config import GENERAL_CONFIG, MODEL_CONFIG, DATASET_CONFIG
from src.data.data_loader import load_evaluation_data, calculate_dataset_priors
from src.core.experiment import run_validation_experiment
from src.analysis.analyzer import analyze_and_plot

logging.set_verbosity_error()

def main():
    # 创建输出目录
    output_dir = GENERAL_CONFIG["OUTPUT_DIR"]
    os.makedirs(output_dir, exist_ok=True)
    
    # 选择数据集 (可以根据需要修改)
    dataset_name = "ACE"
    dataset_config = DATASET_CONFIG[dataset_name]
    
    # A. 加载全部数据计算先验概率
    prior_full_dataset = load_evaluation_data(dataset_config["PRIOR_DATASET_PATH"])
    if not prior_full_dataset:
        print("Error: Could not load dataset.")
        return
    # 计算 Prior Map
    prior_map = calculate_dataset_priors(prior_full_dataset, dataset_config["ALL_TYPES"])
    print(f"prior_map = {prior_map}")
    

    # B. 加载测试集
    full_dataset = load_evaluation_data(dataset_config["DATASET_PATH"])
    if not full_dataset:
        print("Error: Could not load dataset.")
        return
    
    for MODEL_PATH in MODEL_CONFIG["MODEL_PATH_LIST"]:
        model_short_name = os.path.basename(os.path.normpath(MODEL_PATH))
        
        # 采样部分数据用于评估 (为了速度)
        evaluation_data = random.sample(full_dataset, min(GENERAL_CONFIG["NUM_EVALUATION_SENTENCES"], len(full_dataset)))
        
        print(f"\n######## Testing Features for: {model_short_name} ########")
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16, device_map='auto'
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model.eval()
        
        # 1. Examples -> Task
        system_content_1 = dataset_config["FIXED_GOOD_EXAMPLE_TEXT"] + "\n" + dataset_config["FIXED_TASK_TEXT"]
        df1 = run_validation_experiment(
            model, tokenizer, system_content_1, evaluation_data, "Examples->Task", 
            prior_map, dataset_config["ALL_TYPES"]
        )

        # 2. Task -> Examples
        system_content_2 = dataset_config["FIXED_TASK_TEXT"] + "\n" + dataset_config["FIXED_GOOD_EXAMPLE_TEXT"]
        df2 = run_validation_experiment(
            model, tokenizer, system_content_2, evaluation_data, "Task->Examples", 
            prior_map, dataset_config["ALL_TYPES"]
        )
        
        full_df = pd.concat([df1, df2], ignore_index=True)
        
        # 数据统计
        print("\n" + "="*50)
        print(f" Summary for {model_short_name}")
        print("="*50)
        print(full_df.groupby(['order', 'accuracy'])[['label_entropy', 'logit_margin', 'pred_prior_prob']].mean().to_string())
        
        # 绘图
        analyze_and_plot(full_df, model_short_name, output_dir)
        
        # 保存详细CSV
        csv_path = os.path.join(output_dir, f"{model_short_name}_full_metrics.csv")
        full_df.to_csv(csv_path, index=False)
        print(f"Data saved to {csv_path}")

if __name__ == '__main__':
    main()
