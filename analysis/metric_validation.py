import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_auc_score

def metric_validation(csv_path, output_dir):
    """指标验证"""
    # 加载数据
    df = pd.read_csv(csv_path)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算AUC
    metrics = [
        'label_entropy', 
        'first_token_entropy_top50', 
        'first_token_entropy_top500', 
        'first_token_entropy_full',
        'logit_margin', 
        'pred_prior_prob',
        'lsfu_score'
    ]
    
    auc_scores = {}
    for metric in metrics:
        # 对于熵类指标，需要取倒数，因为值越小越好
        if 'entropy' in metric:
            scores = 1 / (df[metric] + 1e-12)
        else:
            scores = df[metric]
        
        # 计算AUC
        try:
            auc = roc_auc_score(df['accuracy'], scores)
            auc_scores[metric] = auc
            print(f"AUC for {metric}: {auc:.4f}")
        except:
            print(f"Could not calculate AUC for {metric}")
    
    # 绘制AUC条形图
    plt.figure(figsize=(12, 6))
    plt.bar(auc_scores.keys(), auc_scores.values())
    plt.xlabel('Metric')
    plt.ylabel('AUC Score')
    plt.title('AUC Scores for Different Metrics')
    plt.ylim(0.5, 1.0)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, 'metric_auc.png')
    plt.savefig(output_path)
    plt.close()
    
    # 长尾分析
    plt.figure(figsize=(12, 6))
    for metric in metrics:
        sorted_values = sorted(df[metric])
        plt.plot(range(len(sorted_values)), sorted_values, label=metric)
    plt.xlabel('Sample Index')
    plt.ylabel('Metric Value')
    plt.title('Long Tail Distribution of Metrics')
    plt.legend()
    plt.grid(True)
    
    # 保存图片
    output_path = os.path.join(output_dir, 'long_tail_analysis.png')
    plt.savefig(output_path)
    plt.close()
    
    print(f"Metric validation results saved to {output_dir}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Validate metrics')
    parser.add_argument('--csv', type=str, required=True, help='Path to the metrics CSV file')
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    args = parser.parse_args()
    
    metric_validation(args.csv, args.output)
