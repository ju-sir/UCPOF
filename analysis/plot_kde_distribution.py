import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_kde_distribution(csv_path, output_dir):
    """绘制KDE分布图"""
    # 加载数据
    df = pd.read_csv(csv_path)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 为正确和错误的预测创建不同的数据集
    correct_df = df[df['accuracy'] == 1]
    incorrect_df = df[df['accuracy'] == 0]
    
    # 需要分析的指标列表
    metrics = [
        'label_entropy', 
        'first_token_entropy_top50', 
        'first_token_entropy_top500', 
        'first_token_entropy_full',
        'logit_margin', 
        'pred_prior_prob',
        'lsfu_score'
    ]
    
    # 绘制每个指标的KDE分布图
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(correct_df[metric], label='Correct', fill=True, alpha=0.5)
        sns.kdeplot(incorrect_df[metric], label='Incorrect', fill=True, alpha=0.5)
        plt.xlabel(metric)
        plt.ylabel('Density')
        plt.title(f'KDE Distribution of {metric}')
        plt.legend()
        plt.grid(True)
        
        # 保存图片
        output_path = os.path.join(output_dir, f'{metric}_kde.png')
        plt.savefig(output_path)
        plt.close()
        
        print(f"KDE plot for {metric} saved to {output_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot KDE distributions')
    parser.add_argument('--csv', type=str, required=True, help='Path to the metrics CSV file')
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    args = parser.parse_args()
    
    plot_kde_distribution(args.csv, args.output)
