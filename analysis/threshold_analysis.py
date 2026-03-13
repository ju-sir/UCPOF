import pandas as pd
import matplotlib.pyplot as plt
import os
from src.metric import MetricCalculator

def threshold_analysis(csv_path, output_dir):
    """阈值分析"""
    # 加载数据
    df = pd.read_csv(csv_path)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化指标计算器
    metric_calculator = MetricCalculator()
    
    # 分析不同指标的阈值
    metrics = [
        'label_entropy', 
        'logit_margin', 
        'lsfu_score'
    ]
    
    for metric in metrics:
        # 拟合阈值
        if metric == 'logit_margin' or metric == 'lsfu_score':
            # 对于这些指标，值越大越好
            thresholds = sorted(df[metric].unique(), reverse=True)
        else:
            # 对于这些指标，值越小越好
            thresholds = sorted(df[metric].unique())
        
        accuracies = []
        coverages = []
        
        for threshold in thresholds:
            if metric == 'logit_margin' or metric == 'lsfu_score':
                subset = df[df[metric] >= threshold]
            else:
                subset = df[df[metric] <= threshold]
            
            if len(subset) > 0:
                accuracy = subset['accuracy'].mean()
                coverage = len(subset) / len(df)
                accuracies.append(accuracy)
                coverages.append(coverage)
        
        # 绘制阈值分析图
        plt.figure(figsize=(10, 6))
        plt.scatter(coverages, accuracies, alpha=0.5)
        plt.plot(coverages, accuracies, 'r-')
        plt.xlabel('Coverage')
        plt.ylabel('Accuracy')
        plt.title(f'Threshold Analysis for {metric}')
        plt.grid(True)
        
        # 保存图片
        output_path = os.path.join(output_dir, f'{metric}_threshold_analysis.png')
        plt.savefig(output_path)
        plt.close()
        
        print(f"Threshold analysis for {metric} saved to {output_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Perform threshold analysis')
    parser.add_argument('--csv', type=str, required=True, help='Path to the metrics CSV file')
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    args = parser.parse_args()
    
    threshold_analysis(args.csv, args.output)
