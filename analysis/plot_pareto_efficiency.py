import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_pareto_efficiency(csv_path, output_dir):
    """绘制帕累托效率图"""
    # 加载数据
    df = pd.read_csv(csv_path)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算不同阈值下的准确率和覆盖率
    thresholds = []
    accuracies = []
    coverages = []
    
    for threshold in sorted(df['lsfu_score'].unique()):
        subset = df[df['lsfu_score'] >= threshold]
        if len(subset) > 0:
            accuracy = subset['accuracy'].mean()
            coverage = len(subset) / len(df)
            thresholds.append(threshold)
            accuracies.append(accuracy)
            coverages.append(coverage)
    
    # 绘制帕累托效率图
    plt.figure(figsize=(10, 6))
    plt.scatter(coverages, accuracies, alpha=0.5)
    plt.plot(coverages, accuracies, 'r-')
    plt.xlabel('Coverage')
    plt.ylabel('Accuracy')
    plt.title('Pareto Efficiency Curve')
    plt.grid(True)
    
    # 保存图片
    output_path = os.path.join(output_dir, 'pareto_efficiency.png')
    plt.savefig(output_path)
    plt.close()
    
    print(f"Pareto efficiency plot saved to {output_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot Pareto efficiency')
    parser.add_argument('--csv', type=str, required=True, help='Path to the metrics CSV file')
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    args = parser.parse_args()
    
    plot_pareto_efficiency(args.csv, args.output)
