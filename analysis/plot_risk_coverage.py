import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_risk_coverage(csv_path, output_dir):
    """绘制风险覆盖率图"""
    # 加载数据
    df = pd.read_csv(csv_path)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算不同阈值下的风险和覆盖率
    thresholds = []
    risks = []
    coverages = []
    
    for threshold in sorted(df['label_entropy'].unique()):
        subset = df[df['label_entropy'] <= threshold]
        if len(subset) > 0:
            risk = 1 - subset['accuracy'].mean()  # 风险 = 1 - 准确率
            coverage = len(subset) / len(df)
            thresholds.append(threshold)
            risks.append(risk)
            coverages.append(coverage)
    
    # 绘制风险覆盖率图
    plt.figure(figsize=(10, 6))
    plt.scatter(coverages, risks, alpha=0.5)
    plt.plot(coverages, risks, 'r-')
    plt.xlabel('Coverage')
    plt.ylabel('Risk')
    plt.title('Risk-Coverage Trade-off')
    plt.grid(True)
    
    # 保存图片
    output_path = os.path.join(output_dir, 'risk_coverage.png')
    plt.savefig(output_path)
    plt.close()
    
    print(f"Risk-coverage plot saved to {output_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot risk-coverage trade-off')
    parser.add_argument('--csv', type=str, required=True, help='Path to the metrics CSV file')
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    args = parser.parse_args()
    
    plot_risk_coverage(args.csv, args.output)
