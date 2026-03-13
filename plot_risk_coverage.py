import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==============================================================================
# 1. 配置
# ==============================================================================
# 指向你刚刚生成的最终结果 CSV
CSV_PATH = "/home/jgy/paper-prompt/prompt-order/last8-ace/final_comprehensive_study/final_experiment_results.csv"
OUTPUT_DIR = "./final_paper_experiment"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设置绘图风格
sns.set_theme(style="whitegrid", context="talk", font_scale=1.0)

def plot_rc_curve(csv_path):
    print(f"Loading results from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Error: 文件未找到，请检查路径。")
        return

    # 必需列检查
    required_cols = ['acc_dynamic', 'focal_score']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV 中缺少必需列 {required_cols}")
        return

    # 1. 数据准备
    # 按不确定性分数从小到大排序 (Score 越低 = 越自信)
    # 我们希望优先保留自信的样本
    df_sorted = df.sort_values(by='focal_score', ascending=True).reset_index(drop=True)
    
    total_samples = len(df_sorted)
    coverages = []     # X轴: 覆盖率
    accuracies = []    # Y轴: 准确率
    risks = []         # 备用Y轴: 错误率 (Risk)

    # 2. 滑动计算
    # 从保留 100% 样本开始，逐步减少到保留 1%
    # 步长可以根据样本量调整，这里设为逐个样本剔除过于密集，改为按百分比步长
    steps = np.linspace(total_samples, int(total_samples * 0.05), 100, dtype=int)
    
    for k in steps:
        # 保留前 k 个最自信的样本
        subset = df_sorted.iloc[:k]
        
        cov = k / total_samples
        acc = subset['acc_dynamic'].mean()
        risk = 1.0 - acc
        
        coverages.append(cov * 100) # 转为百分比
        accuracies.append(acc * 100)
        risks.append(risk * 100)

    # 3. 绘图
    plt.figure(figsize=(10, 7))
    
    # 绘制主曲线
    plt.plot(coverages, accuracies, color='#1f77b4', linewidth=3, label='Dynamic RAG (Sorted by Focal Score)')
    
    # 绘制 Random Baseline (随机剔除)
    # 随机剔除时，准确率的期望值应该保持不变，等于全局准确率
    global_acc = df['acc_dynamic'].mean() * 100
    plt.axhline(y=global_acc, color='gray', linestyle='--', linewidth=2, label=f'Random Baseline ({global_acc:.1f}%)')
    
    # 4. 标注关键点 (Optional)
    # 比如：标出覆盖率为 80% 时的准确率
    # 找到最接近 80% 的点
    idx_80 = (np.abs(np.array(coverages) - 80)).argmin()
    val_cov_80 = coverages[idx_80]
    val_acc_80 = accuracies[idx_80]
    
    plt.plot(val_cov_80, val_acc_80, 'o', color='#d62728', markersize=8)
    plt.annotate(f"Cover: {val_cov_80:.0f}%\nAcc: {val_acc_80:.1f}%", 
                 xy=(val_cov_80, val_acc_80), 
                 xytext=(val_cov_80 - 10, val_acc_80 + 5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                 fontsize=12, fontweight='bold', color='#d62728')

    # 5. 美化图表
    plt.title("Risk-Coverage Curve (Selective Prediction Performance)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Coverage (%) - Percentage of Questions Answered", fontsize=14)
    plt.ylabel("Accuracy (%) on Answered Questions", fontsize=14)
    
    # X轴反转：通常习惯从 100% (左) 到 0% (右)，或者 0 -> 100。
    # 学术界常用 Coverage 从 1 -> 0 (X轴从左到右数值变小) 或者 Rejection Rate 从 0 -> 1。
    # 这里我们设定 X 轴范围从 100 到 0
    plt.xlim(105, 0) 
    plt.ylim(global_acc - 5, 100.5)
    
    plt.legend(loc='lower left', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # 保存
    save_file = os.path.join(OUTPUT_DIR, "Paper_Fig_Risk_Coverage.png")
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存至: {save_file}")
    print("-" * 50)
    print(f"数据摘要:")
    print(f"Global Acc (100% Cover): {global_acc:.2f}%")
    print(f"Acc at ~80% Cover      : {val_acc_80:.2f}%")
    print(f"Acc at ~50% Cover      : {accuracies[-1]:.2f}%")

if __name__ == "__main__":
    plot_rc_curve(CSV_PATH)