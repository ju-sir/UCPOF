import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= 配置区 =================
# 建议使用 LoRA 版本或 Prototype 版本的 CSV，效果最明显
CSV_PATH = "/home/jgy/paper-prompt/prompt-order/last8/v2(have-prototype)/Qwen2.5-7B-Instruct_full_metrics.csv"
OUTPUT_DIR = "/home/jgy/paper-prompt/prompt-order/last8/v2(have-prototype)/application_experiments"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 图表风格
sns.set_theme(style="whitegrid", context="talk", font_scale=1.0)
# =========================================

def load_and_calc_metric(csv_path):
    """加载数据并计算 Log-Scale Focal Uncertainty"""
    df = pd.read_csv(csv_path)
    
    # 筛选
    if 'order' in df.columns:
        df = df[df['order'].isin(['Examples->Prototype->Task'])].copy()
        if len(df) == 0:
             df = df[df['order'].str.contains("Examples")].copy()
    
    # 清洗
    df = df.dropna(subset=['accuracy', 'first_token_entropy_top50', 'pred_prior_prob'])
    
    # ★ 计算指标 (Log-Scale Focal Uncertainty)
    # Score 越低(负值大) = 越自信
    df['Focal_Raw'] = df['first_token_entropy_top50'] * ((1 - df['pred_prior_prob']) ** 2)
    df['Metric_Score'] = np.log10(df['Focal_Raw'] + 1e-9)
    
    return df

# ==============================================================================
# 实验 1: Risk-Coverage Curve (拒答机制验证)
# 逻辑: 按指标排序，优先回答自信的样本。看随着覆盖率(Coverage)降低，准确率(Risk)如何变化。
# ==============================================================================
def plot_risk_coverage_curve(df):
    print("正在进行实验 1: Risk-Coverage Curve (拒答分析)...")
    
    # 1. 排序: 按 Score 从小到大排序 (自信 -> 不自信)
    # 我们希望保留 Score 小的 (Confident)，拒绝 Score 大的 (Uncertain)
    df_sorted = df.sort_values(by='Metric_Score', ascending=True).reset_index(drop=True)
    
    coverages = []
    accuracies = []
    
    # 2. 模拟逐步拒答
    # 从保留 100% 数据开始，逐渐减少到保留 10%
    n_total = len(df_sorted)
    steps = np.arange(1.0, 0.0, -0.01) # 100% -> 1%
    
    for keep_ratio in steps:
        n_keep = int(n_total * keep_ratio)
        if n_keep < 10: break
        
        # 保留最自信的前 n_keep 个样本
        subset = df_sorted.iloc[:n_keep]
        acc = subset['accuracy'].mean()
        
        coverages.append(keep_ratio * 100)
        accuracies.append(acc * 100)
        
    # 3. 绘图
    plt.figure(figsize=(10, 6))
    
    # 绘制曲线
    plt.plot(coverages, accuracies, color='#1f77b4', linewidth=3, label='Sorted by Focal Metric')
    
    # 绘制 Random Baseline (如果不排序随机拒答，准确率应该保持不变，等于全局准确率)
    global_acc = df['accuracy'].mean() * 100
    plt.axhline(y=global_acc, color='gray', linestyle='--', label=f'Random Baseline ({global_acc:.1f}%)')
    
    # 装饰
    plt.title("Risk-Coverage Curve (Selective Prediction)", fontsize=16, fontweight='bold')
    plt.xlabel("Coverage (%) - Percentage of Questions Answered", fontsize=12)
    plt.ylabel("Accuracy (%) on Answered Questions", fontsize=12)
    plt.xlim(100, 10) # X轴反向：从回答所有问题 -> 只回答最有把握的问题
    plt.ylim(global_acc - 5, 100.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 标注关键点
    # 比如：只回答 80% 的问题时，准确率是多少？
    idx_80 = int(len(coverages) * 0.2) # 因为是倒序的 roughly
    if idx_80 < len(coverages):
        cov_val = coverages[idx_80]
        acc_val = accuracies[idx_80]
        plt.plot(cov_val, acc_val, 'ro')
        plt.annotate(f"Cover={cov_val:.0f}%\nAcc={acc_val:.1f}%", 
                     (cov_val, acc_val), xytext=(cov_val-10, acc_val-5),
                     arrowprops=dict(arrowstyle="->", color='red'))

    save_path = os.path.join(OUTPUT_DIR, "App_Risk_Coverage_Curve.png")
    plt.savefig(save_path, dpi=300)
    print(f"图表已保存: {save_path}")
    return coverages, accuracies

# ==============================================================================
# 实验 2: Hard Sample Mining (难例挖掘 / 幻觉检测)
# 逻辑: 找出 "Risk Zone" (指标很自信，但实际答错了) 的样本。
# 这些是用于优化 Prototype 或 Few-shot 的最佳素材。
# ==============================================================================
def export_hard_samples(df):
    print("\n正在进行实验 2: Hard Sample Mining (难例导出)...")
    
    # 1. 找到“风险样本” (Overconfident Errors)
    # 定义：预测错误 (Acc=0) 且 指标分数非常低 (Very Confident)
    # 我们取 Metric Score 最低的前 50 个错误样本
    errors = df[df['accuracy'] == 0].copy()
    if len(errors) == 0:
        print("竟然没有错误样本？跳过。")
        return
        
    risk_samples = errors.sort_values(by='Metric_Score', ascending=True).head(50)
    
    # 2. 导出 CSV
    cols_to_save = ['Metric_Score', 'accuracy', 'true_label', 'pred_label', 'pred_prior_prob', 'sentence']
    # 确保列存在
    save_cols = [c for c in cols_to_save if c in df.columns]
    
    save_path = os.path.join(OUTPUT_DIR, "Mined_Risk_Samples_for_FewShot.csv")
    risk_samples[save_cols].to_csv(save_path, index=False)
    
    print(f"已导出 {len(risk_samples)} 条最高风险样本(幻觉样本)至: {save_path}")
    print("用途：请查看这些样本。它们是模型觉得自己很懂，但完全答错的题。")
    print("建议：将这些样本加入 Few-shot Prompt 的负例，或针对这些 Label 优化 Prototype 定义。")

    # 简单打印前几个看看
    print("\n--- Top 3 幻觉样本 (Model Confident but Wrong) ---")
    for i, row in risk_samples.head(3).iterrows():
        print(f"Score: {row['Metric_Score']:.4f} | GT: {row['true_label']} | Pred: {row.get('pred_label', 'N/A')}")
        print(f"Text: {row.get('sentence', '')[:100]}...")
        print("-" * 30)

# ==============================================================================
# 主程序
# ==============================================================================
if __name__ == "__main__":
    if os.path.exists(CSV_PATH):
        # 1. 准备数据
        df_clean = load_and_calc_metric(CSV_PATH)
        print(f"数据加载完毕，共 {len(df_clean)} 条。全局准确率: {df_clean['accuracy'].mean():.2%}")
        
        # 2. 运行应用实验
        plot_risk_coverage_curve(df_clean)
        export_hard_samples(df_clean)
        
        print("\n所有应用实验完成！请查看 output 目录。")
    else:
        print("文件路径错误，请检查 CSV_PATH")