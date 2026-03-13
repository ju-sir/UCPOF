import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= 配置 =================
# 注意：这里必须用训练集 CSV，因为阈值是基于训练集确定的
TRAIN_CSV_PATH = "/home/jgy/paper-prompt/prompt-order/last7-train/v1(no-prototype)/Qwen2.5-7B-Instruct_full_metrics.csv"
OUTPUT_DIR = "/home/jgy/paper-prompt/prompt-order/last8-ace/final_comprehensive_study"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_threshold_justification(csv_path):
    print(f"Loading Training Data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 数据清洗：只保留目标 Order
    if 'order' in df.columns:
        df = df[df['order'].isin(['Examples->Task'])].copy()
    
    # 1. 计算 LSFU
    df['Metric_Raw'] = df['first_token_entropy_top50'] * ((1 - df['pred_prior_prob']) ** 2)
    df['LSFU'] = np.log10(df['Metric_Raw'] + 1e-9)
    
    # 2. 准备数据：排序
    # 分数越高越不确定，越容易是错误
    df_sorted = df.sort_values(by='LSFU', ascending=False).reset_index(drop=True)
    
    total_samples = len(df_sorted)
    total_errors = (df_sorted['accuracy'] == 0).sum()
    
    print(f"Total Samples: {total_samples}, Total Errors: {total_errors}")
    
    # 3. 模拟扫描过程
    # 我们不逐个样本扫，太慢，取 1000 个采样点
    scan_points = np.linspace(0, total_samples, 1000).astype(int)
    
    results = []
    
    # 预计算累积错误数
    # cumsum_errors[i] 表示前 i 个样本里有多少个错误
    is_error = (df_sorted['accuracy'] == 0).astype(int).values
    cumsum_errors = np.cumsum(is_error)
    
    for idx in scan_points:
        if idx == 0: continue
        
        # 当前阈值就是第 idx 个样本的 LSFU
        # 所有 idx 之前的样本都会被触发 (Score > Threshold)
        trigger_count = idx
        caught_errors = cumsum_errors[idx-1]
        
        trigger_rate = trigger_count / total_samples
        error_coverage = caught_errors / total_errors
        
        results.append({
            'trigger_rate': trigger_rate,
            'error_coverage': error_coverage
        })
        
    res_df = pd.DataFrame(results)
    
    # ================= 绘图 =================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- 图 A: Cost-Benefit Curve (ROI) ---
    ax1.plot(res_df['trigger_rate'], res_df['error_coverage'], color='#2980B9', linewidth=2.5)
    
    # 标注 90% 覆盖率的点
    # 找到最接近 0.90 coverage 的点
    idx_90 = (res_df['error_coverage'] - 0.90).abs().idxmin()
    point_90 = res_df.iloc[idx_90]
    
    # 画虚线
    ax1.axhline(y=0.90, color='gray', linestyle='--', alpha=0.6)
    ax1.axvline(x=point_90['trigger_rate'], color='gray', linestyle='--', alpha=0.6)
    
    # 标注肘点
    ax1.scatter(point_90['trigger_rate'], point_90['error_coverage'], color='#C0392B', s=100, zorder=5)
    ax1.annotate(f"Chosen Point\nCov: 90%\nTrig: {point_90['trigger_rate']:.1%}", 
                 xy=(point_90['trigger_rate'], point_90['error_coverage']), 
                 xytext=(point_90['trigger_rate'] + 0.1, 0.8),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=11, fontweight='bold')
    
    ax1.set_title("Cost-Benefit Analysis (Training Set)", fontsize=14)
    ax1.set_xlabel("Trigger Rate (Cost)", fontsize=12)
    ax1.set_ylabel("Error Coverage (Benefit)", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, 1.05)
    
    # --- 图 B: Marginal Cost Analysis ---
    # 计算每增加 5% 覆盖率，需要增加多少 Trigger Rate
    step = 0.05
    target_coverages = np.arange(0.5, 1.0, step) # 从 50% 到 95%
    marginal_costs = []
    labels = []
    
    colors = []
    
    for cov in target_coverages:
        # 当前区间的起点和终点
        start_cov = cov
        end_cov = cov + step
        
        # 找到对应的 Trigger Rate
        idx_start = (res_df['error_coverage'] - start_cov).abs().idxmin()
        idx_end = (res_df['error_coverage'] - end_cov).abs().idxmin()
        
        trig_start = res_df.iloc[idx_start]['trigger_rate']
        trig_end = res_df.iloc[idx_end]['trigger_rate']
        
        # 边际成本 = (Trigger增加量) / (Coverage增加量)
        # 这里 Coverage 增加量固定是 0.05，所以直接看 Trigger 增加量即可
        cost_increase = trig_end - trig_start
        
        marginal_costs.append(cost_increase)
        labels.append(f"{int(start_cov*100)}-{int(end_cov*100)}%")
        
        # 90-95% 这一档标红
        if start_cov >= 0.89:
            colors.append('#C0392B') # Red
        else:
            colors.append('#2980B9') # Blue

    bars = ax2.bar(labels, marginal_costs, color=colors, alpha=0.8)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_title("Marginal Trigger Cost for Next 5% Error Coverage", fontsize=14)
    ax2.set_xlabel("Error Coverage Interval", fontsize=12)
    ax2.set_ylabel("Additional Trigger Rate Required", fontsize=12)
    ax2.set_xticklabels(labels, rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "threshold_justification.png")
    plt.savefig(save_path, dpi=300)
    print(f"Justification plot saved to: {save_path}")

if __name__ == "__main__":
    plot_threshold_justification(TRAIN_CSV_PATH)