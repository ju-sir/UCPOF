import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= 配置区域 =================
# 1. 输入路径 (对应 comprehensive_experiment_ET.py 的输出)
OUTPUT_DIR = "/home/jgy/paper-prompt/prompt-order/last8-ace/final_comprehensive_study"
CSV_PATH = os.path.join(OUTPUT_DIR, "final_experiment_results.csv")

# 2. 成本假设 (Token估算，用于画 Pareto 轴)
# 依据：System Prompt + Few-shot + Query
COST_STATIC_BASE = 600   # Stage 1 + 2
# 依据：Static + Retrieval Overhead + RAG Prompt (Contexts) + Reasoning
COST_RAG_FULL    = 1400  # Stage 3

# ===========================================

def analyze_efficiency_and_sensitivity(csv_path):
    print(f">>> Loading Test Results from {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # 1. 数据预处理
    # 确保有全量RAG的准确率 (Acc_Full_RAG)
    if 'acc_full_rag' not in df.columns:
        # 如果 csv 里没存，根据 potential prediction 现算
        if 'pred_rag_potential' in df.columns:
            df['acc_full_rag'] = df.apply(lambda r: 1 if str(r['pred_rag_potential']).lower() == str(r['true_label']).lower() else 0, axis=1)
        else:
            print("Error: Missing 'pred_rag_potential' column. Please run experiment with RUN_FULL_RAG_ANALYSIS=True.")
            return

    # 2. 模拟不同阈值下的表现 (Simulation)
    # 逻辑：我们将样本按 LSFU (focal_score) 从高到低排序 (越高越不确定)
    # 然后逐步增加 Trigger Rate (从 0% 到 100%)
    
    df_sorted = df.sort_values(by='focal_score', ascending=False).reset_index(drop=True)
    total_samples = len(df)
    
    simulation_data = []
    
    # 使用向量化计算加速模拟
    # cumsum_static_acc: 如果全都不触发，随着样本增加的累积正确数 (其实没用，我们需要的是剩余部分的static)
    # 我们需要：对于前 k 个样本使用 RAG，后 N-k 个样本使用 Static
    
    # 预计算累积正确数
    rag_correct_cumsum = np.cumsum(df_sorted['acc_full_rag'].values)
    static_correct_cumsum = np.cumsum(df_sorted['acc_static'].values)
    total_static_correct = static_correct_cumsum[-1]
    
    # 扫描点：从 0% 到 100% 触发
    scan_points = np.linspace(0, total_samples, 100).astype(int)
    scan_points = np.unique(scan_points) # 去重
    
    print(">>> Simulating Dynamic Thresholds on Test Set...")
    
    for k in scan_points:
        # k: 触发 RAG 的样本数 (Top-k uncertain samples)
        trigger_rate = k / total_samples
        
        # 混合准确率计算
        # Part A: 前 k 个 (Triggered) -> 使用 RAG 结果
        correct_A = rag_correct_cumsum[k-1] if k > 0 else 0
        
        # Part B: 后 N-k 个 (Not Triggered) -> 使用 Static 结果
        # Static 总正确数 - 前 k 个里的 Static 正确数
        correct_B_total = total_static_correct
        correct_B_excluded = static_correct_cumsum[k-1] if k > 0 else 0
        correct_B = correct_B_total - correct_B_excluded
        
        acc_dynamic = (correct_A + correct_B) / total_samples
        
        # 平均成本
        avg_cost = (trigger_rate * COST_RAG_FULL) + ((1 - trigger_rate) * COST_STATIC_BASE)
        
        simulation_data.append({
            'trigger_rate': trigger_rate,
            'avg_cost': avg_cost,
            'accuracy': acc_dynamic
        })
        
    sim_df = pd.DataFrame(simulation_data)
    
    # 获取我们在 Experiment 中实际跑出来的结果点 (基于训练集阈值)
    actual_trigger_rate = df['triggered'].mean()
    actual_acc = df['acc_dynamic'].mean()
    actual_cost = (actual_trigger_rate * COST_RAG_FULL) + ((1 - actual_trigger_rate) * COST_STATIC_BASE)
    
    print(f"Actual Experiment Point -> Trigger: {actual_trigger_rate:.2%}, Acc: {actual_acc:.2%}")

    # ================= 绘图 1: Pareto Efficiency Curve =================
    plot_pareto(sim_df, df, actual_cost, actual_acc, actual_trigger_rate)

    # ================= 绘图 2: Sensitivity Analysis =================
    plot_sensitivity(sim_df, actual_trigger_rate, actual_acc)

def plot_pareto(sim_df, raw_df, act_cost, act_acc, act_trig):
    """画推理成本与准确率的权衡曲线"""
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(10, 7))
    
    # 1. 模拟曲线
    plt.plot(sim_df['avg_cost'], sim_df['accuracy'], 
             color='#2980B9', linewidth=3, label='UPDRF Simulation Curve')
    
    # 2. 关键基准点
    # Static
    static_acc = raw_df['acc_static'].mean()
    plt.scatter(COST_STATIC_BASE, static_acc, 
                color='gray', s=120, marker='o', edgecolors='k', zorder=5, label='Static Baseline')
    plt.text(COST_STATIC_BASE + 20, static_acc - 0.002, 'Static', va='top', fontsize=10, color='gray')

    # Full RAG
    full_rag_acc = raw_df['acc_full_rag'].mean()
    plt.scatter(COST_RAG_FULL, full_rag_acc, 
                color='#C0392B', s=120, marker='^', edgecolors='k', zorder=5, label='Full RAG')
    plt.text(COST_RAG_FULL - 20, full_rag_acc - 0.002, 'Full RAG', va='top', ha='right', fontsize=10, color='#C0392B')

    # 3. 我们的实际实验点 (Ours)
    plt.scatter(act_cost, act_acc, 
                color='#27AE60', s=250, marker='*', edgecolors='k', zorder=10, label='UPDRF (Ours)')
    
    # 标注 Ours
    plt.annotate(f"Ours (Trigger: {act_trig:.1%})", 
                 xy=(act_cost, act_acc), 
                 xytext=(act_cost + 150, act_acc - 0.015),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                 fontsize=11, fontweight='bold', color='#27AE60')

    plt.title('Figure 3: Efficiency-Accuracy Pareto Frontier', fontsize=16, pad=15)
    plt.xlabel('Average Token Consumption (Cost)', fontsize=13)
    plt.ylabel('Test Accuracy', fontsize=13)
    
    # 自动调整Y轴范围，留出一点空间
    y_min = min(static_acc, full_rag_acc, act_acc) - 0.02
    y_max = max(static_acc, full_rag_acc, act_acc) + 0.02
    plt.ylim(y_min, y_max)
    
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, "pareto_efficiency_curve.png")
    plt.savefig(save_path, dpi=300)
    print(f"Pareto Plot saved to: {save_path}")

def plot_sensitivity(sim_df, act_trig, act_acc):
    """
    敏感性分析图：展示准确率随触发率的变化。
    目的是证明在 20%-30% 左右有一个 Elbow Point，说明我们的选择是鲁棒的。
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # X轴: Trigger Rate
    x = sim_df['trigger_rate'] * 100 # 转百分比
    y = sim_df['accuracy'] * 100     # 转百分比
    
    # --- 左轴: Accuracy 曲线 ---
    ax1.plot(x, y, color='#2c3e50', linewidth=2.5, label='Accuracy Trend')
    ax1.set_xlabel('Trigger Rate (%) - Percentage of Samples sent to RAG', fontsize=12)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12, color='#2c3e50')
    ax1.tick_params(axis='y', labelcolor='#2c3e50')
    
    # 标注实际点
    ax1.scatter(act_trig*100, act_acc*100, color='#27AE60', s=150, marker='*', zorder=10, label='Ours (Selected via Training Set)')
    
    # --- 辅助区域: 收益递减区 ---
    # 找到 "Elbow" 区域：通常是准确率达到了 Full RAG 98%-99% 性能的地方
    max_acc = y.max()
    threshold_acc = max_acc * 0.99
    # 找到第一次达到这个准确率的 trigger rate
    try:
        elbow_idx = np.where(y >= threshold_acc)[0][0]
        elbow_x = x[elbow_idx]
        
        # 画一个阴影区域表示 "Optimal Efficiency Zone"
        ax1.axvspan(elbow_x - 5, elbow_x + 10, color='#27AE60', alpha=0.1, label='Optimal Efficiency Zone')
        ax1.axvline(elbow_x, color='#27AE60', linestyle='--', alpha=0.5)
        ax1.text(elbow_x + 1, min(y) + 1, "Diminishing Returns\nStart Here", color='green', fontsize=9)
    except:
        pass

    # --- 标题和网格 ---
    plt.title('Figure 4: Sensitivity Analysis of Dynamic Threshold', fontsize=15, pad=15)
    plt.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "sensitivity_analysis.png")
    plt.savefig(save_path, dpi=300)
    print(f"Sensitivity Plot saved to: {save_path}")

if __name__ == "__main__":
    analyze_efficiency_and_sensitivity(CSV_PATH)