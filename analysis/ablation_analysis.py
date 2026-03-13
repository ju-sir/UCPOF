import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss
import os

def calculate_ece(probs, accs, n_bins=10):
    """计算期望校准误差 (ECE)"""
    probs = np.array(probs)
    accs = np.array(accs)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (probs > bins[i]) & (probs <= bins[i+1])
        if not np.any(bin_mask):
            continue
        bin_acc = np.mean(accs[bin_mask])
        bin_conf = np.mean(probs[bin_mask])
        ece += np.abs(bin_acc - bin_conf) * (np.sum(bin_mask) / len(probs))
    return ece

def analyze_ablation_results(results_df, output_dir):
    """分析消融实验结果"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # ===================== 计算全部 5 组指标 =====================
    # Baseline
    acc_b = results_df['acc_baseline'].mean()
    ece_b = calculate_ece(results_df['conf_baseline'], results_df['acc_baseline'])
    brier_b = brier_score_loss(results_df['acc_baseline'], results_df['conf_baseline'])
    nll_b = results_df['nll_baseline'].mean()

    # Static
    acc_s = results_df['acc_static'].mean()
    ece_s = calculate_ece(results_df['conf_static'], results_df['acc_static'])
    brier_s = brier_score_loss(results_df['acc_static'], results_df['conf_static'])
    nll_s = results_df['nll_static'].mean()

    # Dynamic 带 (1-prior)^2
    acc_w = results_df['acc_dynamic_with_prior'].mean()
    trig_w = results_df['triggered_with_prior'].mean()
    ece_w = calculate_ece(results_df['conf_dynamic_with_prior'], results_df['acc_dynamic_with_prior'])
    brier_w = brier_score_loss(results_df['acc_dynamic_with_prior'], results_df['conf_dynamic_with_prior'])
    nll_w = results_df['nll_dynamic_with_prior'].mean()

    # Dynamic 无 (1-prior)^2
    acc_n = results_df['acc_dynamic_no_prior'].mean()
    trig_n = results_df['triggered_no_prior'].mean()
    ece_n = calculate_ece(results_df['conf_dynamic_no_prior'], results_df['acc_dynamic_no_prior'])
    brier_n = brier_score_loss(results_df['acc_dynamic_no_prior'], results_df['conf_dynamic_no_prior'])
    nll_n = results_df['nll_dynamic_no_prior'].mean()

    # Full RAG
    results_df['acc_full'] = results_df.apply(lambda r: 1 if r['pred_rag_potential'].lower() == r['true_label'].lower() else 0, axis=1)
    acc_f = results_df['acc_full'].mean()
    ece_f = calculate_ece(results_df['conf_rag_potential'], results_df['acc_full'])
    brier_f = brier_score_loss(results_df['acc_full'], results_df['conf_rag_potential'])
    nll_f = results_df['nll_rag_potential'].mean()

    # ===================== 打印干净表格 =====================
    print("\n" + "="*85)
    print("             🏆 实验核心指标对比表")
    print("="*85)
    print(f"{'模型设置':<28} {'Acc':<10} {'Trigger':<12} {'ECE':<10} {'Brier':<10} {'NLL':<10}")
    print("-"*85)
    print(f"{'Baseline':<28} {acc_b:<10.2%} {'-':<12} {ece_b:<10.4f} {brier_b:<10.4f} {nll_b:<10.4f}")
    print(f"{'Static Optimized':<28} {acc_s:<10.2%} {'-':<12} {ece_s:<10.4f} {brier_s:<10.4f} {nll_s:<10.4f}")
    print(f"{'Dynamic w/ (1-prior)²':<28} {acc_w:<10.2%} {trig_w:<12.2%} {ece_w:<10.4f} {brier_w:<10.4f} {nll_w:<10.4f}")
    print(f"{'Dynamic w/o (1-prior)²':<28} {acc_n:<10.2%} {trig_n:<12.2%} {ece_n:<10.4f} {brier_n:<10.4f} {nll_n:<10.4f}")
    print(f"{'Full RAG':<28} {acc_f:<10.2%} {'100.00%':<12} {ece_f:<10.4f} {brier_f:<10.4f} {nll_f:<10.4f}")
    print("="*85)
    
    # 保存结果到CSV
    metrics_df = pd.DataFrame({
        'Model Setting': ['Baseline', 'Static Optimized', 'Dynamic w/ (1-prior)²', 'Dynamic w/o (1-prior)²', 'Full RAG'],
        'Accuracy': [acc_b, acc_s, acc_w, acc_n, acc_f],
        'Trigger Rate': [0.0, 0.0, trig_w, trig_n, 1.0],
        'ECE': [ece_b, ece_s, ece_w, ece_n, ece_f],
        'Brier Score': [brier_b, brier_s, brier_w, brier_n, brier_f],
        'NLL': [nll_b, nll_s, nll_w, nll_n, nll_f]
    })
    
    output_path = os.path.join(output_dir, 'ablation_metrics.csv')
    metrics_df.to_csv(output_path, index=False)
    print(f"\nMetrics saved to {output_path}")
    
    return metrics_df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Analyze ablation experiment results')
    parser.add_argument('--csv', type=str, required=True, help='Path to the results CSV file')
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    args = parser.parse_args()
    
    # 加载结果数据
    df = pd.read_csv(args.csv)
    # 分析结果
    analyze_ablation_results(df, args.output)
