import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_and_plot(df, model_name, output_dir):
    df['Prediction'] = df['accuracy'].map({1: 'Correct', 0: 'Incorrect'})
    
    # 需要分析的指标列表
    metrics = [
        'label_entropy', 
        'first_token_entropy_top50', 
        'first_token_entropy_top500', 
        'first_token_entropy_full',
        'logit_margin', 
        'nll_ground_truth', 
        'pred_prior_prob'
    ]
    
    # 1. 关系矩阵 (Correlation Matrix)
    plt.figure(figsize=(10, 8))
    # 计算包含 accuracy 的相关性
    corr_cols = ['accuracy'] + metrics
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title(f'Metric Correlation Matrix ({model_name})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_correlation_matrix.png"))
    plt.close()
    
    # 2. 箱线图 (Boxplots) - 预测正确与否和各指标的关系
    # 创建 2行3列 的子图
    fig, axes = plt.subplots(3, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        sns.boxplot(data=df, x='Prediction', y=metric, order=['Correct', 'Incorrect'], palette="Set2", ax=ax)
        
        # 增加标题说明预期的趋势
        trend = ""
        if metric in ['logit_margin', 'pred_prior_prob']:
            trend = "(Higher is usually Better)"
        else:
            trend = "(Lower is usually Better)"
            
        ax.set_title(f"{metric}\n{trend}")
        ax.set_xlabel("")
    
    # 隐藏多余的子图
    for j in range(len(metrics), len(axes)):
        axes[j].axis('off')

    plt.suptitle(f"Metric Distributions by Correctness ({model_name})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f"{model_name}_boxplots.png"))
    plt.close()
    
    print(f"Plots saved to {output_dir}")
