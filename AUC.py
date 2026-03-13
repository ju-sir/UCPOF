import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr, spearmanr
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= 配置区 =================
CSV_PATH = "/home/jgy/paper-prompt/prompt-order/last8-ace/v1(no-prototype)/Llama-3-8B-Instruct_full_metrics.csv"
TARGET_ORDER = "Examples->Task"  # 你的实验标签
# =========================================

def verify_metric():
    # 1. 加载数据
    if not os.path.exists(CSV_PATH):
        print(f"Error: 文件不存在 {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    
    # 筛选特定实验
    if 'order' in df.columns:
        df = df[df['order'] == TARGET_ORDER].copy()
        if len(df) == 0:
            print(f"Warning: 精确匹配 '{TARGET_ORDER}' 失败，尝试模糊匹配...")
            df = df[df['order'].str.contains("Examples")].copy()
    
    # 清洗数据
    req_cols = ['accuracy', 'first_token_entropy_top50', 'pred_prior_prob']
    df = df.dropna(subset=req_cols)
    print(f"成功加载 {len(df)} 条样本。")

    # ==========================================================
    # 2. 计算新指标: Log-Scale Focal Uncertainty
    # ==========================================================
    # 公式: Log10( Entropy * (1 - Prior)^2 )
    # 物理意义: 
    #   - 越低(越负) -> 模型越自信且大概率是对的
    #   - 越高(接近0) -> 模型在瞎猜或者在难样本上犹豫
    
    epsilon = 1e-9
    df['Focal_Raw'] = df['first_token_entropy_top50'] #* ((1 - df['pred_prior_prob']) ** 2)
    df['Log_Focal_Score'] = np.log10(df['Focal_Raw'] + epsilon)

    # ==========================================================
    # 3. 计算相关性
    # ==========================================================
    y_true = df['accuracy']
    x_score = df['Log_Focal_Score']

    # A. Point-Biserial Correlation (点二列相关系数)
    pb_corr, p_value = pointbiserialr(y_true, x_score)

    # B. Spearman Rank Correlation (斯皮尔曼等级相关)
    sp_corr, _ = spearmanr(y_true, x_score)

    # C. ROC AUC Score (区分度)
    auc = roc_auc_score(y_true, -x_score)

    # ==========================================================
    # 4. 输出报告
    # ==========================================================
    print("\n" + "="*50)
    print(" >>> 新指标 (Log-Focal) 验证报告 <<<")
    print("="*50)
    
    print(f"1. AUC Score (区分能力):  {auc:.4f}")
    print(f"   (解读: >0.8 表示非常优秀，能以 {auc:.1%} 的概率正确区分对错)")
    
    print(f"\n2. Point-Biserial Corr: {pb_corr:.4f} (P-value: {p_value:.2e})")
    print(f"   (解读: 负值越强越好。代表 Score 越低，Acc 越高)")

    print(f"\n3. Spearman Rank Corr:  {sp_corr:.4f}")
    print(f"   (解读: 同样应该是负值，衡量排序的一致性)")

    # ==========================================================
    # 5. 可视化验证 (修复后的Boxplot)
    # ==========================================================
    plt.figure(figsize=(8, 5))
    
    # 修复1: 显式指定hue参数，匹配accuracy的数值类型
    # 修复2: 调色板键使用数值类型（int），与accuracy列保持一致
    sns.boxplot(
        x='accuracy', 
        y='Log_Focal_Score', 
        hue='accuracy',  # 新增hue参数，匹配x轴
        data=df, 
        palette={0: "#d62728", 1: "#2ca02c"},  # 键为int类型，匹配accuracy值
        legend=False  # 关闭图例（避免重复）
    )
    
    plt.title(f'Visual Verification: AUC={auc:.3f}\n(Green box should be lower than Red box)', fontsize=12)
    plt.ylabel('Log Focal Score (Lower is Better)')
    plt.xlabel('Accuracy (0=Wrong, 1=Correct)')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    save_path = "verification_boxplot.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')  # 防止标题被截断
    print(f"\n[Visual] 箱线图已保存至: {save_path}")
    print("="*50)

if __name__ == "__main__":
    verify_metric()