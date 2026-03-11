import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

# ================= 配置 =================
CSV_PATH = "/home/jgy/paper-prompt/prompt-order/last8/v1(no-prototype)/Qwen2.5-7B-Instruct_full_metrics.csv"
OUTPUT_DIR = "/home/jgy/paper-prompt/prompt-order/last8/v1(no-prototype)"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"正在读取数据: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

# ================= 数据准备 =================
target_label = "Examples->Task"
df_filtered = df[df['order'] == target_label].copy()
if len(df_filtered) == 0:
    df_filtered = df[df['order'].str.contains("Examples")].copy()

# ★ 1. 计算 Log Focal Metric ★
df_filtered['Metric_Raw'] = df_filtered['first_token_entropy_top50'] * ((1 - df_filtered['pred_prior_prob']) ** 2)
# 取 Log，加极小值
df_filtered['composite_metric'] = np.log10(df_filtered['Metric_Raw'] + 1e-9)

# 填充空值
df_filtered = df_filtered.dropna(subset=['composite_metric'])
df_filtered['Outcome'] = df_filtered['accuracy'].map({1: 'Correct', 0: 'Incorrect'})

# ★ 2. 动态计算 X 轴范围 ★
data_min = df_filtered['composite_metric'].min()
data_max = df_filtered['composite_metric'].max()
padding = (data_max - data_min) * 0.1
X_LIMIT_MIN = data_min - padding
X_LIMIT_MAX = data_max + padding
print(f"Log Range: {data_min:.2f} ~ {data_max:.2f}")

# ================= 辅助函数 =================
def plot_kde_with_peaks(data, color, label, ax, linestyle='-', x_min=-10, x_max=5):
    if len(data) < 2: return 
    try:
        kde = gaussian_kde(data)
        x_grid = np.linspace(x_min, x_max, 1000)
        y_grid = kde(x_grid)
        ax.plot(x_grid, y_grid, color=color, label=label, linewidth=2, linestyle=linestyle)
        ax.fill_between(x_grid, y_grid, color=color, alpha=0.2)
        peaks, _ = find_peaks(y_grid)
        for peak_idx in peaks:
            px, py = x_grid[peak_idx], y_grid[peak_idx]
            if py > 0.05: 
                ax.vlines(x=px, ymin=0, ymax=py, color=color, linestyle=':', linewidth=1)
                ax.text(px, py + (py*0.05), f"{px:.1f}", color=color, fontsize=9, ha='center', fontweight='bold')
    except: pass

# ================= 正式绘图 =================
sns.set_theme(style="whitegrid", font_scale=1.1)

# 数据切片
correct_data = df_filtered[df_filtered['accuracy'] == 1]
incorrect_data = df_filtered[df_filtered['accuracy'] == 0]
global_median = df_filtered['composite_metric'].median()

# 左侧（自信侧）方差计算
left_correct = correct_data[correct_data['composite_metric'] <= global_median]
right_correct = correct_data[correct_data['composite_metric'] > global_median]
var_left = left_correct['composite_metric'].var() if len(left_correct)>0 else 0
var_right = right_correct['composite_metric'].var() if len(right_correct)>0 else 0

fig2, ax2 = plt.subplots(figsize=(14, 8))

# 1. 错误样本 (红)
plot_kde_with_peaks(incorrect_data['composite_metric'], "#d62728", "Incorrect", ax2, x_min=X_LIMIT_MIN, x_max=X_LIMIT_MAX)
# 2. 正确样本 (绿)
plot_kde_with_peaks(correct_data['composite_metric'], "#2ca02c", "Correct", ax2, x_min=X_LIMIT_MIN, x_max=X_LIMIT_MAX)

# 3. 中位线
ax2.axvline(x=global_median, color='black', linestyle='--', linewidth=2, label=f'Global Median ({global_median:.2f})')

# 4. 高置信度区域背景 (Log Scale 下，越小越自信，所以是左边)
y_lim = ax2.get_ylim()
ax2.fill_betweenx(y=y_lim, x1=X_LIMIT_MIN, x2=global_median, color='gray', alpha=0.1, label='High Confidence Zone')

ax2.set_title(f"Correct vs Incorrect Distribution (Focal Log Scale)\nLeft Correct Var: {var_left:.3f} | Right Correct Var: {var_right:.3f}", fontsize=14)
ax2.set_xlabel('Log10(Focal Score) \n← More Confident (Negative) | Less Confident (Positive) →', fontsize=12)
ax2.set_ylabel('Density', fontsize=12)
ax2.set_xlim(X_LIMIT_MIN, X_LIMIT_MAX)
ax2.legend(loc='upper right')

plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "shanfeng.png")
plt.savefig(save_path, dpi=300)
plt.close()
print(f"Peak plot saved to: {save_path}")