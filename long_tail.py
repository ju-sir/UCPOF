import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# 仅保留负号正常显示，关闭中文相关配置（图表用英文）
plt.rcParams['axes.unicode_minus'] = False

# 原始概率数据
prior_map = {
    'declare bankruptcy': 0.009575104727707959,
    'transfer ownership': 0.02154398563734291,
    'transfer money': 0.029024536205864752,
    'marry': 0.019748653500897665,
    'transport': 0.14661879114302812,
    'die': 0.10951526032315978,
    'phone write': 0.03141831238779174,
    'arrest jail': 0.01944943147815679,
    'convict': 0.014063435068821066,
    'sentence': 0.01526032315978456,
    'sue': 0.01436265709156194,
    'end organization': 0.008976660682226212,
    'start organization': 0.0062836624775583485,
    'end position': 0.039796529024536204,
    'start position': 0.023937761819269897,
    'meet': 0.04757630161579892,
    'elect': 0.04009575104727708,
    'attack': 0.27079593058049073,
    'injure': 0.02333931777378815,
    'born': 0.011968880909634948,
    'fine': 0.0038898862956313583,
    'release parole': 0.00807899461400359,
    'charge indict': 0.02154398563734291,
    'extradite': 0.0011968880909634949,
    'trial hearing': 0.02483542788749252,
    'demonstrate': 0.014961101137043686,
    'divorce': 0.0047875523638539795,
    'nominate': 0.002992220227408737,
    'appeal': 0.0062836624775583485,
    'pardon': 0.0005984440454817474,
    'execute': 0.003291442250149611,
    'acquit': 0.0005984440454817474,
    'merge organization': 0.003590664272890485
}

# 1. 数据预处理：按概率降序排序（高概率在前）
sorted_items = sorted(prior_map.items(), key=lambda x: x[1], reverse=True)
sorted_categories = [k for k, v in sorted_items]  # X轴：类别标签（英文，高概率在前）
sorted_probs = [v for k, v in sorted_items]       # Y轴：概率值
rank = np.arange(1, len(sorted_probs)+1)          # 排名（用于量化判断）

# 2. 绘制单一折线图（图表纯英文）
fig, ax = plt.subplots(figsize=(16, 8))

# 绘制折线图：X=类别标签，Y=概率值
ax.plot(sorted_categories, sorted_probs, 'o-', 
        color='#2E86AB', linewidth=2, markersize=6, label='Probability value')

# 标注前3个高概率类别（仅显示概率值，无中文）
for i in range(3):
    ax.annotate(
        f'{sorted_probs[i]:.4f}',
        xy=(sorted_categories[i], sorted_probs[i]),
        xytext=(5, 5), textcoords='offset points',
        fontsize=9, bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7)
    )

# 图表配置（全程英文）
ax.set_xticks(range(len(sorted_categories)))
ax.set_xticklabels(sorted_categories, rotation=45, ha='right', fontsize=8)
ax.set_xlabel('Category labels (descending order of probability)', fontsize=12, fontweight='bold')
ax.set_ylabel('Probability value', fontsize=12, fontweight='bold')
ax.set_title('Probability distribution of categories (High probability first)', 
             fontsize=14, fontweight='bold', pad=20)

# 辅助元素（英文标注）
ax.axhline(y=np.mean(sorted_probs), color='r', linestyle='--', 
           label=f'Mean value ({np.mean(sorted_probs):.4f})')
ax.grid(alpha=0.3)
ax.legend(loc='upper right', fontsize=10)

# 调整布局并保存
plt.tight_layout()
plt.savefig('long_tail_distribution_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. 长尾分布量化判断（终端输出纯中文）
print("=" * 80)
print("长尾分布判断结果")
print("=" * 80)

# 计算核心指标
cumulative_probs = np.cumsum(sorted_probs)
top20_idx = int(len(rank) * 0.2)
top20_ratio = cumulative_probs[top20_idx]

# 幂律拟合
non_zero_probs = [p for p in sorted_probs if p > 0]
log_rank = np.log10(np.arange(1, len(non_zero_probs)+1))
log_probs = np.log10(non_zero_probs)
slope, intercept, r_value, p_value, std_err = stats.linregress(log_rank, log_probs)
r_squared = r_value ** 2

# 中位数/平均值比
median_mean_ratio = np.median(sorted_probs) / np.mean(sorted_probs)

# 输出中文指标
print(f"1. 前20%高概率类别占总概率的比例: {top20_ratio:.4f} ({top20_ratio*100:.2f}%)")
print(f"2. 对数坐标下幂律拟合R²值: {r_squared:.4f} (≥0.7即可判定为长尾分布)")
print(f"3. 中位数/平均值比值: {median_mean_ratio:.4f} (<<1 是长尾分布特征)")

# 最终结论（中文）
if top20_ratio > 0.6 and r_squared > 0.7 and median_mean_ratio < 0.5:
    conclusion = "✅ 判定为典型的长尾分布"
else:
    conclusion = "⚠️ 非典型长尾分布（或弱长尾分布）"
print(f"\n最终结论: {conclusion}")