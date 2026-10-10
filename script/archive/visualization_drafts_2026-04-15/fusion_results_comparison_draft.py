import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据
models = ['round6_best\n(RGB-support\nbaseline)', 'ablation_pose_only\n(Pose-only)', 'ablation_uncert_gate\n(RGB-support +\nuncertainty gate)']
test100_scores = [0.3528, 0.3625, 0.3620]
dev100_scores = [0.3026, 0.2917, 0.3021]

# 创建图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Test100 结果
bars1 = ax1.bar(models, test100_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
ax1.set_title('Test100 Performance Comparison', fontsize=16, fontweight='bold')
ax1.set_ylabel('Average Similarity', fontsize=14)
ax1.set_ylim(0.34, 0.37)
ax1.grid(axis='y', alpha=0.3)

# 添加数值标签
for bar, score in zip(bars1, test100_scores):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005, 
             f'{score:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Dev100 结果
bars2 = ax2.bar(models, dev100_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
ax2.set_title('Dev100 Performance Comparison', fontsize=16, fontweight='bold')
ax2.set_ylabel('Average Similarity', fontsize=14)
ax2.set_ylim(0.28, 0.31)
ax2.grid(axis='y', alpha=0.3)

# 添加数值标签
for bar, score in zip(bars2, dev100_scores):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005, 
             f'{score:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# 添加改进标注
ax1.annotate('Improvement:\n+0.0092', xy=(2, 0.3620), xytext=(1.5, 0.365),
             arrowprops=dict(arrowstyle='->', color='#FF6B6B', linewidth=2),
             fontsize=12, ha='center', fontweight='bold')

ax1.annotate('Very close to\npose-only baseline', xy=(2, 0.3620), xytext=(0.5, 0.364),
             arrowprops=dict(arrowstyle='->', color='#45B7D1', linewidth=2),
             fontsize=11, ha='center')

# 设置标题
fig.suptitle('Reliability-Aware RGB-Pose Fusion Results\nCE-CSL Dataset', 
             fontsize=18, fontweight='bold', y=0.98)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.85)

# 保存图表
plt.savefig('fusion_results_comparison.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()

print("图表已保存为: fusion_results_comparison.png")