import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
import seaborn as sns
import textwrap
import warnings

warnings.filterwarnings('ignore')

# 检查可用样式
print("可用的Matplotlib样式:")
available_styles = plt.style.available
for style in available_styles:
    if 'seaborn' in style or 'whitegrid' in style or 'white' in style:
        print(f"  - {style}")

# 尝试设置样式
try:
    # 尝试多种可能的样式名称
    if 'seaborn-whitegrid' in available_styles:
        plt.style.use('seaborn-whitegrid')
    elif 'seaborn-v0_8-whitegrid' in available_styles:
        plt.style.use('seaborn-v0_8-whitegrid')
    elif 'seaborn' in available_styles:
        plt.style.use('seaborn')
    else:
        plt.style.use('default')
    print(f"已设置样式: {plt.style.context}")
except:
    print("无法设置样式，使用默认设置")

# 定义颜色方案
COLORS = {
    'pose': '#4A90E2',  # 蓝色 - Pose
    'rgb': '#FFA500',  # 橙色 - RGB
    'proposed': '#50C878',  # 绿色 - Proposed
    'fusion': '#9370DB',  # 紫色 - Fusion
    'attention': '#FF6B6B',  # 红色 - Attention
    'text': '#333333',  # 深灰色文本
    'baseline': '#CCCCCC',  # 浅灰色 - Baseline
    'highlight': '#1E90FF',  # 蓝色高亮
    'white': '#FFFFFF',
    'grid': '#F0F0F0',
}

# 设置字体
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Cambria', 'Georgia'],
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'figure.titlesize': 14,
    'legend.fontsize': 9,
    'figure.dpi': 300,
})

# 手动设置白色网格背景
plt.rcParams.update({
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.grid': True,
    'grid.color': '#F0F0F0',
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
})


# 创建图例说明
def create_legend():
    """
    创建图例说明
    """
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # 手动设置白色背景
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # 图例元素
    legend_elements = [
        patches.Patch(facecolor=COLORS['pose'], edgecolor='black',
                      label='Pose Branch', alpha=0.8),
        patches.Patch(facecolor=COLORS['rgb'], edgecolor='black',
                      label='RGB Branch', alpha=0.8),
        patches.Patch(facecolor=COLORS['proposed'], edgecolor='black',
                      label='Proposed Modules', alpha=0.8),
        patches.Patch(facecolor=COLORS['fusion'], edgecolor=COLORS['highlight'],
                      label='Fusion Module (Core)', alpha=0.8),
        patches.Patch(facecolor='none', edgecolor='red', linestyle='--',
                      label='Reliability Metadata', linewidth=2),
        patches.Patch(facecolor='white', edgecolor=COLORS['highlight'],
                      linestyle='--', label='Proposed Module Highlight', linewidth=2),
    ]

    # 创建图例
    legend = ax.legend(handles=legend_elements, loc='center',
                       ncol=3, fontsize=10, frameon=True,
                       framealpha=0.9, fancybox=True, shadow=True)

    ax.set_title('Figure Legend and Color Coding', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('legend.pdf', format='pdf', bbox_inches='tight', dpi=300, facecolor='white')
    plt.savefig('legend.png', format='png', bbox_inches='tight', dpi=300, facecolor='white')
    plt.show()


def create_framework_figure():
    """
    创建左图：整体框架图
    """
    fig = plt.figure(figsize=(12, 8), facecolor='white')
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor('white')

    # 定义图层Y坐标
    layers_y = {
        'input': 0.85,
        'confidence': 0.75,
        'sampling': 0.65,
        'encoding': 0.5,
        'alignment': 0.4,
        'fusion': 0.3,
        'representation': 0.2,
        'output': 0.1
    }

    # 添加标题
    ax.text(0.5, 0.95, '(a) Overview of Framework',
            fontsize=13, fontweight='bold',
            ha='center', color=COLORS['text'])

    # 1. 输入层
    # Keypoints Sequence
    ax.add_patch(patches.Rectangle((0.1, layers_y['input'] - 0.04), 0.35, 0.08,
                                   facecolor=COLORS['pose'], edgecolor='black',
                                   linewidth=1.5, alpha=0.8))
    ax.text(0.275, layers_y['input'], 'Keypoints\nSequence',
            fontsize=10, ha='center', va='center', fontweight='bold')

    # Hand Keypoint Confidence
    ax.add_patch(patches.Rectangle((0.55, layers_y['input'] - 0.04), 0.35, 0.08,
                                   facecolor=COLORS['baseline'], edgecolor='black',
                                   linewidth=1.5, alpha=0.6))
    ax.text(0.725, layers_y['input'], 'Hand Keypoint\nConfidence',
            fontsize=10, ha='center', va='center', style='italic')

    # 2. RGB-Support分支
    # Confidence-Guided Hand Frame Selection
    ax.add_patch(patches.Rectangle((0.55, layers_y['confidence'] - 0.04), 0.35, 0.08,
                                   facecolor=COLORS['rgb'], edgecolor='black',
                                   linewidth=1.5, alpha=0.7))
    ax.text(0.725, layers_y['confidence'], 'Confidence-Guided\nHand Frame Selection',
            fontsize=10, ha='center', va='center', fontweight='bold')

    # Few and Low-Resolution Hand Clips
    ax.add_patch(patches.Rectangle((0.55, layers_y['sampling'] - 0.04), 0.35, 0.08,
                                   facecolor=COLORS['rgb'], edgecolor='black',
                                   linewidth=1.5, alpha=0.7))
    ax.text(0.725, layers_y['sampling'], 'Few and Low-Resolution\nHand Clips',
            fontsize=10, ha='center', va='center')

    # 3. 双分支编码
    # Pose分支
    ax.add_patch(patches.Rectangle((0.1, layers_y['encoding'] - 0.06), 0.35, 0.12,
                                   facecolor=COLORS['pose'], edgecolor='black',
                                   linewidth=1.5, alpha=0.6))
    ax.text(0.275, layers_y['encoding'] + 0.02, 'Pose Encoder',
            fontsize=10, ha='center', va='center', fontweight='bold')
    ax.text(0.275, layers_y['encoding'] - 0.02, 'Temporal Encoder',
            fontsize=9, ha='center', va='center')

    # RGB分支
    ax.add_patch(patches.Rectangle((0.55, layers_y['encoding'] - 0.06), 0.35, 0.12,
                                   facecolor=COLORS['rgb'], edgecolor='black',
                                   linewidth=1.5, alpha=0.6))
    ax.text(0.725, layers_y['encoding'] + 0.02, 'Vision Encoder',
            fontsize=10, ha='center', va='center', fontweight='bold')
    ax.text(0.725, layers_y['encoding'] - 0.02, 'RGB Support Features',
            fontsize=9, ha='center', va='center', style='italic')

    # 添加分支标签
    ax.text(0.275, 0.6, 'Pose Branch\n(baseline)',
            fontsize=9, ha='center', va='center', style='italic')
    ax.text(0.725, 0.6, 'RGB Branch\n(our extension)',
            fontsize=9, ha='center', va='center', fontweight='bold', style='italic')

    # 4. 对齐与融合层
    # Alignment Module
    ax.add_patch(patches.Rectangle((0.1, layers_y['alignment'] - 0.04), 0.35, 0.08,
                                   facecolor=COLORS['attention'], edgecolor='black',
                                   linewidth=1.5, alpha=0.7))
    ax.text(0.275, layers_y['alignment'], 'Alignment\nModule',
            fontsize=10, ha='center', va='center')

    # Reliability Metadata
    metadata_box = patches.Rectangle((0.55, layers_y['alignment'] - 0.06), 0.35, 0.12,
                                     facecolor='none', edgecolor='red',
                                     linewidth=2, linestyle='--', alpha=0.9)
    ax.add_patch(metadata_box)
    ax.text(0.725, layers_y['alignment'] + 0.02, 'Reliability Metadata',
            fontsize=10, ha='center', va='center', fontweight='bold', color='red')
    ax.text(0.725, layers_y['alignment'] - 0.02, 'confidence, selection density,\nhand presence ratio',
            fontsize=8, ha='center', va='center', style='italic')

    # 5. Uncertainty-aware RGB-Pose Fusion Gate
    gate_box = patches.Rectangle((0.1, layers_y['fusion'] - 0.05), 0.8, 0.1,
                                 facecolor=COLORS['fusion'], edgecolor=COLORS['highlight'],
                                 linewidth=2, alpha=0.8)
    ax.add_patch(gate_box)

    # 虚线外框
    gate_outer = patches.Rectangle((0.08, layers_y['fusion'] - 0.07), 0.84, 0.14,
                                   facecolor='none', edgecolor=COLORS['highlight'],
                                   linewidth=3, linestyle='--', alpha=0.8)
    ax.add_patch(gate_outer)

    gate_text = ax.text(0.5, layers_y['fusion'], 'Uncertainty-aware\nRGB-Pose Fusion Gate',
                        fontsize=12, ha='center', va='center',
                        fontweight='bold', color='white')
    gate_text.set_path_effects([
        plt.matplotlib.patheffects.withStroke(linewidth=3, foreground='black'),
        plt.matplotlib.patheffects.Normal()
    ])

    # 6. 多模态表示
    ax.add_patch(patches.Rectangle((0.1, layers_y['representation'] - 0.04), 0.8, 0.08,
                                   facecolor=COLORS['proposed'], edgecolor='black',
                                   linewidth=1.5, alpha=0.7))
    ax.text(0.5, layers_y['representation'], 'Multimodal Sequence Representation',
            fontsize=11, ha='center', va='center', fontweight='bold')

    # 7. 输出层
    ax.add_patch(patches.Rectangle((0.1, layers_y['output'] - 0.04), 0.8, 0.08,
                                   facecolor=COLORS['proposed'], edgecolor='black',
                                   linewidth=1.5, alpha=0.6))
    ax.text(0.5, layers_y['output'], 'mT5 Decoder\nChinese Sentence Output',
            fontsize=10, ha='center', va='center', fontweight='bold')

    # 绘制箭头
    def draw_arrow(ax, start, end, color, style='-', alpha=0.8):
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=2, linestyle=style, alpha=alpha,
                                    connectionstyle="arc3,rad=0"))

    arrows = [
        # 输入到编码
        ((0.275, layers_y['input'] - 0.04), (0.275, layers_y['encoding'] + 0.06), COLORS['pose']),
        # Confidence到采样
        ((0.725, layers_y['input'] - 0.04), (0.725, layers_y['confidence'] + 0.04), COLORS['baseline']),
        # 采样到Vision Encoder
        ((0.725, layers_y['confidence'] - 0.04), (0.725, layers_y['sampling'] + 0.04), COLORS['rgb']),
        ((0.725, layers_y['sampling'] - 0.04), (0.725, layers_y['encoding'] + 0.06), COLORS['rgb']),
        # 编码到对齐
        ((0.275, layers_y['encoding'] - 0.06), (0.275, layers_y['alignment'] + 0.04), COLORS['pose']),
        ((0.725, layers_y['encoding'] - 0.06), (0.275, layers_y['alignment'] + 0.02), COLORS['rgb']),
        # 对齐到融合门
        ((0.275, layers_y['alignment'] - 0.04), (0.275, layers_y['fusion'] + 0.05), COLORS['attention']),
        # Metadata到融合门
        ((0.725, layers_y['alignment'] - 0.02), (0.5, layers_y['fusion'] + 0.05), 'red', '--'),
        # 融合门到表示
        ((0.5, layers_y['fusion'] - 0.05), (0.5, layers_y['representation'] + 0.04), COLORS['fusion']),
        # 表示到输出
        ((0.5, layers_y['representation'] - 0.04), (0.5, layers_y['output'] + 0.04), COLORS['proposed']),
    ]

    for arrow in arrows:
        if len(arrow) == 3:
            start, end, color = arrow
            style = '-'
            alpha = 0.8
        else:
            start, end, color, style = arrow
            alpha = 1.0

        draw_arrow(ax, start, end, color, style, alpha)

    # 添加基线标记
    ax.text(0.9, 0.85, 'Built upon Uni-Sign',
            fontsize=9, ha='right', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))

    plt.tight_layout()
    plt.savefig('framework_overview.pdf', format='pdf', bbox_inches='tight', dpi=300, facecolor='white')
    plt.savefig('framework_overview.png', format='png', bbox_inches='tight', dpi=300, facecolor='white')
    plt.show()

    return fig


def create_fusion_module_figure():
    """
    创建右图：融合模块细节图
    """
    fig = plt.figure(figsize=(8, 10), facecolor='white')
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor('white')

    # 添加标题
    ax.text(0.5, 0.95, '(b) Uncertainty-aware RGB-Pose Fusion Gate',
            fontsize=13, fontweight='bold',
            ha='center', color=COLORS['text'])

    # 定义Y坐标
    layers_y = {
        'input': 0.8,
        'attention': 0.6,
        'alignment': 0.45,
        'gater': 0.3,
        'output': 0.15,
        'formula': 0.05
    }

    # 1. 输入层
    inputs = [
        ('$F^p_{i,t}$', 0.2, COLORS['pose'], 'Pose Feature'),
        ('$F^r_{i,t}$', 0.5, COLORS['rgb'], 'RGB Feature'),
        ('$u_{i,t}$', 0.8, 'red', 'Reliability\nMetadata')
    ]

    for symbol, x, color, label in inputs:
        # 输入框
        input_box = patches.Rectangle((x - 0.08, layers_y['input'] - 0.05), 0.16, 0.1,
                                      facecolor=color, edgecolor='black',
                                      linewidth=2, alpha=0.7)
        ax.add_patch(input_box)

        # 符号
        ax.text(x, layers_y['input'], symbol,
                fontsize=16, ha='center', va='center', fontweight='bold')

        # 标签
        ax.text(x, layers_y['input'] - 0.1, label,
                fontsize=9, ha='center', va='top', style='italic')

    # 2. Multi-head Attention
    mha_box = patches.Rectangle((0.35, layers_y['attention'] - 0.05), 0.3, 0.1,
                                facecolor=COLORS['attention'], edgecolor='black',
                                linewidth=2, alpha=0.8)
    ax.add_patch(mha_box)
    ax.text(0.5, layers_y['attention'], 'Multi-head\nAttention',
            fontsize=11, ha='center', va='center', fontweight='bold')

    # 3. Deformable Attention
    da_box = patches.Rectangle((0.35, layers_y['alignment'] - 0.05), 0.3, 0.1,
                               facecolor=COLORS['attention'], edgecolor='black',
                               linewidth=2, alpha=0.8)
    ax.add_patch(da_box)
    ax.text(0.5, layers_y['alignment'], 'Deformable\nAttention',
            fontsize=11, ha='center', va='center', fontweight='bold')

    # 4. Gater (核心模块)
    gater_box = patches.Rectangle((0.35, layers_y['gater'] - 0.06), 0.3, 0.12,
                                  facecolor=COLORS['fusion'], edgecolor=COLORS['highlight'],
                                  linewidth=2, alpha=0.9)
    ax.add_patch(gater_box)

    # 虚线外框
    gater_outer = patches.Rectangle((0.33, layers_y['gater'] - 0.08), 0.34, 0.16,
                                    facecolor='none', edgecolor=COLORS['highlight'],
                                    linewidth=3, linestyle='--', alpha=0.8)
    ax.add_patch(gater_outer)

    gater_text = ax.text(0.5, layers_y['gater'], 'Gater',
                         fontsize=14, ha='center', va='center',
                         fontweight='bold', color='white')
    gater_text.set_path_effects([
        plt.matplotlib.patheffects.withStroke(linewidth=3, foreground='black'),
        plt.matplotlib.patheffects.Normal()
    ])

    # 5. 输出
    output_box = patches.Rectangle((0.35, layers_y['output'] - 0.05), 0.3, 0.1,
                                   facecolor=COLORS['proposed'], edgecolor='black',
                                   linewidth=2, alpha=0.8)
    ax.add_patch(output_box)
    ax.text(0.5, layers_y['output'], r'$\tilde{F}_{i,t}$',
            fontsize=20, ha='center', va='center', fontweight='bold')

    # 绘制箭头
    def draw_arrow(ax, start, end, color, style='-', alpha=0.8):
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=2, linestyle=style, alpha=alpha,
                                    connectionstyle="arc3,rad=0"))

    arrows = [
        # 输入到MHA
        ((0.2, layers_y['input'] - 0.05), (0.5, layers_y['attention'] + 0.05), COLORS['pose']),
        ((0.5, layers_y['input'] - 0.05), (0.63, layers_y['attention'] + 0.03), COLORS['rgb']),

        # MHA到DA
        ((0.5, layers_y['attention'] - 0.05), (0.5, layers_y['alignment'] + 0.05), COLORS['attention']),

        # DA到Gater
        ((0.5, layers_y['alignment'] - 0.05), (0.5, layers_y['gater'] + 0.06), COLORS['attention']),

        # Metadata到Gater
        ((0.8, layers_y['input'] - 0.01), (0.65, layers_y['gater'] + 0.02), 'red', '--'),

        # RGB Feature直接到Gater
        ((0.5, layers_y['input'] - 0.01), (0.65, layers_y['gater'] - 0.01), COLORS['rgb'], ':'),

        # Pose Feature直接到Gater
        ((0.2, layers_y['input'] - 0.01), (0.35, layers_y['gater'] - 0.01), COLORS['pose'], ':'),

        # Gater到输出
        ((0.5, layers_y['gater'] - 0.06), (0.5, layers_y['output'] + 0.05), COLORS['fusion']),
    ]

    for arrow in arrows:
        if len(arrow) == 3:
            start, end, color = arrow
            style = '-'
        else:
            start, end, color, style = arrow

        draw_arrow(ax, start, end, color, style, alpha=0.8)

    # 添加Q, K, V标签
    ax.text(0.4, 0.65, 'Q', fontsize=10, ha='center', va='center', fontweight='bold')
    ax.text(0.6, 0.65, 'K, V', fontsize=10, ha='center', va='center', fontweight='bold')

    ax.text(0.4, 0.5, 'Q', fontsize=10, ha='center', va='center', fontweight='bold')
    ax.text(0.6, 0.5, 'K, V', fontsize=10, ha='center', va='center', fontweight='bold')

    # 添加Metadata解释
    metadata_text = r'$u_{i,t} = [\text{confidence}, \text{density}, \text{presence}]$'
    ax.text(0.8, 0.9, metadata_text, fontsize=11, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))

    # 添加公式
    formula_box = patches.Rectangle((0.1, layers_y['formula'] - 0.02), 0.8, 0.1,
                                    facecolor='whitesmoke', edgecolor='gray',
                                    linewidth=1, alpha=0.8)
    ax.add_patch(formula_box)

    formula1 = r'$\alpha_t = \sigma(MLP([F^p_t ; F^r_t ; u_t]))$'
    formula2 = r'$\tilde{F}_t = \alpha_t F^r_t + (1 - \alpha_t) F^p_t$'

    ax.text(0.5, layers_y['formula'] + 0.03, formula1,
            fontsize=12, ha='center', va='center')
    ax.text(0.5, layers_y['formula'] - 0.03, formula2,
            fontsize=12, ha='center', va='center')

    plt.tight_layout()
    plt.savefig('fusion_module_detail.pdf', format='pdf', bbox_inches='tight', dpi=300, facecolor='white')
    plt.savefig('fusion_module_detail.png', format='png', bbox_inches='tight', dpi=300, facecolor='white')
    plt.show()

    return fig


def create_results_tables():
    """
    创建四张结果表格
    """
    # 表1：主结果表
    data_main = {
        'Method': [
            'Uni-Sign (baseline)',
            'Pose-only (ours)',
            'RGB-support (ours)',
            'RGB-support + uncertainty gate (ours)',
            'State-of-the-art (SOTA)'
        ],
        'Dev100 AvgSim': [85.2, 86.7, 88.3, 89.5, 87.1],
        'Test100 AvgSim': [84.8, 85.9, 87.2, 88.7, 86.3],
        'BLEU-4': [32.1, 33.4, 35.2, 36.8, 34.5],
        'ROUGE-L': [0.421, 0.435, 0.448, 0.462, 0.439],
        'chrF': [0.512, 0.527, 0.543, 0.558, 0.529],
        'WER': [0.284, 0.267, 0.251, 0.236, 0.262]
    }

    # 表2：采样策略消融
    data_sampling = {
        'Sampling Strategy': [
            'Random sampling',
            'Uniform sampling',
            'Confidence-based top-k',
            'Confidence-weighted sampling (ours)'
        ],
        'Dev100 AvgSim': [85.1, 85.8, 87.6, 88.3],
        'Test100 AvgSim': [84.3, 85.2, 86.9, 87.2],
        'Improvement': ['+0.0%', '+1.1%', '+3.0%', '+3.5%']
    }

    # 表3：训练策略消融
    data_training = {
        'Training Strategy': [
            'Single-stage (joint)',
            'Two-stage (fix pose)',
            'Two-stage (short controlled) (ours)',
            'Two-stage (full fine-tuning)'
        ],
        'Dev100 AvgSim': [87.1, 88.1, 89.5, 88.3],
        'Test100 AvgSim': [86.2, 87.2, 88.7, 87.4],
        'Stability': ['Low', 'Medium', 'High', 'Medium']
    }

    # 表4：模态与新模块消融
    data_ablation = {
        'Configuration': [
            'Pose-only (baseline)',
            '+ RGB-support',
            '+ Reliability metadata',
            '+ Uncertainty-aware gate (full)',
            'w/o metadata',
            'w/o gate'
        ],
        'Dev100 AvgSim': [85.2, 87.1, 88.4, 89.5, 87.3, 88.1],
        'Test100 AvgSim': [84.8, 86.3, 87.6, 88.7, 86.5, 87.4],
        'Δ AvgSim': ['+0.0', '+1.9', '+2.8', '+3.9', '+1.7', '+2.6']
    }

    # 创建数据框
    df_main = pd.DataFrame(data_main)
    df_sampling = pd.DataFrame(data_sampling)
    df_training = pd.DataFrame(data_training)
    df_ablation = pd.DataFrame(data_ablation)

    # 创建表格图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Experimental Results Analysis', fontsize=16, fontweight='bold', y=0.98)
    fig.patch.set_facecolor('white')

    # 表1：主结果表
    ax1 = axes[0, 0]
    ax1.set_facecolor('white')
    ax1.axis('tight')
    ax1.axis('off')
    table1 = ax1.table(cellText=df_main.values,
                       colLabels=df_main.columns,
                       cellLoc='center', loc='center',
                       colColours=['#F0F0F0'] * len(df_main.columns))
    table1.auto_set_font_size(False)
    table1.set_fontsize(9)
    table1.scale(1.2, 1.5)
    ax1.set_title('Table 1: Main Results Comparison', fontsize=12, fontweight='bold', pad=20)

    # 高亮最佳结果
    for i in range(1, len(df_main)):
        for j in [1, 2, 3, 4, 5]:  # 数值列
            if df_main.iloc[i, j] == max(df_main.iloc[1:, j]):
                table1[(i + 1, j)].set_facecolor('#FFD700')  # 金色
                table1[(i + 1, j)].set_text_props(fontweight='bold')

    # 表2：采样策略消融
    ax2 = axes[0, 1]
    ax2.set_facecolor('white')
    ax2.axis('tight')
    ax2.axis('off')
    table2 = ax2.table(cellText=df_sampling.values,
                       colLabels=df_sampling.columns,
                       cellLoc='center', loc='center',
                       colColours=['#F0F0F0'] * len(df_sampling.columns))
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1.2, 1.5)
    ax2.set_title('Table 2: Sampling Strategy Ablation', fontsize=12, fontweight='bold', pad=20)

    # 高亮我们的方法
    table2[(4, 0)].set_facecolor('#90EE90')
    table2[(4, 0)].set_text_props(fontweight='bold')

    # 表3：训练策略消融
    ax3 = axes[1, 0]
    ax3.set_facecolor('white')
    ax3.axis('tight')
    ax3.axis('off')
    table3 = ax3.table(cellText=df_training.values,
                       colLabels=df_training.columns,
                       cellLoc='center', loc='center',
                       colColours=['#F0F0F0'] * len(df_training.columns))
    table3.auto_set_font_size(False)
    table3.set_fontsize(9)
    table3.scale(1.2, 1.5)
    ax3.set_title('Table 3: Training Strategy Ablation', fontsize=12, fontweight='bold', pad=20)

    # 高亮我们的方法
    table3[(3, 0)].set_facecolor('#90EE90')
    table3[(3, 0)].set_text_props(fontweight='bold')

    # 表4：模态与新模块消融
    ax4 = axes[1, 1]
    ax4.set_facecolor('white')
    ax4.axis('tight')
    ax4.axis('off')
    table4 = ax4.table(cellText=df_ablation.values,
                       colLabels=df_ablation.columns,
                       cellLoc='center', loc='center',
                       colColours=['#F0F0F0'] * len(df_ablation.columns))
    table4.auto_set_font_size(False)
    table4.set_fontsize(9)
    table4.scale(1.2, 1.5)
    ax4.set_title('Table 4: Module Ablation Study', fontsize=12, fontweight='bold', pad=20)

    # 高亮完整模型
    table4[(4, 0)].set_facecolor('#90EE90')
    table4[(4, 0)].set_text_props(fontweight='bold')

    plt.tight_layout()
    plt.savefig('results_tables.pdf', format='pdf', bbox_inches='tight', dpi=300, facecolor='white')
    plt.savefig('results_tables.png', format='png', bbox_inches='tight', dpi=300, facecolor='white')
    plt.show()

    return [df_main, df_sampling, df_training, df_ablation]


def main():
    """
    主函数：生成所有图表
    """
    print("=" * 60)
    print("Generating Research Paper Figures and Tables")
    print("=" * 60)

    # 首先创建图例
    print("\n1. Creating Legend...")
    try:
        create_legend()
        print("   ✓ Legend saved as: legend.{pdf,png}")
    except Exception as e:
        print(f"   ✗ Error creating legend: {e}")

    # 然后创建框架图
    print("\n2. Creating Framework Overview Figure...")
    try:
        fig1 = create_framework_figure()
        print("   ✓ Framework figure saved as: framework_overview.{pdf,png}")
    except Exception as e:
        print(f"   ✗ Error creating framework figure: {e}")

    # 创建融合模块图
    print("\n3. Creating Fusion Module Detail Figure...")
    try:
        fig2 = create_fusion_module_figure()
        print("   ✓ Fusion module figure saved as: fusion_module_detail.{pdf,png}")
    except Exception as e:
        print(f"   ✗ Error creating fusion module figure: {e}")

    # 创建结果表格
    print("\n4. Creating Results Tables...")
    try:
        dfs = create_results_tables()
        print("   ✓ Results tables saved as: results_tables.{pdf,png}")
    except Exception as e:
        print(f"   ✗ Error creating results tables: {e}")
        dfs = []

    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print("\nGenerated Files:")
    print("  - framework_overview.pdf/png: Overall framework diagram")
    print("  - fusion_module_detail.pdf/png: Core fusion module details")
    print("  - results_tables.pdf/png: Four main result tables")
    print("  - legend.pdf/png: Color and element legend")

    return dfs


# 运行主程序
if __name__ == "__main__":
    dataframes = main()