import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as path_effects

# 设置字体和风格
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Cambria', 'Georgia'],
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'figure.titlesize': 14,
})

# 创建画布
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 9), gridspec_kw={'wspace': 0.2})

# 移除坐标轴
for ax in [ax_left, ax_right]:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

# 定义颜色
COLORS = {
    'pose': '#87CEEB',  # 浅蓝
    'rgb': '#FFD700',  # 浅橙（原黄色，调整为浅橙）
    'proposed': '#90EE90',  # 浅绿
    'highlight': '#4169E1',  # 蓝色高亮
    'text': '#000000',  # 黑色
    'line_pose': '#000000',  # 黑色实线
    'line_rgb': '#32CD32',  # 绿色虚线
    'background': '#FFFFFF',  # 白色背景
}


# 1. 创建图例（左上角）
def create_legend(ax):
    legend_elements = [
        patches.Patch(facecolor='none', edgecolor=COLORS['line_pose'],
                      label='Pose-only', linewidth=2),
        patches.Patch(facecolor='none', edgecolor=COLORS['line_rgb'],
                      linestyle='--', label='RGB-pose', linewidth=2),
        patches.Patch(facecolor='none', edgecolor=COLORS['highlight'],
                      linestyle='--', label='Proposed (highlight)', linewidth=2)
    ]

    legend_box = patches.Rectangle((0.02, 0.85), 0.2, 0.12,
                                   linewidth=1, edgecolor='gray',
                                   facecolor='white', alpha=0.9)
    ax.add_patch(legend_box)

    ax.text(0.06, 0.94, 'Legend', fontsize=9, fontweight='bold', va='center')

    # 手动绘制图例
    ax.plot([0.05, 0.08], [0.90, 0.90], color=COLORS['line_pose'],
            linewidth=2, label='_nolegend_')
    ax.text(0.09, 0.90, 'Pose-only', fontsize=8, va='center')

    ax.plot([0.05, 0.08], [0.86, 0.86], color=COLORS['line_rgb'],
            linestyle='--', linewidth=2, label='_nolegend_')
    ax.text(0.09, 0.86, 'RGB-pose', fontsize=8, va='center')

    ax.plot([0.05, 0.08], [0.82, 0.82], color=COLORS['highlight'],
            linestyle='--', linewidth=2, label='_nolegend_')
    ax.text(0.09, 0.82, 'Proposed', fontsize=8, va='center')


# 2. 绘制左图 (a) Overview of framework
def draw_left_panel(ax):
    # 设置分层坐标
    layers = {
        'input': 0.1,
        'sampling': 0.25,
        'encoding': 0.4,
        'fusion': 0.55,
        'language': 0.7,
        'output': 0.85
    }

    # 绘制标题
    ax.text(0.5, 0.97, '(a) Overview of framework', fontsize=12,
            fontweight='bold', ha='center')

    # Layer 1: 输入层
    # 左下输入框
    pose_input = patches.Rectangle((0.1, layers['input'] - 0.04), 0.3, 0.08,
                                   facecolor=COLORS['pose'], edgecolor='black', linewidth=1)
    ax.add_patch(pose_input)
    ax.text(0.25, layers['input'], 'Keypoints\nSequence', fontsize=9, ha='center', va='center')

    # 左上输入框
    rgb_input = patches.Rectangle((0.6, layers['input'] - 0.04), 0.3, 0.08,
                                  facecolor=COLORS['rgb'], edgecolor='black', linewidth=1)
    ax.add_patch(rgb_input)
    ax.text(0.75, layers['input'], 'Few and Low-Resolution\nHand Clips',
            fontsize=9, ha='center', va='center')

    # Layer 2: 采样与视觉层
    # Score-aware Sampling
    sampling = patches.Rectangle((0.6, layers['sampling'] - 0.04), 0.3, 0.08,
                                 facecolor='lightgray', edgecolor='black', linewidth=1)
    ax.add_patch(sampling)
    ax.text(0.75, layers['sampling'], 'Score-aware\nSampling',
            fontsize=9, ha='center', va='center')

    # Vision Encoder
    vision_enc = patches.Rectangle((0.6, layers['encoding'] - 0.04), 0.3, 0.08,
                                   facecolor='lightgray', edgecolor='black', linewidth=1)
    ax.add_patch(vision_enc)
    ax.text(0.75, layers['encoding'], 'Vision\nEncoder',
            fontsize=9, ha='center', va='center')

    # Layer 3: Pose / RGB 编码层
    # 三个子姿态流
    for i, x_pos in enumerate([0.1, 0.25, 0.4]):
        # Temporal Encoder
        temp_enc = patches.Rectangle((x_pos + 0.025, layers['encoding'] - 0.04), 0.15, 0.08,
                                     facecolor='lightgray', edgecolor='black', linewidth=1)
        ax.add_patch(temp_enc)
        ax.text(x_pos + 0.1, layers['encoding'] + 0.02, 'Temporal\nEncoder',
                fontsize=8, ha='center', va='center')

        # Pose Encoder
        pose_enc = patches.Rectangle((x_pos + 0.025, layers['sampling'] - 0.04), 0.15, 0.08,
                                     facecolor=COLORS['pose'], edgecolor='black', linewidth=1)
        ax.add_patch(pose_enc)
        ax.text(x_pos + 0.1, layers['sampling'], 'Pose\nEncoder',
                fontsize=8, ha='center', va='center')

    # Sub-pose标签
    ax.text(0.25, layers['sampling'] - 0.1, 'Sub-pose', fontsize=9, ha='center',
            style='italic')

    # Layer 4: 融合模块（新增核心模块）
    # 内部模块
    fusion_inner = patches.Rectangle((0.1, layers['fusion'] - 0.05), 0.2, 0.1,
                                     facecolor=COLORS['proposed'], edgecolor='black', linewidth=1.5)
    ax.add_patch(fusion_inner)
    ax.text(0.2, layers['fusion'], 'Uncertainty-aware\nFusion Module',
            fontsize=9, ha='center', va='center', fontweight='bold')

    # 外部虚线框（高亮）
    fusion_outer = patches.Rectangle((0.08, layers['fusion'] - 0.07), 0.24, 0.14,
                                     facecolor='none', edgecolor=COLORS['highlight'],
                                     linewidth=2, linestyle='--')
    ax.add_patch(fusion_outer)

    # Layer 5: 特征拼接
    concat = patches.Rectangle((0.1, layers['language'] - 0.04), 0.8, 0.08,
                               facecolor='lightgray', edgecolor='black', linewidth=1)
    ax.add_patch(concat)
    ax.text(0.5, layers['language'], 'Feature Concatenation',
            fontsize=10, ha='center', va='center')

    # Layer 6: 预训练语言模型
    llm = patches.Rectangle((0.1, layers['output'] - 0.05), 0.8, 0.1,
                            facecolor='lightgray', edgecolor='black', linewidth=1.5)
    ax.add_patch(llm)
    ax.text(0.5, layers['output'], 'Pre-trained Large\nLanguage Model',
            fontsize=10, ha='center', va='center', fontweight='bold')

    # Layer 7: 输出层
    output = patches.Rectangle((0.1, 0.95 - 0.05), 0.8, 0.1,
                               facecolor='none', edgecolor=COLORS['highlight'],
                               linewidth=2, linestyle='--')
    ax.add_patch(output)
    ax.text(0.5, 0.95, 'Low-resource CE-CSL translation\nwith reliability-aware RGB support',
            fontsize=10, ha='center', va='center')

    # 绘制箭头连接
    arrows = [
        # Keypoints Sequence -> Pose Encoder
        ((0.25, layers['input'] + 0.04), (0.25, layers['sampling'] - 0.04), COLORS['line_pose']),

        # Few and Low-Resolution Hand Clips -> Score-aware Sampling
        ((0.75, layers['input'] + 0.04), (0.75, layers['sampling'] - 0.04), COLORS['line_rgb']),

        # Score-aware Sampling -> Hand Clips (虚线)
        ((0.75, layers['sampling'] + 0.04), (0.75, layers['input'] + 0.08), COLORS['line_rgb'], '--'),

        # Score-aware Sampling -> Vision Encoder
        ((0.75, layers['sampling'] + 0.04), (0.75, layers['encoding'] - 0.04), COLORS['line_rgb']),

        # Pose Encoder (第一组) -> Fusion Module
        ((0.2, layers['sampling'] + 0.04), (0.2, layers['fusion'] - 0.05), COLORS['line_pose']),

        # Vision Encoder -> Fusion Module
        ((0.6, layers['encoding'] + 0.04), (0.3, layers['fusion'] - 0.01), COLORS['line_rgb']),

        # Fusion Module -> Temporal Encoder (第一组)
        ((0.2, layers['fusion'] + 0.05), (0.2, layers['encoding'] - 0.04), COLORS['proposed']),

        # 三个Temporal Encoder -> Feature Concatenation
        ((0.2, layers['encoding'] + 0.04), (0.2, layers['language'] - 0.04), COLORS['line_pose']),
        ((0.35, layers['encoding'] + 0.04), (0.35, layers['language'] - 0.04), COLORS['line_pose']),
        ((0.5, layers['encoding'] + 0.04), (0.5, layers['language'] - 0.04), COLORS['line_pose']),

        # Feature Concatenation -> LLM
        ((0.5, layers['language'] + 0.04), (0.5, layers['output'] - 0.05), 'black'),

        # LLM -> 输出
        ((0.5, layers['output'] + 0.05), (0.5, 0.95 - 0.05), 'black')
    ]

    for arrow in arrows:
        if len(arrow) == 3:
            start, end, color = arrow
            style = '-'
        else:
            start, end, color, style = arrow

        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=1.5, linestyle=style, alpha=0.8))

    # 添加标注
    ax.text(0.45, 0.15, 'confidence-guided\nhand-frame selection',
            fontsize=8, ha='center', style='italic', alpha=0.7)

    ax.text(0.2, layers['fusion'] - 0.12, 'reliability-aware\nRGB-pose fusion',
            fontsize=8, ha='center', style='italic', alpha=0.7, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))


# 3. 绘制右图 (b) Uncertainty-aware fusion module
def draw_right_panel(ax):
    # 设置分层坐标
    layers = {
        'input': 0.1,
        'attention1': 0.3,
        'attention2': 0.5,
        'gater': 0.7,
        'output': 0.85
    }

    # 绘制标题
    ax.text(0.5, 0.97, '(b) Uncertainty-aware fusion module', fontsize=12,
            fontweight='bold', ha='center')

    # 底部输入层
    inputs = [
        ('J^s_{i,t}', 0.2, COLORS['rgb']),  # sampled hand cues
        ('F^p_{i,t}', 0.5, COLORS['pose']),  # pose feature
        ('F^r_{i,t}', 0.8, COLORS['rgb'])  # RGB feature
    ]

    for text, x, color in inputs:
        input_box = patches.Rectangle((x - 0.08, layers['input'] - 0.04), 0.16, 0.08,
                                      facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(input_box)
        ax.text(x, layers['input'], text, fontsize=10, ha='center', va='center')

    # 第一层: Multi-head Attention
    mha = patches.Rectangle((0.5 - 0.12, layers['attention1'] - 0.05), 0.24, 0.1,
                            facecolor='lightgray', edgecolor='black', linewidth=1.5)
    ax.add_patch(mha)
    ax.text(0.5, layers['attention1'], 'Multi-head\nAttention',
            fontsize=10, ha='center', va='center')

    # 第二层: Deformable Attention
    da = patches.Rectangle((0.5 - 0.12, layers['attention2'] - 0.05), 0.24, 0.1,
                           facecolor='lightgray', edgecolor='black', linewidth=1.5)
    ax.add_patch(da)
    ax.text(0.5, layers['attention2'], 'Deformable\nAttention',
            fontsize=10, ha='center', va='center')

    # Position init
    pos_init = patches.Rectangle((0.2 - 0.08, layers['attention2'] - 0.04), 0.16, 0.08,
                                 facecolor='lightblue', edgecolor='black', linewidth=1, alpha=0.7)
    ax.add_patch(pos_init)
    ax.text(0.2, layers['attention2'], 'Position\ninit.', fontsize=9, ha='center', va='center')

    # 第三层: Gater
    gater = patches.Rectangle((0.5 - 0.08, layers['gater'] - 0.04), 0.16, 0.08,
                              facecolor=COLORS['proposed'], edgecolor='black', linewidth=1.5)
    ax.add_patch(gater)
    gater_text = ax.text(0.5, layers['gater'], 'Gater',
                         fontsize=10, ha='center', va='center', fontweight='bold')
    gater_text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])

    # 外部虚线框
    gater_outer = patches.Rectangle((0.5 - 0.1, layers['gater'] - 0.06), 0.2, 0.12,
                                    facecolor='none', edgecolor=COLORS['highlight'],
                                    linewidth=2, linestyle='--')
    ax.add_patch(gater_outer)

    # 输出层
    output = patches.Rectangle((0.5 - 0.08, layers['output'] - 0.04), 0.16, 0.08,
                               facecolor=COLORS['proposed'], edgecolor='black', linewidth=1.5)
    ax.add_patch(output)
    ax.text(0.5, layers['output'], r'$\tilde{F}_{i,t}$',
            fontsize=12, ha='center', va='center')

    # reliability metadata
    metadata_box = patches.Rectangle((0.8 - 0.1, layers['gater'] - 0.08), 0.2, 0.16,
                                     facecolor='none', edgecolor='red',
                                     linewidth=1.5, linestyle='--', alpha=0.7)
    ax.add_patch(metadata_box)
    ax.text(0.8, layers['gater'] + 0.02, r'$u_{i,t} = $', fontsize=9, ha='center', va='center')
    ax.text(0.8, layers['gater'] - 0.02, '[confidence,', fontsize=8, ha='center', va='center')
    ax.text(0.8, layers['gater'] - 0.05, 'density,', fontsize=8, ha='center', va='center')
    ax.text(0.8, layers['gater'] - 0.08, 'presence]', fontsize=8, ha='center', va='center')

    # 绘制箭头连接
    arrows = [
        # 输入到MHA
        ((0.5, layers['input'] + 0.04), (0.5, layers['attention1'] - 0.05), 'black'),  # F^p
        ((0.8, layers['input'] + 0.04), (0.62, layers['attention1'] - 0.02), COLORS['rgb']),  # F^r

        # MHA到DA
        ((0.5, layers['attention1'] + 0.05), (0.5, layers['attention2'] - 0.05), 'black'),

        # Position init到DA
        ((0.2, layers['attention2'] + 0.04), (0.38, layers['attention2'] + 0.02), 'black'),

        # 输入到Gater
        ((0.2, layers['input'] + 0.04), (0.42, layers['gater'] - 0.04), COLORS['rgb']),  # J^s
        ((0.8, layers['input'] + 0.04), (0.58, layers['gater'] - 0.04), COLORS['rgb']),  # F^r的另一路

        # 虚线连接表示特征传递
        ((0.5, layers['attention1'] + 0.02), (0.42, layers['gater'] - 0.02), 'black', '--'),  # MHA输出到Gater

        # DA到Gater
        ((0.5, layers['attention2'] + 0.05), (0.5, layers['gater'] - 0.04), 'black'),

        # metadata到Gater
        ((0.7, layers['gater'] - 0.02), (0.58, layers['gater'] - 0.02), 'red', '--'),

        # Gater到输出
        ((0.5, layers['gater'] + 0.04), (0.5, layers['output'] - 0.04), 'black'),
    ]

    for arrow in arrows:
        if len(arrow) == 3:
            start, end, color = arrow
            style = '-'
        else:
            start, end, color, style = arrow

        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=1.5, linestyle=style, alpha=0.8))

    # 添加标签
    ax.text(0.4, 0.25, 'Q', fontsize=9, fontweight='bold', style='italic')
    ax.text(0.6, 0.25, 'K, V', fontsize=9, fontweight='bold', style='italic')
    ax.text(0.4, 0.45, 'Q', fontsize=9, fontweight='bold', style='italic')
    ax.text(0.6, 0.45, 'K, V', fontsize=9, fontweight='bold', style='italic')

    # 添加公式
    formula_text = (r'$\alpha_t = \sigma(MLP([F^p_t ; F^r_t ; u_t]))$' + '\n' +
                    r'$\tilde{F}_t = \alpha_t F^r_t + (1 - \alpha_t) F^p_t$')
    ax.text(0.5, 0.02, formula_text, fontsize=10, ha='center', va='bottom',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.7))


# 绘制图形
create_legend(ax_left)
draw_left_panel(ax_left)
draw_right_panel(ax_right)

# 添加整体标题
fig.suptitle(
    'Figure X. Overview of the proposed reliability-aware RGB-pose fusion framework\nfor low-resource CE-CSL translation. The left panel presents the overall pipeline\nbuilt upon Uni-Sign, while the right panel details the proposed uncertainty-aware fusion module.',
    fontsize=11, y=0.02, ha='center')

# 调整布局
plt.tight_layout(rect=[0, 0.05, 1, 0.95])

# 保存图形
plt.savefig('research_framework.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.savefig('research_framework.svg', format='svg', dpi=300, bbox_inches='tight')
plt.savefig('research_framework.png', format='png', dpi=300, bbox_inches='tight')

plt.show()

print("图形已保存为: research_framework.pdf, research_framework.svg, research_framework.png")