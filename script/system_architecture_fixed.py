"""
Manim 系统架构动画 - 修复版本
可以直接运行，无需修改
"""

from manim import *

class SystemArchitectureAnimation(Scene):
    def construct(self):
        # 设置背景和分辨率
        self.camera.background_color = WHITE

        # 颜色定义
        pose_color = "#1f77b4"      # 蓝色
        rgb_color = "#ff7f0e"       # 橙色
        fusion_color = "#2ca02c"    # 绿色
        decoder_color = "#d62728"   # 红色
        highlight_color = "#ffbb33" # 黄色

        # 创建标题
        title = Text("Reliability-Aware RGB-Pose Fusion", font_size=40, color=BLACK)
        subtitle = Text("for Low-Resource Chinese Sign Language Translation", font_size=24, color=DARK_GRAY)
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.3)

        self.play(Write(title))
        self.wait(0.5)
        self.play(Write(subtitle))
        self.wait(1)
        self.play(title_group.animate.shift(UP*3.5).scale(0.8))
        self.wait(0.5)

        # ============ 输入层 ============
        input_label = Text("INPUT LAYER", font_size=20, weight=BOLD, color=pose_color)
        input_label.shift(LEFT*5 + UP*2)
        self.play(Write(input_label))

        # 输入组件
        pose_box = Rectangle(width=1.8, height=0.8, color=pose_color, fill_opacity=0.2, stroke_width=2)
        pose_text = Text("Hand Pose\nSequence", font_size=12, color=BLACK)
        pose_group = VGroup(pose_box, pose_text).arrange(CENTER, buff=0)
        pose_group.shift(LEFT*5 + UP*0.8)

        rgb_box = Rectangle(width=1.8, height=0.8, color=rgb_color, fill_opacity=0.2, stroke_width=2)
        rgb_text = Text("RGB\nCrops", font_size=12, color=BLACK)
        rgb_group = VGroup(rgb_box, rgb_text).arrange(CENTER, buff=0)
        rgb_group.shift(LEFT*2.5 + UP*0.8)

        conf_box = Rectangle(width=1.8, height=0.8, color=highlight_color, fill_opacity=0.2, stroke_width=2)
        conf_text = Text("Keypoint\nConfidence", font_size=12, color=BLACK)
        conf_group = VGroup(conf_box, conf_text).arrange(CENTER, buff=0)
        conf_group.shift(RIGHT*0 + UP*0.8)

        self.play(Create(pose_group), Create(rgb_group), Create(conf_group))
        self.wait(1)

        # ============ 特征提取层 ============
        feature_label = Text("FEATURE EXTRACTION", font_size=20, weight=BOLD, color=rgb_color)
        feature_label.shift(RIGHT*3 + UP*2)
        self.play(Write(feature_label))

        # 特征提取组件
        pose_enc = Rectangle(width=2, height=0.8, color=pose_color, fill_opacity=0.3, stroke_width=2)
        pose_enc_text = Text("Pose Encoder\nSTGCN", font_size=11, color=BLACK)
        pose_enc_group = VGroup(pose_enc, pose_enc_text).arrange(CENTER, buff=0)
        pose_enc_group.shift(LEFT*5 + DOWN*0.8)

        frame_sel = Rectangle(width=2, height=0.8, color=highlight_color, fill_opacity=0.3, stroke_width=2)
        frame_sel_text = Text("Frame\nSelection", font_size=11, color=BLACK)
        frame_sel_group = VGroup(frame_sel, frame_sel_text).arrange(CENTER, buff=0)
        frame_sel_group.shift(LEFT*2.5 + DOWN*0.8)

        rgb_enc = Rectangle(width=2, height=0.8, color=rgb_color, fill_opacity=0.3, stroke_width=2)
        rgb_enc_text = Text("RGB Encoder\nViT", font_size=11, color=BLACK)
        rgb_enc_group = VGroup(rgb_enc, rgb_enc_text).arrange(CENTER, buff=0)
        rgb_enc_group.shift(RIGHT*0 + DOWN*0.8)

        self.play(Create(pose_enc_group), Create(frame_sel_group), Create(rgb_enc_group))

        # 连接输入到特征提取
        arrow1 = Arrow(pose_group.get_bottom(), pose_enc_group.get_top(), color=pose_color, buff=0.1)
        arrow2 = Arrow(conf_group.get_bottom(), frame_sel_group.get_top(), color=highlight_color, buff=0.1)
        arrow3 = Arrow(rgb_group.get_bottom(), rgb_enc_group.get_top(), color=rgb_color, buff=0.1)

        self.play(Create(arrow1), Create(arrow2), Create(arrow3))
        self.wait(1)

        # ============ 融合层 (核心创新) ============
        fusion_label = Text("MULTIMODAL FUSION LAYER", font_size=20, weight=BOLD, color=fusion_color)
        fusion_label.shift(RIGHT*3 + DOWN*2.5)
        self.play(Write(fusion_label))

        # 融合输入
        pose_feat = Rectangle(width=1.5, height=0.6, color=pose_color, fill_opacity=0.4, stroke_width=2)
        pose_feat_text = Text("p_t", font_size=12, color=BLACK, weight=BOLD)
        pose_feat_group = VGroup(pose_feat, pose_feat_text).arrange(CENTER, buff=0)
        pose_feat_group.shift(LEFT*5 + DOWN*2.8)

        rgb_feat = Rectangle(width=1.5, height=0.6, color=rgb_color, fill_opacity=0.4, stroke_width=2)
        rgb_feat_text = Text("r_t", font_size=12, color=BLACK, weight=BOLD)
        rgb_feat_group = VGroup(rgb_feat, rgb_feat_text).arrange(CENTER, buff=0)
        rgb_feat_group.shift(LEFT*2.5 + DOWN*2.8)

        metadata = Rectangle(width=1.5, height=0.6, color=highlight_color, fill_opacity=0.4, stroke_width=2)
        metadata_text = Text("u_t", font_size=12, color=BLACK, weight=BOLD)
        metadata_group = VGroup(metadata, metadata_text).arrange(CENTER, buff=0)
        metadata_group.shift(RIGHT*0 + DOWN*2.8)

        self.play(Create(pose_feat_group), Create(rgb_feat_group), Create(metadata_group))

        # 核心融合门 - 高亮显示
        fusion_gate_box = Rectangle(width=5, height=1.2, color=highlight_color, fill_opacity=0.6, stroke_width=3, stroke_color=BLACK)
        fusion_gate_text1 = Text("Uncertainty-Aware Fusion Gate", font_size=14, weight=BOLD, color=BLACK)
        fusion_gate_text2 = Text("Alpha = sigmoid(MLP(p_t, r_t, u_t))", font_size=11, color=BLACK)
        fusion_gate_text3 = Text("h_t = Alpha * r_t + (1 - Alpha) * p_t", font_size=11, color=BLACK)

        fusion_gate_group = VGroup(fusion_gate_box, fusion_gate_text1, fusion_gate_text2, fusion_gate_text3)
        fusion_gate_group.arrange(CENTER, buff=0.05)
        fusion_gate_group.shift(RIGHT*3 + DOWN*4.5)

        self.play(Create(fusion_gate_box))
        self.play(Write(fusion_gate_text1))
        self.play(Write(fusion_gate_text2), Write(fusion_gate_text3))

        # 连接到融合门
        arrow4 = Arrow(pose_feat_group.get_right(), fusion_gate_box.get_bottom() + LEFT*1.3, color=pose_color, buff=0.05)
        arrow5 = Arrow(rgb_feat_group.get_right(), fusion_gate_box.get_bottom(), color=rgb_color, buff=0.05)
        arrow6 = Arrow(metadata_group.get_right(), fusion_gate_box.get_bottom() + RIGHT*1.3, color=highlight_color, buff=0.05)

        self.play(Create(arrow4), Create(arrow5), Create(arrow6))

        # 高亮核心创新
        highlight_circle = Circle(radius=2.8, color=RED, stroke_width=2).move_to(RIGHT*3 + DOWN*4.5)
        self.play(Create(highlight_circle))

        core_text = Text("CORE INNOVATION!", font_size=16, weight=BOLD, color=RED)
        core_text.shift(RIGHT*3 + DOWN*6.2)
        self.play(Write(core_text))

        self.wait(2)

        # ============ 解码层 ============
        # 融合输出
        fused_output = Rectangle(width=2, height=0.7, color=fusion_color, fill_opacity=0.4, stroke_width=2)
        fused_output_text = Text("Fused h_t", font_size=11, color=BLACK)
        fused_output_group = VGroup(fused_output, fused_output_text).arrange(CENTER, buff=0)
        fused_output_group.shift(RIGHT*3 + DOWN*6)

        # mT5 解码器
        mt5_decoder = Rectangle(width=2, height=0.7, color=decoder_color, fill_opacity=0.4, stroke_width=2)
        mt5_text = Text("mT5 Decoder", font_size=11, color=BLACK)
        mt5_group = VGroup(mt5_decoder, mt5_text).arrange(CENTER, buff=0)
        mt5_group.shift(RIGHT*3 + DOWN*7)

        # 输出
        output_box = Rectangle(width=2, height=0.7, color=decoder_color, fill_opacity=0.5, stroke_width=2)
        output_text = Text("Chinese Text", font_size=11, color=BLACK)
        output_group = VGroup(output_box, output_text).arrange(CENTER, buff=0)
        output_group.shift(RIGHT*3 + DOWN*8)

        self.play(Create(fused_output_group), Create(mt5_group), Create(output_group))

        # 连接融合门到解码
        arrow7 = Arrow(fusion_gate_box.get_bottom(), fused_output_group.get_top(), color=fusion_color, buff=0.1)
        arrow8 = Arrow(fused_output_group.get_bottom(), mt5_group.get_top(), color=fusion_color, buff=0.1)
        arrow9 = Arrow(mt5_group.get_bottom(), output_group.get_top(), color=decoder_color, buff=0.1)

        self.play(Create(arrow7), Create(arrow8), Create(arrow9))
        self.wait(2)

        # 最终总结
        summary = Text("RGB Reliability-Aware Fusion Improves Translation Quality",
                      font_size=18, weight=BOLD, color=GREEN)
        summary.shift(UP*7)
        self.play(Write(summary))
        self.wait(2)

        # 淡出
        self.play(FadeOut(VGroup(*self.mobjects)))

