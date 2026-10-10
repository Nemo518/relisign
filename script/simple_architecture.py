"""
Manim 系统架构动画 - 超简易版本
最快的可运行版本，适合快速测试
"""

from manim import *

class SimpleSystemArchitecture(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        # ===== 标题 =====
        title = Text("Reliability-Aware RGB-Pose Fusion", font_size=36, color=BLACK, weight=BOLD)
        self.add(title)
        self.wait(1)

        # ===== 第一层：输入 =====
        input_title = Text("INPUT", font_size=24, color=BLUE)
        input_title.shift(LEFT*4 + UP*2)
        self.add(input_title)

        pose = Rectangle(width=1.5, height=0.6, color=BLUE, fill_opacity=0.3, stroke_width=2)
        pose.shift(LEFT*4 + UP*0.8)
        pose_text = Text("Pose", font_size=12, color=BLACK)
        pose_text.shift(LEFT*4 + UP*0.8)
        self.add(pose, pose_text)

        rgb = Rectangle(width=1.5, height=0.6, color=RED, fill_opacity=0.3, stroke_width=2)
        rgb.shift(LEFT*1.5 + UP*0.8)
        rgb_text = Text("RGB", font_size=12, color=BLACK)
        rgb_text.shift(LEFT*1.5 + UP*0.8)
        self.add(rgb, rgb_text)

        conf = Rectangle(width=1.5, height=0.6, color=GOLD, fill_opacity=0.3, stroke_width=2)
        conf.shift(RIGHT*1 + UP*0.8)
        conf_text = Text("Confidence", font_size=12, color=BLACK)
        conf_text.shift(RIGHT*1 + UP*0.8)
        self.add(conf, conf_text)

        # ===== 第二层：特征提取 =====
        feature_title = Text("FEATURE EXTRACTION", font_size=24, color=PURPLE)
        feature_title.shift(RIGHT*3 + UP*2)
        self.add(feature_title)

        stgcn = Rectangle(width=1.5, height=0.6, color=BLUE, fill_opacity=0.4, stroke_width=2)
        stgcn.shift(LEFT*4 + DOWN*0.3)
        stgcn_text = Text("STGCN", font_size=11, color=BLACK)
        stgcn_text.shift(LEFT*4 + DOWN*0.3)
        self.add(stgcn, stgcn_text)

        frame_sel = Rectangle(width=1.5, height=0.6, color=GOLD, fill_opacity=0.4, stroke_width=2)
        frame_sel.shift(LEFT*1.5 + DOWN*0.3)
        frame_sel_text = Text("TopK Select", font_size=11, color=BLACK)
        frame_sel_text.shift(LEFT*1.5 + DOWN*0.3)
        self.add(frame_sel, frame_sel_text)

        vit = Rectangle(width=1.5, height=0.6, color=RED, fill_opacity=0.4, stroke_width=2)
        vit.shift(RIGHT*1 + DOWN*0.3)
        vit_text = Text("ViT", font_size=11, color=BLACK)
        vit_text.shift(RIGHT*1 + DOWN*0.3)
        self.add(vit, vit_text)

        # ===== 第三层：融合（核心） =====
        fusion_box = Rectangle(width=4, height=1.2, color=YELLOW, fill_opacity=0.6, stroke_width=3, stroke_color=BLACK)
        fusion_box.shift(RIGHT*3 + DOWN*1.5)

        fusion_text1 = Text("CORE: Fusion Gate", font_size=16, weight=BOLD, color=BLACK)
        fusion_text1.shift(RIGHT*3 + DOWN*1.2)

        fusion_text2 = Text("Alpha = sigmoid(MLP(p,r,u))", font_size=12, color=BLACK)
        fusion_text2.shift(RIGHT*3 + DOWN*1.8)

        self.add(fusion_box)
        self.add(fusion_text1)
        self.add(fusion_text2)

        highlight = Circle(radius=2.3, color=RED, stroke_width=2).shift(RIGHT*3 + DOWN*1.5)
        self.add(highlight)

        # ===== 第四层：解码 =====
        decoder_title = Text("OUTPUT", font_size=24, color=GREEN)
        decoder_title.shift(RIGHT*3 + DOWN*3.2)
        self.add(decoder_title)

        fused = Rectangle(width=1.5, height=0.6, color=GREEN, fill_opacity=0.4, stroke_width=2)
        fused.shift(RIGHT*3 + DOWN*3.8)
        fused_text = Text("h_t", font_size=12, color=BLACK, weight=BOLD)
        fused_text.shift(RIGHT*3 + DOWN*3.8)
        self.add(fused, fused_text)

        mt5 = Rectangle(width=1.5, height=0.6, color=GREEN, fill_opacity=0.5, stroke_width=2)
        mt5.shift(RIGHT*3 + DOWN*4.6)
        mt5_text = Text("mT5", font_size=12, color=BLACK)
        mt5_text.shift(RIGHT*3 + DOWN*4.6)
        self.add(mt5, mt5_text)

        output = Rectangle(width=1.5, height=0.6, color=GREEN, fill_opacity=0.5, stroke_width=2)
        output.shift(RIGHT*3 + DOWN*5.4)
        output_text = Text("Chinese", font_size=12, color=BLACK)
        output_text.shift(RIGHT*3 + DOWN*5.4)
        self.add(output, output_text)

        # ===== 最后总结 =====
        summary = Text("RGB Reliability-Aware Fusion", font_size=18, weight=BOLD, color=GREEN)
        summary.shift(UP*3.5)
        self.add(summary)

        self.wait(2)

