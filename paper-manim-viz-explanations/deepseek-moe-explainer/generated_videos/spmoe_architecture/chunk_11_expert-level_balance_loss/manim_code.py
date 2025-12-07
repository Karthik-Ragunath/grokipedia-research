from manim import *

class ConceptScene(Scene):
    def construct(self):
        # Step 1: Title and Introduction
        title = Text("Expert-Level Balance Loss", font_size=40).to_edge(UP)
        intro = Text(
            "Prevents routing collapse in MoE",
            font_size=24
        ).next_to(title, DOWN, buff=0.5)
        
        self.play(Write(title))
        self.play(FadeIn(intro))
        self.wait(1)
        
        self.play(FadeOut(intro))
        
        # Step 2: Mathematical Formula (simplified to avoid LaTeX issues)
        formula_title = Text("Mathematical Formulation", font_size=32).next_to(title, DOWN, buff=0.5)
        self.play(Write(formula_title))
        
        # Use simpler MathTex without \mathds
        loss_eq = MathTex(
            r"\mathcal{L}_{\text{ExpBal}}", "=", r"\alpha_1", r"\sum_{i=1}^{N'}", "f_i \cdot P_i"
        ).scale(1.0).next_to(formula_title, DOWN, buff=0.5)
        
        fi_def = MathTex(
            r"f_i", ":", r"\text{fraction of tokens selecting expert } i"
        ).scale(0.8).next_to(loss_eq, DOWN, buff=0.4)
        
        pi_def = MathTex(
            r"P_i", ":", r"\text{average gating score for expert } i"
        ).scale(0.8).next_to(fi_def, DOWN, buff=0.3)
        
        # Color components
        loss_eq[0].set_color(BLUE)
        loss_eq[2].set_color(GREEN)
        loss_eq[4].set_color(YELLOW)
        fi_def[0].set_color(YELLOW)
        pi_def[0].set_color(ORANGE)
        
        self.play(Write(loss_eq))
        self.wait(0.5)
        self.play(Write(fi_def))
        self.play(Write(pi_def))
        self.wait(2)
        
        self.play(FadeOut(formula_title), FadeOut(loss_eq), FadeOut(fi_def), FadeOut(pi_def))
        
        # Step 3: Visual Representation
        visual_title = Text("Visualizing Expert Selection", font_size=32).next_to(title, DOWN, buff=0.5)
        self.play(Write(visual_title))
        
        # Draw experts
        experts = VGroup(*[
            Rectangle(height=1, width=1.2, color=BLUE).set_fill(BLUE, opacity=0.3)
            for _ in range(4)
        ]).arrange(RIGHT, buff=0.4).shift(DOWN*0.5)
        
        expert_labels = VGroup(*[
            Text(f"E{i+1}", font_size=18).next_to(experts[i], DOWN, buff=0.15)
            for i in range(4)
        ])
        
        # Draw tokens
        tokens = VGroup(*[
            Circle(radius=0.2, color=YELLOW).set_fill(YELLOW, opacity=0.5)
            for _ in range(6)
        ]).arrange(RIGHT, buff=0.25).next_to(experts, UP, buff=1.2)
        
        token_labels = VGroup(*[
            Text(f"T{i+1}", font_size=14).next_to(tokens[i], UP, buff=0.1)
            for i in range(6)
        ])
        
        self.play(FadeIn(experts), FadeIn(expert_labels))
        self.play(FadeIn(tokens), FadeIn(token_labels))
        self.wait(0.5)
        
        # Arrows showing imbalanced routing (most go to Expert 2)
        connections = [(0, 1), (1, 1), (2, 0), (3, 1), (4, 2), (5, 1)]
        arrows = VGroup()
        for token_idx, expert_idx in connections:
            arrow = Arrow(
                start=tokens[token_idx].get_bottom(),
                end=experts[expert_idx].get_top(),
                color=GREEN,
                buff=0.1,
                stroke_width=2
            )
            arrows.add(arrow)
        
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.15))
        self.wait(1)
        
        # Highlight imbalance
        imbalance = Text("Imbalanced! Expert 2 overloaded", color=RED, font_size=22).to_edge(DOWN)
        self.play(experts[1].animate.set_color(RED), Write(imbalance))
        self.wait(1.5)
        
        self.play(FadeOut(visual_title), FadeOut(experts), FadeOut(expert_labels),
                  FadeOut(tokens), FadeOut(token_labels), FadeOut(arrows), FadeOut(imbalance))
        
        # Step 4: Code Snippet
        code_title = Text("PyTorch Implementation", font_size=32).next_to(title, DOWN, buff=0.5)
        self.play(Write(code_title))
        
        code_lines = [
            "# Compute expert balance loss",
            "ce = mask.float().mean(0)  # selection freq",
            "Pi = scores.mean(0)        # avg gating score", 
            "fi = ce * n_routed_experts",
            "aux_loss = (Pi * fi).sum() * alpha"
        ]
        
        code_group = VGroup(*[
            Text(line, font="Monospace", font_size=18)
            for line in code_lines
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(code_title, DOWN, buff=0.5)
        
        for i, line in enumerate(code_group):
            self.play(FadeIn(line), run_time=0.4)
        self.wait(2)
        
        self.play(FadeOut(code_title), FadeOut(code_group))
        
        # Summary
        summary_lines = [
            "Expert-Level Balance Loss:",
            "• Prevents routing collapse",
            "• Balances expert utilization",
            "• Improves MoE efficiency"
        ]
        summary = VGroup(*[
            Text(line, font_size=24) for line in summary_lines
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(title, DOWN, buff=0.8)
        
        self.play(Write(summary))
        self.wait(2)
        
        self.play(FadeOut(summary), FadeOut(title))
        self.wait(0.5)
