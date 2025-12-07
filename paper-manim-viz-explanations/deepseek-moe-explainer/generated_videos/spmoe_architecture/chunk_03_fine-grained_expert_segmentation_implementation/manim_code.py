from manim import *

class ConceptScene(Scene):
    def construct(self):
        # Step 1: Title and Introduction
        title = Text("Fine-Grained Expert Segmentation", font_size=36).to_edge(UP)
        intro_text = Text(
            "Segmenting large experts into smaller ones\nwhile keeping total computation fixed.",
            font_size=24
        ).next_to(title, DOWN, buff=0.5)

        self.play(Write(title))
        self.play(FadeIn(intro_text))
        self.wait(1)

        self.play(FadeOut(intro_text))

        # Step 2: Traditional vs Segmented Experts
        traditional_label = Text("Traditional MoE", color=BLUE, font_size=28).to_edge(LEFT).shift(UP*2)
        segmented_label = Text("Fine-Grained MoE", color=GREEN, font_size=28).to_edge(RIGHT).shift(UP*2)

        # Draw traditional expert
        expert_large = Rectangle(width=2, height=3, color=BLUE).next_to(traditional_label, DOWN, buff=0.5)
        expert_large_label = Text("1 Expert", font_size=20).next_to(expert_large, DOWN, buff=0.2)

        # Draw segmented experts
        small_experts_group = VGroup()
        for i in range(4):
            exp = Rectangle(width=0.8, height=1.2, color=GREEN).shift(LEFT*1.5 + RIGHT*(i*1.0)).next_to(segmented_label, DOWN, buff=0.8)
            small_experts_group.add(exp)
        segmented_label_group = Text("4 Smaller Experts", font_size=20).next_to(small_experts_group, DOWN, buff=0.2)

        self.play(Write(traditional_label), Write(segmented_label))
        self.play(Create(expert_large), Write(expert_large_label))
        self.play(Create(small_experts_group), Write(segmented_label_group))
        self.wait(1)

        # Step 3: Mathematical Explanation
        math_eq = MathTex(
            r"\text{Original Hidden Dim} \rightarrow \frac{1}{m} \times \text{Original}",
            font_size=30
        ).to_edge(DOWN).shift(UP*0.5)

        increase_text = Text(
            "Number of activated experts increased by m",
            font_size=24
        ).next_to(math_eq, DOWN, buff=0.3)

        self.play(Write(math_eq))
        self.play(FadeIn(increase_text))
        self.wait(2)

        # Cleanup before next section
        self.play(
            FadeOut(title),
            FadeOut(traditional_label),
            FadeOut(segmented_label),
            FadeOut(expert_large),
            FadeOut(expert_large_label),
            FadeOut(small_experts_group),
            FadeOut(segmented_label_group),
            FadeOut(math_eq),
            FadeOut(increase_text)
        )

        # Step 4: Code Snippet - DeepseekMoE Class
        code_title = Text("Implementation Example", font_size=32).to_edge(UP)
        self.play(Write(code_title))

        code_snippet_1 = Text(
            "class DeepseekMoE(nn.Module):\n"
            "    def __init__(self, config):\n"
            "        self.experts = nn.ModuleList([\n"
            "            DeepseekMLP(config,\n"
            "                intermediate_size=config.moe_intermediate_size)\n"
            "            for i in range(config.n_routed_experts)])\n"
            "        self.num_experts_per_tok = config.num_experts_per_tok",
            font="Monospace",
            font_size=18
        ).next_to(code_title, DOWN, buff=0.5)

        self.play(FadeIn(code_snippet_1))
        self.wait(2)

        # Step 5: Configuration Parameters
        param_text = Text(
            "Key Config Parameters:\n"
            "- moe_intermediate_size\n"
            "- num_experts_per_tok\n"
            "- n_routed_experts",
            font_size=24,
            line_spacing=1.2
        ).to_edge(RIGHT)

        self.play(FadeIn(param_text))
        self.wait(2)

        # Final cleanup
        self.play(
            FadeOut(code_title),
            FadeOut(code_snippet_1),
            FadeOut(param_text)
        )

        # Conclusion
        conclusion = Text(
            "Fine-grained segmentation allows:\n"
            "• More flexible expert combinations\n"
            "• Maintains computational efficiency",
            font_size=28,
            line_spacing=1.3
        ).move_to(ORIGIN)

        self.play(Write(conclusion))
        self.wait(2)