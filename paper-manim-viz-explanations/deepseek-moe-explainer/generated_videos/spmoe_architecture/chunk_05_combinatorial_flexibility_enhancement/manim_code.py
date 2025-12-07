from manim import *

class ConceptScene(Scene):
    def construct(self):
        # Step 1: Title and Introduction
        title = Text("Combinatorial Flexibility Enhancement", font_size=36).to_edge(UP)
        self.play(Write(title))
        self.wait(1)

        desc = Text(
            "Fine-grained expert segmentation increases\n"
            "the number of possible expert combinations.",
            font_size=24
        ).next_to(title, DOWN, buff=0.5)
        self.play(Write(desc))
        self.wait(2)

        self.play(FadeOut(desc))

        # Step 2: Mathematical Comparison
        math_group = VGroup()
        eq1 = MathTex(r"N = 16,\ \text{top-2 routing} \Rightarrow \binom{16}{2} = 120", font_size=36)
        eq2 = MathTex(r"\text{Split each expert into 4} \Rightarrow 16 \times 4 = 64 \text{ experts}", font_size=36)
        eq3 = MathTex(r"\text{Select 8 experts} \Rightarrow \binom{64}{8} = 4,426,165,368", font_size=36)

        eq1.next_to(title, DOWN, buff=1)
        eq2.next_to(eq1, DOWN, buff=0.5)
        eq3.next_to(eq2, DOWN, buff=0.5)

        math_group.add(eq1, eq2, eq3)
        self.play(Write(math_group))
        self.wait(3)

        self.play(FadeOut(math_group))

        # Step 3: Visual Metaphor - Circles Representing Experts
        expert_group = VGroup()
        big_experts = VGroup(*[Circle(radius=0.3, color=BLUE) for _ in range(16)])
        big_experts.arrange_in_grid(rows=4, cols=4, buff=0.5).shift(LEFT*3)
        big_labels = VGroup(*[
            Text(f"E{i+1}", font_size=18).move_to(big_experts[i].get_center())
            for i in range(16)
        ])

        small_expert_grid = VGroup()
        for i in range(16):
            sub_experts = VGroup(*[Circle(radius=0.15, color=GREEN) for _ in range(4)])
            sub_experts.arrange(buff=0.1).move_to(big_experts[i].get_center())
            small_expert_grid.add(sub_experts)

        small_expert_grid.shift(RIGHT*3)
        self.play(FadeIn(big_experts), FadeIn(big_labels))
        self.wait(1)
        self.play(Transform(big_experts.copy(), small_expert_grid), run_time=2)
        self.wait(2)

        # Highlight combinations
        combo_text = Text("More Combinations!", color=YELLOW, font_size=30).to_edge(DOWN)
        self.play(Write(combo_text))
        self.wait(2)

        self.play(FadeOut(big_experts), FadeOut(big_labels), FadeOut(small_expert_grid), FadeOut(combo_text))

        # Step 4: Code Snippet - Core Mechanism
        code_title = Text("Core Implementation: DeepseekMoE", font_size=28).next_to(title, DOWN)
        self.play(Write(code_title))

        code_snippet = Text(
            "class DeepseekMoE(nn.Module):\n"
            "    def __init__(self, config):\n"
            "        self.num_experts_per_tok = config.num_experts_per_tok\n"
            "        self.experts = nn.ModuleList([...])\n"
            "        self.gate = MoEGate(config)\n\n"
            "    def forward(self, hidden_states):\n"
            "        topk_idx, topk_weight, aux_loss = self.gate(...)\n"
            "        # Routing logic here",
            font="Monospace",
            font_size=20
        ).next_to(code_title, DOWN, buff=0.5)

        self.play(Write(code_snippet))
        self.wait(3)

        self.play(FadeOut(code_title), FadeOut(code_snippet))

        # Step 5: Final Summary
        summary = Text(
            "Fine-grained segmentation & top-K routing\n"
            "→ Exponential increase in expert combinations\n"
            "→ Enhanced model flexibility & performance",
            font_size=24,
            line_spacing=1.2
        ).move_to(ORIGIN)
        self.play(Write(summary))
        self.wait(3)

        self.play(FadeOut(title), FadeOut(summary))
        self.wait(1)