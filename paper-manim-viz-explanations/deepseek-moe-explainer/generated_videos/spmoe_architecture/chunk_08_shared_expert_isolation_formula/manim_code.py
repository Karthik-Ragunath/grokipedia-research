from manim import *

class ConceptScene(Scene):
    def construct(self):
        # Step 1: Title and Introduction
        title = Text("Shared Expert Isolation Formula", font_size=36).to_edge(UP)
        self.play(Write(title))
        self.wait(1)

        desc = Text(
            "In DeepSeekMoE, experts are split into shared and routed.",
            font_size=24
        ).next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(desc))
        self.wait(2)

        self.play(FadeOut(desc))

        # Step 2: Mathematical Formula
        formula = MathTex(
            r"\mathbf{h}_{t}^{l} &= \sum_{i=1}^{K_{s}} {\operatorname{FFN}_{i}\left( \mathbf{u}_{t}^{l} \right)}",
            r"+ \sum_{i=K_{s} + 1}^{mN} \left( {g_{i,t} \operatorname{FFN}_{i}\left( \mathbf{u}_{t}^{l} \right)} \right)",
            r"+ \mathbf{u}_{t}^{l}"
        ).scale(0.8).next_to(title, DOWN, buff=1)

        self.play(Write(formula[0]), Write(formula[2]))
        self.wait(1)
        self.play(Write(formula[1]))
        self.wait(2)

        # Highlight parts
        shared_brace = Brace(formula[0], DOWN, color=BLUE)
        shared_label = Text("Shared Experts", color=BLUE, font_size=20).next_to(shared_brace, DOWN, buff=0.1)
        routed_brace = Brace(formula[1], DOWN, color=GREEN)
        routed_label = Text("Routed Experts", color=GREEN, font_size=20).next_to(routed_brace, DOWN, buff=0.1)

        self.play(
            GrowFromCenter(shared_brace),
            FadeIn(shared_label)
        )
        self.wait(1)
        self.play(
            GrowFromCenter(routed_brace),
            FadeIn(routed_label)
        )
        self.wait(2)

        # Clean up
        self.play(
            FadeOut(formula),
            FadeOut(shared_brace),
            FadeOut(shared_label),
            FadeOut(routed_brace),
            FadeOut(routed_label)
        )

        # Step 3: Gating Mechanism Explanation
        gate_title = Text("Gating Mechanism", font_size=30).next_to(title, DOWN)
        self.play(Write(gate_title))
        self.wait(1)

        gate_formula = MathTex(
            r"s_{i,t} &= \operatorname{Softmax}_i \left( {\mathbf{u}_{t}^{l}}^{T} \mathbf{e}_{i}^{l} \right)",
            r"g_{i,t} &= \begin{cases} s_{i,t}, & s_{i,t} \in \operatorname{Topk}(...) \\ 0, & \text{otherwise} \end{cases}"
        ).scale(0.7).next_to(gate_title, DOWN, buff=0.5)

        self.play(Write(gate_formula[0]))
        self.wait(1)
        self.play(Write(gate_formula[1]))
        self.wait(2)

        # Clean up
        self.play(FadeOut(gate_title), FadeOut(gate_formula))

        # Step 4: Code Snippet Visualization
        code_title = Text("Implementation in PyTorch", font_size=30).next_to(title, DOWN)
        self.play(Write(code_title))
        self.wait(1)

        code_snippet = Text(
            "class DeepseekMoE(nn.Module):\n"
            "    def forward(self, hidden_states):\n"
            "        # Shared experts always active\n"
            "        y = y + self.shared_experts(identity)\n"
            "        # Routed experts with gating\n",
            font="Monospace",
            font_size=20
        ).next_to(code_title, DOWN, buff=0.5)

        self.play(FadeIn(code_snippet))
        self.wait(3)

        # Clean up
        self.play(
            FadeOut(code_title),
            FadeOut(code_snippet)
        )

        # Final Summary
        summary = Text(
            "Shared experts are always active.\nRouted experts selected via gating.\nThis improves efficiency and performance.",
            font_size=24,
            line_spacing=1.2
        ).next_to(title, DOWN, buff=1)

        self.play(FadeIn(summary))
        self.wait(3)

        self.play(
            FadeOut(title),
            FadeOut(summary)
        )
        self.wait(1)