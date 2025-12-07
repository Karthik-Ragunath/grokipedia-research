from manim import *

class ConceptScene(Scene):
    def construct(self):
        # Step 1: Title and Introduction
        title = Text("Fine-Grained Expert Segmentation in MoE", font_size=36).to_edge(UP)
        self.play(Write(title))
        self.wait(1)

        desc = Text(
            "Mixture of Experts (MoE) layers route tokens to\n"
            "specialized experts for efficient computation.",
            font_size=24
        ).next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(desc))
        self.wait(2)

        self.play(FadeOut(desc))

        # Step 2: Mathematical Formula Overview
        formula_title = Text("Core MoE Output Formula:", font_size=30).next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(formula_title))

        eq1 = MathTex(
            r"\mathbf{h}_{t}^{l} = \sum_{i=1}^{mN} \left( g_{i,t} \operatorname{FFN}_{i}\left( \mathbf{u}_{t}^{l} \right) \right) + \mathbf{u}_{t}^{l}",
            font_size=30
        ).next_to(formula_title, DOWN, buff=0.5)
        self.play(Write(eq1))
        self.wait(2)

        eq2 = MathTex(
            r"g_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} \in \text{Topk}(...), \\ 0, & \text{otherwise} \end{cases}",
            font_size=30
        ).next_to(eq1, DOWN, buff=0.3)
        self.play(Write(eq2))
        self.wait(2)

        eq3 = MathTex(
            r"s_{i,t} = \operatorname{Softmax}_i \left( {\mathbf{u}_{t}^{l}}^{T} \mathbf{e}_{i}^{l} \right)",
            font_size=30
        ).next_to(eq2, DOWN, buff=0.3)
        self.play(Write(eq3))
        self.wait(2)

        self.play(FadeOut(VGroup(formula_title, eq1, eq2, eq3)))

        # Step 3: Visualizing Token Routing
        routing_title = Text("Token Routing to Experts", font_size=30).next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(routing_title))

        # Input token
        token = Circle(radius=0.5, color=BLUE).shift(LEFT*3)
        token_label = Text("Token", font_size=20).move_to(token.get_center())
        input_group = VGroup(token, token_label)

        # Experts
        experts = VGroup(*[
            Rectangle(height=1, width=1.5, color=GREEN).set_fill(GREEN, opacity=0.3)
            for _ in range(4)
        ]).arrange(RIGHT, buff=0.5).shift(RIGHT*0.5)
        expert_labels = VGroup(*[
            Text(f"E{i+1}", font_size=20).move_to(experts[i].get_center())
            for i in range(4)
        ])

        # Arrows from token to experts
        arrows = VGroup(*[
            Arrow(start=token.get_right(), end=experts[i].get_left(), color=YELLOW)
            for i in range(4)
        ])

        self.play(FadeIn(input_group), FadeIn(experts), FadeIn(expert_labels))
        self.play(Create(arrows))
        self.wait(2)

        # Highlight selected experts (Top-K)
        selected_experts = VGroup(experts[1], experts[3])
        self.play(selected_experts.animate.set_color(RED))
        self.wait(1)

        weights = VGroup(*[
            MathTex(f"w_{i+1}", color=RED, font_size=24).next_to(arrows[i], UP, buff=0.1)
            for i in [1, 3]
        ])
        self.play(FadeIn(weights))
        self.wait(2)

        self.play(FadeOut(VGroup(input_group, experts, expert_labels, arrows, weights, routing_title)))

        # Step 4: Code Snippet - Gating Mechanism
        code_title = Text("Gating Mechanism in Code", font_size=30).next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(code_title))

        code_snippet = Text(
            "class MoEGate(nn.Module):\n"
            "    def forward(self, hidden_states):\n"
            "        logits = F.linear(...)\n"
            "        scores = logits.softmax(dim=-1)\n"
            "        topk_weight, topk_idx = torch.topk(scores, k=self.top_k)",
            font="Monospace",
            font_size=20,
            line_spacing=0.7
        ).next_to(code_title, DOWN, buff=0.5)

        self.play(FadeIn(code_snippet))
        self.wait(3)

        self.play(FadeOut(VGroup(code_title, code_snippet)))

        # Step 5: Final Summary
        summary = Text(
            "Fine-grained segmentation increases expert usage\n"
            "and improves model performance through better routing.",
            font_size=24
        ).next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(summary))
        self.wait(3)

        self.play(FadeOut(VGroup(title, summary)))
        self.wait(1)