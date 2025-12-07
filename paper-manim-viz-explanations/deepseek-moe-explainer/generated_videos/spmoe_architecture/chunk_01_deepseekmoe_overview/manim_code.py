from manim import *

class ConceptScene(Scene):
    def construct(self):
        # Step 1: Title and Core Concept
        title = Text("DeepSeekMoE Overview", font_size=40).to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        desc = Text(
            "Mixture-of-Experts (MoE) model with specialized experts\n"
            "Two key strategies: Fine-grained segmentation & Shared isolation",
            font_size=24,
            line_spacing=1.2
        ).next_to(title, DOWN, buff=0.5)
        self.play(Write(desc))
        self.wait(2)

        # Cleanup
        self.play(FadeOut(title), FadeOut(desc))
        self.wait(0.5)

        # Step 2: Visualizing Expert Structure
        structure_title = Text("Expert Architecture", font_size=36).to_edge(UP)
        self.play(Write(structure_title))
        self.wait(0.5)

        # Routed Experts
        routed_label = Text("Routed Experts", color=BLUE, font_size=28)
        routed_experts = VGroup(
            *[Rectangle(height=1, width=1.5, color=BLUE).set_fill(BLUE, opacity=0.3) for _ in range(4)]
        ).arrange(RIGHT, buff=0.3).next_to(routed_label, DOWN, buff=0.5)
        routed_label.next_to(routed_experts, UP)

        # Shared Experts
        shared_label = Text("Shared Experts", color=GREEN, font_size=28)
        shared_expert = Rectangle(height=1.2, width=2, color=GREEN).set_fill(GREEN, opacity=0.3)
        shared_expert.next_to(routed_experts, DOWN, buff=1.5)
        shared_label.next_to(shared_expert, UP)

        # Input and Output Arrows
        input_arrow = Arrow(LEFT*3, routed_experts.get_left(), color=YELLOW)
        output_arrow = Arrow(routed_experts.get_right(), RIGHT*3, color=YELLOW)
        shared_arrow = Arrow(input_arrow.start, shared_expert.get_left(), color=ORANGE)
        combine_point = shared_expert.get_right() + RIGHT*1.5
        combine_arrow_1 = Arrow(output_arrow.end, combine_point, color=YELLOW)
        combine_arrow_2 = Arrow(shared_expert.get_right(), combine_point, color=ORANGE)
        final_output = Arrow(combine_point, combine_point + RIGHT*1.5, color=PURPLE)

        self.play(
            Write(routed_label),
            *[Create(expert) for expert in routed_experts]
        )
        self.wait(0.5)

        self.play(
            Write(shared_label),
            Create(shared_expert)
        )
        self.wait(0.5)

        self.play(GrowArrow(input_arrow))
        self.play(GrowArrow(output_arrow))
        self.play(GrowArrow(shared_arrow))
        self.play(GrowArrow(combine_arrow_1), GrowArrow(combine_arrow_2))
        self.play(GrowArrow(final_output))

        self.wait(2)

        # Cleanup
        all_objects = VGroup(
            structure_title,
            routed_label, routed_experts,
            shared_label, shared_expert,
            input_arrow, output_arrow, shared_arrow,
            combine_arrow_1, combine_arrow_2, final_output
        )
        self.play(FadeOut(all_objects))
        self.wait(0.5)

        # Step 3: Gating Mechanism
        gate_title = Text("Gating Mechanism", font_size=36).to_edge(UP)
        self.play(Write(gate_title))
        self.wait(0.5)

        gate_desc = Text(
            "Selects top-k experts per token\nSoftmax-based routing",
            font_size=24
        ).next_to(gate_title, DOWN)
        self.play(Write(gate_desc))
        self.wait(1)

        # Token representation
        tokens = VGroup(
            Circle(radius=0.3, color=WHITE).set_fill(WHITE, opacity=0.5),
            Circle(radius=0.3, color=WHITE).set_fill(WHITE, opacity=0.5),
            Text("...", font_size=24),
            Circle(radius=0.3, color=WHITE).set_fill(WHITE, opacity=0.5)
        ).arrange(RIGHT, buff=0.5).shift(UP*1)
        token_label = Text("Input Tokens", font_size=24).next_to(tokens, UP)

        # Gate box
        gate_box = Rectangle(width=3, height=1.5, color=BLUE).set_fill(BLUE, opacity=0.3)
        gate_text = Text("Gate", font_size=24).move_to(gate_box)
        gate_group = VGroup(gate_box, gate_text).next_to(tokens, DOWN, buff=1)

        # Expert representations
        experts_small = VGroup(
            *[Rectangle(height=0.8, width=1, color=BLUE).set_fill(BLUE, opacity=0.3) for _ in range(4)]
        ).arrange(RIGHT, buff=0.2).next_to(gate_group, DOWN, buff=1)

        arrows_to_gate = VGroup(
            *[Arrow(token.get_bottom(), gate_box.get_top() + LEFT*0.4 + RIGHT*(i*0.3), color=YELLOW) for i, token in enumerate(tokens[:3])]
        )

        arrows_to_experts = VGroup(
            *[Arrow(gate_box.get_bottom(), expert.get_top(), color=YELLOW) for expert in experts_small]
        )

        self.play(
            Write(token_label),
            *[Create(token) for token in tokens if isinstance(token, Circle)]
        )
        self.play(Write(tokens[2]))  # "..."
        self.wait(0.5)

        self.play(Create(gate_box), Write(gate_text))
        self.play(Create(arrows_to_gate))
        self.wait(0.5)

        self.play(
            *[Create(expert) for expert in experts_small]
        )
        self.play(Create(arrows_to_experts))
        self.wait(2)

        # Cleanup
        self.play(FadeOut(VGroup(
            gate_title, gate_desc, token_label, tokens,
            gate_group, experts_small, arrows_to_gate, arrows_to_experts
        )))
        self.wait(0.5)

        # Step 4: Code Snippet Highlight
        code_title = Text("Key Implementation", font_size=36).to_edge(UP)
        self.play(Write(code_title))
        self.wait(0.5)

        code_snippet = Text(
            "class DeepseekMoE(nn.Module):\n"
            "    def __init__(self, config):\n"
            "        self.experts = nn.ModuleList([...])\n"
            "        self.gate = MoEGate(config)\n"
            "        self.shared_experts = DeepseekMLP(...)\n\n"
            "    def forward(self, hidden_states):\n"
            "        topk_idx, topk_weight, aux_loss = self.gate(...)\n"
            "        # Combine routed and shared experts\n"
            "        if self.shared_experts:\n"
            "            y = y + self.shared_experts(identity)",
            font="Monospace",
            font_size=20,
            line_spacing=1.1
        ).next_to(code_title, DOWN, buff=0.5)

        self.play(Write(code_snippet))
        self.wait(3)

        # Final cleanup
        self.play(FadeOut(code_title), FadeOut(code_snippet))
        self.wait(0.5)

        # Ending
        thanks = Text("Thanks for watching!", font_size=36)
        self.play(Write(thanks))
        self.wait(2)