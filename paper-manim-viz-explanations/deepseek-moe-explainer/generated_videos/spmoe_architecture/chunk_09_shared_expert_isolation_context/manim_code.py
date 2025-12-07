from manim import *

class ConceptScene(Scene):
    def construct(self):
        # Step 1: Title and Introduction
        title = Text("Shared Expert Isolation Context", font_size=36).to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        desc1 = Text(
            "In MoE models, shared experts are processed\n"
            "separately from routed experts.",
            font_size=24
        ).next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(desc1))
        self.wait(1)

        # Step 2: Visualizing Routed vs Shared Experts
        self.play(FadeOut(desc1))

        # Create routed experts group
        routed_label = Text("Routed Experts", color=BLUE, font_size=24)
        routed_experts = VGroup(
            *[Rectangle(height=1, width=0.5, color=BLUE) for _ in range(4)]
        ).arrange(RIGHT, buff=0.3).next_to(routed_label, DOWN)
        routed_group = VGroup(routed_label, routed_experts)

        # Create shared experts group
        shared_label = Text("Shared Experts", color=GREEN, font_size=24)
        shared_experts = Rectangle(height=1, width=2, color=GREEN).next_to(shared_label, DOWN)
        shared_group = VGroup(shared_label, shared_experts)

        # Position groups
        routed_group.to_edge(LEFT, buff=1.5)
        shared_group.to_edge(RIGHT, buff=1.5)

        self.play(FadeIn(routed_group), FadeIn(shared_group))
        self.wait(1)

        # Step 3: Arrows showing data flow
        input_text = Text("Input", font_size=20).to_edge(DOWN).shift(UP*1.5)
        input_arrow = Arrow(input_text.get_top(), routed_experts.get_bottom(), color=YELLOW)
        shared_arrow = Arrow(input_text.get_top(), shared_experts.get_bottom(), color=YELLOW)

        self.play(Write(input_text), GrowArrow(input_arrow), GrowArrow(shared_arrow))
        self.wait(1)

        # Output arrow combining results
        output_text = Text("Output = Routed + Shared", font_size=20).to_edge(DOWN)
        output_arrow_routed = Arrow(routed_experts.get_bottom(), output_text.get_left()+LEFT*0.5, color=YELLOW)
        output_arrow_shared = Arrow(shared_experts.get_bottom(), output_text.get_right()+RIGHT*0.5, color=YELLOW)

        self.play(
            GrowArrow(output_arrow_routed),
            GrowArrow(output_arrow_shared),
            Write(output_text)
        )
        self.wait(1)

        # Step 4: Show code snippet for shared expert handling
        self.play(
            FadeOut(routed_group),
            FadeOut(shared_group),
            FadeOut(input_text),
            FadeOut(input_arrow),
            FadeOut(shared_arrow),
            FadeOut(output_text),
            FadeOut(output_arrow_routed),
            FadeOut(output_arrow_shared)
        )

        code_title = Text("Code: Shared Expert Handling", font_size=28).to_edge(UP)
        self.play(Transform(title, code_title))

        code_snippet = Text(
            "if config.n_shared_experts is not None:\n"
            "    intermediate_size = config.moe_intermediate_size * config.n_shared_experts\n"
            "    self.shared_experts = DeepseekMLP(...)\n\n"
            "# In forward pass:\n"
            "if self.config.n_shared_experts is not None:\n"
            "    y = y + self.shared_experts(identity)",
            font="Monospace",
            font_size=20
        ).next_to(code_title, DOWN, buff=0.5)

        self.play(FadeIn(code_snippet))
        self.wait(2)

        # Final cleanup
        self.play(
            FadeOut(title),
            FadeOut(code_snippet)
        )
        self.wait(0.5)