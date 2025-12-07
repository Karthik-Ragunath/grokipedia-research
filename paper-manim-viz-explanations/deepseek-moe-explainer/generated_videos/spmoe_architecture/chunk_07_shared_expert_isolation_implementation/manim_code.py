from manim import *

class ConceptScene(Scene):
    def construct(self):
        # Step 1: Title and Core Concept
        title = Text("Shared Expert Isolation", font_size=40).to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        desc1 = Text(
            "In MoE models, some experts are designated\n"
            "as 'shared' — used by ALL tokens.",
            font_size=28
        ).next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(desc1))
        self.wait(2)

        self.play(FadeOut(desc1))

        # Step 2: Visualizing Shared vs Routed Experts
        box_shared = Rectangle(height=1.5, width=4, color=YELLOW).shift(LEFT*3)
        label_shared = Text("Shared Experts", font_size=24).next_to(box_shared, UP)
        shared_group = VGroup(box_shared, label_shared)

        box_routed = Rectangle(height=1.5, width=4, color=BLUE).shift(RIGHT*3)
        label_routed = Text("Routed Experts", font_size=24).next_to(box_routed, UP)
        routed_group = VGroup(box_routed, label_routed)

        self.play(Create(shared_group), Create(routed_group))
        self.wait(1)

        # Tokens flowing to both
        tokens = VGroup(*[
            Circle(radius=0.2, color=GREEN).move_to(DOWN*2 + LEFT*3 + RIGHT*i*1.5)
            for i in range(4)
        ])
        token_labels = VGroup(*[
            Text(f"T{i+1}", font_size=18).move_to(tokens[i].get_center())
            for i in range(4)
        ])
        all_tokens = VGroup(tokens, token_labels)
        self.play(FadeIn(all_tokens))
        self.wait(1)

        # Arrows to shared experts (all tokens go there)
        arrows_to_shared = VGroup(*[
            Arrow(start=token.get_top(), end=box_shared.get_bottom()+RIGHT*(i-1.5), color=YELLOW)
            for i, token in enumerate(tokens)
        ])
        self.play(*[GrowArrow(arrow) for arrow in arrows_to_shared])
        self.wait(1)

        # Arrows to routed experts (selective)
        arrows_to_routed = VGroup(
            Arrow(start=tokens[0].get_top(), end=box_routed.get_bottom()+LEFT*0.5, color=BLUE),
            Arrow(start=tokens[2].get_top(), end=box_routed.get_bottom()+RIGHT*0.5, color=BLUE)
        )
        self.play(*[GrowArrow(arrow) for arrow in arrows_to_routed])
        self.wait(2)

        # Clean up for next step
        self.play(FadeOut(arrows_to_shared, arrows_to_routed, all_tokens, shared_group, routed_group))

        # Step 3: Mathematical Notation
        math_notation = MathTex(
            r"K_s", r"\text{ shared experts} \quad \Rightarrow \quad",
            r"\text{Tokens assigned } \textbf{deterministically}",
            font_size=36
        ).shift(UP*1)
        self.play(Write(math_notation))
        self.wait(2)

        cost_note = MathTex(
            r"\text{Total compute} = \text{Shared} + \text{Routed} - K_s",
            font_size=36
        ).next_to(math_notation, DOWN, buff=0.6)
        self.play(Write(cost_note))
        self.wait(2)

        self.play(FadeOut(math_notation, cost_note))

        # Step 4: Code Snippet
        code_text = Text(
            "if self.config.n_shared_experts is not None:\n"
            "    y = y + self.shared_experts(identity)",
            font="Monospace",
            font_size=24,
            color=GRAY_A
        ).shift(DOWN*0.5)
        
        code_box = SurroundingRectangle(code_text, color=GRAY, buff=0.2)
        code_group = VGroup(code_text, code_box)

        code_label = Text("Code: Shared Expert Integration", font_size=28).next_to(code_group, UP)
        self.play(FadeIn(code_label), Create(code_box), Write(code_text))
        self.wait(3)

        self.play(FadeOut(code_label, code_group))

        # Final Summary
        summary = Text(
            "Shared experts enhance model capacity\n"
            "without increasing per-token compute.",
            font_size=30
        )
        self.play(Write(summary))
        self.wait(2)

        self.play(FadeOut(summary, title))
        self.wait(1)