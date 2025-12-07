from manim import *

class ConceptScene(Scene):
    def construct(self):
        # Step 1: Introduce the problem with limited experts
        title = Text("Fine-Grained Expert Segmentation", font_size=36).to_edge(UP)
        prob_text = Text(
            "When few experts exist,\n"
            "each must handle diverse knowledge types.",
            font_size=24
        ).next_to(title, DOWN, buff=0.5)

        self.play(Write(title))
        self.play(Write(prob_text))
        self.wait(2)

        # Visualize limited experts handling diverse tokens
        tokens = VGroup(*[Circle(radius=0.3, color=BLUE) for _ in range(5)]).arrange(RIGHT, buff=0.5).shift(LEFT*2)
        token_labels = VGroup(*[Text(f"T{i+1}", font_size=18).move_to(tokens[i]) for i in range(5)])
        experts = VGroup(*[Rectangle(width=1.5, height=2, color=RED) for _ in range(2)]).arrange(RIGHT, buff=2).shift(RIGHT*2)
        expert_labels = VGroup(*[Text(f"E{i+1}", font_size=20).move_to(experts[i]) for i in range(2)])

        arrows_to_e1 = VGroup(*[
            Arrow(start=t.get_right(), end=experts[0].get_left(), color=YELLOW)
            for t in tokens[:3]
        ])
        arrows_to_e2 = VGroup(*[
            Arrow(start=t.get_right(), end=experts[1].get_left(), color=GREEN)
            for t in tokens[3:]
        ])

        self.play(FadeIn(tokens), FadeIn(token_labels))
        self.play(FadeIn(experts), FadeIn(expert_labels))
        self.play(Create(arrows_to_e1), Create(arrows_to_e2))
        self.wait(2)

        # Clean up
        self.play(
            FadeOut(prob_text),
            FadeOut(tokens),
            FadeOut(token_labels),
            FadeOut(experts),
            FadeOut(expert_labels),
            FadeOut(arrows_to_e1),
            FadeOut(arrows_to_e2)
        )

        # Step 2: Introduce solution - route tokens to multiple experts
        sol_text = Text(
            "Routing each token to multiple experts\n"
            "enables better knowledge decomposition.",
            font_size=24
        ).next_to(title, DOWN, buff=0.5)

        self.play(Write(sol_text))
        self.wait(1)

        # Visualize routing to multiple experts
        tokens2 = VGroup(*[Circle(radius=0.3, color=BLUE) for _ in range(3)]).arrange(RIGHT, buff=0.7).shift(LEFT*2)
        token_labels2 = VGroup(*[Text(f"T{i+1}", font_size=18).move_to(tokens2[i]) for i in range(3)])
        experts2 = VGroup(*[Rectangle(width=1.5, height=2, color=RED) for _ in range(3)]).arrange(RIGHT, buff=1.5).shift(RIGHT*2)
        expert_labels2 = VGroup(*[Text(f"E{i+1}", font_size=20).move_to(experts2[i]) for i in range(3)])

        # Each token connects to two experts
        arrows_t1 = VGroup(
            Arrow(start=tokens2[0].get_right(), end=experts2[0].get_left(), color=YELLOW),
            Arrow(start=tokens2[0].get_right(), end=experts2[1].get_left(), color=YELLOW)
        )
        arrows_t2 = VGroup(
            Arrow(start=tokens2[1].get_right(), end=experts2[1].get_left(), color=GREEN),
            Arrow(start=tokens2[1].get_right(), end=experts2[2].get_left(), color=GREEN)
        )
        arrows_t3 = VGroup(
            Arrow(start=tokens2[2].get_right(), end=experts2[0].get_left(), color=PURPLE),
            Arrow(start=tokens2[2].get_right(), end=experts2[2].get_left(), color=PURPLE)
        )

        self.play(FadeIn(tokens2), FadeIn(token_labels2))
        self.play(FadeIn(experts2), FadeIn(expert_labels2))
        self.play(Create(arrows_t1), Create(arrows_t2), Create(arrows_t3))
        self.wait(2)

        # Clean up
        self.play(
            FadeOut(sol_text),
            FadeOut(tokens2),
            FadeOut(token_labels2),
            FadeOut(experts2),
            FadeOut(expert_labels2),
            FadeOut(arrows_t1),
            FadeOut(arrows_t2),
            FadeOut(arrows_t3)
        )

        # Step 3: Show configuration parameters controlling this behavior
        param_title = Text("Key Configuration Parameters", font_size=30).next_to(title, DOWN, buff=0.5)
        param_code = Text(
            "n_routed_experts = 64\n"
            "num_experts_per_tok = 6",
            font="Monospace",
            font_size=24,
            color=ORANGE
        ).next_to(param_title, DOWN, buff=0.5)

        self.play(Write(param_title))
        self.play(Write(param_code))
        self.wait(2)

        # Final summary
        self.play(FadeOut(param_title), FadeOut(param_code))

        summary = Text(
            "Each token routes to multiple experts.\n"
            "Knowledge is decomposed and specialized.\n"
            "Experts remain focused and efficient.",
            font_size=24
        ).next_to(title, DOWN, buff=0.5)

        self.play(Write(summary))
        self.wait(3)

        self.play(FadeOut(summary), FadeOut(title))
        self.wait(1)