from manim import *

class ConceptScene(Scene):
    def construct(self):
        # Step 1: Title and Introduction
        title = Text("Outcome Supervision with GRPO", font_size=40).to_edge(UP)
        self.play(Write(title))
        self.wait(1)

        intro_text = Text(
            "GRPO uses outcome supervision\nby scoring generated outputs\nand normalizing rewards.",
            font_size=24,
            line_spacing=1.2
        ).next_to(title, DOWN, buff=0.8)
        self.play(FadeIn(intro_text))
        self.wait(2)

        self.play(FadeOut(title), FadeOut(intro_text))

        # Step 2: Sampling Outputs and Reward Scoring
        step1_title = Text("Step 1: Sample Outputs & Score Rewards", font_size=32).to_edge(UP)
        self.play(Write(step1_title))
        self.wait(0.5)

        # Visualize group of outputs
        outputs_group = VGroup(
            *[Text(f"Output {i+1}", font_size=20).move_to(
                3 * LEFT + (i - 1.5) * RIGHT * 1.5 + UP * 1
            ) for i in range(4)]
        )
        outputs_label = Text("Outputs from Policy π_θ_old", font_size=20).next_to(outputs_group, UP, buff=0.5)
        self.play(FadeIn(outputs_group), Write(outputs_label))
        self.wait(1)

        # Add reward scores
        rewards = VGroup(
            *[MathTex(f"r_{i+1}", color=YELLOW).next_to(outputs_group[i], DOWN, buff=0.5) for i in range(4)]
        )
        rewards_label = Text("Rewards from Model", font_size=20).next_to(rewards, DOWN, buff=0.3)
        self.play(FadeIn(rewards), Write(rewards_label))
        self.wait(2)

        self.play(FadeOut(outputs_group), FadeOut(outputs_label), FadeOut(rewards), FadeOut(rewards_label))

        # Step 3: Reward Normalization
        step2_title = Text("Step 2: Normalize Rewards", font_size=32).to_edge(UP)
        self.play(Transform(step1_title, step2_title))
        self.wait(0.5)

        raw_rewards = MathTex(r"\mathbf{r} = \{r_1, r_2, ..., r_G\}").scale(1.2)
        self.play(Write(raw_rewards))
        self.wait(1)

        norm_formula = MathTex(
            r"\widetilde{r}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}"
        ).next_to(raw_rewards, DOWN, buff=0.8)
        norm_label = Text("Normalized Reward", font_size=24).next_to(norm_formula, DOWN, buff=0.3)
        self.play(Write(norm_formula), Write(norm_label))
        self.wait(2)

        self.play(FadeOut(raw_rewards), FadeOut(norm_formula), FadeOut(norm_label))

        # Step 4: Advantage Computation
        step3_title = Text("Step 3: Set Advantages", font_size=32).to_edge(UP)
        self.play(Transform(step1_title, step3_title))
        self.wait(0.5)

        advantage_eq = MathTex(
            r"\hat{A}_{i,t} = \widetilde{r}_i"
        ).scale(1.3)
        advantage_text = Text("Advantage for all tokens\nin output i", font_size=24).next_to(advantage_eq, DOWN, buff=0.5)
        self.play(Write(advantage_eq), Write(advantage_text))
        self.wait(2)

        self.play(FadeOut(advantage_eq), FadeOut(advantage_text))

        # Step 5: Code Snippet Highlight
        step4_title = Text("Implementation Example", font_size=32).to_edge(UP)
        self.play(Transform(step1_title, step4_title))
        self.wait(0.5)

        code_snippet = Text(
            "_compute_loss(...):\n"
            "  advantages = inputs['advantages']\n"
            "  per_token_loss = -min(c1*A, c2*A)\n"
            "  loss = mean(per_token_loss)",
            font="Monospace",
            font_size=20,
            line_spacing=1.2
        ).center()
        code_box = SurroundingRectangle(code_snippet, color=BLUE, buff=0.2)
        self.play(FadeIn(code_snippet), Create(code_box))
        self.wait(3)

        self.play(FadeOut(code_snippet), FadeOut(code_box), FadeOut(step1_title))

        # Final Summary
        final_summary = Text(
            "GRPO leverages outcome supervision\nby normalizing rewards\nand applying them as advantages.",
            font_size=28,
            line_spacing=1.3
        ).center()
        self.play(Write(final_summary))
        self.wait(3)
        self.play(FadeOut(final_summary))