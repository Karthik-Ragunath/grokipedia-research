from manim import *

class ConceptScene(Scene):
    def construct(self):
        # Step 1: Title and Core Concept
        title = Text("Training and Evaluation Setup", font_size=40).to_edge(UP)
        self.play(Write(title))
        self.wait(1)

        concept_text = Text(
            "SPmath-RL uses GRPO with specific hyperparameters\n"
            "and evaluation benchmarks.",
            font_size=28
        ).next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(concept_text))
        self.wait(2)

        # Cleanup
        self.play(FadeOut(title), FadeOut(concept_text))

        # Step 2: Hyperparameters Overview
        hparams_title = Text("Key Hyperparameters", font_size=36).to_edge(UP)
        self.play(Write(hparams_title))

        lr_policy = Text("Policy LR: 1e-6", font_size=28).shift(UP * 1.5)
        kl_coeff = Text("KL Coefficient: 0.04", font_size=28).next_to(lr_policy, DOWN, buff=0.5)
        samples_per_q = Text("Samples per Question: 64", font_size=28).next_to(kl_coeff, DOWN, buff=0.5)
        max_len = Text("Max Length: 1024", font_size=28).next_to(samples_per_q, DOWN, buff=0.5)
        batch_size = Text("Batch Size: 1024", font_size=28).next_to(max_len, DOWN, buff=0.5)

        hparams_group = VGroup(lr_policy, kl_coeff, samples_per_q, max_len, batch_size)
        self.play(LaggedStart(*[FadeIn(obj) for obj in hparams_group], lag_ratio=0.3))
        self.wait(3)

        # Cleanup
        self.play(FadeOut(hparams_title), FadeOut(hparams_group))

        # Step 3: Visualizing Sampling Process
        sampling_title = Text("Sampling Strategy", font_size=36).to_edge(UP)
        self.play(Write(sampling_title))

        prompt_box = Rectangle(height=1, width=3, color=BLUE).shift(LEFT*3)
        prompt_label = Text("Prompt", font_size=24).next_to(prompt_box, UP)
        self.play(Create(prompt_box), Write(prompt_label))

        arrows = VGroup()
        outputs = VGroup()
        for i in range(4):
            arrow = Arrow(start=prompt_box.get_right(), end=RIGHT * 1.5 + UP*(1.5 - i), color=YELLOW)
            output = Rectangle(height=0.8, width=2.5, color=GREEN).next_to(arrow, RIGHT)
            arrows.add(arrow)
            outputs.add(output)

        # Add dots to imply more samples
        dots = Text("...", font_size=36).next_to(outputs[-1], RIGHT, buff=0.5)

        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.2))
        self.play(LaggedStart(*[Create(o) for o in outputs], lag_ratio=0.2))
        self.play(Write(dots))
        self.wait(2)

        # Label total samples
        sample_count = MathTex("64 \\text{ samples}", color=ORANGE).to_edge(DOWN)
        self.play(Write(sample_count))
        self.wait(2)

        # Cleanup
        self.play(FadeOut(sampling_title), FadeOut(VGroup(prompt_box, prompt_label, arrows, outputs, dots)), FadeOut(sample_count))

        # Step 4: Code Snippets Highlighting Configurations
        code_title = Text("Code Implementation", font_size=36).to_edge(UP)
        self.play(Write(code_title))

        code_snippet = Text(
            "learning_rate: float = 1e-6\n"
            "num_generations: int = 8\n"
            "max_completion_length: int = 256",
            font="Monospace",
            font_size=24
        ).scale(0.9).shift(UP * 0.5)

        box = SurroundingRectangle(code_snippet, color=GRAY, buff=0.2)
        self.play(Create(box))
        self.play(Write(code_snippet))
        self.wait(3)

        # Cleanup
        self.play(FadeOut(code_title), FadeOut(box), FadeOut(code_snippet))

        # Final Summary Slide
        summary_title = Text("Summary", font_size=36).to_edge(UP)
        self.play(Write(summary_title))

        summary_points = VGroup(
            Text("• GRPO-based training with KL penalty", font_size=28),
            Text("• Evaluated on GSM8K and MATH (in-domain)", font_size=28),
            Text("• Other benchmarks treated as out-of-domain", font_size=28)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4).next_to(summary_title, DOWN, buff=0.8)

        self.play(LaggedStart(*[Write(point) for point in summary_points], lag_ratio=0.4))
        self.wait(3)

        # Final Fade Out
        self.play(FadeOut(summary_title), FadeOut(summary_points))
        self.wait(1)