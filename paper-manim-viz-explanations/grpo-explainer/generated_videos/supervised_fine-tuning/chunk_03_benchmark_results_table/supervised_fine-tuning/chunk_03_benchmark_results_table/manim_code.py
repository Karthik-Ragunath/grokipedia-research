from manim import *

class ConceptScene(Scene):
    def construct(self):
        # Step 1: Title and Introduction
        title = Text("Benchmark Results Table", font_size=40).to_edge(UP)
        self.play(Write(title))
        self.wait(1)

        desc = Text(
            "Comparing AI models on math reasoning tasks\nin English and Chinese",
            font_size=24,
        ).next_to(title, DOWN, buff=0.5)
        self.play(Write(desc))
        self.wait(2)

        self.play(FadeOut(title), FadeOut(desc))

        # Step 2: Table Visualization
        table_title = Text("Key Models & Benchmarks", font_size=36).to_edge(UP)
        self.play(Write(table_title))
        self.wait(1)

        # Simplified table headers
        headers = VGroup(
            Text("Model", font_size=24),
            Text("GSM8K", font_size=24),
            Text("MATH", font_size=24),
            Text("MGSM-zh", font_size=24),
            Text("CMATH", font_size=24),
        ).arrange(RIGHT, buff=1).next_to(table_title, DOWN, buff=1)

        separator_line = Line(
            start=headers.get_left(), end=headers.get_right()
        ).next_to(headers, DOWN, buff=0.2)
        self.play(FadeIn(headers), Create(separator_line))
        self.wait(1)

        # Sample model rows
        model_rows = VGroup(
            Text("GPT-4", font_size=20),
            Text("92.0%", font_size=20),
            Text("52.9%", font_size=20),
            Text("-", font_size=20),
            Text("86.0%", font_size=20),
        ).arrange(RIGHT, buff=1.2).next_to(separator_line, DOWN, buff=0.3)

        model_rows2 = VGroup(
            Text("DeepSeek 67B", font_size=20),
            Text("84.1%", font_size=20),
            Text("32.6%", font_size=20),
            Text("74.0%", font_size=20),
            Text("80.3%", font_size=20),
        ).arrange(RIGHT, buff=1.2).next_to(model_rows, DOWN, buff=0.3)

        model_rows3 = VGroup(
            Text("SPM-RL 7B", font_size=20, color=YELLOW),
            Text("88.2%", font_size=20, color=GREEN),
            Text("51.7%", font_size=20, color=GREEN),
            Text("79.6%", font_size=20, color=GREEN),
            Text("88.8%", font_size=20, color=GREEN),
        ).arrange(RIGHT, buff=1.2).next_to(model_rows2, DOWN, buff=0.3)

        self.play(FadeIn(model_rows), FadeIn(model_rows2), FadeIn(model_rows3))
        self.wait(2)

        highlight = SurroundingRectangle(model_rows3, color=YELLOW, buff=0.1)
        self.play(Create(highlight))
        self.wait(1)

        note = Text(
            "SPM-RL 7B outperforms larger models!",
            font_size=24,
            color=YELLOW
        ).to_edge(DOWN)
        self.play(Write(note))
        self.wait(2)

        self.play(
            FadeOut(table_title),
            FadeOut(headers),
            FadeOut(separator_line),
            FadeOut(model_rows),
            FadeOut(model_rows2),
            FadeOut(model_rows3),
            FadeOut(highlight),
            FadeOut(note)
        )

        # Step 3: Explain Training Method
        algo_title = Text("What Powers SPM-RL?", font_size=36).to_edge(UP)
        self.play(Write(algo_title))
        self.wait(1)

        algo_desc = Text(
            "Uses GRPO: Group Relative Policy Optimization",
            font_size=28
        ).next_to(algo_title, DOWN, buff=0.5)
        self.play(Write(algo_desc))
        self.wait(1)

        formula = MathTex(
            r"\mathcal{L}_{\text{GRPO}} = \mathbb{E} \left[ \min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t) \right]",
            font_size=36
        ).next_to(algo_desc, DOWN, buff=0.7)
        self.play(Write(formula))
        self.wait(2)

        self.play(FadeOut(algo_title), FadeOut(algo_desc), FadeOut(formula))

        # Step 4: Code Snippet Preview
        code_title = Text("GRPO Trainer Snippet", font_size=36).to_edge(UP)
        self.play(Write(code_title))
        self.wait(1)

        code_snippet = Text(
            "class GRPOTrainer(BaseTrainer):\n"
            "    def compute_loss(...):\n"
            "        advantages = inputs['advantages']\n"
            "        ratio = exp(logprobs - old_logprobs)\n"
            "        loss = min(ratio * A, clip(ratio, ...) * A)",
            font="Monospace",
            font_size=20,
        ).next_to(code_title, DOWN, buff=0.5)

        self.play(FadeIn(code_snippet))
        self.wait(3)

        conclusion = Text(
            "GRPO enables efficient training\nof powerful math reasoning models!",
            font_size=28,
            color=BLUE
        ).to_edge(DOWN)
        self.play(Write(conclusion))
        self.wait(2)

        self.play(
            FadeOut(code_title),
            FadeOut(code_snippet),
            FadeOut(conclusion)
        )

        # Final Summary
        final_text = Text(
            "Benchmark success comes from advanced training algorithms",
            font_size=32
        ).move_to(ORIGIN)
        self.play(Write(final_text))
        self.wait(2)
        self.play(FadeOut(final_text))