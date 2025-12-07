from manim import *

class ConceptScene(Scene):
    def construct(self):
        # Step 1: Introduce the core concept
        title = Text("Tool-Integrated Reasoning Performance", font_size=36).to_edge(UP)
        desc = Text(
            "Models combine natural language reasoning\nwith program-based tools for problem-solving",
            font_size=24,
            line_spacing=1.2
        ).next_to(title, DOWN, buff=0.5)

        self.play(Write(title))
        self.play(FadeIn(desc))
        self.wait(2)

        # Clean up
        self.play(FadeOut(title), FadeOut(desc))

        # Step 2: Show benchmark performance
        perf_title = Text("Performance on Benchmarks", font_size=32).to_edge(UP)
        math_label = Text("MATH Benchmark:", font_size=24).shift(UP)
        math_result = MathTex(r"\text{Accuracy} \approx 60\%").next_to(math_label, RIGHT, buff=0.5).set_color(GREEN)

        comparison_text = Text(
            "Competitive with\nDeepSeek-LLM-Chat 67B\n(10× larger model)",
            font_size=24,
            line_spacing=1.2
        ).next_to(math_label, DOWN, buff=1)

        self.play(Write(perf_title))
        self.play(Write(math_label), Write(math_result))
        self.play(FadeIn(comparison_text))
        self.wait(2)

        # Clean up
        self.play(
            FadeOut(perf_title),
            FadeOut(math_label),
            FadeOut(math_result),
            FadeOut(comparison_text)
        )

        # Step 3: Explain GRPO Training Approach
        grpo_title = Text("Group Relative Policy Optimization (GRPO)", font_size=28).to_edge(UP)
        explanation = Text(
            "Optimizes policies for complex reasoning tasks\nby balancing rewards and KL divergence",
            font_size=22,
            line_spacing=1.2
        ).next_to(grpo_title, DOWN, buff=0.5)

        # Visual metaphor: balance scale
        left_side = Rectangle(width=2, height=1, color=BLUE).shift(LEFT*2)
        right_side = Rectangle(width=2, height=1, color=YELLOW).shift(RIGHT*2)
        balance_beam = Line(LEFT*3, RIGHT*3, color=GREY).move_to(ORIGIN)
        pivot = Dot(color=RED)

        reward_label = Text("Rewards", font_size=20).next_to(left_side, UP)
        kl_label = Text("KL Penalty", font_size=20).next_to(right_side, UP)

        self.play(Write(grpo_title))
        self.play(FadeIn(explanation))
        self.play(Create(balance_beam), FadeIn(pivot))
        self.play(FadeIn(left_side), FadeIn(right_side))
        self.play(Write(reward_label), Write(kl_label))
        self.wait(2)

        # Clean up
        self.play(
            FadeOut(grpo_title),
            FadeOut(explanation),
            FadeOut(balance_beam),
            FadeOut(pivot),
            FadeOut(left_side),
            FadeOut(right_side),
            FadeOut(reward_label),
            FadeOut(kl_label)
        )

        # Step 4: Show Key Code Components
        code_title = Text("Key Implementation Components", font_size=32).to_edge(UP)

        # Simplified code snippets
        trainer_snippet = Text(
            "class GRPOTrainer(BaseTrainer):\n    def __init__(self, model, reward_funcs, ...):",
            font="Monospace",
            font_size=20
        ).shift(UP*1.5)

        config_snippet = Text(
            "class GRPOConfig(TrainingArguments):\n    beta = 0.0  # KL coefficient\n    num_iterations = 1",
            font="Monospace",
            font_size=20
        ).next_to(trainer_snippet, DOWN, buff=0.8)

        reward_snippet = Text(
            "reward_funcs = {\n    'accuracy_reward': ..., \n    'think_format_reward': ...\n}",
            font="Monospace",
            font_size=20
        ).next_to(config_snippet, DOWN, buff=0.8)

        self.play(Write(code_title))
        self.play(FadeIn(trainer_snippet))
        self.play(FadeIn(config_snippet))
        self.play(FadeIn(reward_snippet))
        self.wait(2)

        # Final cleanup
        self.play(
            FadeOut(code_title),
            FadeOut(trainer_snippet),
            FadeOut(config_snippet),
            FadeOut(reward_snippet)
        )

        # Step 5: Summary
        summary = Text(
            "GRPO enables strong mathematical reasoning\nby optimizing tool-integrated policies",
            font_size=28,
            line_spacing=1.3
        )
        self.play(Write(summary))
        self.wait(2)