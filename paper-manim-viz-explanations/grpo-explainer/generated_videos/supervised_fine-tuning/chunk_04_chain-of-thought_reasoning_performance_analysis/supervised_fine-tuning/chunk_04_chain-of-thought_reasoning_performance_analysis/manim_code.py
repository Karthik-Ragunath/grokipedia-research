from manim import *

class ConceptScene(Scene):
    def construct(self):
        # Step 1: Introduce Chain-of-Thought Reasoning
        title = Text("Chain-of-Thought Reasoning", font_size=40).to_edge(UP)
        description = Text(
            "Step-by-step reasoning for complex tasks like math problems.",
            font_size=24
        ).next_to(title, DOWN, buff=0.5)

        self.play(Write(title))
        self.play(FadeIn(description))
        self.wait(1)

        self.play(FadeOut(title), FadeOut(description))

        # Step 2: Performance Comparison
        comparison_title = Text("Performance on MATH Dataset", font_size=36).to_edge(UP)
        model_comparison = Text(
            "spmath-Instruct 7B > All open-source & most proprietary models",
            font_size=24,
            color=YELLOW
        ).next_to(comparison_title, DOWN, buff=0.5)

        note = Text(
            "Even outperforms larger models like Qwen 72B!",
            font_size=24,
            color=GREEN
        ).next_to(model_comparison, DOWN, buff=0.3)

        self.play(Write(comparison_title))
        self.play(Write(model_comparison))
        self.play(Write(note))
        self.wait(2)

        self.play(FadeOut(comparison_title), FadeOut(model_comparison), FadeOut(note))

        # Step 3: GRPO Trainer Overview
        grpo_title = Text("GRPO Trainer for Math Reasoning", font_size=36).to_edge(UP)
        grpo_desc = Text(
            "Optimizes step-by-step reasoning via reward-based training.",
            font_size=24
        ).next_to(grpo_title, DOWN, buff=0.5)

        self.play(Write(grpo_title))
        self.play(Write(grpo_desc))
        self.wait(1)

        # Show simplified code snippet
        code_snippet = Text(
            "class GRPOTrainer(BaseTrainer):\n"
            "    def __init__(self, model, reward_funcs, ...):\n"
            "        ...\n"
            "\n"
            "    def train(self):\n"
            "        # Optimizes policy using rewards\n"
            "        ...",
            font="Monospace",
            font_size=20,
            color=BLUE
        ).next_to(grpo_desc, DOWN, buff=0.5)

        self.play(Write(code_snippet))
        self.wait(2)

        self.play(FadeOut(grpo_title), FadeOut(grpo_desc), FadeOut(code_snippet))

        # Step 4: Key Training Parameters
        param_title = Text("Key GRPO Configurations", font_size=36).to_edge(UP)
        
        params = VGroup(
            Text("• temperature = 1.0", font_size=24),
            Text("• top_p = 1.0", font_size=24),
            Text("• max_completion_length = 256", font_size=24),
            Text("• beta = 0.0 (KL coefficient)", font_size=24),
            Text("• num_iterations = 1", font_size=24),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(param_title, DOWN, buff=0.5)

        self.play(Write(param_title))
        self.play(LaggedStart(*[Write(p) for p in params], lag_ratio=0.3))
        self.wait(2)

        self.play(FadeOut(param_title), FadeOut(params))

        # Step 5: Reward Function Mechanism
        reward_title = Text("Reward Function", font_size=36).to_edge(UP)
        reward_desc = Text(
            "Guides model toward correct reasoning steps.",
            font_size=24
        ).next_to(reward_title, DOWN, buff=0.5)

        reward_code = Text(
            "def reward_func(completions, **kwargs):\n"
            "    # Returns reward scores for completions\n"
            "    return [score_for(c) for c in completions]",
            font="Monospace",
            font_size=20,
            color=GREEN
        ).next_to(reward_desc, DOWN, buff=0.5)

        self.play(Write(reward_title))
        self.play(Write(reward_desc))
        self.play(Write(reward_code))
        self.wait(2)

        self.play(
            FadeOut(reward_title),
            FadeOut(reward_desc),
            FadeOut(reward_code)
        )

        # Final Summary
        summary = Text(
            "GRPO enables strong mathematical reasoning\nthrough reward-guided training",
            font_size=30,
            color=PURPLE
        )
        self.play(Write(summary))
        self.wait(2)