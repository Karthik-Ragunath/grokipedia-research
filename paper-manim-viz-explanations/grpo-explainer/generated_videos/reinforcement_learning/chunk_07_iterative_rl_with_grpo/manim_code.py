from manim import *

class ConceptScene(Scene):
    def construct(self):
        # Step 1: Title and Core Idea
        title = Text("Iterative RL with GRPO", font_size=40).to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        idea_text = Text(
            "Reward models become outdated as policies improve.\n"
            "Solution: Continuously retrain reward models with new data.",
            font_size=24,
            line_spacing=1.2
        ).next_to(title, DOWN, buff=0.5)

        self.play(FadeIn(idea_text))
        self.wait(2)

        # Clean up
        self.play(FadeOut(title), FadeOut(idea_text))
        self.wait(0.5)

        # Step 2: Visualizing the Loop
        loop_title = Text("Training Loop Overview", font_size=36).to_edge(UP)
        self.play(Write(loop_title))

        # Create components
        policy_box = Rectangle(color=BLUE, height=1.5, width=3).shift(LEFT*3)
        policy_label = Text("Policy Model", font_size=20).move_to(policy_box)

        reward_box = Rectangle(color=GREEN, height=1.5, width=3).shift(RIGHT*3)
        reward_label = Text("Reward Model", font_size=20).move_to(reward_box)

        data_box = Rectangle(color=ORANGE, height=1.5, width=2.5).shift(DOWN*2)
        data_label = Text("New Data", font_size=20).move_to(data_box)

        # Arrows
        arrow_pr = Arrow(policy_box.get_right(), reward_box.get_left(), color=YELLOW)
        arrow_rd = Arrow(reward_box.get_bottom(), data_box.get_right(), color=YELLOW)
        arrow_dp = Arrow(data_box.get_left(), policy_box.get_bottom(), color=YELLOW)
        loop_arrow = CurvedArrow(reward_box.get_top(), policy_box.get_top(), angle=PI/3, color=PURPLE)

        loop_group = VGroup(
            policy_box, policy_label,
            reward_box, reward_label,
            data_box, data_label,
            arrow_pr, arrow_rd, arrow_dp, loop_arrow
        )

        self.play(
            Create(policy_box),
            Write(policy_label),
            Create(reward_box),
            Write(reward_label),
            Create(data_box),
            Write(data_label)
        )
        self.wait(0.5)

        self.play(
            GrowArrow(arrow_pr),
            GrowArrow(arrow_rd),
            GrowArrow(arrow_dp),
            GrowArrow(loop_arrow)
        )
        self.wait(2)

        # Clean up
        self.play(FadeOut(loop_title), FadeOut(loop_group))
        self.wait(0.5)

        # Step 3: Replay Mechanism
        replay_title = Text("Replay Mechanism", font_size=36).to_edge(UP)
        self.play(Write(replay_title))

        # Timeline
        timeline = NumberLine(x_range=[0, 10, 1], length=8, include_numbers=True)
        timeline.shift(DOWN*0.5)

        # Data points
        history_dot = Dot(timeline.n2p(2), color=GRAY)
        history_label = Text("History (10%)", font_size=20).next_to(history_dot, UP, buff=0.2)

        new_data_dot = Dot(timeline.n2p(8), color=ORANGE)
        new_data_label = Text("New Data (90%)", font_size=20).next_to(new_data_dot, UP, buff=0.2)

        self.play(Create(timeline))
        self.play(FadeIn(history_dot, history_label))
        self.play(FadeIn(new_data_dot, new_data_label))
        self.wait(2)

        # Clean up
        self.play(FadeOut(replay_title), FadeOut(timeline), FadeOut(history_dot), FadeOut(history_label),
                  FadeOut(new_data_dot), FadeOut(new_data_label))
        self.wait(0.5)

        # Step 4: Key Code Snippet
        code_title = Text("GRPO Trainer Initialization", font_size=36).to_edge(UP)
        self.play(Write(code_title))

        code_snippet = Text(
            "self.num_iterations = args.num_iterations\n"
            "self.reward_funcs = reward_funcs\n"
            "for i, reward_func in enumerate(reward_funcs):\n"
            "    if isinstance(reward_func, str):\n"
            "        reward_funcs[i] = AutoModel...\n"
            "self._step = 0  # Track iterations",
            font="Monospace",
            font_size=20,
            line_spacing=1.1
        ).next_to(code_title, DOWN, buff=0.5)

        self.play(FadeIn(code_snippet))
        self.wait(3)

        # Clean up
        self.play(FadeOut(code_title), FadeOut(code_snippet))
        self.wait(0.5)

        # Final Summary
        summary_title = Text("Key Takeaways", font_size=36).to_edge(UP)
        self.play(Write(summary_title))

        summary_points = VGroup(
            Text("• Policies and rewards co-evolve", font_size=24),
            Text("• New data improves reward modeling", font_size=24),
            Text("• Replay preserves historical context", font_size=24),
            Text("• Multiple iterations refine both models", font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(summary_title, DOWN, buff=0.5)

        for point in summary_points:
            self.play(Write(point))
            self.wait(0.3)

        self.wait(2)
        self.play(FadeOut(summary_title), FadeOut(summary_points))
        self.wait(1)