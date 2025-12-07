from manim import *

class ConceptScene(Scene):
    def construct(self):
        # Step 1: Title and Core Idea
        title = Text("Iterative RL with GRPO", font_size=40).to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        idea_text = Text(
            "Reward models become outdated as policies improve.\n"
            "We iteratively retrain reward models with new data.",
            font_size=24,
            line_spacing=1.2
        ).next_to(title, DOWN, buff=0.5)
        self.play(Write(idea_text))
        self.wait(2)

        # Cleanup
        self.play(FadeOut(title), FadeOut(idea_text))
        self.wait(0.5)

        # Step 2: Visualizing the Loop
        loop_title = Text("Training Loop Overview", font_size=36).to_edge(UP)
        self.play(Write(loop_title))

        # Components
        policy_box = Rectangle(color=BLUE, height=1.5, width=3).set_fill(BLUE, opacity=0.3)
        policy_label = Text("Policy Model", font_size=20).move_to(policy_box)
        policy_group = VGroup(policy_box, policy_label).shift(LEFT*3)

        reward_box = Rectangle(color=GREEN, height=1.5, width=3).set_fill(GREEN, opacity=0.3)
        reward_label = Text("Reward Model", font_size=20).move_to(reward_box)
        reward_group = VGroup(reward_box, reward_label).shift(RIGHT*3)

        arrow_pr = Arrow(policy_group.get_right(), reward_group.get_left(), color=YELLOW)
        arrow_rp = Arrow(reward_group.get_bottom(), policy_group.get_bottom(), color=YELLOW).shift(DOWN*0.5)
        loop_label = Text("Iterative Updates", font_size=18, color=YELLOW).next_to(arrow_rp, DOWN, buff=0.1)

        self.play(FadeIn(policy_group), FadeIn(reward_group))
        self.play(GrowArrow(arrow_pr))
        self.play(GrowArrow(arrow_rp), Write(loop_label))
        self.wait(2)

        # Cleanup
        self.play(FadeOut(loop_title), FadeOut(policy_group), FadeOut(reward_group),
                  FadeOut(arrow_pr), FadeOut(arrow_rp), FadeOut(loop_label))
        self.wait(0.5)

        # Step 3: Replay Mechanism
        replay_title = Text("Replay Mechanism", font_size=36).to_edge(UP)
        self.play(Write(replay_title))

        # Data Bars
        history_bar = Rectangle(color=GRAY, width=4, height=0.5).set_fill(GRAY, opacity=0.5).shift(LEFT*2 + UP*1)
        history_label = Text("Historical Data (10%)", font_size=18).next_to(history_bar, DOWN, buff=0.1)

        new_data_bar = Rectangle(color=ORANGE, width=4, height=0.5).set_fill(ORANGE, opacity=0.7).shift(RIGHT*2 + UP*1)
        new_data_label = Text("New Samples (90%)", font_size=18).next_to(new_data_bar, DOWN, buff=0.1)

        combined_bar = Rectangle(color=PURPLE, width=4, height=0.7).set_fill(PURPLE, opacity=0.6).shift(DOWN*1)
        combined_label = Text("Training Set (Mixed)", font_size=18).next_to(combined_bar, DOWN, buff=0.1)

        self.play(FadeIn(history_bar), Write(history_label))
        self.play(FadeIn(new_data_bar), Write(new_data_label))
        self.wait(1)
        self.play(Transform(VGroup(history_bar.copy(), new_data_bar.copy()), combined_bar), Write(combined_label))
        self.wait(2)

        # Cleanup
        self.play(FadeOut(replay_title), FadeOut(history_bar), FadeOut(new_data_bar),
                  FadeOut(combined_bar), FadeOut(history_label), FadeOut(new_data_label),
                  FadeOut(combined_label))
        self.wait(0.5)

        # Step 4: Code Snippet Highlight
        code_title = Text("Code: Reward Function Handling", font_size=32).to_edge(UP)
        self.play(Write(code_title))

        code_snippet = Text(
            "for i, reward_func in enumerate(reward_funcs):\n"
            "    if isinstance(reward_func, str):\n"
            "        reward_funcs[i] = from_pretrained(...)\n"
            "self.reward_funcs = reward_funcs\n"
            "# Load or update reward models dynamically",
            font="Monospace",
            font_size=20,
            line_spacing=1.1
        ).next_to(code_title, DOWN, buff=0.5)

        self.play(Write(code_snippet))
        self.wait(3)

        # Final cleanup
        self.play(FadeOut(code_title), FadeOut(code_snippet))
        self.wait(0.5)

        # Step 5: Summary
        summary_title = Text("Key Takeaways", font_size=36).to_edge(UP)
        self.play(Write(summary_title))

        points = VGroup(
            Text("• Policies evolve over time", font_size=24),
            Text("• Reward models must adapt", font_size=24),
            Text("• Iterative training improves alignment", font_size=24),
            Text("• Replay helps retain knowledge", font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(summary_title, DOWN, buff=0.5)

        for point in points:
            self.play(Write(point))
            self.wait(0.5)

        self.wait(2)
        self.play(FadeOut(summary_title), FadeOut(points))
        self.wait(1)