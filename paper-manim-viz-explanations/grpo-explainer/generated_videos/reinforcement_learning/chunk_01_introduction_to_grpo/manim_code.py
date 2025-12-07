from manim import *

class ConceptScene(Scene):
    def construct(self):
        # Step 1: Introduce GRPO
        title = Text("Group Relative Policy Optimization (GRPO)", font_size=36, weight=BOLD)
        title.to_edge(UP)
        
        desc1 = Text("A Reinforcement Learning algorithm for improving", font_size=24)
        desc2 = Text("mathematical reasoning in Large Language Models.", font_size=24)
        desc_group = VGroup(desc1, desc2).arrange(DOWN, buff=0.3).next_to(title, DOWN, buff=0.5)

        self.play(Write(title))
        self.play(FadeIn(desc_group))
        self.wait(2)

        # Clean up
        self.play(FadeOut(title), FadeOut(desc_group))
        self.wait(0.5)

        # Step 2: Key Concepts of GRPO
        concept_title = Text("Core Concepts of GRPO", font_size=32, weight=BOLD)
        concept_title.to_edge(UP)

        kl_div = MathTex(r"\text{KL}(\pi_\theta || \pi_{\text{ref}})").set_color(BLUE)
        kl_text = Text("Minimize divergence from reference policy", font_size=24).next_to(kl_div, RIGHT, buff=0.5)

        clip_loss = MathTex(r"\text{Clipped Objective: } \min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)").set_color(GREEN)
        clip_text = Text("Clip importance ratios for stable training", font_size=24).next_to(clip_loss, RIGHT, buff=0.5)

        kl_group = VGroup(kl_div, kl_text).arrange(RIGHT, buff=0.5).shift(UP)
        clip_group = VGroup(clip_loss, clip_text).arrange(RIGHT, buff=0.5).shift(DOWN)

        self.play(Write(concept_title))
        self.play(FadeIn(kl_group))
        self.wait(1)
        self.play(FadeIn(clip_group))
        self.wait(2)

        # Clean up
        self.play(FadeOut(concept_title), FadeOut(kl_group), FadeOut(clip_group))
        self.wait(0.5)

        # Step 3: GRPO in Code - Class Definition
        code_title = Text("GRPO Implementation in Code", font_size=32, weight=BOLD)
        code_title.to_edge(UP)

        class_def = Text("class GRPOTrainer(BaseTrainer):", font="Monospace", font_size=20, color=YELLOW)
        init_method = Text("def __init__(self, model, reward_funcs, ...):", font="Monospace", font_size=20, color=ORANGE)
        param_example = Text("args = GRPOConfig(...)", font="Monospace", font_size=20, color=PURPLE)

        code_group = VGroup(class_def, init_method, param_example).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        code_group.next_to(code_title, DOWN, buff=0.5).shift(LEFT * 1.5)

        self.play(Write(code_title))
        self.play(FadeIn(class_def))
        self.wait(0.5)
        self.play(FadeIn(init_method))
        self.wait(0.5)
        self.play(FadeIn(param_example))
        self.wait(2)

        # Clean up
        self.play(FadeOut(code_title), FadeOut(code_group))
        self.wait(0.5)

        # Step 4: GRPO Config Parameters
        config_title = Text("Key GRPO Configuration Parameters", font_size=32, weight=BOLD)
        config_title.to_edge(UP)

        beta_param = Text("beta: KL coefficient (default: 0.0)", font_size=22)
        iter_param = Text("num_iterations: Training iterations per batch", font_size=22)
        eps_param = Text("epsilon: Clipping bounds for importance ratios", font_size=22)
        scale_param = Text("scale_rewards: Reward scaling strategy", font_size=22)

        params_group = VGroup(beta_param, iter_param, eps_param, scale_param).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        params_group.next_to(config_title, DOWN, buff=0.5)

        self.play(Write(config_title))
        self.play(FadeIn(params_group))
        self.wait(2)

        # Clean up
        self.play(FadeOut(config_title), FadeOut(params_group))
        self.wait(0.5)

        # Step 5: Summary
        summary_title = Text("Summary", font_size=32, weight=BOLD)
        summary_title.to_edge(UP)

        summary1 = Text("• GRPO improves math reasoning in LLMs", font_size=24)
        summary2 = Text("• Uses clipped objectives & KL regularization", font_size=24)
        summary3 = Text("• Implemented via GRPOTrainer & GRPOConfig", font_size=24)

        summary_group = VGroup(summary1, summary2, summary3).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        summary_group.next_to(summary_title, DOWN, buff=0.5)

        self.play(Write(summary_title))
        self.play(Write(summary_group))
        self.wait(3)

        # Final cleanup
        self.play(FadeOut(summary_title), FadeOut(summary_group))
        self.wait(1)