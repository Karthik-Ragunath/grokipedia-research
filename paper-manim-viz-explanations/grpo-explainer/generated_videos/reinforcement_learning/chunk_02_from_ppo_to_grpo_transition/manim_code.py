from manim import *

class ConceptScene(Scene):
    def construct(self):
        # Step 1: Introduce PPO
        title_ppo = Text("Proximal Policy Optimization (PPO)", font_size=36).to_edge(UP)
        self.play(Write(title_ppo))
        self.wait(1)

        ppo_formula = MathTex(
            r"\mathcal{J}_{PPO}(\theta) = \mathbb{E} \left[ \min \left( r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t \right) \right]",
            font_size=30
        ).next_to(title_ppo, DOWN, buff=0.5)
        ratio_def = MathTex(r"r_t = \frac{\pi_\theta(o_t|q,o_{<t})}{\pi_{\theta_{old}}(o_t|q,o_{<t})}", font_size=28).next_to(ppo_formula, DOWN, buff=0.3)

        self.play(Write(ppo_formula))
        self.play(Write(ratio_def))
        self.wait(2)

        # Clean up
        self.play(FadeOut(title_ppo), FadeOut(ppo_formula), FadeOut(ratio_def))

        # Step 2: Reward with KL Penalty
        kl_title = Text("Reward with KL Penalty", font_size=36).to_edge(UP)
        self.play(Write(kl_title))

        reward_eq = MathTex(
            r"r_t = r_\phi(q, o_{\le t}) - \beta \log \frac{\pi_\theta(o_t|q,o_{<t})}{\pi_{ref}(o_t|q,o_{<t})}",
            font_size=30
        ).next_to(kl_title, DOWN, buff=0.5)

        kl_explanation = Text("KL penalty prevents overfitting to reward model", font_size=24).next_to(reward_eq, DOWN, buff=0.3)

        self.play(Write(reward_eq))
        self.play(Write(kl_explanation))
        self.wait(2)

        # Clean up
        self.play(FadeOut(kl_title), FadeOut(reward_eq), FadeOut(kl_explanation))

        # Step 3: Transition to GRPO Core Idea
        grpo_title = Text("Group-Relative Policy Optimization (GRPO)", font_size=32).to_edge(UP)
        self.play(Write(grpo_title))

        grpo_core = Text("Optimizes policies relative to a reference within groups", font_size=24).next_to(grpo_title, DOWN, buff=0.5)
        self.play(Write(grpo_core))
        self.wait(1)

        # Visual metaphor: group of policies
        ref_circle = Circle(radius=0.5, color=BLUE).shift(LEFT*2)
        ref_label = Text("Ref Model", font_size=20).next_to(ref_circle, DOWN)
        policy_circles = VGroup(*[
            Circle(radius=0.3, color=GREEN).move_to(ref_circle.get_center() + np.array([np.cos(i*TAU/5)*1.5, np.sin(i*TAU/5)*1.5, 0]))
            for i in range(5)
        ])
        policy_labels = VGroup(*[
            Text(f"Policy {i+1}", font_size=18).next_to(policy_circles[i], DOWN)
            for i in range(5)
        ])

        group_brace = Brace(VGroup(ref_circle, *policy_circles), direction=RIGHT)
        group_label = Text("Group", font_size=24).next_to(group_brace, RIGHT)

        self.play(Create(ref_circle), Write(ref_label))
        self.play(LaggedStart(*[Create(circle) for circle in policy_circles], lag_ratio=0.2))
        self.play(LaggedStart(*[Write(label) for label in policy_labels], lag_ratio=0.2))
        self.play(GrowFromCenter(group_brace), Write(group_label))
        self.wait(2)

        # Clean up
        self.play(FadeOut(grpo_title), FadeOut(grpo_core), FadeOut(ref_circle), FadeOut(ref_label),
                  FadeOut(policy_circles), FadeOut(policy_labels), FadeOut(group_brace), FadeOut(group_label))

        # Step 4: GRPO Loss Formula
        grpo_loss_title = Text("GRPO Loss Function", font_size=36).to_edge(UP)
        self.play(Write(grpo_loss_title))

        grpo_loss = MathTex(
            r"\mathcal{L} = -\min \left( r^{(g)}_t A_t, \text{clip}(r^{(g)}_t, 1-\epsilon, 1+\epsilon) A_t \right)",
            font_size=30
        ).next_to(grpo_loss_title, DOWN, buff=0.5)

        ratio_grpo = MathTex(
            r"r^{(g)}_t = \frac{\pi_\theta(o_t|q,o_{<t})}{\pi_{ref}(o_t|q,o_{<t})}",
            font_size=28
        ).next_to(grpo_loss, DOWN, buff=0.3)

        kl_term = MathTex(
            r"+ \beta \cdot D_{KL}(\pi_\theta || \pi_{ref})",
            font_size=28
        ).next_to(ratio_grpo, DOWN, buff=0.3)

        self.play(Write(grpo_loss))
        self.play(Write(ratio_grpo))
        self.play(Write(kl_term))
        self.wait(2)

        # Clean up
        self.play(FadeOut(grpo_loss_title), FadeOut(grpo_loss), FadeOut(ratio_grpo), FadeOut(kl_term))

        # Step 5: Code Snippet Visualization
        code_title = Text("GRPO Implementation (Loss Computation)", font_size=32).to_edge(UP)
        self.play(Write(code_title))

        code_snippet = Text(
            '''def compute_loss(self, model, inputs):
    per_token_logps = get_log_probs(...)
    ref_logps = inputs["ref_per_token_logps"]
    
    ratio = torch.exp(per_token_logps - ref_logps)
    clipped = torch.clamp(ratio, 1-e, 1+e)
    
    loss = -min(ratio * A, clipped * A)
    loss += beta * kl_divergence
    
    return loss.mean()''',
            font="Monospace",
            font_size=20
        ).next_to(code_title, DOWN, buff=0.5)

        self.play(Write(code_snippet))
        self.wait(3)

        # Final cleanup
        self.play(FadeOut(code_title), FadeOut(code_snippet))
        self.wait(1)