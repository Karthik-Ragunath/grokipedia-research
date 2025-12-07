from manim import *

class ConceptScene(Scene):
    def construct(self):
        # Step 1: Introduction to GRPO vs PPO
        title = Text("Group Relative Policy Optimization (GRPO)", font_size=36).to_edge(UP)
        comparison_text = Text(
            "PPO uses a value model for baseline\nGRPO estimates baseline from group scores",
            font_size=24
        ).next_to(title, DOWN, buff=0.5)

        ppo_box = Rectangle(height=1, width=3, color=BLUE).shift(LEFT*3)
        ppo_label = Text("PPO", color=BLUE).next_to(ppo_box, UP)
        value_model = Text("Value Model", font_size=20).move_to(ppo_box)

        grpo_box = Rectangle(height=1, width=3, color=GREEN).shift(RIGHT*3)
        grpo_label = Text("GRPO", color=GREEN).next_to(grpo_box, UP)
        group_scores = Text("Group Scores\nBaseline", font_size=20).move_to(grpo_box)

        arrow = Arrow(LEFT, RIGHT, color=YELLOW).next_to(ppo_box, RIGHT, buff=0)

        self.play(Write(title))
        self.wait(0.5)
        self.play(Write(comparison_text))
        self.wait(0.5)
        self.play(Create(ppo_box), Write(ppo_label), Write(value_model))
        self.play(GrowArrow(arrow))
        self.play(Create(grpo_box), Write(grpo_label), Write(group_scores))
        self.wait(2)

        # Clean up
        self.play(FadeOut(VGroup(title, comparison_text, ppo_box, ppo_label, value_model, arrow, grpo_box, grpo_label, group_scores)))
        self.wait(0.5)

        # Step 2: Mathematical formulation
        math_title = Text("GRPO Objective Function", font_size=36).to_edge(UP)
        
        objective_eq = MathTex(
            r"\mathcal{J}_{GRPO}(\theta) = \mathbb{E}{[q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)]}",
            font_size=24
        ).next_to(math_title, DOWN, buff=0.5)
        
        summation_eq = MathTex(
            r"\frac{1}{G}\sum_{i=1}^G\frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left\{ \min \left[ \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} | q, o_{i,<t})} \hat{A}_{i,t}, \text{clip}(\cdot) \hat{A}_{i,t} \right] - \beta \mathbb{D}_{KL}[\pi_{\theta} || \pi_{ref}]\right\}",
            font_size=20
        ).next_to(objective_eq, DOWN, buff=0.3)
        
        kl_div_eq = MathTex(
            r"\mathbb{D}_{KL}[\pi_{\theta} || \pi_{ref}] = \frac{\pi_{ref}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}- \log\frac{\pi_{ref}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta}(o_{i,t}|q,o_{i,<t})} - 1",
            font_size=20
        ).next_to(summation_eq, DOWN, buff=0.3)
        
        advantage_text = Text("Advantage calculated from group-relative rewards", font_size=20).next_to(kl_div_eq, DOWN, buff=0.3)
        
        self.play(Write(math_title))
        self.wait(0.5)
        self.play(Write(objective_eq))
        self.wait(0.5)
        self.play(Write(summation_eq))
        self.wait(0.5)
        self.play(Write(kl_div_eq))
        self.wait(0.5)
        self.play(Write(advantage_text))
        self.wait(2)

        # Clean up
        self.play(FadeOut(VGroup(math_title, objective_eq, summation_eq, kl_div_eq, advantage_text)))
        self.wait(0.5)

        # Step 3: Visualizing group sampling
        group_title = Text("Group-Based Sampling", font_size=36).to_edge(UP)
        
        # Draw a question
        question = Text("Question: Explain quantum computing", font_size=24).next_to(group_title, DOWN, buff=1)
        
        # Draw group responses
        responses_group = VGroup()
        for i in range(4):
            response = Rectangle(height=1, width=3, color=ORANGE).shift(DOWN*(i-1.5))
            label = Text(f"Response {i+1}", font_size=20).move_to(response)
            responses_group.add(VGroup(response, label))
            
        brace = Brace(responses_group, LEFT, color=YELLOW)
        group_label = Text("Group G", font_size=24).next_to(brace, LEFT)
        
        avg_reward = Text("Average Reward = Baseline", font_size=24).next_to(responses_group, RIGHT, buff=1)
        arrow_to_avg = Arrow(responses_group.get_right(), avg_reward.get_left(), color=YELLOW)
        
        self.play(Write(group_title))
        self.wait(0.5)
        self.play(Write(question))
        self.wait(0.5)
        self.play(Create(responses_group), Create(brace), Write(group_label))
        self.wait(0.5)
        self.play(GrowArrow(arrow_to_avg), Write(avg_reward))
        self.wait(2)

        # Clean up
        self.play(FadeOut(VGroup(group_title, question, responses_group, brace, group_label, avg_reward, arrow_to_avg)))
        self.wait(0.5)

        # Step 4: Code implementation snippet
        code_title = Text("GRPO Implementation", font_size=36).to_edge(UP)
        
        code_snippet = Text(
            "def compute_loss(self, model, inputs):\n"
            "    # Compute per-token log probabilities\n"
            "    per_token_logps = self._get_per_token_logps(...)\n\n"
            "    # Calculate advantages using group-relative rewards\n"
            "    advantages = inputs['advantages']\n\n"
            "    # Apply clipped PPO objective\n"
            "    ratio = torch.exp(per_token_logps - old_per_token_logps)\n"
            "    loss = -torch.min(ratio * advantages, \n"
            "                      clip(ratio, 1-ε, 1+ε) * advantages)\n\n"
            "    # Add KL divergence regularization\n"
            "    if self.beta != 0:\n"
            "        kl_div = self._compute_kl_divergence(...)\n"
            "        loss += self.beta * kl_div\n\n"
            "    return loss.mean()",
            font="Monospace",
            font_size=18
        ).next_to(code_title, DOWN, buff=0.5)
        
        self.play(Write(code_title))
        self.wait(0.5)
        self.play(Write(code_snippet))
        self.wait(2)

        # Clean up
        self.play(FadeOut(VGroup(code_title, code_snippet)))
        self.wait(0.5)

        # Step 5: Key benefits
        benefits_title = Text("Key Benefits of GRPO", font_size=36).to_edge(UP)
        
        benefit1 = Text("• No value function approximation needed", font_size=24).next_to(benefits_title, DOWN, buff=1).shift(LEFT*2)
        benefit2 = Text("• Reduced memory/compute requirements", font_size=24).next_to(benefit1, DOWN, buff=0.5).align_to(benefit1, LEFT)
        benefit3 = Text("• Aligns with comparative reward models", font_size=24).next_to(benefit2, DOWN, buff=0.5).align_to(benefit1, LEFT)
        benefit4 = Text("• Direct KL regularization", font_size=24).next_to(benefit3, DOWN, buff=0.5).align_to(benefit1, LEFT)
        
        self.play(Write(benefits_title))
        self.wait(0.5)
        self.play(Write(benefit1))
        self.wait(0.5)
        self.play(Write(benefit2))
        self.wait(0.5)
        self.play(Write(benefit3))
        self.wait(0.5)
        self.play(Write(benefit4))
        self.wait(2)

        # Final cleanup
        self.play(FadeOut(VGroup(benefits_title, benefit1, benefit2, benefit3, benefit4)))
        self.wait(0.5)