from manim import *

class ConceptScene(Scene):
    def construct(self):
        # Step 1: Title and Core Idea
        title = Text("GRPO Algorithm Implementation", font_size=36).to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        core_idea = Text(
            "Group Relative Policy Optimization (GRPO)\n"
            "Optimizes policies using relative advantages\n"
            "across groups of generated outputs.",
            font_size=24,
            line_spacing=1.2
        ).next_to(title, DOWN, buff=0.5)
        
        self.play(Write(core_idea))
        self.wait(2)
        
        self.play(FadeOut(core_idea))

        # Step 2: Key Components Overview
        components_title = Text("Key Components", font_size=32).next_to(title, DOWN, buff=0.5)
        self.play(Write(components_title))
        self.wait(0.5)

        # Boxes for components
        box_policy = Rectangle(height=1.5, width=4, color=BLUE).shift(LEFT*4)
        label_policy = Text("Policy Model π_θ", font_size=20).move_to(box_policy)
        
        box_reward = Rectangle(height=1.5, width=4, color=GREEN).shift(RIGHT*4)
        label_reward = Text("Reward Model r_φ", font_size=20).move_to(box_reward)
        
        box_data = Rectangle(height=1.5, width=4, color=YELLOW).shift(DOWN*2)
        label_data = Text("Task Prompts D", font_size=20).move_to(box_data)

        arrow_pr = Arrow(box_policy.get_right(), box_reward.get_left(), buff=0.1)
        arrow_pd = Arrow(box_policy.get_bottom(), box_data.get_top(), buff=0.1)

        self.play(Create(box_policy), Write(label_policy))
        self.play(Create(box_reward), Write(label_reward))
        self.play(Create(box_data), Write(label_data))
        self.play(GrowArrow(arrow_pr), GrowArrow(arrow_pd))
        self.wait(2)

        # Clean up
        self.play(FadeOut(VGroup(box_policy, label_policy, box_reward, label_reward, box_data, label_data, arrow_pr, arrow_pd, components_title)))

        # Step 3: Algorithm Flow Diagram
        flow_title = Text("Algorithm Flow", font_size=32).next_to(title, DOWN, buff=0.5)
        self.play(Write(flow_title))
        self.wait(0.5)

        # Steps
        step1 = Text("1. Sample G outputs", font_size=20).shift(UP*2 + LEFT*3)
        step2 = Text("2. Compute rewards", font_size=20).shift(UP*1 + LEFT*3)
        step3 = Text("3. Estimate advantage", font_size=20).shift(DOWN*0 + LEFT*3)
        step4 = Text("4. Update policy (μ times)", font_size=20).shift(DOWN*1 + LEFT*3)
        step5 = Text("5. Retrain reward model", font_size=20).shift(DOWN*2 + LEFT*3)

        steps_group = VGroup(step1, step2, step3, step4, step5)
        self.play(Write(steps_group))
        self.wait(1)

        # Visual representation of data flow
        gen_circle = Circle(radius=0.3, color=BLUE).next_to(step1, RIGHT*3)
        gen_text = Text("G", font_size=20).move_to(gen_circle)
        gen_group = VGroup(gen_circle, gen_text)

        reward_rect = Rectangle(width=0.6, height=0.6, color=GREEN).next_to(step2, RIGHT*3)
        reward_text = MathTex(r"r_\phi(o_i)").scale(0.6).move_to(reward_rect)

        adv_triangle = Triangle(color=YELLOW).scale(0.3).next_to(step3, RIGHT*3)
        adv_text = MathTex(r"\hat{A}_{i,t}").scale(0.6).next_to(adv_triangle, UP, buff=0.1)

        update_square = Square(side_length=0.6, color=ORANGE).next_to(step4, RIGHT*3)
        update_text = MathTex(r"\pi_\theta \leftarrow \pi_\theta + \Delta").scale(0.5).move_to(update_square)

        retrain_hex = RegularPolygon(n=6, color=PURPLE).scale(0.3).next_to(step5, RIGHT*3)
        retrain_text = MathTex(r"r_\phi \leftarrow r_\phi + \delta").scale(0.5).next_to(retrain_hex, UP, buff=0.1)

        self.play(FadeIn(gen_group))
        self.wait(0.5)
        self.play(FadeIn(reward_rect), Write(reward_text))
        self.wait(0.5)
        self.play(FadeIn(adv_triangle), Write(adv_text))
        self.wait(0.5)
        self.play(FadeIn(update_square), Write(update_text))
        self.wait(0.5)
        self.play(FadeIn(retrain_hex), Write(retrain_text))
        self.wait(2)

        # Clean up
        self.play(FadeOut(VGroup(steps_group, gen_group, reward_rect, reward_text, adv_triangle, adv_text, update_square, update_text, retrain_hex, retrain_text, flow_title)))

        # Step 4: Simplified Code Snippet
        code_title = Text("Code: GRPOTrainer Initialization", font_size=32).next_to(title, DOWN, buff=0.5)
        self.play(Write(code_title))
        self.wait(0.5)

        code_snippet = Text(
            "class GRPOTrainer:\n"
            "    def __init__(self, model, reward_funcs, ...):\n"
            "        self.num_generations = args.num_generations  # G\n"
            "        self.max_completion_length = args.max_completion_length  # |o_i|\n"
            "        self.num_iterations = args.num_iterations  # μ\n"
            "        ...\n"
            "    def _generate(self, prompts):\n"
            "        # Sample G outputs per prompt\n"
            "        ...\n"
            "    def _calculate_rewards(self, ...):\n"
            "        # Run reward model r_φ\n"
            "        ...",
            font="Monospace",
            font_size=18
        ).next_to(code_title, DOWN, buff=0.5)

        self.play(Write(code_snippet))
        self.wait(3)

        # Final cleanup
        self.play(FadeOut(VGroup(title, code_title, code_snippet)))
        self.wait(1)