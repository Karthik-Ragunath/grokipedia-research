from manim import *

class ConceptScene(Scene):
    def construct(self):
        # Step 1: Title and Introduction
        title = Text("Process Supervision with GRPO", font_size=36).to_edge(UP)
        self.play(Write(title))
        self.wait(1)

        intro_text = Text(
            "In complex tasks like math reasoning,\n"
            "rewarding only the final answer isn't enough.\n"
            "We want to guide the model step-by-step.",
            font_size=24
        ).next_to(title, DOWN, buff=0.5)
        self.play(Write(intro_text))
        self.wait(2)

        self.play(FadeOut(title), FadeOut(intro_text))

        # Step 2: Outcome vs Process Supervision
        header = Text("Two Types of Supervision", font_size=32).to_edge(UP)
        self.play(Write(header))
        self.wait(0.5)

        outcome_box = Rectangle(height=1.5, width=4, color=RED).shift(LEFT*3)
        outcome_label = Text("Outcome\nSupervision", font_size=20, color=RED).move_to(outcome_box)
        outcome_desc = Text("→ Single reward at the end", font_size=18).next_to(outcome_box, RIGHT, buff=0.5)

        process_box = Rectangle(height=1.5, width=4, color=GREEN).shift(RIGHT*3)
        process_label = Text("Process\nSupervision", font_size=20, color=GREEN).move_to(process_box)
        process_desc = Text("→ Reward at each reasoning step", font_size=18).next_to(process_box, LEFT, buff=0.5)

        self.play(Create(outcome_box), Write(outcome_label), Write(outcome_desc))
        self.wait(1)
        self.play(Create(process_box), Write(process_label), Write(process_desc))
        self.wait(2)

        self.play(FadeOut(header), FadeOut(outcome_box), FadeOut(outcome_label), FadeOut(outcome_desc),
                  FadeOut(process_box), FadeOut(process_label), FadeOut(process_desc))

        # Step 3: Process Reward Model (PRM)
        prm_title = Text("Process Reward Model (PRM)", font_size=32).to_edge(UP)
        self.play(Write(prm_title))
        self.wait(0.5)

        question = Text("Question: q", font_size=24).shift(UP*2 + LEFT*4)
        outputs_group = VGroup(
            Text("Output 1: o₁", font_size=20),
            Text("Output 2: o₂", font_size=20),
            Text("Output G: o_G", font_size=20)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(question, DOWN, buff=0.5)

        dots = MathTex(r"\vdots", font_size=24).next_to(outputs_group[-1], DOWN, buff=0.2)

        self.play(Write(question), Write(outputs_group), Write(dots))
        self.wait(1)

        arrow = Arrow(start=LEFT, end=RIGHT, color=YELLOW).next_to(outputs_group, RIGHT, buff=1)
        prm_label = Text("PRM", color=YELLOW, font_size=24).next_to(arrow, UP, buff=0.1)
        self.play(GrowArrow(arrow), Write(prm_label))
        self.wait(1)

        rewards_group = VGroup(
            MathTex(r"\{r_1^{(1)}, ..., r_1^{(K_1)}\}", font_size=20),
            MathTex(r"\{r_2^{(1)}, ..., r_2^{(K_2)}\}", font_size=20),
            MathTex(r"\{r_G^{(1)}, ..., r_G^{(K_G)}\}", font_size=20)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(arrow, RIGHT, buff=1)
        rewards_dots = MathTex(r"\vdots", font_size=24).next_to(rewards_group[-1], DOWN, buff=0.2)

        self.play(Write(rewards_group), Write(rewards_dots))
        self.wait(2)

        self.play(FadeOut(prm_title), FadeOut(question), FadeOut(outputs_group), FadeOut(dots),
                  FadeOut(arrow), FadeOut(prm_label), FadeOut(rewards_group), FadeOut(rewards_dots))

        # Step 4: Reward Normalization and Advantage Calculation
        norm_title = Text("Reward Normalization", font_size=32).to_edge(UP)
        self.play(Write(norm_title))
        self.wait(0.5)

        raw_reward = MathTex(r"r_i^{(j)}", font_size=36).shift(UP)
        norm_formula = MathTex(
            r"\tilde{r}_i^{(j)} = \frac{r_i^{(j)} - \text{mean}(\mathbf{R})}{\text{std}(\mathbf{R})}",
            font_size=30
        ).next_to(raw_reward, DOWN, buff=0.8)
        self.play(Write(raw_reward), Write(norm_formula))
        self.wait(2)

        self.play(FadeOut(norm_title), FadeOut(raw_reward), FadeOut(norm_formula))

        adv_title = Text("Advantage Calculation", font_size=32).to_edge(UP)
        self.play(Write(adv_title))
        self.wait(0.5)

        adv_formula = MathTex(
            r"\hat{A}_{i,t} = \sum_{\text{index}(j) \geq t} \tilde{r}_i^{\text{index}(j)}",
            font_size=36
        )
        self.play(Write(adv_formula))
        self.wait(2)

        self.play(FadeOut(adv_title), FadeOut(adv_formula))

        # Step 5: GRPO Objective
        obj_title = Text("GRPO Objective", font_size=32).to_edge(UP)
        self.play(Write(obj_title))
        self.wait(0.5)

        obj_formula = MathTex(
            r"\mathcal{L} = \mathbb{E}\left[ \hat{A}_{i,t} \cdot \nabla \log \pi_\theta(a_t|s_t) \right]",
            font_size=36
        )
        self.play(Write(obj_formula))
        self.wait(2)

        conclusion = Text(
            "GRPO uses process supervision to guide\n"
            "models through complex reasoning paths,\n"
            "optimizing step-by-step performance.",
            font_size=24
        ).next_to(obj_formula, DOWN, buff=1)
        self.play(Write(conclusion))
        self.wait(3)

        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)