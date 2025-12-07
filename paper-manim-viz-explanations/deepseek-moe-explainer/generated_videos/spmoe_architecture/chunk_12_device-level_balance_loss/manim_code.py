from manim import *

class ConceptScene(Scene):
    def construct(self):
        # Step 1: Title and Introduction
        title = Text("Device-Level Balance Loss", font_size=40).to_edge(UP)
        intro_text = Text(
            "Balancing computation across devices\ninstead of individual experts",
            font_size=24,
            line_spacing=1.2
        ).next_to(title, DOWN, buff=0.5)

        self.play(Write(title))
        self.play(FadeIn(intro_text))
        self.wait(2)

        self.play(FadeOut(intro_text))

        # Step 2: Expert vs Device Level
        comparison_title = Text("Two Levels of Balance", font_size=32).next_to(title, DOWN, buff=0.8)
        expert_box = Rectangle(height=1.5, width=4, color=BLUE).shift(LEFT*3)
        expert_label = Text("Expert-Level", font_size=20, color=BLUE).move_to(expert_box)
        expert_desc = Text("Balance load per expert", font_size=18).next_to(expert_box, DOWN, buff=0.2)

        device_box = Rectangle(height=1.5, width=4, color=GREEN).shift(RIGHT*3)
        device_label = Text("Device-Level", font_size=20, color=GREEN).move_to(device_box)
        device_desc = Text("Balance load per device", font_size=18).next_to(device_box, DOWN, buff=0.2)

        self.play(Write(comparison_title))
        self.play(Create(expert_box), Write(expert_label), Write(expert_desc))
        self.play(Create(device_box), Write(device_label), Write(device_desc))
        self.wait(2)

        self.play(FadeOut(VGroup(comparison_title, expert_box, expert_label, expert_desc, device_box, device_label, device_desc)))

        # Step 3: Mathematical Formula
        formula_title = Text("Device-Level Balance Loss Formula", font_size=28).next_to(title, DOWN, buff=0.8)
        
        loss_formula = MathTex(
            r"\mathcal{L}_{\mathrm{DevBal}}", "=", r"\alpha_{2}", r"\sum_{i=1}^{D}{f_i^{\prime} P_i^{\prime}}"
        ).next_to(formula_title, DOWN, buff=0.8)
        
        fi_formula = MathTex(
            r"f_i^{\prime}", "=", r"\frac{1}{|\mathcal{E}_i|}", r"\sum_{j \in \mathcal{E}_i}{ f_j }"
        ).next_to(loss_formula, DOWN, buff=0.4)
        
        pi_formula = MathTex(
            r"P_i^{\prime}", "=", r"\sum_{j \in \mathcal{E}_i}{ P_j }"
        ).next_to(fi_formula, DOWN, buff=0.4)
        
        legend = Text("Where:\nα₂ = device-level balance factor\nD = number of devices\nℰᵢ = experts on device i", 
                      font_size=18, line_spacing=1.2).to_edge(DOWN)

        self.play(Write(formula_title))
        self.play(Write(loss_formula))
        self.play(Write(fi_formula))
        self.play(Write(pi_formula))
        self.play(FadeIn(legend))
        self.wait(3)

        self.play(FadeOut(VGroup(formula_title, loss_formula, fi_formula, pi_formula, legend)))

        # Step 4: Visual Representation
        visual_title = Text("Visualizing Device Groups", font_size=32).next_to(title, DOWN, buff=0.8)
        
        # Create devices
        device1 = Rectangle(height=2, width=5, color=GREEN).shift(LEFT*3)
        device2 = Rectangle(height=2, width=5, color=GREEN).shift(RIGHT*3)
        
        device1_label = Text("Device 1", font_size=20, color=GREEN).next_to(device1, UP)
        device2_label = Text("Device 2", font_size=20, color=GREEN).next_to(device2, UP)
        
        # Create experts in groups
        experts_d1 = VGroup(*[Circle(radius=0.2, color=BLUE) for _ in range(3)]).arrange(RIGHT, buff=0.3).move_to(device1)
        experts_d2 = VGroup(*[Circle(radius=0.2, color=BLUE) for _ in range(3)]).arrange(RIGHT, buff=0.3).move_to(device2)
        
        expert_labels = []
        for i, exp_group in enumerate([experts_d1, experts_d2]):
            for j, exp in enumerate(exp_group):
                label = Text(f"E{j+1+3*i}", font_size=14).next_to(exp, DOWN, buff=0.1)
                expert_labels.append(label)
        
        expert_labels_vg = VGroup(*expert_labels)
        
        self.play(Write(visual_title))
        self.play(Create(device1), Create(device2))
        self.play(Write(device1_label), Write(device2_label))
        self.play(FadeIn(experts_d1), FadeIn(experts_d2))
        self.play(FadeIn(expert_labels_vg))
        self.wait(2)

        self.play(FadeOut(VGroup(visual_title, device1, device2, device1_label, device2_label, 
                                 experts_d1, experts_d2, expert_labels_vg)))

        # Step 5: Code Implementation
        code_title = Text("Implementation Approach", font_size=32).next_to(title, DOWN, buff=0.8)
        
        code_snippet = Text(
            "def device_balance_loss(expert_loads, device_groups):\n"
            "    total_loss = 0\n"
            "    for group in device_groups:\n"
            "        avg_freq = mean([expert_loads[i] for i in group])\n"
            "        total_prob = sum([expert_probs[i] for i in group])\n"
            "        total_loss += avg_freq * total_prob\n"
            "    return alpha2 * total_loss",
            font="Monospace",
            font_size=18
        ).next_to(code_title, DOWN, buff=0.5)
        
        note = Text(
            "Note: Actual implementation combines with\nexpert-level loss in training loop",
            font_size=20
        ).next_to(code_snippet, DOWN, buff=0.5)

        self.play(Write(code_title))
        self.play(FadeIn(code_snippet))
        self.play(Write(note))
        self.wait(3)

        # Final fadeout
        self.play(FadeOut(VGroup(title, code_title, code_snippet, note)))
        self.wait(1)