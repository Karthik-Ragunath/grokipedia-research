from manim import *

class ConceptScene(Scene):
    def construct(self):
        # Step 1: Title and Introduction
        title = Text("SFT Dataset Construction", font_size=40).to_edge(UP)
        description = Text(
            "Building a math instruction-tuning dataset\nin English and Chinese",
            font_size=24,
        ).next_to(title, DOWN, buff=0.5)

        self.play(Write(title))
        self.play(FadeIn(description))
        self.wait(1)

        self.play(FadeOut(title), FadeOut(description))

        # Step 2: Dataset Overview
        overview_title = Text("Dataset Composition", font_size=36).to_edge(UP)
        self.play(Write(overview_title))

        languages = VGroup(
            Text("English", color=BLUE, font_size=30),
            Text("Chinese", color=GREEN, font_size=30),
        ).arrange(RIGHT, buff=2).next_to(overview_title, DOWN, buff=1)

        english_desc = Text(
            "GSM8K, MATH, MathInstruct,\nLila-OOD with CoT/PoT/TIR",
            font_size=20,
        ).next_to(languages[0], DOWN, buff=0.5)

        chinese_desc = Text(
            "K-12 problems across 76 topics\nwith CoT and TIR formats",
            font_size=20,
        ).next_to(languages[1], DOWN, buff=0.5)

        total_examples = MathTex("776K\\ \\text{training examples}", font_size=30).to_edge(DOWN)

        self.play(FadeIn(languages), Write(english_desc), Write(chinese_desc))
        self.play(Write(total_examples))
        self.wait(2)

        self.play(
            FadeOut(overview_title),
            FadeOut(languages),
            FadeOut(english_desc),
            FadeOut(chinese_desc),
            FadeOut(total_examples),
        )

        # Step 3: Annotation Formats
        format_title = Text("Annotation Formats", font_size=36).to_edge(UP)
        self.play(Write(format_title))

        formats = VGroup(
            Text("Chain-of-Thought (CoT)", color=YELLOW, font_size=24),
            Text("Program-of-Thought (PoT)", color=ORANGE, font_size=24),
            Text("Tool-Integrated Reasoning (TIR)", color=PURPLE, font_size=24),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5).next_to(format_title, DOWN, buff=1)

        cot_example = MathTex(
            "\\text{Q: } 2x + 3 = 7 \\quad \\text{A: } x = 2",
            font_size=24,
        ).next_to(formats[0], RIGHT, buff=1)

        pot_example = Text(
            "def solve():\n    return (7 - 3) / 2",
            font="Monospace",
            font_size=20,
        ).next_to(formats[1], RIGHT, buff=1)

        tir_example = Text(
            "[Calculator] 7 - 3 = 4\n[Divide] 4 / 2 = 2",
            font="Monospace",
            font_size=20,
        ).next_to(formats[2], RIGHT, buff=1)

        self.play(FadeIn(formats), Write(cot_example), Write(pot_example), Write(tir_example))
        self.wait(2)

        self.play(
            FadeOut(format_title),
            FadeOut(formats),
            FadeOut(cot_example),
            FadeOut(pot_example),
            FadeOut(tir_example),
        )

        # Step 4: Visual Representation
        visual_title = Text("Visual Summary", font_size=36).to_edge(UP)
        self.play(Write(visual_title))

        # Diagram Elements
        eng_box = Rectangle(height=2, width=3, color=BLUE).shift(LEFT * 3)
        chn_box = Rectangle(height=2, width=3, color=GREEN).shift(RIGHT * 3)
        eng_label = Text("English", font_size=24).move_to(eng_box)
        chn_label = Text("Chinese", font_size=24).move_to(chn_box)

        arrow = Arrow(LEFT, RIGHT, color=WHITE).shift(UP * 0.5)
        annotation_label = Text("Annotated with\nCoT/PoT/TIR", font_size=20).next_to(arrow, UP, buff=0.2)

        combined_text = Text("776K Training Examples", font_size=24).to_edge(DOWN)

        self.play(Create(eng_box), Write(eng_label), Create(chn_box), Write(chn_label))
        self.play(GrowArrow(arrow), Write(annotation_label))
        self.play(Write(combined_text))
        self.wait(2)

        self.play(
            FadeOut(visual_title),
            FadeOut(eng_box),
            FadeOut(chn_box),
            FadeOut(eng_label),
            FadeOut(chn_label),
            FadeOut(arrow),
            FadeOut(annotation_label),
            FadeOut(combined_text),
        )

        # Final message
        final_message = Text("Ready for Instruction Tuning!", font_size=36)
        self.play(Write(final_message))
        self.wait(1)