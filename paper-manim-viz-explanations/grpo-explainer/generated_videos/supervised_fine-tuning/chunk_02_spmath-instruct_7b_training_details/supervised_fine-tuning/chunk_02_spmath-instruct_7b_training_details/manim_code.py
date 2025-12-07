from manim import *

class ConceptScene(Scene):
    def construct(self):
        # Step 1: Title and Introduction
        title = Text(r"\spmath-Instruct 7B Training Details", font_size=36, weight=BOLD)
        title.to_edge(UP)
        
        intro = Text(
            "Instruction-tuned model for mathematical reasoning\n"
            "Built upon \spmath-Base with specialized training",
            font_size=24, line_spacing=1.2
        ).next_to(title, DOWN, buff=0.5)
        
        self.play(Write(title))
        self.play(FadeIn(intro))
        self.wait(2)
        
        # Clean up
        self.play(FadeOut(title), FadeOut(intro))
        
        # Step 2: Training Process Visualization
        process_title = Text("Training Process", font_size=32, weight=BOLD)
        process_title.to_edge(UP)
        
        # Data flow diagram
        data_blocks = VGroup(*[
            Rectangle(width=1.5, height=1.0, color=BLUE)
            for _ in range(4)
        ]).arrange(RIGHT, buff=0.3)
        
        concat_arrow = Arrow(
            start=data_blocks.get_right(),
            end=data_blocks.get_right() + RIGHT * 2,
            color=YELLOW
        )
        
        context_box = Rectangle(width=3.0, height=1.5, color=GREEN)
        context_label = Text("4K Context", font_size=20).move_to(context_box)
        
        context_group = VGroup(context_box, context_label).next_to(concat_arrow, RIGHT, buff=0.3)
        
        # Training parameters
        params = Text(
            "• 500 training steps\n"
            "• Batch size: 256\n"
            "• Learning rate: 5e-5",
            font_size=24, line_spacing=1.3
        ).to_edge(LEFT, buff=1.5)
        
        self.play(Write(process_title))
        self.play(LaggedStart(*[Create(block) for block in data_blocks], lag_ratio=0.3))
        self.play(GrowArrow(concat_arrow))
        self.play(Create(context_box), Write(context_label))
        self.play(FadeIn(params))
        self.wait(2)
        
        # Clean up
        self.play(FadeOut(process_title), FadeOut(data_blocks), FadeOut(concat_arrow),
                  FadeOut(context_group), FadeOut(params))
        
        # Step 3: Evaluation Benchmarks
        eval_title = Text("Evaluation Benchmarks", font_size=32, weight=BOLD)
        eval_title.to_edge(UP)
        
        # Benchmark categories
        english_bench = Text("English Benchmarks", font_size=24, color=BLUE)
        chinese_bench = Text("Chinese Benchmarks", font_size=24, color=RED)
        
        bench_group = VGroup(english_bench, chinese_bench).arrange(DOWN, buff=1.0).shift(LEFT*3)
        
        # Model comparison
        closed_source = Text("Closed-Source Models", font_size=20, color=PURPLE)
        open_source = Text("Open-Source Models", font_size=20, color=ORANGE)
        
        model_group = VGroup(closed_source, open_source).arrange(DOWN, buff=1.0).shift(RIGHT*3)
        
        separator = Line(UP*2, DOWN*2, color=GRAY).shift(RIGHT*0.5)
        
        self.play(Write(eval_title))
        self.play(FadeIn(bench_group))
        self.play(Create(separator))
        self.play(FadeIn(model_group))
        self.wait(2)
        
        # Clean up
        self.play(FadeOut(eval_title), FadeOut(bench_group), FadeOut(separator), FadeOut(model_group))
        
        # Step 4: Code Implementation
        code_title = Text("Configuration Code", font_size=32, weight=BOLD)
        code_title.to_edge(UP)
        
        # Code snippet
        code_snippet = Text(
            "class SFTConfig(TrainingArguments):\n"
            "    learning_rate: float = field(\n"
            "        default=2e-5,\n"
            "        metadata={\"help\": \"Initial learning rate\"}\n"
            "    )\n"
            "    # Other training parameters...",
            font="Monospace",
            font_size=20,
            line_spacing=1.2
        ).shift(DOWN*0.5)
        
        note = Text(
            "Note: Paper mentions 5e-5 learning rate\n"
            "Default config shows 2e-5 (can be overridden)",
            font_size=20,
            color=YELLOW
        ).next_to(code_snippet, DOWN, buff=0.5)
        
        self.play(Write(code_title))
        self.play(Write(code_snippet))
        self.play(FadeIn(note))
        self.wait(2)
        
        # Final cleanup
        self.play(FadeOut(code_title), FadeOut(code_snippet), FadeOut(note))
        
        # Conclusion
        conclusion = Text(
            "\spmath-Instruct 7B:\n"
            "• Specialized mathematical instruction tuning\n"
            "• Evaluated on multilingual benchmarks\n"
            "• Configurable training parameters",
            font_size=28, line_spacing=1.3
        )
        
        self.play(Write(conclusion))
        self.wait(2)