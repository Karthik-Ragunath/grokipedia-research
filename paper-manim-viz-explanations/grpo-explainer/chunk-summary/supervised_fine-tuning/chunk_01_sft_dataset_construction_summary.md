# SFT Dataset Construction for Mathematical Reasoning

## 1. Intuition & Core Idea

Think of teaching a student advanced mathematics - you wouldn't just give them final answers. Instead, you'd show them the complete problem-solving process: how to think through the problem, which steps to take, and why each step makes sense.

This is exactly what **Supervised Fine-Tuning (SFT) Dataset Construction** aims to do for AI models. Rather than training models on raw input-output pairs, we curate datasets that teach models *how* to solve complex mathematical problems through structured reasoning processes.

The core insight is that mathematical problem-solving involves multiple thinking patterns:
- **Chain-of-Thought (CoT)**: Step-by-step logical reasoning like "First I'll identify what's given, then I'll apply the quadratic formula..."
- **Program-of-Thought (PoT)**: Breaking problems into algorithmic steps that could be coded
- **Tool-integrated reasoning**: Using external tools (like calculators or symbolic solvers) at appropriate moments

By mixing these different reasoning styles across thousands of problems, we're essentially creating a comprehensive "mathematics textbook" that teaches the model not just what the answers are, but how to think like a mathematician.

## 2. Technical Deep Dive

While there isn't explicit mathematical formulation in this section, the technical architecture can be understood through data composition principles:

### Dataset Composition Framework

The SFT dataset construction follows a multi-dimensional coverage strategy:

$$\text{Dataset} = \sum_{i=1}^{n} (\text{Problem}_i, \text{Solution}_{\text{format}_j})$$

Where:
- $\text{Problem}_i$: Individual mathematical problems with varying complexity
- $\text{Solution}_{\text{format}_j}$: Solution expressed in one of three formats:
  - $\text{CoT}$: Natural language step-by-step reasoning
  - $\text{PoT}$: Programmatic solution approach  
  - $\text{Tool-integrated}$: Solutions using external computational tools

### Coverage Dimensions

The dataset spans multiple axes of diversity:

$$\text{Coverage} = \text{Languages} \times \text{Mathematical Fields} \times \text{Complexity Levels} \times \text{Reasoning Formats}$$

Where:
- **Languages**: English, Chinese
- **Fields**: Algebra, Probability, Number Theory, Calculus, Geometry, etc.
- **Complexity**: K-12 level through advanced undergraduate
- **Formats**: CoT, PoT, Tool-integrated

### Quality Control Metrics

Dataset quality is measured by:
$$\text{Quality Score} = \alpha \cdot \text{Accuracy} + \beta \cdot \text{Diversity} + \gamma \cdot \text{Completeness}$$

With weights $\alpha, \beta, \gamma$ balancing different quality aspects.

## 3. Code Implementation Walkthrough

Since no direct code was provided for the dataset construction itself, here's a representative implementation of how such a dataset curation pipeline might be structured:

```python
class MathDatasetConstructor:
    def __init__(self):
        self.datasets = {}
        self.annotation_formats = ['cot', 'pot', 'tool_integrated']
        
    def collect_english_datasets(self):
        """Collect and integrate English mathematical datasets"""
        # GSM8K with tool-integrated annotations
        gsm8k_data = self.annotate_gsm8k_with_tools()
        
        # MATH problems with tool integration
        math_data = self.annotate_math_with_tools()
        
        # MathInstruct subset (already contains CoT/PoT)
        mathinstruct_data = self.load_mathinstruct_subset()
        
        # Lila-OOD training set
        lila_data = self.load_lila_ood_training()
        
        return self.merge_datasets([
            gsm8k_data, math_data, mathinstruct_data, lila_data
        ])
    
    def collect_chinese_datasets(self):
        """Collect Chinese K-12 mathematical problems"""
        chinese_problems = []
        
        # 76 sub-topics like linear equations, geometry, etc.
        topics = self.get_chinese_math_topics()  # Returns 76 topics
        
        for topic in topics:
            problems = self.collect_problems_for_topic(topic)
            # Annotate with both CoT and tool-integrated reasoning
            annotated_problems = self.annotate_chinese_problems(problems)
            chinese_problems.extend(annotated_problems)
            
        return chinese_problems
    
    def annotate_problem(self, problem, format_type):
        """Annotate a single problem with specified reasoning format"""
        if format_type == 'cot':
            return self.generate_chain_of_thought(problem)
        elif format_type == 'pot':
            return self.generate_program_of_thought(problem)
        elif format_type == 'tool_integrated':
            return self.generate_tool_integrated_solution(problem)
    
    def construct_final_dataset(self):
        """Main method to construct the complete SFT dataset"""
        english_data = self.collect_english_datasets()
        chinese_data = self.collect_chinese_datasets()
        
        # Combine all data
        full_dataset = english_data + chinese_data
        
        # Validate dataset size (should be ~776K examples)
        assert len(full_dataset) >= 700000, "Dataset size insufficient"
        
        return {
            'total_examples': len(full_dataset),
            'english_examples': len(english_data),
            'chinese_examples': len(chinese_data),
            'data': full_dataset
        }
```

Key implementation considerations:
1. **Multi-source Integration**: Combines existing datasets with new annotations
2. **Format Diversity**: Ensures each problem has multiple reasoning path annotations
3. **Language Coverage**: Handles both English and Chinese mathematical conventions
4. **Scalability**: Designed to handle the large scale (776K examples)

## 4. Worked Example

Let's walk through constructing a small sample from this dataset:

```python
# Example problem construction
constructor = MathDatasetConstructor()

# Sample English problem from GSM8K
english_problem = {
    'question': "A train travels 120 miles in 2 hours. What is its average speed?",
    'answer': 60,
    'domain': 'algebra',
    'difficulty': 'elementary'
}

# Generate different reasoning formats
cot_solution = constructor.annotate_problem(english_problem, 'cot')
# Output: "To find average speed, divide distance by time. 
#          Distance = 120 miles, Time = 2 hours. 
#          Speed = 120 ÷ 2 = 60 mph."

pot_solution = constructor.annotate_problem(english_problem, 'pot')
# Output: "def calculate_speed(distance, time):
#          return distance / time
#          speed = calculate_speed(120, 2)
#          print(speed)  # 60"

tool_solution = constructor.annotate_problem(english_problem, 'tool_integrated')
# Output: "Using calculator: 120 miles ÷ 2 hours = 60 mph"

# Sample Chinese problem
chinese_problem = {
    'question': "解方程：2x + 5 = 15",  # Solve equation: 2x + 5 = 15
    'answer': "x = 5",
    'domain': 'linear_equations',
    'difficulty': 'middle_school'
}

# Chinese problem gets both CoT and tool-integrated annotations
chinese_cot = constructor.annotate_problem(chinese_problem, 'cot')
chinese_tool = constructor.annotate_problem(chinese_problem, 'tool_integrated')

# Final dataset entry would look like:
dataset_entry = {
    'problem_id': 'math_001',
    'language': 'english',
    'question': english_problem['question'],
    'solutions': {
        'cot': cot_solution,
        'pot': pot_solution,
        'tool_integrated': tool_solution
    },
    'ground_truth': english_problem['answer'],
    'metadata': {
        'domain': 'algebra',
        'difficulty': 'elementary',
        'source': 'GSM8K'
    }
}
```

This demonstrates how a single problem gets transformed into a richly annotated training example that teaches the model multiple ways to approach mathematical reasoning.

## 5. Mathematical Derivation

While there's no explicit mathematical derivation in the original content, we can formalize the dataset construction optimization problem:

### Optimization Objective

The goal is to maximize learning effectiveness while maintaining dataset diversity:

$$\max_{\mathcal{D}} \mathbb{E}_{(x,y) \sim \mathcal{D}}[\text{LearningGain}(x,y)]$$

Subject to:
$$\text{Coverage}(\mathcal{D}) \geq \tau_{\text{coverage}}$$
$$|\mathcal{D}| = N_{\text{target}}$$

Where:
- $\mathcal{D}$: The constructed dataset
- $\text{LearningGain}(x,y)$: Expected improvement in model performance from example $(x,y)$
- $\text{Coverage}(\mathcal{D})$: Diversity metric across problem types, languages, and reasoning formats
- $\tau_{\text{coverage}}$: Minimum required coverage threshold
- $N_{\text{target}}$: Target dataset size (776K in this case)

### Diversity Maximization

To ensure adequate coverage, we can frame this as:

$$\text{Diversity}(\mathcal{D}) = \sum_{i=1}^{k} w_i \cdot H(\mathcal{D}_{\text{attribute}_i})$$

Where:
- $H(\cdot)$ represents entropy (diversity measure)
- $\mathcal{D}_{\text{attribute}_i}$ is the distribution of attribute $i$ in dataset
- $w_i$ are weights for different attributes (languages, domains, formats, difficulty)

## 6. Key Takeaways

### Main Insights:
1. **Multi-modal Reasoning Training**: Teaching models multiple ways to solve problems (natural language, programming, tool usage) creates more robust mathematical reasoning capabilities
2. **Cross-linguistic Coverage**: Including both English and Chinese ensures broader applicability and captures different mathematical pedagogical traditions
3. **Structured Problem-Solving**: Chain-of-thought annotation explicitly teaches the model sequential reasoning processes rather than just memorizing answers

### Common Pitfalls:
- **Annotation Consistency**: Maintaining quality across different annotators and reasoning formats
- **Domain Balance**: Ensuring adequate representation across mathematical subfields
- **Cultural Adaptation**: Mathematical problem presentation varies significantly between cultures

### Best Practices:
1. **Multiple Solution Paths**: Each problem should have annotations in multiple reasoning formats
2. **Progressive Complexity**: Include problems ranging from elementary to advanced levels
3. **Source Verification**: Validate that source datasets maintain their quality when combined

### Further Reading:
- **Chain-of-Thought Prompting**: Wei et al. (2022) - Original CoT work
- **Program-of-Thought**: Gao et al. (2023) - Programming-based reasoning
- **Tool-Augmented Reasoning**: Qin et al. (2023) - Integration with external tools
- **Mathematical Dataset Curation**: Hendrycks et al. (2021) - GSM8K and MATH datasets

This SFT dataset construction approach represents a sophisticated methodology for creating comprehensive mathematical reasoning training data that goes beyond simple question-answer pairs to teach models genuine problem-solving strategies.