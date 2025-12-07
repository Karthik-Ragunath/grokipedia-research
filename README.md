# Grokipedia For Research

> *Democratizing cutting-edge AI research for the world — one (actually many) animated explanation(s) at a time.*

---

## 🚀 Transform Any Research Paper Into Engaging Video Explanations

**Now you can convert any research paper into beautiful, animated explainer videos—ready to publish on Grokipedia (Research).**

Simply provide:
1. 📄 A LaTeX paper
2. 💻 The associated codebase

And our pipeline automatically generates:
- 🎥 **Animated Manim videos** for each concept
- 📝 **Comprehensive summaries** with math, code, and intuition
- ❓ **FAQ explainers** answering common questions
- 🔗 **Theory-to-code mappings** showing how ideas become implementation

**Just research in, education out.**

---

## How We Built It

We built Grokipedia using a **multi-stage LLM-powered pipeline** that transforms raw research into educational content:

### 🧠 Large Coder Models at Every Stage

We leverage **state-of-the-art large coder models** with strong mathematical reasoning capabilities at each pipeline stage:

> *Examples: NVIDIA Nemotron Qwen3-Coder-480B, Qwen3-235B-A22B-Instruct, DeepSeek-Coder-V2, Grok-3, and similar frontier code models*

1. **LaTeX Parsing & Section Extraction** (`latex_section_extractor.py`)
   - Custom regex-based parser handles complex LaTeX commands (`\spmoe{}`, nested braces, figures)
   - Converts mathematical environments to Markdown-compatible format

2. **Semantic Chunking** (`semantic_chunker.py`)
   - Large coder model intelligently splits sections based on semantic meaning
   - Model understands paper structure: keeps formulas with explanations, groups related concepts
   - Outputs structured XML chunks with titles and content

3. **Theory-to-Code Alignment** (`code_chunk_aligner.py`)
   - Model analyzes paper chunks and codebases simultaneously
   - Identifies which code sections implement which concepts
   - Returns relevance scores (0-10) and precise line number ranges

4. **Manim Video Generation** (`chunk_video_generator.py`)
   - Model generates complete, executable Manim Python code
   - Includes retry mechanism—if Manim fails, model regenerates with error context
   - Produces 3Blue1Brown-style educational animations

5. **Summary Generation** (`chunk_summary_generator.py`)
   - Creates comprehensive markdown documents with:
     - Intuition & analogies
     - Mathematical deep dives
     - Code walkthroughs
     - Worked examples with expected outputs

### 🔧 Tech Stack
- **LLMs**: Large coder models with mathematical reasoning (e.g., Grok-3, NVIDIA Nemotron Qwen3-Coder-480B, Qwen3-235B-A22B-Instruct)
- **Animation**: Manim Community Edition v0.19.0
- **Languages**: Python, LaTeX, Markdown
- **Environment**: pyenv + virtual environments

---

## Challenges We Ran Into

### 🔗 Devising the Theory-to-Code Alignment Algorithm
- Mapping abstract paper concepts to concrete code implementations is non-trivial
- Papers describe ideas at a high level; code has implementation details, edge cases, and optimizations
- Multiple code sections may relate to one concept, or one function may implement multiple ideas
- **Solution**: Designed a multi-step LLM prompt that analyzes chunks and entire codebases together, returning relevance scores (0-10) and precise line ranges. Only sections scoring ≥5 are included.

### 🎬 Manim Code Generation is Hard
- LLM-generated Manim code often fails on first attempt (syntax errors, deprecated APIs, positioning issues)
- **Solution**: Built a retry mechanism that feeds error messages back to the LLM for self-correction (up to 5 retries)

### 📐 LaTeX Parsing Edge Cases
- Papers use custom commands (`\spmoe{}`, `\spmath{}`) and deeply nested structures
- Figure environments and tables caused `IndexError` exceptions
- **Solution**: Iteratively refined regex patterns and added robust error handling

### 🔗 API Key and Environment Management
- `.env` file paths broke when scripts moved between directories
- pyenv activation required for dependencies but wasn't persisting across commands
- **Solution**: Calculated relative paths from `__file__` and created activation one-liners

### 📊 Output Structure Complexity
- Needed consistent directory structures across chunks, videos, summaries
- Video assets (texts/, images/) needed proper copying after Manim generation
- **Solution**: Mirrored input structure in outputs with dedicated asset directories

### ⏱️ Long-Running Generation
- 14 videos × 3+ minutes each = hours of processing
- Some chunks would timeout or freeze mid-generation
- **Solution**: Individual chunk processing with loop-based fallback for stuck items

---

## Accomplishments We're Proud Of

### ✅ End-to-End Automation
From a LaTeX file and codebase to 27 animated videos and summaries—fully automated.

### ✅ Pipeline to convert any paper + code to explainer videos
For proof of concept, we attached intermediates and final outputs for two papers which we generated:
- **DeepSeek MoE**: 12 chunks + FAQ video + comprehensive summaries
- **GRPO (DeepSeek Math)**: 14 chunks across RL and SFT sections

### ✅ Theory-Code Bridge
Our aligner successfully maps abstract paper concepts to actual implementation code, with relevance scores averaging 8/10.

### ✅ Self-Healing Video Generation
The retry mechanism means even complex mathematical concepts eventually render—the LLM learns from Manim errors and fixes its own code.

### ✅ Educational Quality
Generated summaries include:
- Intuitive analogies for complex concepts
- Step-by-step mathematical derivations
- Actual code snippets with explanations
- Worked examples with expected outputs

---

## What We Learned

### 🎓 LLMs as Code Generators Need Guardrails
Raw LLM output rarely works first try for complex libraries like Manim. Retry loops with error feedback are essential.

### 🎓 Semantic Understanding Beats Rule-Based Parsing
Using LLMs for chunking produces far better results than heuristic approaches—they understand what concepts belong together.

### 🎓 Research Papers Have Hidden Structure
Papers aren't just linear text—they have implicit hierarchies, cross-references, and dependencies that LLMs can surface.

### 🎓 Animation is Powerful for Learning
Seeing formulas animate step-by-step creates "aha moments" that static text can't match.

### 🎓 The Gap Between Theory and Code is Bridgeable
With the right prompting, LLMs can identify how abstract math becomes concrete implementation.

---

## What's Next for Grokipedia - Research For Everyone

### 🧠 Custom RL-Trained Alignment Model
- Fine-tune a specialized model using reinforcement learning for theory-to-code alignment
- Train on curated dataset of paper-code pairs with human feedback on alignment quality
- Achieve higher precision in mapping abstract concepts to implementation details

### ⚡ Parallel Video Generation at Scale
- Implement distributed processing to generate multiple videos simultaneously
- Leverage GPU clusters for parallel Manim rendering
- Reduce end-to-end processing time from hours to minutes

### 🌍 Scale to Every Research Paper
- Build infrastructure to process papers from arXiv, ACL, NeurIPS, ICML automatically
- Create a continuously growing library of animated explanations
- Enable real-time generation as new papers are published

### 🎯 The Ultimate Goal
**Make every research paper as accessible as a `3Blue1Brown` video / something of quality of `StatQuesy With Josh Starmer` series to democratize research to more people.**

---

## Why?

### 🌍 Research Belongs to Everyone

Research is one of humanity's most powerful engines of progress—yet it remains locked behind walls of jargon, paywalls, and assumed expertise. We believe there's no better place to democratize research than **XAI's truth-seeking platform, Grok**. 

Grokipedia is built on the principle that understanding should flow freely. By transforming dense academic papers into engaging, accessible content, we're not just explaining research—we're **unlocking potential** in every curious mind that encounters it.

---

### 🎬 Breaking the Jargon Barrier with Visual Learning

Research papers are notoriously difficult to parse. Terms like *"Mixture of Experts routing"*, *"Rotary Positional Embeddings"*, and *"auxiliary load balancing losses"* can make brilliant ideas feel impenetrable.

**Our solution: Transform complexity into clarity through engaging video explanations.**

We leverage **Manim**—the mathematical animation engine created by 3Blue1Brown—to create beautiful, step-by-step animated explanations that:

- 📊 **Visualize abstract concepts** like attention mechanisms and expert routing
- 🔢 **Animate mathematical derivations** so formulas come alive
- 🧩 **Build intuition progressively** from simple analogies to full technical depth
- 💡 **Show code in context** with highlighted implementations

**Plus comprehensive text summaries** that include:
- Core intuitions and analogies
- Mathematical deep dives with LaTeX equations
- Code walkthroughs with worked examples
- Key takeaways and common pitfalls

*Example: Our DeepSeek MoE explainer generates 12+ animated FAQ videos and detailed summaries covering everything from forward pass architecture to RoPE embeddings.*

---

### 🚀 From Theory to Code: The Elite Understanding

What separates a **good understanding** of research from an **elite understanding**?

> *The ability to not just comprehend the theory, but to translate it into working code.*

Most explanations stop at the paper. We go further.

**Our pipeline bridges theory and implementation:**

```
📄 Research Paper (LaTeX)
    ↓
🔍 Semantic Chunking (AI-powered section extraction)
    ↓
🔗 Code Alignment (Maps paper concepts → actual implementation)
    ↓
🎥 Manim Video Generation (Animated explanations with code)
    ↓
📝 Comprehensive Summaries (Math + Code + Intuition)
    ↓
❓ FAQ Deep Dives (Real questions, animated answers)
```

**What we deliver for each paper:**

| Output | Description |
|--------|-------------|
| **Extracted Sections** | Clean markdown from LaTeX papers |
| **Semantic Chunks** | AI-split coherent concept units |
| **Code-Aligned Chunks** | Paper concepts linked to actual code |
| **Animated Videos** | Manim visualizations per concept |
| **Chunk Summaries** | Deep educational markdown documents |
| **FAQ Videos** | Animated answers to common questions |
| **FAQ Summaries** | Comprehensive Q&A reference guides |

---

### 🛠️ The Technical Pipeline

Our codebase is a fully automated research-to-video pipeline:

```python
# 1. Extract sections from LaTeX papers
latex_section_extractor.py  →  introduction.md, architecture.md, ...

# 2. Semantically chunk content using LLMs
semantic_chunker.py  →  chunk_01_overview.md, chunk_02_mechanism.md, ...

# 3. Align paper chunks with code implementations
code_chunk_aligner.py  →  chunks-with-code/*.md (theory + implementation)

# 4. Generate Manim animations for each concept
chunk_video_generator.py  →  video.mp4, manim_code.py, texts/, images/

# 5. Create detailed educational summaries
chunk_summary_generator.py  →  *_summary.md (intuition + math + examples)

# 6. Process FAQs into videos and summaries
faq_video_generator.py  →  faq_combined_video.mp4
faq_summary_generator.py  →  comprehensive_summary.md
```

**Powered by:**
- 🤖 **Large Language Models** with strong mathematical reasoning capabilities
- 🎨 **Manim Community Edition** for mathematical animations
- 🔗 **LLM-powered alignment** to connect theory and code

---

### 💡 See It In Action

We've built **two complete proof-of-concepts** demonstrating the full pipeline:

---

#### 📘 **DeepSeek MoE Explainer**

From the DeepSeek MoE research paper + codebase:
- ✅ 12 semantic chunks with code alignment
- ✅ 12 animated Manim videos explaining each concept
- ✅ 12 comprehensive chunk summaries with math + code
- ✅ 1 combined FAQ video (all 12 questions animated)
- ✅ 1 comprehensive FAQ summary (34KB educational document)

**Topics covered:**
- Mixture of Experts architecture
- Fine-grained expert segmentation
- Shared expert isolation
- Load balancing mechanisms
- RoPE embeddings and context extension
- Forward pass implementation details

---

#### 📗 **GRPO (Group Relative Policy Optimization) Explainer**

From the DeepSeek Math paper + TRL library codebase:
- ✅ 14 semantic chunks across 2 major sections
- ✅ 14 animated Manim videos (9 RL + 5 SFT)
- ✅ 14 comprehensive chunk summaries with math + code
- ✅ Code aligned from `grpo_trainer.py`, `grpo_config.py`, and `sft_trainer.py`

**Sections covered:**

| Section | Chunks | Topics |
|---------|--------|--------|
| **Reinforcement Learning** | 9 | PPO → GRPO transition, algorithm design, outcome/process supervision, iterative RL, training setup |
| **Supervised Fine-Tuning** | 5 | SFT data curation, training details, benchmark results, CoT & tool-integrated reasoning |

**Key concepts explained:**
- From PPO to GRPO: Why eliminate the value function?
- Group-relative advantage estimation
- Outcome vs. Process supervision
- KL divergence regularization
- Iterative reward model updates
- Chain-of-thought reasoning performance

---

### 📊 Deliverables Summary

| Paper | Chunks | Videos | Summaries | Code Files Aligned |
|-------|--------|--------|-----------|-------------------|
| **DeepSeek MoE** | 12 | 12 + 1 FAQ | 12 + 1 FAQ | `modeling_deepseek.py` |
| **DeepSeek Math (GRPO)** | 14 | 14 | 14 | `grpo_trainer.py`, `grpo_config.py`, `sft_trainer.py`, `modelling_llama.py` |
| **Total** | **26** | **27** | **27** | **5 files** |

---

### 🎯 The Vision

**Grokipedia is a platform where anyone can:**

- 📤 **Upload papers and code** to create educational collections
- 🎬 **Auto-generate video series** explaining complex research
- 🌐 **Contribute to a growing library** of research explanations
- 🎓 **Learn cutting-edge AI** from first principles to implementation
- 🔗 **Bridge theory and code** with aligned, animated walkthroughs

**Post to Grokipedia. Democratize knowledge. Change the world.**

---

## Architecture

![Grokipedia Research Architecture](Grokipedia-Research.drawio.png)

---

## Sample Project Structure

```
grokipedia-research/
├── paper-manim-viz-explanations/
│   ├── ai/                           # Core pipeline scripts
│   │   ├── latex_section_extractor.py
│   │   ├── semantic_chunker.py
│   │   ├── aligner/
│   │   │   └── code_chunk_aligner.py
│   │   ├── manim_video_generator/
│   │   │   └── chunk_video_generator.py
│   │   ├── manim_video_generator_faqs/
│   │   │   └── faq_video_generator.py
│   │   └── summary_generator/
│   │       ├── chunk_summary_generator.py
│   │       └── faq_summary_generator.py
│   │
│   ├── deepseek-moe-explainer/       # MoE Paper Outputs
│   │   ├── extracted_sections/
│   │   ├── chunks/
│   │   ├── chunks-with-code/
│   │   ├── generated_videos/
│   │   ├── chunk-summary/
│   │   └── faq-summary/
│   │
│   └── grpo-explainer/               # GRPO Paper Outputs
│       ├── grpo-paper/               # LaTeX source
│       ├── grpo-code/                # TRL library code
│       ├── extracted_sections/
│       ├── chunks/
│       ├── chunks-with-code/
│       ├── generated_videos/
│       └── chunk-summary/
│
├── requirements.txt
└── README.md
```
