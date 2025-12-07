# Chain-of-Thought Reasoning Performance Analysis

As shown in Table (ref: tab:sft_rl_math), under the evaluation setting where tool use is disallowed, \spmath-Instruct 7B demonstrates strong performance of step-by-step reasoning.
Notably, on the competition-level MATH dataset, our model surpasses all open-source models and the majority of proprietary models (e.g., Inflection-2 and Gemini Pro) by at least 9% absolute.
This is true even for models that are substantially larger (e.g., Qwen 72B) or have been specifically enhanced through math-focused reinforcement learning (e.g., WizardMath-v1.1 7B).
While \spmath-Instruct rivals the Chinese proprietary models GLM-4 and Baichuan-3 on MATH, it still underperforms GPT-4 and Gemini Ultra.