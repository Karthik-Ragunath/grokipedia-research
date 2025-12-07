# \spmath-Instruct 7B Training Details

## Training and Evaluating \spmath-Instruct 7B

In this section, we introduce \spmath-Instruct 7B which undergoes mathematical instruction tuning based on \spmath-Base.
Training examples are randomly concatenated until reaching a maximum context length of 4K tokens.
We train the model for 500 steps with a batch size of 256 and a constant learning rate of 5e-5.

We evaluate models' mathematical performance both without and with tool use, on 4 quantitative reasoning benchmarks in English and Chinese.
We benchmark our model against the leading models of the time:
[topsep=0pt]
    
- **Closed-source models** include:
    (1) the GPT family among which GPT-4 [gpt4] and GPT-4 Code Interpreter  (footnote: \url{https://openai.com/blog/chatgpt-plugins\#\#code-interpreter)} are the most capable ones,
    (2) Gemini Ultra and Pro [gemini],
    (3) Inflection-2 [inflection-2],
    (4) Grok-1  (footnote: \url{https://x.ai/model-card)},
    as well as models recently released by Chinese companies including
    (5) Baichuan-3  (footnote: \url{https://www.baichuan-ai.com)},
    (6) the latest GLM-4  (footnote: \url{https://open.bigmodel.cn/dev/api\#glm-4)}
    % and ChatGLM-Turbo  (footnote: \url{https://open.bigmodel.cn/dev/api\#glm-3-turbo)}
    from the GLM family [glm].
    % and (7) Ernie-bot-4.0  (footnote: \url{https://yiyan.baidu.com/)}.
    These models are for general purposes, most of which have undergone a series of alignment procedures.
    
- **Open-source models** include:
    general models like (1) DeepSeek-LLM-Chat 67B [deepseek-llm], (2) Qwen 72B [qwen], (3) SeaLLM-v2 7B [seallm], and (4) ChatGLM3 6B [chatglm3],
    as well as models with enhancements in mathematics including
    (5) InternLM2-Math 20B  (footnote: \url{https://github.com/InternLM/InternLM-Math)} which builds on InternLM2 and underwent math training followed by instruction tuning,
    (6) Math-Shepherd-Mistral 7B which applys PPO training [schulman2017proximal] to Mistral 7B [mistral] with a process-supervised reward model,
    (7) the WizardMath series [wizardmath] which improves mathematical reasoning in Mistral 7B and Llama-2 70B [llama2] using evolve-instruct (i.e., a version of instruction tuning that uses AI-evolved instructions) and PPO training with training problems primarily sourced from GSM8K and MATH,
    (8) MetaMath 70B [metamath] which is Llama-2 70B fine-tuned on an augmented version of GSM8K and MATH,
    (9) ToRA 34B [tora] which is CodeLlama 34B fine-tuned to do tool-integrated mathematical reasoning,
    (10) MAmmoTH 70B [MathInstruct] which is Llama-2 70B instruction-tuned on MathInstruct.