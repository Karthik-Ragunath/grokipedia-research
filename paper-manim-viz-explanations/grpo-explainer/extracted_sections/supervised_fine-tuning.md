# Supervised Fine-Tuning

## SFT Data Curation

We construct a mathematical instruction-tuning dataset covering English and Chinese problems from different mathematical fields and of varying complexity levels:
problems are paired with solutions in chain-of-thought (CoT) [cot], program-of-thought (PoT) [pot,pal], and tool-integrated reasoning format [tora].
The total number of training examples is 776K.
[topsep=0pt]
    
- **English mathematical datasets**:
    We annotate GSM8K and MATH problems with tool-integrated solutions, and adopt a subset of MathInstruct [MathInstruct] along with the training set of Lila-OOD [lila] where problems are solved with CoT or PoT.
    Our English collection covers diverse fields of mathematics, e.g., algebra, probability, number theory, calculus, and geometry.
    
- **Chinese mathematical datasets**:
    We collect Chinese K-12 mathematical problems spanning 76 sub-topics such as linear equations, with solutions annotated in both CoT and tool-integrated reasoning format.

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

\begingroup
\setlength{\tabcolsep}{3pt} % Default value: 6pt
\renewcommand{\arraystretch}{1} % Default value: 1
\begin{table*}[t!]
    
\adjustbox{max width=\textwidth}{
\begin{tabular}{llcccc} 
\toprule
\multicolumn{1}{l}{\multirow{2}{*}{Model}}  & \multicolumn{1}{l}{\multirow{2}{*}{Size}}                & \multicolumn{2}{c}{English Benchmarks}           & \multicolumn{2}{c}{Chinese Benchmarks}                                                        \\ 
\cmidrule(r){3-4} \cmidrule(r){5-6} 
& & GSM8K                          & MATH            & MGSM-zh         & CMATH   \\
\midrule
\midrule
\multicolumn{6}{c}{**Chain-of-Thought Reasoning**}                                                                                                                         \\ 
\midrule
\multicolumn{6}{c}{Closed-Source Model}                                                                                                                                         \\ 
\midrule
Gemini Ultra  &     -           & \textcolor{gray}{94.4%} & 53.2%               &        -              &        -                                                                    \\
GPT-4 &       -                & 92.0%                         & 52.9%               &         -             &   86.0%                                                                \\
Inflection-2 &     -            & 81.4%                         & 34.8%               &            -          &       -                                                               \\
GPT-3.5 &    -                & 80.8%                         & 34.1%               &           -           &     73.8%                                                         \\
Gemini Pro &     -             & \textcolor{gray}{86.5%} & 32.6%               &              -        &      -                                                                   \\
Grok-1 &          -            & 62.9%                         & 23.9%               &           -           &       -                                                                 \\
\midrule
Baichuan-3 &      -            & 88.2%                         & 49.2%               &            -          &         -                                                            \\
GLM-4 &           -            & 87.6%                         & 47.9%               &            -          &       -                                                          \\
\midrule
\multicolumn{6}{c}{Open-Source Model}                                                                                                                                           \\ 
\midrule
InternLM2-Math & 20B           & 82.6%                         & 37.7%          &           -      &             -                                                     \\
Qwen & 72B                     & 78.9%                         & 35.2%          &          -       &       -                                                         \\
Math-Shepherd-Mistral & 7B & 84.1%                         & 33.0%          &             -    &               -                                                     \\
WizardMath-v1.1 & 7B           & 83.2%                         & 33.0%          &            -     &               -                                                  \\
DeepSeek-LLM-Chat & 67B        & 84.1%                         & 32.6%               & 74.0%               & 80.3%                                                            \\
MetaMath & 70B                 & 82.3%                         & 26.6%          & 66.4%          & 70.9%                                                             \\
SeaLLM-v2 & 7B & 78.2% & 27.5% & 64.8% & - \\
ChatGLM3 & 6B                  & 72.3%                         & 25.7%          &         -        &          -                                                      \\
WizardMath-v1.0 & 70B          & 81.6%                         & 22.7%          & 64.8%          & 65.4%                                                            \\ 
\midrule
**\spmath-Instruct** & 7B     & 82.9%                         & 46.8%          & 73.2%          & 84.6%                                                     \\
**\spmath-RL** & 7B           & **88.2%**                & **51.7%** & **79.6%** & **88.8%**                                        \\
\midrule
\midrule
\multicolumn{6}{c}{**Tool-Integrated Reasoning**}                                                                                                                          \\ 
\midrule
\multicolumn{6}{c}{Closed-Source Model}                                                                                                                                         \\ 
\midrule
GPT-4 Code Interpreter &   -   & 97.0%                         & 69.7%          &            -     &          -                                                      \\ 
\midrule
\multicolumn{6}{c}{Open-Source Model}                                                                                                                                           \\ 
\midrule
InternLM2-Math & 20B           & 80.7%                         & 54.3%          &          -       &      -                                                               \\
DeepSeek-LLM-Chat & 67B        & 86.7%                & 51.1%          & 76.4%          & 85.4%                                                            \\
ToRA & 34B                     & 80.7%                         & 50.8%          & 41.2%          & 53.4%                                                              \\
MAmmoTH & 70B                  & 76.9%                         & 41.8%          &          -       &            -                                                         \\ 
\midrule
**\spmath-Instruct** & 7B     & 83.7%                         & 57.4%          & 72.0%          & 84.3%                                                                \\
**\spmath-RL** & 7B           & **86.7%**                         & **58.8%** & **78.4%** & **87.6%**                                                       \\
\bottomrule
\end{tabular}
}
    \caption{
   Performance of Open- and Closed-Source models with both Chain-of-Thought and Tool-Integrated Reasoning on English and Chinese Benchmarks.
   Scores in \textcolor{gray}{gray} denote majority votes with 32 candidates; The others are Top1 scores.
   \spmath-RL 7B beats all open-source models from 7B to 70B, as well as the majority of closed-source models. Although \spmath-RL 7B is only further trained on chain-of-thought-format instruction tuning data of GSM8K and MATH, it improves over \spmath-Instruct 7B on all benchmarks.
   }
    
\end{table*}
\endgroup

As shown in Table (ref: tab:sft_rl_math), under the evaluation setting where tool use is disallowed, \spmath-Instruct 7B demonstrates strong performance of step-by-step reasoning.
Notably, on the competition-level MATH dataset, our model surpasses all open-source models and the majority of proprietary models (e.g., Inflection-2 and Gemini Pro) by at least 9% absolute.
This is true even for models that are substantially larger (e.g., Qwen 72B) or have been specifically enhanced through math-focused reinforcement learning (e.g., WizardMath-v1.1 7B).
While \spmath-Instruct rivals the Chinese proprietary models GLM-4 and Baichuan-3 on MATH, it still underperforms GPT-4 and Gemini Ultra.

Under the evaluation setting where models are allowed to integrate natural language reasoning and program-based tool use for problem solving, \spmath-Instruct 7B approaches an accuracy of 60% on MATH, surpassing all existing open-source models.
On the other benchmarks, our model is competitive with DeepSeek-LLM-Chat 67B, the prior state-of-the-art that is 10 times larger.