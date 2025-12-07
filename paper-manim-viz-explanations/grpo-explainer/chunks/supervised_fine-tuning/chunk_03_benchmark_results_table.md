# Benchmark Results Table

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