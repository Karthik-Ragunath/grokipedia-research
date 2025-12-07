# Scaling to 16B and Beyond

Leveraging our architecture, we subsequently scale up the model parameters to 16B and train DeepSeekMoE 16B on a large-scale corpus with 2T tokens. 
Evaluation results reveal that with only about 40% of computations, DeepSeekMoE 16B achieves comparable performance with DeepSeek 7B[deepseek_llm], a dense model trained on the same 2T corpus. 
We also compare DeepSeekMoE with open source models and the evaluations demonstrate that DeepSeekMoE 16B consistently outperforms models with a similar number of activated parameters by a large margin, and achieves comparable performance with LLaMA2 7B[llama2], which has approximately 2.5 times the activated parameters. 
Figure (ref: fig:openllm) demonstrates the evaluation results on the Open LLM Leaderboard (footnote: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard). 
Additionally, we conduct supervised fine-tuning (SFT) for alignment, transforming the model into a chat model.
Evaluation results show that DeepSeekMoE Chat 16B also achieves comparable performance with DeepSeek Chat 7B and LLaMA2 SFT 7B in the chat setting. 
Encouraged by these results, we further undertake a preliminary endeavor to scale up DeepSeekMoE to 145B. 
The experimental results still validate its substantial advantages over the GShard architecture consistently.
In addition, it shows performance comparable with DeepSeek 67B, using only 28.5% (maybe even 18.2%) of computations.