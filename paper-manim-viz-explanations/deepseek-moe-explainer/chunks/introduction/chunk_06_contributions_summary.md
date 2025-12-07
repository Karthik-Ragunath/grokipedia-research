# Contributions Summary

Our contributions are summarized as follows: 

    
- **Architectural Innovation.**
    We introduce DeepSeekMoE, an innovative MoE architecture aiming at achieving ultimate expert specialization, which employs two principal strategies of fine-grained expert segmentation and shared expert isolation. 
    
- **Empirical Validation.**
    We conduct extensive experiments to empirically validate the effectiveness of the DeepSeekMoE architecture. 
    Experimental results validate the high level of expert specialization in DeepSeekMoE 2B, and indicate that DeepSeekMoE 2B can nearly approach the upper bound performance for MoE models
    
- **Scalability.**
    We scale up DeepSeekMoE to train a 16B model and show that with only about 40% of computations, DeepSeekMoE 16B achieves comparable performance with DeepSeek 7B and LLaMA2 7B. 
    We also undertake a preliminary endeavor to scale up DeepSeekMoE to 145B, highlighting its consistent advantages over the GShard architecture and showing a comparable performance with DeepSeek 67B.
    
- **Alignment for MoE.**
    We successfully perform supervised fine-tuning on DeepSeekMoE 16B to create an aligned chat model, showcasing the adaptability and versatility of DeepSeekMoE 16B.
    
- **Public Release.**
    In the spirit of open research, we release the model checkpoint of DeepSeekMoE 16B to the public. 
    Notably, this model can be deployed on a single GPU with 40GB of memory without the need for quantization.