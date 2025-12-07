# Training and Evaluation Setup

## Training and Evaluating \spmath-RL

We conduct RL based on \spmath-Instruct 7B.
The training data of RL are chain-of-thought-format questions related to GSM8K and MATH from the SFT data, which consists of around 144K questions. 
We exclude other SFT questions to investigate the impact of RL on benchmarks that lack data throughout the RL phase.
We construct the training set of reward models following [wang2023math].
We train our initial reward model based on the \spmath-Base 7B with a learning rate of 2e-5.
For GRPO, we set the learning rate of the policy model as 1e-6. The KL coefficient is 0.04. For each question, we sample $64$ outputs.  The max length is set to 1024, and the training batch size is 1024.
The policy model only has a single update following each
exploration stage.
We evaluate \spmath-RL 7B on benchmarks following \spmath-Instruct 7B.
For \spmath-RL 7B, GSM8K and MATH with chain-of-thought reasoning can be regarded as in-domain tasks and all the other benchmarks can be regarded as out-of-domain tasks.