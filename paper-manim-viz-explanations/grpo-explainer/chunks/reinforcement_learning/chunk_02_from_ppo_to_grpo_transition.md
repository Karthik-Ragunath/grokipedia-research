# From PPO to GRPO Transition

### From PPO to GRPO
 Proximal Policy Optimization (PPO) [schulman2017proximal] is an actor-critic RL algorithm that is widely used in the RL fine-tuning stage of LLMs [ouyang2022training]. In particular, it optimizes LLMs by maximizing the following surrogate objective:

$$

\footnotesize
    \mathcal{J}_{PPO}(\theta) = \mathbb{E}{[q \sim P(Q), o \sim \pi_{\theta_{old}}(O|q)]} \frac{1}{|o|} \sum_{t=1}^{|o|} \min \left[ \frac{\pi_\theta(o_{t} | q, o_{<t})}{\pi_{\theta_{old}}(o_{t} | q, o_{<t})} A_{t}, \text{clip} \left( \frac{\pi_\theta(o_{t} | q, o_{<t})}{\pi_{\theta_{old}}(o_{t} | q, o_{<t})}, 1 - \epsilon, 1 + \epsilon \right)  A_{t} \right] ,

$$

where $\pi_{\theta}$ and $\pi_{\theta_{old}}$ are the current and old policy models, and $q, o$  are questions and outputs sampled from the question dataset and the old policy  $\pi_{\theta_{old}}$, respectively.  $\epsilon$ is a clipping-related hyper-parameter introduced in PPO for stabilizing training.  $A_t$  is the advantage, which is computed by applying Generalized Advantage Estimation (GAE) [gae], based on the rewards  $\{r_{\ge t}\}$  and a learned value function  $V_{\psi}$. Thus, in PPO, a value function needs to be trained alongside the policy model and to mitigate over-optimization of the reward model, the standard approach is to add a per-token KL penalty from a reference model in the reward at each token [ouyang2022training], i.e., 

$$

    r_{t} = r_\phi(q, o_{\le t}) - \beta \log\frac{\pi_{\theta}(o_{t}|q, o_{<t})}{\pi_{ref}(o_{t}|q, o_{<t})},

$$

where $r_\phi$  is the reward model, $\pi_{ref}$ is the reference model, which is usually the initial SFT model, and $\beta$  is the coefficient of the KL penalty.