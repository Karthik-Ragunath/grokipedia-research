# GRPO Algorithm Implementation

% This section details our approach, which includes both outcome- and process-supervised GRPO.
\begin{algorithm}[t]
  \small
  \caption{Iterative Group Relative Policy Optimization}
  **Input** initial policy model $\pi_{\theta_{\text{init}}}$; reward models $r_\phi$; task prompts $\mathcal{D}$; 
  hyperparameters $\epsilon$, $\beta$, $\mu$
  \begin{algorithmic}[1]
    \State policy model $\pi_\theta \leftarrow \pi_{\theta_{\text{init}}}$
    \For{iteration = 1, \dots, I}
       \State reference model $\pi_{ref} \leftarrow \pi_{\theta}$
      \For{step = 1, \dots, M}
      \State Sample a batch $\mathcal{D}_b$ from $\mathcal{D}$
      \State Update the old policy model $\pi_{\theta_{old}} \leftarrow \pi_{\theta}$ 
      \State Sample $G$ outputs $\{o_i\}_{i=1}^G \sim \pi_{\theta_{old}} (\cdot \mid q) $ for each question $q \in \mathcal{D}_b$
      \State Compute rewards $\{r_i\}_{i=1}^{G}$ for each sampled output $o_i$ by running $r_{\phi}$ 
      \State Compute $\hat{A}_{i,t}$ for the $t$-th token of $o_i$ through group relative advantage estimation.
      \For{GRPO iteration = 1, \dots, $\mu$}
        \State Update the policy model $\pi_{\theta}$ by maximizing the GRPO objective (Equation (ref: eq:GC-GRPO))
      \EndFor
    \EndFor 
    \State Update $r_\phi$ through continuous training using a replay mechanism. 
    \EndFor 
  \end{algorithmic}
  **Output** $\pi_\theta$
  
\end{algorithm>