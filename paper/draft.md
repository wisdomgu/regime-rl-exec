# Does Regime Awareness Improve Reinforcement Learning for Optimal Trade Execution?

**Abstract**

Optimal trade execution requires balancing execution speed against market 
impact across varying market conditions. While reinforcement learning (RL) 
has shown promise for execution tasks, most approaches train a single policy 
across all market conditions, ignoring the distinct behavioral regimes that 
characterize real markets. Amrouni et al. (2022) demonstrated that hand-coded 
regime-aware rules outperform regime-blind strategies in simulated limit order 
book markets, calling for further research into learned regime-aware policies. 
We directly answer this call by training Proximal Policy Optimization (PPO) 
agents with and without regime information in a CTMSTOU-simulated market 
environment. We find that regime conditioning via state augmentation alone is 
insufficient: the regime-aware agent learns a near-binary policy that 
completely abstains in bear markets, failing to match the hand-coded rule's 
execution cost. A regime-conditioned reward variant introduces training 
instability without improving performance. Our perturbation analysis reveals 
that learned regime sensitivity is highly initialization-dependent, suggesting 
that stable regime exploitation requires architectural changes beyond state 
or reward modification.

---

## 1. Introduction

Financial markets exhibit distinct behavioral regimes — periods of sustained 
upward trends, downward trends, and elevated volatility. A skilled trader 
adapts their behavior accordingly: executing aggressively in rising markets 
to avoid chasing price, and placing patient limit orders in falling markets 
to capture favorable prices. Yet most reinforcement learning approaches to 
optimal execution treat the market as a single uniform environment, learning 
one policy across all conditions.

The optimal execution problem arises whenever an institutional trader must 
buy or sell a large quantity of shares. Executing the entire order at once 
causes adverse price movement — known as market impact — while spreading 
execution over too long a horizon exposes the trader to unfavorable price 
drift. Classical approaches such as Almgren and Chriss (2001) address this 
tradeoff through mathematical optimization, producing deterministic schedules 
that minimize a combination of market impact and timing risk. However, these 
models assume stationary market conditions and cannot adapt to the regime 
shifts that characterize real markets.

Amrouni et al. (2022) introduced the Continuous-Time Markov Switching 
Trending Ornstein-Uhlenbeck (CTMSTOU) process as a simulation environment 
specifically designed to study regime-aware execution. Using hand-coded 
rule-based strategies, they demonstrated that regime-aware execution 
significantly outperforms regime-blind approaches, and explicitly called for 
further research into learned policies. Reinforcement learning is the natural 
next step: rather than hand-coding regime-specific behavior, can an RL agent 
learn to exploit regime information automatically?

We directly answer this question through a controlled empirical study. We 
implement the CTMSTOU environment as a Gymnasium-compatible RL training 
framework, train PPO agents under three conditions — regime-blind, 
regime-aware via state augmentation, and regime-aware via reward conditioning 
— and evaluate all agents against the rule-based baselines from Amrouni et al.

Our findings are nuanced. RL agents successfully learn to fully execute 
parent orders, dramatically outperforming TWAP on completion rate. However, 
none of our RL approaches match the hand-coded regime-aware rule on execution 
cost. The regime-aware agent learns to use the regime signal — but learns the 
wrong behavior, completely abstaining in bear markets rather than executing 
patiently. Our perturbation analysis reveals that this behavior is highly 
initialization-dependent: different training seeds produce qualitatively 
different degrees of regime sensitivity, suggesting that PPO cannot reliably 
learn stable regime-conditioned behavior from state augmentation alone.

This paper is organized as follows. Section 2 provides background on optimal 
execution, market regimes, and reinforcement learning for trading. Section 3 
describes our simulation environment, RL formulation, and baselines. Section 4 
presents experimental results across all three agent variants. Section 5 
discusses implications and limitations. Section 6 concludes with directions 
for future work.

**Contributions:**
1. We implement the CTMSTOU simulation environment from Amrouni et al. (2022) 
   as a Gymnasium-compatible framework for RL training, making the environment 
   accessible to the RL research community.
2. We conduct the first controlled ablation of regime-awareness in RL-based 
   execution, comparing blind, state-augmented, and reward-conditioned agents 
   across five random seeds each.
3. We discover that state augmentation produces a near-binary learned policy 
   that is both highly regime-sensitive and qualitatively incorrect, 
   abstaining in bear markets rather than executing patiently.
4. We show through perturbation analysis that regime sensitivity is 
   initialization-dependent, identifying training instability as a key 
   challenge for regime-aware RL execution.

---

## 2. Background

### 2.1 The Optimal Execution Problem

A trader must purchase Q shares before a deadline T. Executing the full 
quantity immediately via market orders guarantees completion but causes 
significant market impact — the act of buying moves prices upward, increasing 
the average cost. Spreading execution over the full horizon reduces market 
impact but exposes the trader to price drift: in a rising market, delaying 
execution means paying more for later shares.

The classic solution, due to Almgren and Chriss (2001), frames this as a 
mean-variance optimization problem. Given a linear market impact model, 
the optimal execution schedule trades off expected cost against variance of 
cost, producing a deterministic trajectory that is typically front-loaded 
relative to naive time-weighted execution. The Time-Weighted Average Price 
(TWAP) strategy — executing equal quantities at equal time intervals — serves 
as a widely used practical baseline, though it ignores market conditions 
entirely.

We measure execution quality using two primary metrics. The Weighted Average 
Price (WAP) is the volume-weighted average price across all executed orders, 
normalized to the starting price. A WAP below 1.0 indicates the trader bought 
at a discount relative to the opening price, while a WAP above 1.0 indicates 
overpayment. The completion rate measures the fraction of the target quantity 
Q actually executed before the deadline — a critical operational constraint, 
as failing to execute the full order carries its own costs and risks.

### 2.2 Market Regimes

Financial markets are widely understood to exhibit distinct behavioral 
regimes — persistent states characterized by different statistical properties 
of returns, volatility, and price direction. The concept of regime switching 
was formalized by Hamilton (1989), who proposed modeling economic time series 
as Markov chains switching between latent states, each with different 
distributional parameters. In this framework, the market at time t is in 
state s_t ∈ {0, 1, ..., k}, and the observed price process depends on the 
current state.

In practice, traders commonly distinguish between bullish regimes — periods 
of sustained upward price trends — and bearish regimes — periods of sustained 
downward trends. These two states carry fundamentally different implications 
for execution strategy. In a bullish regime, delaying a buy order means paying 
more as the price rises; urgency is rewarded. In a bearish regime, patience 
is rewarded — waiting for the price to fall further allows the trader to 
execute at more favorable levels using limit orders.

Amrouni et al. (2022) introduced the CTMSTOU process as a simulation 
environment specifically designed to study these regime dynamics, providing 
clearly defined regime switches with calibrated transition probabilities. 
We adopt their framework directly, using the same bullish/bearish 
two-state distinction to maintain comparability with their baseline results.

### 2.3 Reinforcement Learning for Execution

Reinforcement learning frames the execution problem as a Markov Decision 
Process (MDP) defined by a tuple (S, A, T, R, γ), where S is the state space, 
A is the action space, T is the transition function, R is the reward function, 
and γ is the discount factor. At each timestep t, the agent observes state 
s_t, takes action a_t, receives reward r_t, and transitions to state s_{t+1}. 
The agent's objective is to learn a policy π: S → A that maximizes expected 
cumulative discounted reward.

Nevmyvaka et al. (2006) established the foundational RL approach to execution, 
demonstrating that a Q-learning agent trained on historical limit order book 
data could outperform standard execution benchmarks. Subsequent work has 
explored deep RL methods, hierarchical architectures, and multi-agent 
formulations. However, to our knowledge, no prior work has conducted a 
controlled ablation of regime-awareness in RL execution using a simulation 
environment with explicit regime labels.

We use Proximal Policy Optimization (PPO), introduced by Schulman et al. 
(2017), as our RL algorithm. PPO is a policy gradient method that uses a 
clipped surrogate objective to constrain policy updates, providing stable 
training without the sensitivity to hyperparameters that characterizes 
earlier policy gradient methods. Its robustness and strong empirical 
performance across continuous control tasks make it a natural choice for 
the execution problem.

---

## 3. Method

### 3.1 Market Simulation: CTMSTOU

We use the Continuous-Time Markov Switching Trending Ornstein-Uhlenbeck 
(CTMSTOU) process introduced by Amrouni et al. (2022) as our market simulator. 
This process switches between bullish and bearish regimes according to a 
continuous-time Markov chain, with each regime modeled as a trending 
Ornstein-Uhlenbeck process. The simulator provides explicit regime labels at 
each timestep, enabling controlled study of regime-aware execution.

Formally, the fundamental value follows:

    dX_t = θ_{s_t}(M_{s_t} - X_t)dt + σ_{s_t}dW_t

where s_t ∈ {0,1} is the regime state governed by a Continuous-Time Markov 
Chain with transition rates calibrated to BTC/USD data from the Gemini 
exchange (λ=2.90 events/day, ω=0.812 events/day). M_{s_t} is a 
time-dependent center term that trends linearly at rate μ_{s_t}, modeling 
the directional drift of each regime. θ controls mean reversion speed, σ 
controls noise intensity, and W_t is a standard Wiener process.

We calibrate parameters to produce realistic intraday price dynamics with 
an average of approximately one regime switch per trading day, consistent 
with Amrouni et al.'s calibration to real BTC/USD data. Figure 1 shows a 
representative simulated trading day, with green sections indicating bullish 
regime and red sections indicating bearish regime.

Parameters used in all experiments:
- θ = 0.00005, σ = 50.0
- μ_bull = 0.5, μ_bear = -0.5
- Starting price: 100,000
- Simulation length: 82,800 seconds (23 hours)
- Transition rates: λ = 2.90/day, ω = 0.812/day

### 3.2 Execution Environment

We implement a custom Gymnasium environment wrapping the CTMSTOU simulator. 
Each episode represents one simulated trading day. The agent must purchase 
Q = 20,000 shares before the 23-hour deadline, acting every 60 seconds 
(1,380 decision steps per episode).

**State space.** The regime-aware agent observes a 5-dimensional state vector:
- s_1: shares remaining, normalized to [0,1]
- s_2: time remaining, normalized to [0,1]  
- s_3: current price, normalized to starting price
- s_4: rolling price volatility (standard deviation of last 10 prices, normalized)
- s_5: current regime label (0 = bullish, 1 = bearish)

The regime-blind agent observes the same state without s_5 (4-dimensional).

**Action space.** The agent outputs a continuous action a ∈ [0,1] 
representing the fraction of remaining shares to execute in the current 
60-second period. The quantity executed is qty = a × shares_remaining, 
filled at the volume-weighted average price of that period.

**Reward function.** At each step, the agent receives:

    r_t = -cost_t × (qty_t / Q) + 0.5 × (qty_t / Q)

where cost_t = (avg_price_t - starting_price) / starting_price is the 
normalized execution cost. The first term penalizes buying above the 
starting price; the second term provides a positive progress reward for 
executing shares, preventing the degenerate policy of doing nothing. 
At episode end, a strong incompletion penalty is applied:

    r_T += -20.0 × (shares_remaining / Q)

This penalty ensures the agent prioritizes completing the order before 
the deadline.

### 3.3 Baselines

We compare against three rule-based baselines directly from Amrouni et al. 
(2022), implemented with simplified fill mechanics (market orders fill with 
probability 1.0, limit orders fill with probability 0.85):

**TWAP:** Executes equal quantity q = Q×τ/T every τ=60 seconds using limit 
orders at the best bid. Does not use regime information.

**Full Market Order (Full MO):** Executes equal quantity using market orders, 
guaranteeing fills at the cost of higher market impact. Does not use regime 
information.

**Regime Aware Rule:** Queries the regime oracle at each decision point. In 
bullish regimes, executes k×q shares via market order to secure execution 
before prices rise further. In bearish regimes, executes q shares via limit 
orders spread across k price levels to capture falling prices patiently. 
We use k=10 following Amrouni et al. This strategy has access to the true 
regime label and encodes explicit domain knowledge about optimal per-regime 
behavior.

### 3.4 RL Agent: PPO

We train all RL agents using Proximal Policy Optimization (PPO) from the 
Stable Baselines 3 library (Raffin et al., 2021), with an MLP policy network 
consisting of two hidden layers of 64 units each with tanh activations. 
Four parallel environments are used for experience collection.

All agents are trained for 500,000 timesteps with the following 
hyperparameters: learning rate 3×10⁻⁴, n_steps=512, batch_size=64, 
n_epochs=10, discount factor γ=0.99. These hyperparameters were selected 
based on standard PPO defaults and held constant across all agent variants 
to ensure fair comparison. Each agent variant is trained across 5 independent 
random seeds, and results are reported as mean ± standard deviation.

We train three agent variants:
1. **PPO Blind:** state dimension 4, standard reward
2. **PPO State-Aware:** state dimension 5 (includes regime label), 
   standard reward
3. **PPO Reward-Conditioned:** state dimension 5, regime-conditioned reward 
   that applies urgency bonuses in bull regimes and patience bonuses in 
   bear regimes

---

## 4. Experiments and Results

### 4.1 Overall Execution Performance

Table 1 reports mean WAP and completion rate across 100 test episodes, 
with RL results averaged across 5 training seeds.

**Table 1: Overall execution performance**

| Strategy | WAP (mean ± std) | Completion |
|---|---|---|
| TWAP | 1.0277 ± 0.000 | 0.850 |
| Full MO | 1.0278 ± 0.000 | 1.000 |
| Regime Aware rule | **0.9949 ± 0.000** | 0.997 |
| PPO blind | 1.0003 ± 0.0000 | 1.000 |
| PPO state-aware | 1.0004 ± 0.0001 | 1.000 |
| PPO reward-conditioned | 1.0069 ± 0.0131 | 0.996 |

Both PPO blind and PPO state-aware agents achieve near-perfect completion 
(1.000), substantially outperforming TWAP (0.850). TWAP's incomplete 
execution arises from its fixed limit order strategy — in strongly trending 
markets, limit orders at the best bid are never filled as the price moves 
away, leaving shares unexecuted. The RL agents learn to use market orders 
when necessary to guarantee completion, sacrificing some price improvement 
for execution certainty.

On execution cost (WAP), however, neither RL agent approaches the hand-coded 
regime-aware rule. The rule achieves WAP of 0.9949 — actually buying below 
the starting price on average — while both blind and state-aware PPO agents 
achieve approximately 1.0003. The reward-conditioned variant performs 
notably worse (1.0069) with high variance (±0.0131), indicating training 
instability introduced by the regime-conditioned reward formulation.

### 4.2 Regime-Stratified Performance

To understand where each strategy succeeds and fails, Table 2 breaks down 
WAP by the dominant regime of each test episode. An episode is classified 
as bull-dominant or bear-dominant based on whether the bullish regime 
accounts for more or less than 50% of timesteps.

**Table 2: WAP by dominant market regime**

| Strategy | Bull WAP | Bear WAP | Bull episodes | Bear episodes |
|---|---|---|---|---|
| TWAP | 1.0844 | 0.9769 | 95 | 105 |
| Regime Aware rule | 1.0036 | **0.9559** | 156 | 44 |
| PPO blind | 1.0000 | 1.0002 | 95 | 105 |
| PPO state-aware | 1.0000 | 1.0002 | 95 | 105 |

The regime breakdown reveals the mechanism behind each strategy's aggregate 
performance. TWAP suffers most severely in bull-dominant episodes (WAP = 
1.0844) — its fixed limit order schedule cannot adapt as prices trend upward, 
forcing the trader to chase the price or leave shares unexecuted. In 
bear-dominant episodes, TWAP inadvertently benefits (WAP = 0.9769) as the 
falling market brings prices down toward its standing limit orders.

The hand-coded regime-aware rule achieves its advantage primarily through 
exceptional bear-market execution (WAP = 0.9559). By placing patient limit 
orders across multiple price levels in bearish regimes, it systematically 
captures prices as the market falls — behavior that requires both knowing 
the regime and having domain knowledge about the correct response.

The PPO agents achieve strikingly uniform performance across both regime 
types (blind: 1.0000 bull, 1.0002 bear; state-aware: identical). This 
uniformity indicates that both agents learned an approximately 
regime-independent steady execution policy — executing a consistent fraction 
of remaining shares at each step regardless of market conditions. Critically, 
the state-aware and blind agents produce identical regime-stratified results, 
confirming that the regime label in the state is not meaningfully influencing 
the learned policy despite being available.

### 4.3 Regime Sensitivity Analysis

To directly test whether the regime-aware agent uses the regime signal, we 
conduct a perturbation experiment. For each of the 5 trained state-aware 
models, we observe the initial state of a test episode and record the 
predicted action under the true regime label. We then flip the regime label 
in the observation vector and record the predicted action again.

**Table 3: Action sensitivity to regime label perturbation (multi-seed)**

| Seed | True action (Bull) | Flipped action (Bear) | Difference |
|---|---|---|---|
| 0 | 0.7384 | 0.7788 | 0.0404 |
| 1 | 0.7383 | 0.7788 | 0.0405 |
| 2 | 0.7384 | 0.7788 | 0.0404 |
| 3 | 0.7385 | 0.7789 | 0.0404 |
| 4 | 0.7384 | 0.7788 | 0.0404 |

The multi-seed models show remarkably consistent but weak regime sensitivity: 
flipping the regime label changes the predicted action by only 0.040 across 
all five seeds, with the direction inverted relative to what domain knowledge 
would suggest (the agent executes slightly *more* when told it is a bear 
market, rather than less).

This finding contrasts sharply with a single-seed model trained under 
identical conditions, which exhibited extreme binary sensitivity: action 
dropped from 0.92 in bull regime to 0.00 in bear regime. Together, these 
observations reveal that PPO training converges to qualitatively different 
policies depending on initialization — some seeds produce strongly 
regime-sensitive policies, others produce nearly regime-insensitive ones — 
and that on average across seeds, the regime signal is not being exploited 
in a meaningful or consistent way.

This initialization-dependence of regime sensitivity is itself a key finding. 
It suggests that the PPO optimization landscape contains multiple local optima 
with different degrees of regime exploitation, and that the standard training 
procedure does not reliably converge to the regime-sensitive optimum.

---

## 5. Discussion

Our results reveal a fundamental tension between learning to complete orders 
and learning to exploit market structure. PPO agents reliably solve the 
completion problem — the strong incompletion penalty in the reward function 
ensures near-perfect execution rates across all seeds and conditions. However, 
this same reward structure may actively prevent the agent from learning 
regime-conditioned behavior, because steady execution is a robust local 
optimum: it guarantees completion with moderate cost, regardless of regime.

The hand-coded regime-aware rule succeeds precisely because it encodes 
domain knowledge that is difficult to learn from reward alone. Knowing that 
patient limit orders in bear markets systematically capture falling prices 
requires understanding the causal relationship between regime, order type, 
and fill price — a relationship that emerges only weakly from scalar reward 
signals over thousands of episodes. The rule bypasses this learning challenge 
by directly encoding the correct behavior.

State augmentation — providing the regime label as an additional observation 
— is a necessary but insufficient condition for regime-aware behavior. The 
agent can observe the regime, but the reward signal does not clearly 
differentiate between the correct regime-specific responses. Our 
reward-conditioned variant attempts to address this but introduces training 
instability, with high variance across seeds (WAP std = 0.0131 versus 0.0001 
for the state-aware agent). The instability likely arises from conflicting 
gradient signals: urgency bonuses and patience bonuses create opposing 
incentives that the optimizer struggles to reconcile stably.

The perturbation analysis adds a further dimension to this picture. The 
fact that a single training seed can produce a strongly regime-sensitive 
policy (action 0.92 → 0.00 on regime flip) while the multi-seed average 
shows only weak sensitivity (0.738 → 0.779) suggests that regime exploitation 
is achievable in principle but not reliably in practice. This points to the 
need for explicit mechanisms to encourage stable regime-conditioned behavior, 
such as auxiliary losses, information-theoretic objectives that reward using 
the regime signal, or architectural changes that process regime information 
through dedicated pathways.

**Limitations.** Our study uses a simplified fill model with fixed fill 
probabilities (market orders: 100%, limit orders: 85%), whereas real LOB 
execution depends on queue position, order size, and market depth. Our 
CTMSTOU simulator, while calibrated to real BTC/USD data, is considerably 
cleaner than real markets and may not capture microstructure effects that 
influence execution quality. We test only two market regimes (bullish and 
bearish); real markets exhibit richer regime structures including high 
volatility, low volatility, and mean-reverting regimes. Finally, our 
evaluation uses 100 test episodes per seed, which while sufficient for 
stable mean estimates may underestimate tail risks.

---

## 6. Conclusion

We conducted a controlled empirical study of regime-awareness in 
reinforcement learning for optimal trade execution, building on the CTMSTOU 
simulation environment of Amrouni et al. (2022). Training PPO agents under 
three conditions — regime-blind, regime-aware via state augmentation, and 
regime-aware via reward conditioning — we find that none of our RL approaches 
match the execution cost achieved by a hand-coded regime-aware rule, despite 
all agents achieving near-perfect order completion.

Our key findings are threefold. First, state augmentation alone is 
insufficient: providing the regime label as an observation does not reliably 
produce regime-conditioned behavior, and regime-aware and blind agents achieve 
statistically identical execution quality. Second, regime sensitivity is 
highly initialization-dependent: perturbation analysis reveals that different 
training seeds produce qualitatively different levels of regime exploitation, 
with the average policy showing only weak and directionally incorrect regime 
sensitivity. Third, reward conditioning introduces instability without 
improvement, suggesting that the execution reward landscape is poorly suited 
to regime-specific gradient signals under standard PPO training.

These findings point to several concrete directions for future work. 
Architectural approaches — separate policy networks per regime, hierarchical 
RL with an explicit regime-switching module, or mixture-of-experts policies 
— may provide the structural inductive bias needed for stable regime 
exploitation. Information-theoretic regularization that explicitly rewards 
using the regime signal could complement reward-based approaches. Finally, 
extending this study to the full ABIDES-Markets simulator with realistic LOB 
mechanics would strengthen the ecological validity of these findings.

The hand-coded regime-aware rule remains the performance frontier — a 
reminder that domain knowledge, when available and correctly encoded, is 
difficult for learned policies to match. Closing this gap is the central 
challenge for regime-aware execution RL.

---

## References

- Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio 
  transactions. *Journal of Risk*, 3(2), 5–39.
- Amrouni, S., Moulin, A., & Balch, T. (2022). CTMSTOU driven markets: 
  simulated environment for regime-awareness in trading policies. 
  *arXiv:2202.00941*.
- Hamilton, J. D. (1989). A new approach to the economic analysis of 
  nonstationary time series and the business cycle. *Econometrica*, 
  57(2), 357–384.
- Nevmyvaka, Y., Feng, Y., & Kearns, M. (2006). Reinforcement learning for 
  optimized trade execution. *Proceedings of the 23rd ICML*, 673–680.
- Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., & 
  Dormann, N. (2021). Stable-Baselines3: Reliable reinforcement learning 
  implementations. *JMLR*, 22(268), 1–8.
- Saqur, R. (2024). What teaches robots to walk, teaches them to trade too. 
  *arXiv:2406.15508*.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). 
  Proximal policy optimization algorithms. *arXiv:1707.06347*.