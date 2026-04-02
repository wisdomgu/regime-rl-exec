# Regime-Aware Reinforcement Learning for Optimal Trade Execution

Research paper implementation exploring whether regime-awareness improves 
reinforcement learning agents for optimal execution in limit order book markets.

## Motivation
Most RL execution agents treat the market as a single uniform environment. 
But markets shift between regimes — trending, volatile, mean-reverting. 
This paper asks: if you tell the agent what regime it's in, does it learn 
a better execution policy than a regime-blind agent or hand-coded rules?

## Builds directly on
- Amrouni et al. (2022) — CTMSTOU simulation environment (JP Morgan AI Research)
- ABIDES-Markets simulator

## Structure
```
regime-aware-execution/
├── ctmstou.py          # CTMSTOU market simulator
├── baselines.py        # TWAP, Full MO, Regime-Aware rule baselines  
├── environment.py      # Custom Gym environment for RL training
├── train.py            # PPO agent training (with/without regime)
├── evaluate.py         # Experiment runner and results
└── paper/              # LaTeX source
```

## Baseline Results (reproducing Amrouni et al. Table 1)
| Strategy | WAP Normalized | Completion |
|---|---|---|
| TWAP | 1.0277 | 0.850 |
| Full Market Order | 1.0278 | 1.000 |
| Regime Aware (rule) | 0.9949 | 0.997 |

## Research Question
Can a learned RL policy conditioned on market regime outperform 
hand-coded regime-aware rules?

## Status
🔄 In progress — building RL environment
