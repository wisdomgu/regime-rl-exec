# Regime-Aware Reinforcement Learning for Optimal Trade Execution

Empirical study examining whether market regime information improves 
reinforcement learning agents for optimal trade execution in simulated 
limit order book markets. Built on the CTMSTOU simulation environment 
from Amrouni et al. (2022) - JP Morgan AI Research.

## Background

The optimal execution problem: an institutional trader must buy a large 
quantity of shares at minimal cost before a deadline. Markets shift between 
**regimes**, bullish (rising prices) and bearish (falling prices), and 
a good trader behaves differently in each. This study asks whether an RL 
agent can learn this regime-conditional behavior automatically.

## Contribution

This work provides a controlled empirical study of regime-aware reinforcement learning for optimal trade execution. We evaluate whether standard PPO-based agents can exploit regime information when it is explicitly available in the state or reward. Through multi-seed experiments and ablation analysis, we show that such agents fail to reliably leverage regime signals, indicating a structural limitation rather than an issue of insufficient training or hyperparameter tuning.

## Research Questions

1. Can a learned RL policy conditioned on market regime match hand-coded regime-aware rules?
2. Is regime information in the state sufficient, or is reward conditioning also needed?
3. Why does flat RL fail to exploit regime information even when it has access to it?

## Key Findings

- RL agents achieve near-perfect order completion (1.000) vs TWAP (0.850)
- Neither state augmentation nor reward conditioning matches the hand-coded rule on cost (WAP 1.0003 vs 0.9949)
- The regime-aware agent exhibits highly polarized behavior across regimes, often deviating from the qualitatively optimal strategy.
- Regime sensitivity is **initialization-dependent**; single seeds show extreme sensitivity (action 0.92→0.00 on regime flip) while multi-seed average shows near-zero sensitivity
- Reward conditioning introduces training instability (WAP std 0.0131) without gains
- Hyperparameter sensitivity analysis confirms the gap persists regardless of training budget, the failure is structural, not a matter of sample efficiency

## Results

| Strategy | WAP (mean ± std) | Completion |
|---|---|---|
| TWAP | 1.0277 ± 0.000 | 0.850 |
| Full Market Order | 1.0278 ± 0.000 | 1.000 |
| **Regime Aware Rule** | **0.9949 ± 0.000** | **0.997** |
| PPO Blind | 1.0003 ± 0.0000 | 1.000 |
| PPO State-Aware | 1.0004 ± 0.0001 | 1.000 |
| PPO Reward-Conditioned | 1.0069 ± 0.0131 | 0.996 |

WAP normalized to starting price, lower is better. Below 1.0 means buying cheaper than the opening price.
Standard deviations are computed across 5 independent training seeds.

## Structure
```
├── figures/              # All paper figures
├── models/               # Trained PPO models
├── results/              # Raw results
├── src/                  # All python files
    ├── baselines.py          # TWAP, Full MO, Regime-Aware rule baselines
    ├── ctmstou.py            # CTMSTOU market simulator
    ├── environment.py        # Gymnasium execution environment
    ├── plot_results.py       # Figure generation
    ├── run_curves.py         # Generate learning curves
    ├── train.py              # PPO training + multi-seed evaluation
└── README.md
```

## Setup
```bash
conda create -n regime-exec python=3.9
conda activate regime-exec
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install stable-baselines3 gymnasium numpy matplotlib
```

## Reproducing Results
```bash
cd src
python baselines.py      # rule-based baselines (~1 min)
python train.py          # all RL agents, 5 seeds each (~3 hours CPU)
python run_curves.py     # generate learning curves
python plot_results.py   # generate all figures
```

## Paper

This repository accompanies a research paper. A preprint link will be added upon publication.

## Built On

- [Amrouni et al. (2022)](https://arxiv.org/abs/2202.00941) - CTMSTOU driven markets (JP Morgan AI Research)
- [Schulman et al. (2017)](https://arxiv.org/abs/1707.06347) - Proximal Policy Optimization
- [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3)