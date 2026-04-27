# An Empirical Study on Regime Awareness in Reinforcement Learning for Optimal Trade Execution
[![DOI](https://zenodo.org/badge/1199326073.svg)](https://doi.org/10.5281/zenodo.19441357)

Empirical study examining whether market regime information improves 
reinforcement learning agents for optimal trade execution in simulated 
limit order book markets. Built on the CTMSTOU simulation environment 
from Amrouni et al. (2022) - JP Morgan AI Research.

## Background

The optimal execution problem: an institutional trader must buy a large 
quantity of shares at minimal cost before a deadline. Markets shift between 
**regimes**, bullish (rising prices) and bearish (falling prices), and 
a good trader behaves differently in each.

This study asks whether an RL agent can learn this regime-conditional behavior automatically.
We conduct a controlled empirical study evaluating whether PPO-based agents can exploit regime information when introduced via state augmentation or reward conditioning, using multi-seed experiments and ablation analysis.

## Contribution

This work provides controlled empirical evidence that flat RL approaches fail to learn qualitatively correct regime-dependent execution behavior, motivating hierarchical formulations.

## Research Questions

1. Can a learned RL policy conditioned on market regime match hand-coded regime-aware rules?
2. Is regime information in the state sufficient, or is reward conditioning also needed?
3. Why does flat RL fail to exploit regime information even when it has access to it?

## Key Findings

- RL agents achieve near-perfect order completion (1.000) vs TWAP (0.850)
- Neither state augmentation nor reward conditioning matches the hand-coded rule on cost (WAP 1.0003 vs 0.9950)
- The regime-aware agent exhibits highly polarized behavior across regimes, often deviating from the qualitatively optimal strategy.
- Regime sensitivity is **initialization-dependent**; single seeds show extreme sensitivity (action 0.92→0.00 on regime flip) while multi-seed average shows near-zero sensitivity
- Reward conditioning introduces training instability (WAP std 0.0131) without gains
- Hyperparameter sensitivity analysis confirms the gap persists regardless of training budget, suggesting the limitation is structural rather than due to insufficient training

## Results

| Strategy | WAP (mean ± std) | Completion |
|---|---|---|
| TWAP | 1.0278 ± 0.000 | 0.850 |
| Full Market Order | 1.0278 ± 0.000 | 1.000 |
| **Regime Aware Rule** | **0.9950 ± 0.000** | **0.996** |
| PPO Blind | 1.0003 ± 0.0000 | 1.000 |
| PPO State-Aware | 1.0004 ± 0.0001 | 1.000 |
| PPO Reward-Conditioned | 1.0069 ± 0.0131 | 0.996 |

WAP normalized to starting price, lower is better. Below 1.0 means buying cheaper than the opening price.
Standard deviations are computed across 5 independent training seeds.

## Structure
```
├── curves/               # Learning curves
├── figures/              # All paper figures
├── src/                  # All python files
    ├── baselines.py          # TWAP, Full MO, Regime-Aware rule baselines
    ├── ctmstou.py            # CTMSTOU market simulator
    ├── environment.py        # Gymnasium execution environment
    ├── plot_results.py       # Figure generation
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
Expected runtime: several hours depending on CPU (multi-seed PPO training is the main cost).
```bash
cd src
python baselines.py      # rule-based baselines (~1 min)
python train.py          # all RL agents, 5 seeds each (several hours)
python plot_results.py   # generate all figures
```

## Notes on Artifacts

Pretrained model checkpoints are not included. All results are generated 
via controlled multi-seed experiments using the provided training pipeline, 
ensuring full reproducibility without reliance on fixed model artifacts.

## Citation

If you use this code, please cite:

**Code (Zenodo):**  
DOI: https://doi.org/10.5281/zenodo.19441357

## Paper

Preprint: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6559598

## Built On

- [Amrouni et al. (2022)](https://arxiv.org/abs/2202.00941) - CTMSTOU driven markets (JP Morgan AI Research)
- [Schulman et al. (2017)](https://arxiv.org/abs/1707.06347) - Proximal Policy Optimization
- [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3)
