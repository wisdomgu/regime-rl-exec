import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from ctmstou import CTMSTOUFundamental

# ---------------------------------------------------------------------------
# Market impact parameters must match environment.py exactly so that
# baselines and RL agents are evaluated under identical assumptions.
# ---------------------------------------------------------------------------
MAX_PARTICIPATION_RATE = 0.10   # max fraction of Q per 60-second step
ETA = 0.0005                    # impact coeff: 5bps at 10% participation


def _apply_impact(qty, price, total_shares, eta=ETA):
    """
    Apply linear temporary market impact (Almgren-Chriss).
    Returns effective execution price.
    """
    impact_fraction = eta * (qty / total_shares)
    return price * (1.0 + impact_fraction)


def simulate_day(strategy, seed, total_shares=20000, total_seconds=82800,
                 max_participation=MAX_PARTICIPATION_RATE, eta=ETA):
    """
    Simulate one trading day using a baseline strategy.

    Market impact and participation rate cap are applied consistently
    with environment.py so that baselines and RL agents are compared
    on equal footing.

    Args:
        strategy:         'twap', 'regime_aware_1', or 'full_mo'
        seed:             RNG seed for the CTMSTOU simulator
        total_shares:     parent order size Q
        total_seconds:    episode length in seconds
        max_participation: max fraction of Q executable per 60s step
        eta:              linear temporary impact coefficient

    Returns:
        dict with wap, wap_norm, completion or None if nothing executed
    """
    fund = CTMSTOUFundamental(seed=seed)
    rng = np.random.default_rng(seed)

    shares_remaining = float(total_shares)
    executed_value = 0.0
    executed_shares = 0.0

    period = 60
    q = (total_shares / total_seconds) * period   # TWAP slice size ~14.5
    k = 10                                         # regime-aware multiplier

    # Hard cap: no strategy may exceed this per step (same as RL agent)
    max_qty_per_step = max_participation * total_shares

    for t in range(total_seconds):
        price, regime = fund.step()

        if shares_remaining <= 0:
            break
        if t % period != 0:
            continue

        # --- Determine desired quantity and fill probability ---
        if strategy == 'twap':
            qty_desired = min(q, shares_remaining)
            fill_prob = 0.85
            order_type = 'limit'

        elif strategy == 'regime_aware_1':
            if regime == 0:   # bull: aggressive market order
                qty_desired = min(k * q, shares_remaining)
                fill_prob = 1.0
                order_type = 'market'
            else:             # bear: patient limit order
                qty_desired = min(q, shares_remaining)
                fill_prob = 0.80
                order_type = 'limit'

        elif strategy == 'full_mo':
            qty_desired = min(q, shares_remaining)
            fill_prob = 1.0
            order_type = 'market'

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # --- Apply participation rate cap ---
        qty = min(qty_desired, max_qty_per_step)

        # --- Fill decision ---
        if order_type == 'market' or rng.random() < fill_prob:
            effective_price = _apply_impact(qty, price, total_shares, eta)
            executed_value += qty * effective_price
            executed_shares += qty
            shares_remaining -= qty

    if executed_shares == 0:
        return None

    wap = executed_value / executed_shares
    return {
        'wap': wap,
        'wap_norm': wap / 100000.0,
        'completion': executed_shares / total_shares,
    }


def simulate_day_detailed(strategy, seed, total_shares=20000,
                          total_seconds=82800,
                          max_participation=MAX_PARTICIPATION_RATE,
                          eta=ETA):
    """
    Simulate one day and track dominant regime for regime-stratified analysis.
    Same market impact model as simulate_day.

    Returns:
        dict with wap_norm, completion, dominant_regime or None
    """
    fund = CTMSTOUFundamental(seed=seed)
    rng = np.random.default_rng(seed)

    shares_remaining = float(total_shares)
    executed_value = 0.0
    executed_shares = 0.0
    regimes_seen = []

    period = 60
    q = (total_shares / total_seconds) * period
    k = 10
    max_qty_per_step = max_participation * total_shares

    for t in range(total_seconds):
        price, regime = fund.step()
        regimes_seen.append(regime)

        if shares_remaining <= 0:
            break
        if t % period != 0:
            continue

        if strategy == 'twap':
            qty_desired = min(q, shares_remaining)
            fill_prob = 0.85
            order_type = 'limit'

        elif strategy == 'regime_aware_1':
            if regime == 0:
                qty_desired = min(k * q, shares_remaining)
                fill_prob = 1.0
                order_type = 'market'
            else:
                qty_desired = min(q, shares_remaining)
                fill_prob = 0.80
                order_type = 'limit'

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        qty = min(qty_desired, max_qty_per_step)

        if order_type == 'market' or rng.random() < fill_prob:
            effective_price = _apply_impact(qty, price, total_shares, eta)
            executed_value += qty * effective_price
            executed_shares += qty
            shares_remaining -= qty

    if executed_shares == 0:
        return None

    dominant_regime = 0 if np.mean(regimes_seen) < 0.5 else 1
    return {
        'wap_norm': executed_value / executed_shares / 100000.0,
        'completion': executed_shares / total_shares,
        'dominant_regime': dominant_regime,
    }


# ---------------------------------------------------------------------------
# Main: run baseline evaluation (called directly or imported by train.py)
# ---------------------------------------------------------------------------
if __name__ == '__main__':

    n_days = 100
    strategies = ['twap', 'regime_aware_1', 'full_mo']

    print("=== BASELINE RESULTS (with market impact) ===")
    for strat in strategies:
        results = [simulate_day(strat, seed=i) for i in range(n_days)]
        results = [r for r in results if r is not None]
        avg_wap  = np.mean([r['wap_norm']   for r in results])
        avg_comp = np.mean([r['completion'] for r in results])
        print(f"{strat:20s} | WAP_norm: {avg_wap:.4f} | Completion: {avg_comp:.3f}")

    print("\n=== REGIME BREAKDOWN (with market impact) ===")
    for strat in ['twap', 'regime_aware_1']:
        results = [simulate_day_detailed(strat, seed=i) for i in range(200)]
        results = [r for r in results if r]
        bull = [r for r in results if r['dominant_regime'] == 0]
        bear = [r for r in results if r['dominant_regime'] == 1]
        print(f"\n{strat}:")
        if bull:
            print(f"  Bull (n={len(bull):3d}): WAP={np.mean([r['wap_norm'] for r in bull]):.4f}")
        if bear:
            print(f"  Bear (n={len(bear):3d}): WAP={np.mean([r['wap_norm'] for r in bear]):.4f}")