import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ctmstou import CTMSTOUFundamental


# ---------------------------------------------------------------------------
# Market impact parameters (Almgren-Chriss linear temporary impact model)
# ---------------------------------------------------------------------------
# The simulation uses abstract share units (Q=20,000) rather than literal
# BTC quantities. Market impact is therefore defined relative to Q,
# following the self-contained framing of the Almgren-Chriss framework.
#
# Participation rate cap: the agent cannot execute more than
# MAX_PARTICIPATION_RATE * total_shares in a single 60-second step.
# At 10% of Q the order can be completed in as few as 10 steps out of
# 1380 available, allowing aggressive strategies while ruling out
# fully instantaneous execution that would be impossible in real markets.
#
# Temporary impact coefficient ETA: the effective execution price is
# raised by   impact_fraction = ETA * (qty / total_shares)
# so executing 10% of Q in one step costs ~5 basis points of temporary
# impact. Both parameters are constructor arguments to facilitate the
# parameter sensitivity analysis recommended by reviewers.

MAX_PARTICIPATION_RATE = 0.10   # max fraction of Q per 60-second step
ETA = 0.0005                    # impact coeff: 5bps at 10% participation


class ExecutionEnv(gym.Env):
    """
    RL environment for optimal trade execution in CTMSTOU markets.

    Agent must buy `total_shares` before time runs out.
    At each step it decides what fraction of remaining shares to execute.

    Market impact model (Almgren-Chriss linear temporary impact):
        impact_fraction_t = ETA * (qty_t / total_shares)
    The effective execution price is avg_price * (1 + impact_fraction),
    raising costs for aggressive execution and preventing exploitation of
    the zero-impact price simulator.

    A participation rate cap enforces:
        qty_t <= MAX_PARTICIPATION_RATE * total_shares  per step.

    Modes:
    - regime_aware=True:  agent sees regime label in state
    - regime_aware=False: agent does not see regime label
    """

    def __init__(self, regime_aware=True, total_shares=20000,
                 total_steps=1380, seed=None, reward_mode='standard',
                 max_participation=MAX_PARTICIPATION_RATE, eta=ETA):
        super().__init__()
        self.total_steps = total_steps
        self.total_shares = total_shares
        self.regime_aware = regime_aware
        self.init_seed = seed
        self.starting_price = 100000.0
        self.reward_mode = reward_mode
        self.max_participation = max_participation
        self.eta = eta

        self.action_space = spaces.Box(
            low=np.array([0.0]),
            high=np.array([1.0]),
            dtype=np.float32
        )

        obs_dim = 5 if regime_aware else 4
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """
        Start a new episode / trading day.

        Returns:
            obs (np.array): initial observation
        """
        super().reset(seed=seed)
        ep_seed = seed if seed is not None else np.random.randint(0, 100000)
        self.fund = CTMSTOUFundamental(seed=ep_seed)

        self.step_num = 0
        self.shares_remaining = float(self.total_shares)
        self.executed_value = 0.0
        self.executed_shares = 0.0
        self.price_history = []

        price, regime = self.fund.step()
        self.current_price = price
        self.current_regime = regime
        self.price_history.append(price)

        return self._get_obs(), {}

    def _get_obs(self):
        """
        Construct current observation.

        Returns:
            np.array: observation vector
        """
        shares_norm = self.shares_remaining / self.total_shares
        time_norm = 1.0 - (self.step_num / self.total_steps)
        price_norm = self.current_price / self.starting_price

        vol = (np.std(self.price_history[-10:]) / self.starting_price
               if len(self.price_history) >= 2 else 0.0)

        if self.regime_aware:
            return np.array([shares_norm, time_norm, price_norm,
                             vol, float(self.current_regime)], dtype=np.float32)
        else:
            return np.array([shares_norm, time_norm, price_norm, vol],
                            dtype=np.float32)

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action (np.array): fraction of remaining shares to buy (0-1)

        Returns:
            obs (np.array):   next observation
            reward (float):   reward for this step
            done (bool):      whether episode finished
            truncated (bool): always False
            info (dict):      execution diagnostics including impact_fraction
        """
        fraction = float(np.clip(action[0], 0.0, 1.0))
        qty_desired = fraction * self.shares_remaining

        # --- Participation rate cap ---
        # Prevents instantaneous execution regardless of price impact.
        max_qty = self.max_participation * self.total_shares
        qty = min(qty_desired, max_qty)

        # advance market by 60 seconds (price evolves independently of agent)
        prices_this_step = []
        for _ in range(60):
            price, regime = self.fund.step()
            prices_this_step.append(price)
            self.price_history.append(price)

        self.current_price = prices_this_step[-1]
        self.current_regime = regime
        avg_price = np.mean(prices_this_step)

        # --- Linear temporary market impact (Almgren-Chriss) ---
        # Temporary impact only; permanent impact omitted for consistency
        # with the original CTMSTOU framework (Amrouni et al. 2022).
        impact_fraction = self.eta * (qty / self.total_shares)
        effective_price = avg_price * (1.0 + impact_fraction)

        if qty > 0:
            self.executed_value += qty * effective_price
            self.executed_shares += qty
            self.shares_remaining -= qty

        self.step_num += 1
        done = self.step_num >= self.total_steps

        # --- Reward computation ---
        # cost reflects both price drift and market impact
        if qty > 0:
            cost = (effective_price - self.starting_price) / self.starting_price

            if self.reward_mode == 'regime_conditioned':
                if self.current_regime == 0:   # bull: reward urgency
                    urgency_bonus = 0.3 * (qty / self.total_shares)
                    cost_penalty = -2.0 * cost * (qty / self.total_shares)
                    reward = cost_penalty + urgency_bonus
                else:                          # bear: reward patience
                    savings = max(0, -cost)
                    patience_reward = 1.0 * savings * (qty / self.total_shares)
                    progress_reward = 0.3 * (qty / self.total_shares)
                    reward = patience_reward + progress_reward
            else:
                cost_penalty = -cost * (qty / self.total_shares)
                progress_reward = 0.5 * (qty / self.total_shares)
                reward = cost_penalty + progress_reward
        else:
            reward = -0.001

        if done and self.shares_remaining > 0:
            incompletion_penalty = -20.0 * (self.shares_remaining / self.total_shares)
            reward += incompletion_penalty

        obs = self._get_obs()
        info = {
            'impact_fraction': impact_fraction,
            'qty_executed': qty,
            'qty_desired': qty_desired,
            'effective_price': effective_price,
            'avg_price': avg_price,
        }
        return obs, reward, done, False, info

    def get_results(self):
        """
        Return final WAP and completion stats for episode.

        Returns:
            dict or None
        """
        if self.executed_shares == 0:
            return None
        wap = self.executed_value / self.executed_shares
        return {
            'wap': wap,
            'wap_norm': wap / self.starting_price,
            'completion': self.executed_shares / self.total_shares
        }