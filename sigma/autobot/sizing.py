"""Position maths for the multiplier model used by the bot (Deriv-style, also used for paper).

  profit ($) = exposure x direction x (price_now - entry) / entry - costs
  exposure   = stake x multiplier
  The broker stops a position out when the loss reaches the stake, so the stake must
  always be larger than the planned stop loss in dollars.

Given the dollar take profit we want and the price distance to it, the exposure follows:
  exposure = take_profit_$ x entry / tp_distance
and the stop loss in dollars is then fixed by the stop distance:
  stop_loss_$ = exposure x sl_distance / entry
"""
from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MULTIPLIERS = {
    "synthetic": [20, 40, 60, 100, 200, 400],
    "forex": [50, 100, 150, 250, 300, 500],
    "metal": [50, 100, 200, 300, 400, 500],
    "crypto": [10, 20, 30, 40, 50, 100],
    "index": [20, 50, 100, 150],
    "commodity": [20, 50, 100, 200],
    "basket": [50, 100, 200],
}
STAKE_BUFFER = 1.15   # stake at least 15% above the planned stop, so the broker's stop-out never comes first


@dataclass
class Size:
    exposure: float
    stake: float
    multiplier: int
    tp_usd: float
    sl_usd: float
    cost_usd: float


def auto_take_profit(equity: float) -> float:
    """Per-position profit target: 50c on small accounts, $1 on mid-size, 0.1% of equity above $1,000."""
    if equity < 300:
        return 0.50
    if equity < 1000:
        return 1.00
    return round(equity * 0.001, 2)


def max_scalp_positions(equity: float) -> int:
    """More simultaneous scalps as the account grows: $100 -> 2, $500 -> 6, $1,000+ -> 10."""
    return int(max(2, min(10, equity // 100 + 1)))


def size_position(entry: float, tp_dist: float, sl_dist: float, tp_usd: float, max_loss_usd: float,
                  multipliers: list[int], min_stake: float, cost_fraction: float) -> Size | None:
    """Returns the size, or None when the trade cannot be placed within the limits."""
    if entry <= 0 or tp_dist <= 0 or sl_dist <= 0 or tp_usd <= 0:
        return None
    edge = tp_dist / entry - cost_fraction           # profit per $1 of exposure at the target, after costs
    if edge <= 0 or cost_fraction > 0.5 * tp_dist / entry:
        return None                                  # costs would eat half the move or more: not worth it
    exposure = tp_usd / edge                         # so the NET profit at the target is exactly tp_usd
    sl_usd = exposure * (sl_dist / entry + cost_fraction)   # loss at the stop, costs included
    if sl_usd > max_loss_usd:            # shrink the trade so the stop never risks more than allowed
        scale = max_loss_usd / sl_usd
        exposure *= scale
        sl_usd = max_loss_usd
        tp_usd *= scale
    cost = exposure * cost_fraction
    need = max(min_stake, sl_usd * STAKE_BUFFER)
    for m in sorted(multipliers, reverse=True):        # highest multiplier = smallest stake
        stake = exposure / m
        if stake >= need:
            return Size(round(exposure, 4), round(stake, 2), m, round(tp_usd, 4), round(sl_usd, 4), round(cost, 4))
    return None   # even the lowest multiplier needs less than the minimum stake: too small for this market


def pnl(exposure: float, direction: int, entry: float, price: float, cost: float = 0.0) -> float:
    return exposure * direction * (price - entry) / entry - cost
