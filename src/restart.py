"""Restart (kick-off / throw-in / free kick / corner / goal kick) detector from tracking only.

A restart looks like: players walking and the ball slow or not visible for a while, then the
ball suddenly moving fast. At the first fast frame we make the D -> H/A transition cheaper, so the
dead-ball stretch ends at the kick instead of drifting a few seconds early or late. It only
lowers a switch cost; the emissions still decide whether, and to which team, play restarts.
No frame numbers, no labels: general rules on per-frame features.
"""

import numpy as np


def kicks(feat, cfg, fps=25):
    """(N,) bool: first frame of a fast ball after a calm stretch."""
    rc = cfg["restart"]
    v = feat["ball_speed"]
    obs = feat["ball_obs"] == 1
    fast = obs & (np.nan_to_num(v, nan=0.0) > rc["kick_mps"])
    rise = fast & ~np.concatenate(([False], fast[:-1]))
    pre = max(1, int(round(rc["pre_s"] * fps)))
    # calm before: no observed ball faster than ball_slow_mps, players walking on average
    ball_busy = (obs & (np.nan_to_num(v, nan=0.0) > rc["ball_slow_mps"])).astype(np.int32)
    c = np.concatenate(([0], np.cumsum(ball_busy)))
    t = np.arange(len(v))
    lo = np.maximum(t - pre, 0)
    busy_before = c[t] - c[lo]                     # frames [t-pre, t)
    ps = np.nan_to_num(feat["player_speed"], nan=np.inf)
    cp = np.concatenate(([0.0], np.cumsum(np.where(np.isfinite(ps), ps, 0.0))))
    cn = np.concatenate(([0], np.cumsum(np.isfinite(ps))))
    cnt = np.maximum(cn[t] - cn[lo], 1)
    walk_before = (cp[t] - cp[lo]) / cnt < rc["players_walk_mps"]
    return rise & (busy_before == 0) & walk_before & (t >= pre)


def bonus_matrix(feat, cfg, fps=25):
    """(N, 3, 3) nats added to the D -> H and D -> A transitions at detected kicks."""
    k = kicks(feat, cfg, fps)
    B = np.zeros((len(k), 3, 3))
    B[k, 2, 0] = cfg["restart"]["switch_bonus"]
    B[k, 2, 1] = cfg["restart"]["switch_bonus"]
    return B, int(k.sum())
