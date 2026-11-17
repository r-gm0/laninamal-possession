"""Per-frame log-emission scores for the states H, A, D.

These are log-likelihood *scores* (relative, per frame), not normalised probabilities: Viterbi
only needs differences between states. A missing ball gives an all-zero row, i.e. no evidence,
so the decoded state is carried by the transition model -- the HMM equivalent of "X inside a
possession". A missing ball is NOT evidence of a dead ball.
"""

import numpy as np

STATES = ("H", "A", "D")
H, A, D = 0, 1, 2


def log_sigmoid(z):
    return -np.logaddexp(0.0, -z)


def compute(feat, cfg):
    ec = cfg["emission"]
    n = len(feat["bx"])
    dh, da = feat["d_home"], feat["d_away"]
    have = ~np.isnan(dh) & ~np.isnan(da)

    # Which team is controlling: closer nearest player, plus local numerical superiority.
    logit = np.zeros(n)
    logit[have] = ((da[have] - dh[have]) / ec["dist_scale_m"]
                   + ec["crowd_beta"] * (feat["n_home"][have] - feat["n_away"][have]))
    logit = np.clip(logit, -ec["max_logit"], ec["max_logit"])

    # How much to trust this frame as evidence of control.
    w = np.where(feat["ball_obs"] == 1, ec["weight_observed"],
                 np.where(feat["ball_obs"] == 2, ec["weight_interpolated"], 0.0))
    w = np.where(have, w, 0.0)
    dmin = np.where(have, np.fmin(dh, da), np.inf)
    w = w * np.exp(-np.clip(dmin - ec["control_radius_m"], 0, None) / ec["prox_decay_m"])
    v = np.nan_to_num(feat["ball_speed"], nan=0.0)
    w = w * np.exp(-np.clip(v - ec["ball_speed_free_mps"], 0, None) / ec["ball_speed_decay_mps"])

    E = np.zeros((n, 3))
    E[:, H] = w * log_sigmoid(logit)
    E[:, A] = w * log_sigmoid(-logit)

    # Stage 1 dead-ball evidence.
    out = feat["out"].astype(bool)
    E[out, D] += ec["dead_out_of_bounds"]
    alive = (w > 0) & ~out
    E[alive, D] -= ec["alive_ball_near_player"] * w[alive]

    # Stage 2 dead-ball evidence. Each term is a log-odds bonus added to D; weight 0 disables it.
    dc = cfg["dead"]
    ps = feat.get("player_speed")
    if ps is not None:
        # How "walking" the players are: 0 at >= walk speed, 1 when standing still.
        slow = np.clip((dc["walk_speed_mps"] - np.nan_to_num(ps, nan=dc["walk_speed_mps"]))
                       / dc["walk_speed_mps"], 0.0, 1.0)
        # A slow team with a player on the ball is a slow possession, not a stoppage: gate the
        # slowness terms by the absence of control evidence (w is the control weight above).
        if dc.get("gate_slow_by_control", True):
            slow = slow * (1.0 - np.clip(w, 0.0, 1.0))
        E[:, D] += dc["w_players_slow"] * slow
        # Ball (almost) still while players walk: placed for a restart.
        bstill = (feat["ball_obs"] == 1) & (np.nan_to_num(feat["ball_speed"], nan=99.0)
                                            < dc["ball_still_mps"])
        E[bstill, D] += dc["w_ball_still_and_slow"] * slow[bstill]
        # Long ball absence with players slow -> likely stoppage (absence alone is NOT evidence).
        long_miss = feat["ball_missing_run"] >= int(dc["long_missing_s"] * dc["fps"])
        E[long_miss, D] += dc["w_long_missing_and_slow"] * slow[long_miss]
    # A fast-moving ball is in play.
    fast = np.nan_to_num(feat["ball_speed"], nan=0.0) > dc["ball_fast_mps"]
    E[fast & ~out, D] -= dc["w_fast_ball_alive"]
    return add_tagging(E, feat, cfg)


def add_tagging(E, feat, cfg, phase_scale=1.0):
    """Smart Tagging terms, added in place to any (n, 3) emission matrix (absent on external games).

    phase_scale rescales the team-phase prior so it keeps its weight relative to the tracking
    evidence when the tracking emissions change scale (learned emissions, log E17)."""
    # Smart Tagging restarts (soft prior, approximate timing).
    sp = feat.get("set_piece_prior")
    if sp is not None:
        E[:, D] += cfg["dead"]["w_set_piece"] * sp

    # Stage 3: Smart Tagging phases with a Team (+ = home has the ball). Split symmetrically so
    # it tilts H vs A without moving D on average. Capped so overlapping tags cannot dominate.
    tp = feat.get("team_phase_prior")
    if tp is not None:
        cap = cfg["tagging"]["phase_cap"]
        t = phase_scale * np.clip(tp, -cap, cap)
        E[:, H] += 0.5 * t
        E[:, A] -= 0.5 * t
    return E
