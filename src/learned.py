"""Learned per-frame evidence (log E17): tracking features -> gradient-boosted trees -> log-odds.

The feature set (multi-scale team activity, ball visibility/out-of-play timing, nearest-player
advantage with past/future context, touches, team shape, closing speed, ball heading) is ported
from our teammate's pipeline (fermartinezm/laninamal-mtah2027, possession/learned.py, features
version 2). What differs here is the TRAINING DATA: the trees are fitted on our degraded Metrica
games (ball removed on 41% of frames in occlusion-like bursts, so "ball missing" no longer leaks
"ball dead") against our event-derived reference, and their log-odds feed our minimum-duration
Viterbi. Training: src/train_learned.py (scikit-learn, external games only). Inference here is
numpy only, from the committed src/learned_model.json, so the submission stays deterministic.
"""

import json
import os
import warnings

import numpy as np

FPS = 25
MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "learned_model.json")
OUT_MARGIN_M = 0.5     # m beyond the lines before the ball counts as out
OUT_INVALID_M = 12.0   # m beyond the lines = impossible position, treated as missing


# ------------------------------------------------------------------------------------ helpers
def _mean_window(x, k):
    """Centred moving mean over k frames, NaN-aware."""
    ok = np.isfinite(x)
    w = np.ones(k)
    num = np.convolve(np.where(ok, x, 0.0), w, "same")
    den = np.convolve(ok.astype(float), w, "same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _shift_mean(x, k, past=True):
    """Mean of x over the k frames before (past=True) or after the current frame."""
    c = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x))])
    n = len(x)
    i = np.arange(n)
    if past:
        a, b = np.maximum(0, i - k), i
    else:
        a, b = i + 1, np.minimum(n, i + 1 + k)
    return (c[b] - c[a]) / np.maximum(1, b - a)


def _run_since(mask):
    """Length of the run of True ending at each frame (0 where False)."""
    m = np.asarray(mask, dtype=bool)
    idx = np.arange(len(m))
    last_false = np.where(~m, idx, -1)
    np.maximum.accumulate(last_false, out=last_false)
    return (idx - last_false).astype(np.float64)


def _since_true(mask, cap):
    """Frames since mask was last True, capped."""
    m = np.asarray(mask, dtype=bool)
    idx = np.arange(len(m))
    last = np.where(m, idx, -10 ** 9)
    np.maximum.accumulate(last, out=last)
    return np.minimum(cap, idx - last).astype(np.float64)


def _ffill(x):
    idx = np.where(np.isfinite(x), np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    return x[idx]


def to_metres(atd, pitch):
    """ATD dict (normalised 0-1) -> players (n, S, 3: x, y, speed m/s), ball (n, 2), metres."""
    L, W = pitch
    xy = atd["players"].astype(np.float64) * np.array([L, W])
    ball = atd["ball"].astype(np.float64) * np.array([L, W])
    sp = np.full(xy.shape[:2], np.nan)
    sp[5:] = np.hypot(xy[5:, :, 0] - xy[:-5, :, 0], xy[5:, :, 1] - xy[:-5, :, 1]) * FPS / 5.0
    return np.concatenate([xy, sp[:, :, None]], axis=2), ball


def _ball_kinematics(ball, pitch):
    length, width = pitch
    x, y = ball[:, 0].copy(), ball[:, 1].copy()
    with np.errstate(invalid="ignore"):
        beyond = np.maximum.reduce([-x, x - length, -y, y - width])
        invalid = np.isfinite(beyond) & (beyond > OUT_INVALID_M)
        x[invalid] = np.nan
        y[invalid] = np.nan
        visible = np.isfinite(x)
        out = visible & (beyond > OUT_MARGIN_M)
    sp = np.full(len(x), np.nan)
    sp[3:] = np.hypot(x[3:] - x[:-3], y[3:] - y[:-3]) * FPS / 3.0
    return x, y, visible, out, sp


# ----------------------------------------------------------------------------------- features
def frame_features(atd, pitch):
    """(F_dead (n, 27), F_team (n, 17)) float32. Team features are home-minus-away, so negating
    them describes the same frame with the teams swapped."""
    players, ball = to_metres(atd, pitch)
    team = np.asarray(atd["team"])
    length, width = pitch
    x, y, visible, out, sp = _ball_kinematics(ball, pitch)
    n = len(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pspeed = players[:, :, 2]
        mean_sp = np.nanmean(pspeed, axis=1)
        fast_frac = np.nanmean(np.where(np.isfinite(pspeed), pspeed > 2.0, np.nan), axis=1)
        n_tracked = np.isfinite(players[:, :, 0]).sum(1).astype(float)

    act05 = _mean_window(mean_sp, 12)
    act2 = _mean_window(mean_sp, 50)
    act6 = _mean_window(mean_sp, 150)
    fast2 = _mean_window(fast_frac, 50)
    act_past = _shift_mean(np.nan_to_num(mean_sp, nan=1.5), 75, past=True)
    act_fut = _shift_mean(np.nan_to_num(mean_sp, nan=1.5), 75, past=False)

    miss = ~visible
    miss_back = _run_since(miss) / FPS
    miss_fwd = _run_since(miss[::-1])[::-1] / FPS
    vis1 = _mean_window(visible.astype(float), 50)
    vis3 = _mean_window(visible.astype(float), 150)
    out_since = _since_true(out, 250) / FPS
    out_until = _since_true(out[::-1], 250)[::-1] / FPS

    spd = np.where(visible, np.nan_to_num(sp), np.nan)
    bs1 = _mean_window(spd, 25)
    bs_max = np.nan_to_num(_mean_window(np.where(np.isfinite(spd), spd ** 2, np.nan), 50)) ** 0.5

    lx = np.nan_to_num(_ffill(x), nan=length / 2)
    ly = np.nan_to_num(_ffill(y), nan=width / 2)
    d_goal = np.minimum(np.abs(lx), np.abs(length - lx))
    d_touch = np.minimum(np.abs(ly), np.abs(width - ly))

    px, py = players[:, :, 0], players[:, :, 1]
    d = np.hypot(px - lx[:, None], py - ly[:, None])
    home = team[None, :] == 0
    big = 60.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dh = np.nan_to_num(np.nanmin(np.where(home, d, big), axis=1), nan=big)
        da = np.nan_to_num(np.nanmin(np.where(~home, d, big), axis=1), nan=big)
        near = np.minimum(dh, da)
        within3 = np.nansum(d < 3, axis=1).astype(float)
        within10 = np.nansum(d < 10, axis=1).astype(float)
        h5 = np.nansum((d < 5) & home, axis=1).astype(float)
        a5 = np.nansum((d < 5) & ~home, axis=1).astype(float)
        h10 = np.nansum((d < 10) & home, axis=1).astype(float)
        a10 = np.nansum((d < 10) & ~home, axis=1).astype(float)

    # touches: sharp change of ball speed next to a player
    dv = np.zeros(n)
    s0 = np.nan_to_num(sp)
    dv[5:] = np.abs(s0[5:] - s0[:-5])
    touch = (visible & (dv > 4.0) & (near < 2.5)).astype(float)
    touch_side = touch * np.where(dh <= da, 1.0, -1.0)

    # which team is goal-side of the ball (team centroids along x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        hcx = np.nan_to_num(np.nanmean(np.where(home, px, np.nan), axis=1), nan=length / 2)
        acx = np.nan_to_num(np.nanmean(np.where(~home, px, np.nan), axis=1), nan=length / 2)
    shape = np.sign(acx - hcx) * (lx - (hcx + acx) / 2) / 10.0

    adv = np.clip(da - dh, -15, 15)
    slow = (np.nan_to_num(sp, nan=20) < 7).astype(float)
    adv_vis = np.where(visible, adv, 0.0)
    F_dead = [act05, act2, act6, act2 - act6, act_fut - act_past, fast2, n_tracked,
              miss_back, miss_fwd, vis1, vis3, out.astype(float), out_since, out_until,
              np.nan_to_num(bs1, nan=-1), bs_max, d_goal, d_touch, near, within3, within10,
              visible.astype(float)]
    F_team = [adv_vis, adv_vis * slow, _shift_mean(adv_vis, 25, True), _shift_mean(adv_vis, 25, False),
              _shift_mean(adv_vis, 75, True), _shift_mean(adv_vis, 75, False), touch_side,
              _shift_mean(touch_side, 50, True) * 50, _shift_mean(touch_side, 50, False) * 50,
              h5 - a5, h10 - a10, shape, visible.astype(float) * np.sign(adv)]

    # version-2 extras: closing speed of each team, ball heading, ball stillness, long context
    ddh = np.zeros(n)
    dda = np.zeros(n)
    ddh[12:] = dh[12:] - dh[:-12]
    dda[12:] = da[12:] - da[:-12]
    bvx = np.zeros(n)
    bvy = np.zeros(n)
    bvx[3:] = lx[3:] - lx[:-3]
    bvy[3:] = ly[3:] - ly[:-3]
    bnorm = np.hypot(bvx, bvy) + 1e-6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dhm = np.where(home, d, big)
        dam = np.where(~home, d, big)
        ih = np.argmin(np.where(np.isfinite(dhm), dhm, np.inf), axis=1)
        ia = np.argmin(np.where(np.isfinite(dam), dam, np.inf), axis=1)
    rows = np.arange(n)
    hx_, hy_ = np.nan_to_num(px[rows, ih] - lx), np.nan_to_num(py[rows, ih] - ly)
    ax_, ay_ = np.nan_to_num(px[rows, ia] - lx), np.nan_to_num(py[rows, ia] - ly)
    cos_h = (bvx * hx_ + bvy * hy_) / (bnorm * (np.hypot(hx_, hy_) + 1e-6))
    cos_a = (bvx * ax_ + bvy * ay_) / (bnorm * (np.hypot(ax_, ay_) + 1e-6))
    heading = (cos_h - cos_a) * (bnorm * FPS / 3 > 3).astype(float)
    still = _mean_window((np.nan_to_num(sp, nan=0) < 0.5).astype(float) * visible, 75)
    act15 = _mean_window(mean_sp, 375)
    close = np.clip(dda - ddh, -10, 10)
    F_dead += [still, act15, act2 - act15, _mean_window(visible.astype(float), 250),
               _shift_mean(out.astype(float), 125, False)]
    F_team += [close, heading, _shift_mean(heading, 25, False), _shift_mean(close, 25, False)]

    F_dead = np.nan_to_num(np.stack(F_dead, 1), nan=-1.0).astype(np.float32)
    F_team = np.nan_to_num(np.stack(F_team, 1), nan=0.0).astype(np.float32)
    return F_dead, F_team


# ------------------------------------------------------------------------------ tree inference
def _tree_predict(tree, X):
    f = np.asarray(tree["feature"])
    t = np.asarray(tree["threshold"], dtype=np.float32)
    left = np.asarray(tree["left"])
    right = np.asarray(tree["right"])
    v = np.asarray(tree["value"])
    node = np.zeros(len(X), dtype=np.int64)
    rows = np.arange(len(X))
    for _ in range(64):
        leaf = left[node] < 0
        if leaf.all():
            break
        go_left = X[rows, np.maximum(f[node], 0)] <= t[node]
        node = np.where(leaf, node, np.where(go_left, left[node], right[node]))
    return v[node]


def ensemble_logodds(model, X):
    out = np.full(len(X), model["init"], dtype=np.float64)
    for tree in model["trees"]:
        out += model["lr"] * _tree_predict(tree, X)
    return out


def raw_logodds(model, F_dead, F_team):
    """(dead log-odds, home-vs-away log-odds), the team score symmetrised: (f(x) - f(-x)) / 2."""
    d = ensemble_logodds(model["dead"], F_dead)
    t = 0.5 * (ensemble_logodds(model["team"], F_team) - ensemble_logodds(model["team"], -F_team))
    return d, t


def emission_matrix(d, t, alpha, beta):
    """(n, 3) log-emission scores for H, A, D from the two log-odds, scaled per frame."""
    return np.stack([-alpha * d / 2 + beta * t / 2, -alpha * d / 2 - beta * t / 2, alpha * d / 2], 1)


def load_model(path=MODEL_FILE):
    with open(path) as fh:
        return json.load(fh)


def compute(atd, pitch, model):
    Fd, Ft = frame_features(atd, pitch)
    d, t = raw_logodds(model, Fd, Ft)
    return emission_matrix(d, t, model["alpha"], model["beta"])
