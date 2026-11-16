"""Per-frame features from the ATD arrays. Everything in metres and m/s.

Returned dict, every array of length N (video frames):
    ball_obs      0 = missing, 1 = observed, 2 = interpolated (short gap filled)
    bx, by        ball position (m), NaN where missing and not interpolated
    d_home,d_away distance (m) from the ball to the nearest player of each team (NaN if no ball)
    n_home,n_away players of each team within crowd_radius_m of the ball
    ball_speed    m/s, NaN where unknown
    out           1 where the ball is beyond a touch/goal line by more than out_margin_m
    n_players     tracked players on the frame
"""

import numpy as np

MISSING, OBSERVED, INTERPOLATED = 0, 1, 2


def interpolate_short_gaps(x, max_gap):
    """Linear interpolation over NaN runs of length <= max_gap with known values on both sides."""
    x = x.astype(np.float64).copy()
    filled = np.zeros(len(x), dtype=bool)
    nan = np.isnan(x)
    if not nan.any() or max_gap <= 0:
        return x, filled
    edges = np.diff(np.concatenate(([0], nan.astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    for a, b in zip(starts, stops):
        if a == 0 or b == len(x) or b - a > max_gap:
            continue
        x[a:b] = np.interp(np.arange(a, b), [a - 1, b], [x[a - 1], x[b]])
        filled[a:b] = True
    return x, filled


def moving_mean_nan(x, k):
    """Centred moving mean of width k ignoring NaN (NaN if the whole window is NaN)."""
    if k <= 1:
        return x
    valid = ~np.isnan(x)
    v = np.where(valid, x, 0.0)
    kern = np.ones(k)
    s = np.convolve(v, kern, mode="same")
    c = np.convolve(valid.astype(float), kern, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = s / c
    out[c == 0] = np.nan
    return out


def compute(atd, pitch, cfg, fps=25):
    length, width = pitch
    fc = cfg["features"]
    ball = atd["ball"].astype(np.float64)
    bx_raw, by_raw = ball[:, 0] * length, ball[:, 1] * width
    observed = ~np.isnan(bx_raw)
    bx, fx = interpolate_short_gaps(bx_raw, fc["interp_max_gap"])
    by, _ = interpolate_short_gaps(by_raw, fc["interp_max_gap"])
    ball_obs = np.where(observed, OBSERVED, np.where(fx, INTERPOLATED, MISSING)).astype(np.int8)

    px = atd["players"][:, :, 0].astype(np.float32) * length
    py = atd["players"][:, :, 1].astype(np.float32) * width
    team = atd["team"]
    n = len(bx)
    d_home = np.full(n, np.nan)
    d_away = np.full(n, np.nan)
    n_home = np.zeros(n, dtype=np.int16)
    n_away = np.zeros(n, dtype=np.int16)
    R = fc["crowd_radius_m"]
    chunk = 20000
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        dx = px[a:b] - bx[a:b, None]
        dy = py[a:b] - by[a:b, None]
        d = np.sqrt(dx * dx + dy * dy)  # NaN where player or ball missing
        for t, dist, cnt in ((0, d_home, n_home), (1, d_away, n_away)):
            dt = d[:, team == t]
            allnan = np.all(np.isnan(dt), axis=1)
            dmin = np.full(b - a, np.nan)
            if (~allnan).any():
                dmin[~allnan] = np.nanmin(dt[~allnan], axis=1)
            dist[a:b] = dmin
            with np.errstate(invalid="ignore"):
                cnt[a:b] = np.sum(dt <= R, axis=1)

    with np.errstate(invalid="ignore"):
        vx = np.gradient(bx) * fps
        vy = np.gradient(by) * fps
    speed = moving_mean_nan(np.sqrt(vx * vx + vy * vy), fc["ball_speed_smooth"])
    speed[np.isnan(bx)] = np.nan

    m = fc["out_margin_m"]
    with np.errstate(invalid="ignore"):
        out = ((bx < -m) | (bx > length + m) | (by < -m) | (by > width + m)).astype(np.int8)

    # Collective player speed (m/s): ATD speed channel if present, else from positions. Slot
    # identity switches produce teleports, so per-slot speeds are clipped before averaging.
    if "pspeed" in atd and np.isfinite(atd["pspeed"]).any():
        ps = atd["pspeed"].astype(np.float32)
    else:
        ps = np.sqrt(np.gradient(px, axis=0) ** 2 + np.gradient(py, axis=0) ** 2) * fps
    ps = np.where(ps > fc["player_speed_clip_mps"], np.nan, ps)
    cnt = np.sum(np.isfinite(ps), axis=1)
    mean_ps = np.where(cnt > 0, np.nansum(ps, axis=1) / np.maximum(cnt, 1), np.nan)
    mean_ps = moving_mean_nan(mean_ps, fc["player_speed_smooth"])

    # Length of the current ball-missing run each frame sits in (0 where the ball is known).
    miss = np.isnan(bx)
    miss_run = np.zeros(n, dtype=np.int32)
    edges = np.diff(np.concatenate(([0], miss.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        miss_run[a:b] = b - a

    return {
        "player_speed": mean_ps, "ball_missing_run": miss_run,
        "ball_obs": ball_obs, "bx": bx, "by": by,
        "d_home": d_home, "d_away": d_away, "n_home": n_home, "n_away": n_away,
        "ball_speed": speed, "out": out,
        "n_players": np.sum(~np.isnan(px), axis=1).astype(np.int16),
    }
