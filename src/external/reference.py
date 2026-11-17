"""Reference possession from human-annotated EVENTS of other matches, and ATD-like degradation.

Possession rules (CHALLENGE.md, applied mechanically to events):
  * A team gains possession with a controlled action: PASS, SHOT, SET PIECE, RECOVERY, CARRY,
    or BALL LOST (the losing team had the ball up to that moment).
  * CHALLENGE, FAULT RECEIVED and CARD never change possession by themselves; a loose ball
    stays with the last team that had it.
  * Dead ball: from a stoppage (BALL OUT, a SHOT going out or in, a FAULT RECEIVED) to the next
    SET PIECE -- unless a controlled action happens first (advantage played -> no stoppage).
    Each period starts dead until its kick-off set piece.
Caveat, documented in TECHNICAL.md: the challenge ground truth comes from corrected *tracking*
ownership, not events, so this reference approximates the target definition, it is not it.
"""

import numpy as np

CONTROL = {"PASS", "SHOT", "SET PIECE", "RECOVERY", "CARRY", "BALL LOST"}


def _is_stoppage(ev):
    t, s = ev["type"], ev["subtype"]
    if t == "BALL OUT" or t == "FAULT RECEIVED":
        return True
    if t == "SHOT" and ("OUT" in s or "GOAL" in s):
        return True
    return False


def _require_control(state, evs, first, rc):
    """CHALLENGE.md: a touch, a deflection or an aerial duel without follow-up does not change
    possession; proxies for control are keeping the ball a few seconds or making passes. A team
    run that follows the OTHER team's possession is kept only if that team makes a controlled
    action in it (rc['controlled_actions']) or the run lasts >= rc['min_keep_s']; otherwise it
    is given back to the previous team ("stays with the last team that had it")."""
    acts = {}
    for e in evs:
        if e["type"] in rc["controlled_actions"] and e["team"] is not None:
            acts.setdefault(e["team"], []).append(e["start"] - first)
    acts = {t: np.array(sorted(v)) for t, v in acts.items()}
    min_keep = int(round(rc["min_keep_s"] * 25))
    d = np.flatnonzero(np.diff(state)) + 1
    starts, ends = np.concatenate(([0], d)), np.concatenate((d, [len(state)]))
    prev = -1
    for a, b in zip(starts, ends):
        t = int(state[a])
        if t >= 0 and prev >= 0 and t != prev:
            ts = acts.get(t, np.array([]))
            has = np.any((ts >= a) & (ts < b))
            if not has and b - a < min_keep:
                state[a:b] = prev
                t = prev
        prev = t
    return state


def possession_labels(events, periods, n, rule=None):
    """(n,) array of 'H'/'A'/'D', 'X' outside periods. rule = cfg['external']['reference']."""
    lab = np.full(n, "X", dtype="<U1")
    for p, first, last in periods:
        evs = sorted((e for e in events if e["period"] == p), key=lambda e: (e["start"], e["end"]))
        state = np.full(last - first + 1, -1, dtype=np.int8)       # -1 = dead
        cur = -1                                                    # dead until kick-off
        t_prev = first
        pending_dead = None
        for i, e in enumerate(evs):
            f = min(max(e["start"], first), last)
            state[t_prev - first:f - first] = cur
            t_prev = f
            if e["type"] in CONTROL and e["team"] is not None:
                cur = e["team"]
            if _is_stoppage(e):
                # dead from the end of the stoppage event to the next set piece, unless a
                # controlled non-set-piece action comes first (advantage)
                nxt = next((x for x in evs[i + 1:] if x["type"] in CONTROL), None)
                if nxt is not None and nxt["type"] == "SET PIECE":
                    g = min(max(e["end"], f), last)
                    state[t_prev - first:g - first] = cur
                    t_prev = g
                    cur = -1
        state[t_prev - first:] = cur
        if rule and rule.get("require_control"):
            state = _require_control(state, evs, first, rule)
        lab[first:last + 1] = np.where(state == 0, "H", np.where(state == 1, "A", "D"))
    return lab


def _fill_gaps_hold(ball):
    """Interior NaN runs: first half = last seen position, second half = next seen position."""
    out = ball.copy()
    miss = np.isnan(out[:, 0])
    if not miss.any() or miss.all():
        return out
    d = np.diff(np.concatenate(([0], miss.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if a == 0 or b == len(out):
            continue                                   # leading/trailing gaps stay missing
        mid = a + (b - a) // 2
        out[a:mid] = out[a - 1]
        out[mid:b] = out[b]
    return out


def _fill_gaps_linear(ball):
    """Interior NaN runs: straight line from the last seen to the next seen position."""
    out = ball.copy()
    t = np.arange(len(out))
    ok = ~np.isnan(out[:, 0])
    if ok.sum() < 2:
        return out
    inside = (t > t[ok][0]) & (t < t[ok][-1]) & ~ok
    for j in range(out.shape[1]):
        out[inside, j] = np.interp(t[inside], t[ok], out[ok, j])
    return out


def _smooth_noise(rng, shape, k, std_m):
    z = rng.normal(0.0, 1.0, shape)
    if k > 1:
        c = np.cumsum(np.concatenate([np.zeros((k,) + shape[1:]), z]), axis=0)
        z = (c[k:] - c[:-k]) / np.sqrt(k)       # moving sum / sqrt(k): unit variance kept
    return std_m * z


def degrade(atd, periods, pitch, seed, cfg):
    """Make clean tracking look like the challenge ATD. Deterministic for a given seed."""
    dc = cfg["external"]["degrade"]
    rng = np.random.default_rng(seed)
    L, W = pitch
    ball = atd["ball"].astype(np.float64).copy()
    players = atd["players"].astype(np.float64).copy()
    n, S, _ = players.shape
    team = atd["team"]

    # Metrica's clean tracking drops the ball mostly while it is OUT of play, so "missing" would
    # leak "dead". The challenge ATD loses the ball to occlusion instead. Fill clean gaps with a
    # physical guess (ball rests where it left play for the first half of the gap, then at the
    # restart spot), and only then apply occlusion-like dropouts. Uses tracking only, no labels.
    if dc.get("fill_clean_gaps", True):
        ball = (_fill_gaps_linear(ball) if dc.get("fill_mode", "linear") == "linear"
                else _fill_gaps_hold(ball))

    # position noise (metres -> normalised). Temporally correlated: real ATD positions are smooth
    # (its speed distribution matches clean tracking), so i.i.d. per-frame noise would invent
    # ~5 m/s of fake speed. White noise -> moving average over noise_smooth_frames -> rescaled.
    k = int(dc.get("noise_smooth_frames", 1))
    players += _smooth_noise(rng, players.shape, k, dc["player_noise_m"]) / np.array([L, W])
    ball += _smooth_noise(rng, ball.shape, k, dc["ball_noise_m"]) / np.array([L, W])

    # identity switches: re-shuffle slot order within each team at random times
    t = 0
    while t < n:
        t += int(rng.exponential(dc["id_switch_every_s"] * 25)) + 1
        for tm in (0, 1):
            cols = np.flatnonzero(team == tm)
            perm = rng.permutation(cols)
            players[t:, cols] = players[t:, perm]

    # player dropouts in bursts
    for s in range(S):
        u = rng.random(n) < dc["player_drop_rate"] / dc["player_drop_len"]
        for a in np.flatnonzero(u):
            players[a:a + int(rng.exponential(dc["player_drop_len"])) + 1, s] = np.nan

    # ball jitter (log E18): the real ATD ball position shakes frame to frame, so a ball is almost
    # never "still" over 3 frames (5% vs 13-15% with smooth noise only). i.i.d. noise, metres.
    if dc.get("ball_jitter_m", 0) > 0:
        ball += rng.normal(0.0, dc["ball_jitter_m"], ball.shape) / np.array([L, W])

    graded = np.zeros(n, dtype=bool)
    for _, a, b in periods:
        graded[a:b + 1] = True
    if dc.get("ball_drop_mode", "bursty") == "gaps":
        # log E18: the real ATD loses the ball in ~150 LONG gaps (median ~11 s), not in short
        # bursts. Gap lengths are drawn from the real gap-length deciles (aggregate, no labels),
        # placed uniformly at random -- independent of the game state, so "ball missing" stays
        # uninformative about dead ball -- until the target fraction of graded frames is missing.
        qs = np.asarray(dc["ball_gap_deciles_frames"], dtype=float)
        drop = np.isnan(ball[:, 0]) & graded
        while drop[graded].mean() < dc["ball_missing_target"]:
            ln = int(round(np.interp(rng.random(), np.linspace(0, 1, len(qs)), qs)))
            a = int(rng.integers(0, n))
            if drop[max(0, a - 1):a + max(1, ln) + 1].any():
                continue                      # keep gaps separate, as in the real ATD
            drop[a:a + max(1, ln)] = True
        ball[drop] = np.nan
        out = dict(atd)
        out["ball"] = ball.astype(np.float32)
        out["players"] = players.astype(np.float32)
        out["pspeed"] = np.full((n, S), np.nan, np.float32)
        return out, float(drop[graded].mean())

    # ball dropouts: bursty, more likely in crowds; hazard scaled to hit the target fraction
    bx, by = ball[:, 0] * L, ball[:, 1] * W
    with np.errstate(invalid="ignore"):
        d = np.hypot(players[:, :, 0] * L - bx[:, None], players[:, :, 1] * W - by[:, None])
        crowd = np.nansum(d < 3.0, axis=1)
    weight = 1.0 + dc["ball_drop_crowd_gain"] * crowd
    u = rng.random(n)
    lengths = rng.exponential(dc["ball_drop_len"], n).astype(int) + 1
    already = np.isnan(ball[:, 0]) & graded
    target = dc["ball_missing_target"]

    def mask_for(h0):
        starts = np.flatnonzero(u < h0 * weight)
        m = np.zeros(n + 1, dtype=np.int32)
        np.add.at(m, starts, 1)
        np.add.at(m, np.minimum(starts + lengths[starts], n), -1)
        return (np.cumsum(m)[:n] > 0) | already

    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if mask_for(mid)[graded].mean() < target:
            lo = mid
        else:
            hi = mid
    drop = mask_for(hi)
    ball[drop] = np.nan
    out = dict(atd)
    out["ball"] = ball.astype(np.float32)
    out["players"] = players.astype(np.float32)
    out["pspeed"] = np.full((n, S), np.nan, np.float32)
    return out, float(drop[graded].mean())
