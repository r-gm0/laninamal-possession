"""Smart Tagging -> per-frame priors. Evidence only, never copied as a label.

Stage 2 uses only SET PIECES (restarts): around a tagged restart the ball is likely dead, then
alive. The tag intervals are approximate (+-1-3 s), so the prior is a soft, configurable bump
over a window, and the tracking decides the exact frame.
Time base (DATA.md): video_frame = round(seconds * 25).
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))


def load_events(game=None, data_dir=None):
    import load_smart_tagging as lst  # stdlib-only loader shipped with the repo
    from games import GAME
    return lst.load(game or GAME, data_dir)


def set_piece_prior(events, n_frames, cfg, fps=25):
    """(n,) float in [0,1]: 1 inside the window where a tagged restart's dead ball is expected.

    Window = [start - pre_s, start + frac * (end - start)] in frames, per set piece. Which types
    are used is configurable (all of them by default).
    """
    tc = cfg["tagging"]
    types = {t.lower() for t in tc["set_piece_types"]}
    prior = np.zeros(n_frames, dtype=np.float32)
    n_used = 0
    for ev in events:
        if (ev["code"] or "").strip().lower() != "set pieces":
            continue
        typ = (ev["tags"].get("Type") or "").strip().lower()
        if types and typ not in types:
            continue
        a = ev["start_frame"] - int(round(tc["set_piece_pre_s"] * fps))
        b = ev["start_frame"] + int(round(tc["set_piece_frac"] * (ev["end_frame"] - ev["start_frame"])))
        a, b = max(0, a), min(n_frames, b)
        if b > a:
            prior[a:b] = 1.0
            n_used += 1
    return prior, n_used


# -- Stage 3: phases with a Team as evidence of who has the ball ---------------------------------

def map_team_names(tag_names, atd_names, overrides=None):
    """{tagging team string -> 0 (home) / 1 (away)} by string similarity to the ATD club names.

    The two exports spell clubs differently, so this is fuzzy, automatic and reported. 'N/A' and
    empty strings map to nothing (handled explicitly: no team prior from those tags).
    """
    import difflib
    overrides = overrides or {}
    out = {}
    for name in tag_names:
        key = (name or "").strip()
        if not key or key.lower() in ("n/a", "na", "none"):
            continue
        if key in overrides:
            out[key] = {"H": 0, "A": 1}[overrides[key]]
            continue
        scores = [difflib.SequenceMatcher(None, key.lower(), a.lower()).ratio() for a in atd_names]
        out[key] = int(np.argmax(scores))
    return out


def team_phase_prior(events, n_frames, atd_names, cfg):
    """(n,) float: positive = evidence home has the ball, negative = away. Plus a report dict.

    Each phase code has a signed weight in cfg['tagging']['phase_weights']: + means the tagged
    Team is in possession (Build Up, Progression...), - means the tagged Team is defending, so
    the OTHER team has the ball. Codes absent from the table give nothing.
    """
    tc = cfg["tagging"]
    weights = {k.lower(): float(v) for k, v in tc["phase_weights"].items()}
    names = {(ev["tags"].get("Team") or "").strip() for ev in events}
    mapping = map_team_names(names, list(atd_names), tc.get("team_name_overrides"))
    prior = np.zeros(n_frames, dtype=np.float64)
    report = {"n_used": 0, "n_no_team": 0, "mapping": mapping}
    for ev in events:
        wgt = weights.get((ev["code"] or "").strip().lower(), 0.0)
        if wgt == 0.0:
            continue
        team = mapping.get((ev["tags"].get("Team") or "").strip())
        if team is None:
            report["n_no_team"] += 1
            continue
        sign = 1.0 if team == 0 else -1.0
        a, b = max(0, ev["start_frame"]), min(n_frames, ev["end_frame"])
        if b > a:
            prior[a:b] += sign * wgt
            report["n_used"] += 1
    return prior, report


def transition_hints(events, n_frames, cfg, fps=25):
    """(n,) float in [0,1]: 1 near the start of a tagged transition (possession likely changes)."""
    tc = cfg["tagging"]
    codes = {c.lower() for c in tc["transition_codes"]}
    pre, post = int(round(tc["transition_pre_s"] * fps)), int(round(tc["transition_post_s"] * fps))
    hint = np.zeros(n_frames, dtype=np.float64)
    n_used = 0
    for ev in events:
        if (ev["code"] or "").strip().lower() not in codes:
            continue
        a, b = max(0, ev["start_frame"] - pre), min(n_frames, ev["start_frame"] + post)
        if b > a:
            hint[a:b] = 1.0
            n_used += 1
    return hint, n_used
