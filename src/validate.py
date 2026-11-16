"""Stage 5: score the pipeline on OTHER matches (Metrica open sample data), never the challenge one.

    python -m src.validate [--config src/config.yaml] [--games Sample_Game_1 Sample_Game_2]

For each game: reference possession from its events -> degrade its tracking to look like the
challenge ATD (seeded) -> run the exact same decoder (no Smart Tagging: these games have none)
-> score with the official metric library. The starter baseline (nearest player + majority
vote +-50 frames) is scored on the same degraded data, so the comparison is like for like.
Aggregates only.
"""

import argparse
import copy
import os
import sys
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "starter"))
import possession_metrics as pm  # noqa: E402
from src import features, run  # noqa: E402
from src.external import metrica, reference  # noqa: E402

PITCH = (105.0, 68.0)


def external_cfg(cfg):
    """The pipeline config with the Smart Tagging terms off (external games have no tagging)."""
    c = copy.deepcopy(cfg)
    c["dead"]["use_smart_tagging"] = False
    c["tagging"]["use_team_phases"] = False
    c["tagging"]["use_transition_hints"] = False
    return c


def prepare(cfg, games=None, base=metrica.DEFAULT_DIR, verbose=True):
    """[(name, degraded_atd, reference_labels, periods)] -- loaded once, reused by tuning."""
    games = games or metrica.available_games(base)
    if not games:
        raise IOError("no external games found in {} -- see src/external/metrica.py".format(base))
    out = []
    for name in games:
        atd, events, periods = metrica.load_game(name, base)
        ref = reference.possession_labels(events, periods, len(atd["ball"]),
                                          cfg["external"].get("reference"))
        seed = cfg["seed"] + zlib.crc32(name.encode())       # deterministic per game
        deg, miss = reference.degrade(atd, periods, PITCH, seed, cfg)
        if verbose:
            spells = pm.possession_stats(pm.label_codes(ref), pm.valid_runs(pm.label_codes(ref)))
            print("{}: {} periods, reference {} possessions (mean {:.1f}s), "
                  "D {:.0%} of graded, ball missing after degradation {:.0%}".format(
                      name, len(periods), spells["n_spells"], spells["mean_duration_s"],
                      np.mean(ref[ref != "X"] == "D"), miss))
        out.append((name, deg, ref, periods))
    return out


def baseline_smooth50(atd, periods, n):
    """The starter baseline logic on arrays: nearest player's team, D where no ball, +-50 vote."""
    import baseline as bl
    c = {"features": {"interp_max_gap": 0, "crowd_radius_m": 5.0, "ball_speed_smooth": 1,
                      "out_margin_m": 0.5, "player_speed_clip_mps": 12.0,
                      "player_speed_smooth": 1}}
    f = features.compute(atd, PITCH, c)
    codes = np.where(np.isnan(f["d_home"]) & np.isnan(f["d_away"]), -1,
                     np.where(np.fmin(f["d_home"], np.inf) <= np.fmin(f["d_away"], np.inf), 0, 1))
    codes = np.where(np.isnan(f["bx"]), -1, codes).astype(np.int8)
    lab = bl.smooth(codes, 50)
    out = np.full(n, "X", dtype="<U1")
    for _, a, b in periods:
        out[a:b + 1] = lab[a:b + 1]
    return out


_FEAT_CACHE = {}


def _features(name, deg, cfg):
    """Per-frame features depend only on cfg['features']; cache them across tuning runs."""
    key = (name, repr(sorted(cfg["features"].items())))
    if key not in _FEAT_CACHE:
        _FEAT_CACHE[key] = features.compute(deg, PITCH, cfg, 25)
    return _FEAT_CACHE[key]


def score(cfg, prepared, with_baseline=False, verbose=True):
    ecfg = external_cfg(cfg)
    rows = []
    for name, deg, ref, periods in prepared:
        lab, _ = run.decode_match(deg, ecfg, periods, len(ref), PITCH, 25, None,
                                  feat=_features(name, deg, ecfg))
        gt = pm.label_codes(ref)
        r = pm.evaluate(gt, pm.label_codes(lab))
        row = {"game": name, "spell": r["spell_f1_mean"], "macro": r["macro_f1"],
               "poss": r["n_pred_spells"], "gt_poss": r["n_gt_spells"],
               "dur": r["mean_spell_duration_s"]}
        if with_baseline:
            b = pm.evaluate(gt, pm.label_codes(baseline_smooth50(deg, periods, len(ref))))
            row.update({"b_spell": b["spell_f1_mean"], "b_macro": b["macro_f1"],
                        "b_poss": b["n_pred_spells"]})
        rows.append(row)
        if verbose:
            msg = "  {:<14} spellF1 {:.3f} macroF1 {:.3f} poss {} (ref {}) dur {:.1f}s".format(
                name, row["spell"], row["macro"], row["poss"], row["gt_poss"], row["dur"])
            if with_baseline:
                msg += "   | baseline_smooth50 spellF1 {:.3f} macroF1 {:.3f} poss {}".format(
                    row["b_spell"], row["b_macro"], row["b_poss"])
            print(msg)
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "game"}
    if verbose:
        print("  {:<14} spellF1 {:.3f} macroF1 {:.3f}".format("MEAN", mean["spell"], mean["macro"])
              + ("   | baseline spellF1 {:.3f} macroF1 {:.3f}".format(mean["b_spell"], mean["b_macro"])
                 if with_baseline else ""))
    return mean, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=os.path.join(ROOT, "src", "config.yaml"))
    ap.add_argument("--games", nargs="*")
    ap.add_argument("--base", default=metrica.DEFAULT_DIR)
    args = ap.parse_args()
    cfg = run.load_config(args.config)
    prepared = prepare(cfg, args.games, args.base)
    score(cfg, prepared, with_baseline=True)


if __name__ == "__main__":
    main()
