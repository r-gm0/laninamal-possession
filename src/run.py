"""Regenerate the submission from the provided data with one command.

    python -m src.run --out submissions/<team_name>.csv [--config src/config.yaml] [--check]

Deterministic: no randomness, fixed tie-breaking, so two runs produce byte-identical files.
Prints aggregates only (class counts, possession counts and durations) -- never per-frame values.
--check additionally scores the file against the PUBLIC 5-minute window as a pipeline sanity
check. That number must not be used to choose parameters (RULES.md, EVALUATION.md).
"""

import argparse
import hashlib
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks, source_frames  # noqa: E402
from src import atd as atd_mod  # noqa: E402
from src import emissions, features, learned, restart, tagging, viterbi, xpolicy  # noqa: E402

LABELS = np.array(["H", "A", "D"])
# Transitions a Smart Tagging "transition" hint makes cheaper: H <-> A only.
SWITCH_MASK = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)


def load_config(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def hmm_params(cfg):
    hc = cfg["hmm"]
    sw, de, dx = hc["switch_penalty"], hc["dead_enter_penalty"], hc["dead_exit_penalty"]
    penalty = np.array([[0, sw, de],
                        [sw, 0, de],
                        [dx, dx, 0]], dtype=float)
    min_dur = [hc["min_dur_possession"], hc["min_dur_possession"], hc["min_dur_dead"]]
    prior = [hc["init_logprior"][s] for s in emissions.STATES]
    return min_dur, penalty, prior


def predict(atd, cfg, game=GAME, events=None):
    """Label array (N,) of 'H'/'A'/'D'/'X' for the challenge match. X outside the graded halves.

    events: Smart Tagging instances (as starter/load_smart_tagging.load returns). None = load
    them from the data bundle when a tagging feature is switched on in the config.
    """
    uses_tags = (cfg["dead"].get("use_smart_tagging") or cfg["tagging"].get("use_team_phases")
                 or cfg["tagging"].get("use_transition_hints"))
    if uses_tags and events is None:
        events = tagging.load_events(game)
    return decode_match(atd, cfg, half_video_blocks(game), source_frames("tactical", game),
                        GAMES[game]["pitch"], GAMES[game]["fps"], events)


def decode_match(atd, cfg, blocks, n, pitch, fps=25, events=None, feat=None):
    """Game-agnostic core: ATD arrays -> labels. blocks = [(half, first, last)] frames to decode.

    Used for the challenge match and, identically, for the external validation matches (Stage 5),
    where events=None and the Smart Tagging terms are simply absent.
    """
    # feat may be passed precomputed (it depends only on cfg['features']); copied, never mutated
    feat = dict(feat) if feat is not None else features.compute(atd, pitch, cfg, fps)
    tc = cfg["tagging"]
    bonus = None
    if events is not None:
        if cfg["dead"].get("use_smart_tagging"):
            feat["set_piece_prior"], feat["n_set_pieces_used"] = tagging.set_piece_prior(
                events, n, cfg, fps)
        if tc.get("use_team_phases"):
            feat["team_phase_prior"], feat["team_phase_report"] = tagging.team_phase_prior(
                events, n, atd["team_names"], cfg)
            mapped = sorted(set(feat["team_phase_report"]["mapping"].values()))
            if mapped != [0, 1]:
                raise ValueError("Smart Tagging team names did not map onto both ATD teams; "
                                 "set tagging.team_name_overrides in the config")
        if tc.get("use_transition_hints"):
            hint, feat["n_transition_hints"] = tagging.transition_hints(events, n, cfg, fps)
            bonus = hint * tc["transition_switch_bonus"]
    if cfg["emission"].get("mode", "rules") == "learned":
        # log E17: tracking evidence from trees trained on external games; Smart Tagging on top
        lm = learned.load_model(os.path.join(ROOT, cfg["emission"]["learned_model"]))
        E = learned.compute(atd, pitch, lm)
        E[:, 2] += cfg["emission"].get("learned_dead_bias", 0.0)   # log E19
        E = emissions.add_tagging(E, feat, cfg, cfg["tagging"].get("learned_phase_scale", 1.0))
    else:
        E = emissions.compute(feat, cfg)
    min_dur, penalty, prior = hmm_params(cfg)
    if cfg.get("restart", {}).get("enabled"):
        B, feat["n_restart_kicks"] = restart.bonus_matrix(feat, cfg, fps)
        if bonus is not None:
            B = B + bonus[:, None, None] * SWITCH_MASK
        bonus = B
    labels = np.full(n, "X", dtype="<U1")
    for _, first, last in blocks:
        path = viterbi.decode(E[first:last + 1], min_dur, penalty, prior,
                              None if bonus is None else bonus[first:last + 1], SWITCH_MASK)
        labels[first:last + 1] = LABELS[path]
    labels = xpolicy.apply(labels, E, blocks, cfg)
    return labels, feat


def summarise(labels, game=GAME, fps=25):
    """Aggregate stats of a prediction inside the graded halves."""
    n_spells, durs, counts = 0, [], {k: 0 for k in "HADX"}
    for _, first, last in half_video_blocks(game):
        w = labels[first:last + 1]
        for k in counts:
            counts[k] += int(np.sum(w == k))
        brk = np.flatnonzero(w[1:] != w[:-1]) + 1
        bounds = np.concatenate(([0], brk, [len(w)]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            if w[a] in ("H", "A"):
                n_spells += 1
                durs.append((b - a) / fps)
    durs = np.array(durs) if durs else np.zeros(1)
    return {"counts": counts, "n_possessions": n_spells, "mean_dur_s": float(durs.mean()),
            "median_dur_s": float(np.median(durs)), "n_under_2s": int(np.sum(durs < 2.0))}


def write_csv(path, labels):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def public_check(path):
    sys.path.insert(0, os.path.join(ROOT, "evaluation"))
    import possession_metrics as pm
    gt_path = os.path.join(ROOT, "evaluation", "gt", "{}_possession_gt_public.csv".format(GAME))
    gt = pm.label_codes(pm.read_label_csv(gt_path))
    pred = pm.label_codes(pm.read_label_csv(path))
    r = pm.evaluate(gt, pred, bootstrap_n=1000)
    print("\n[pipeline check only -- public 5 min, 17 possessions; NOT a basis for tuning]")
    print("  spellF1 {:.4f} (90% CI {:.3f}-{:.3f})  macroF1 {:.4f}  poss {} (gt {})  dur {:.1f}s (gt {:.1f}s)"
          .format(r["spell_f1_mean"], r["spell_bootstrap_low"], r["spell_bootstrap_high"],
                  r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"],
                  r["mean_spell_duration_s"], r["gt_mean_spell_duration_s"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=os.path.join(ROOT, "src", "config.yaml"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true", help="pipeline check on the public window")
    args = ap.parse_args()

    cfg = load_config(args.config)
    np.random.seed(cfg.get("seed", 0))
    atd = atd_mod.load(os.environ.get("MTAH_CACHE_DIR") or cfg["data"]["cache_dir"])
    labels, feat = predict(atd, cfg)
    digest = write_csv(args.out, labels)
    s = summarise(labels)
    print("wrote {}  sha256 {}".format(args.out, digest[:16]))
    if "n_set_pieces_used" in feat:
        print("smart tagging: {} set-piece windows used as D prior".format(feat["n_set_pieces_used"]))
    if "team_phase_report" in feat:
        rep = feat["team_phase_report"]
        print("smart tagging: {} team phases used, {} skipped for Team=N/A; name->side {}".format(
            rep["n_used"], rep["n_no_team"],
            {k: "HA"[v] for k, v in rep["mapping"].items()}))
    if "n_transition_hints" in feat:
        print("smart tagging: {} transition hints".format(feat["n_transition_hints"]))
    if "n_restart_kicks" in feat:
        print("restart detector: {} kicks after a calm stretch".format(feat["n_restart_kicks"]))
    d_runs = []
    for _, first, last in half_video_blocks(GAME):
        w = labels[first:last + 1] == "D"
        e = np.diff(np.concatenate(([0], w.astype(np.int8), [0])))
        d_runs += list((np.flatnonzero(e == -1) - np.flatnonzero(e == 1)) / 25.0)
    if d_runs:
        print("dead-ball stretches {}  mean {:.1f}s".format(len(d_runs), float(np.mean(d_runs))))
    print("graded frames: H {H} A {A} D {D} X {X}".format(**s["counts"]))
    print("possessions {}  mean {:.1f}s  median {:.1f}s  shorter than 2s: {}".format(
        s["n_possessions"], s["mean_dur_s"], s["median_dur_s"], s["n_under_2s"]))
    if s["n_possessions"] > 600 or s["mean_dur_s"] < 4.0:
        print("WARNING: possession count/duration looks like flicker")
    if args.check:
        public_check(args.out)


if __name__ == "__main__":
    main()
