"""Train the learned evidence model (log E17) on EXTERNAL games only and export it for numpy inference.

    pip install -r requirements-train.txt        # scikit-learn, training only
    python -m src.train_learned [--out src/learned_model.json]

1. Metrica sample games 1-3: our event-derived reference + our degradation (src/external), the
   same data as src.validate. The challenge match is never read here.
2. Leave-one-game-out: trees fitted on two games give out-of-fold log-odds on the third.
3. The emission scales (alpha for dead ball, beta for home-vs-away) are chosen on those
   out-of-fold log-odds, decoded with the configured minimum-duration HMM, by mean spellF1.
4. The final trees are fitted on all three games and written with the chosen scales.
Deterministic (fixed random_state, fixed degradation seeds). Prints aggregates only.
"""

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import GradientBoostingClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from src import learned, run, validate, viterbi  # noqa: E402

GB = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.7, random_state=0)
STRIDE = 4                     # every 4th graded frame: neighbouring frames are near-duplicates
ALPHAS = [0.2, 0.35, 0.6]      # dead-ball scale grid (log E17 picked 0.2-0.35)
BETAS = [0.05, 0.1, 0.2]       # home-vs-away scale grid (log E17 picked 0.1, the grid's edge)
PITCH = validate.PITCH
LAB = np.array(["H", "A", "D"])


def _export(clf):
    trees = []
    for est in clf.estimators_[:, 0]:
        t = est.tree_
        trees.append(dict(feature=t.feature.tolist(), threshold=[float(v) for v in t.threshold],
                          left=t.children_left.tolist(), right=t.children_right.tolist(),
                          value=[float(v) for v in t.value[:, 0, 0]]))
    init = float(clf._raw_predict_init(np.zeros((1, clf.n_features_in_), dtype=np.float32))[0, 0])
    return dict(init=init, lr=float(clf.learning_rate), trees=trees)


def fit(games):
    Xd, yd, Xt, yt = [], [], [], []
    for g in games:
        ref = g["ref"]
        idx = np.flatnonzero(ref != "X")[::STRIDE]
        Xd.append(g["Fd"][idx])
        yd.append(ref[idx] == "D")
        alive = idx[ref[idx] != "D"]
        Xt.append(g["Ft"][alive])
        yt.append(ref[alive] == "H")
    Xd, yd, Xt, yt = map(np.concatenate, (Xd, yd, Xt, yt))
    Xt, yt = np.concatenate([Xt, -Xt]), np.concatenate([yt, ~yt])   # home/away symmetry
    return dict(dead=_export(GradientBoostingClassifier(**GB).fit(Xd, yd)),
                team=_export(GradientBoostingClassifier(**GB).fit(Xt, yt)))


def decode_score(g, d, t, alpha, beta, hmm):
    E = learned.emission_matrix(d, t, alpha, beta)
    lab = np.full(len(g["ref"]), "X", dtype="<U1")
    for _, a, b in g["periods"]:
        lab[a:b + 1] = LAB[viterbi.decode(E[a:b + 1], *hmm)]
    r = pm.evaluate(pm.label_codes(g["ref"]), pm.label_codes(lab))
    return r["spell_f1_mean"], r["macro_f1"], float(np.mean(lab[g["ref"] != "X"] == "D"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=os.path.join(ROOT, "src", "config.yaml"))
    ap.add_argument("--out", default=learned.MODEL_FILE)
    args = ap.parse_args()
    t0 = time.time()
    cfg = run.load_config(args.config)
    hmm = run.hmm_params(cfg)
    games = []
    for name, deg, ref, periods in validate.prepare(cfg):
        Fd, Ft = learned.frame_features(deg, PITCH)
        games.append(dict(name=name, ref=ref, periods=periods, Fd=Fd, Ft=Ft))
    print("features {} dead + {} team ({:.0f}s)".format(Fd.shape[1], Ft.shape[1], time.time() - t0),
          flush=True)

    oof = []
    for h, g in enumerate(games):
        m = fit([x for i, x in enumerate(games) if i != h])
        oof.append(learned.raw_logodds(m, g["Fd"], g["Ft"]))
        print("  out-of-fold log-odds for {} ({:.0f}s)".format(g["name"], time.time() - t0), flush=True)

    grid = {}
    for a, b in itertools.product(ALPHAS, BETAS):
        grid[(a, b)] = [decode_score(g, d, t, a, b, hmm) for g, (d, t) in zip(games, oof)]
        print("  alpha {:.2f} beta {:.2f}  held-out spellF1 {}  mean {:.3f}".format(
            a, b, " ".join("{:.3f}".format(s[0]) for s in grid[(a, b)]),
            np.mean([s[0] for s in grid[(a, b)]])), flush=True)
    best = max(grid, key=lambda k: (np.mean([s[0] for s in grid[k]]), -k[0], -k[1]))
    rows = np.array(grid[best])
    print("chosen alpha {} beta {}: held-out spellF1 {:.3f} macroF1 {:.3f} D {:.0%}".format(
        best[0], best[1], rows[:, 0].mean(), rows[:, 1].mean(), rows[:, 2].mean()))

    model = fit(games)
    model.update(alpha=best[0], beta=best[1], features="src/learned.py frame_features (27 dead + 17 team)",
                 trained_on="Metrica sample games 1-3, degraded (src/external), event-derived reference",
                 sklearn=sklearn_version, gb_params=GB, stride=STRIDE,
                 logo_spellF1=[round(float(s), 4) for s in rows[:, 0]])
    with open(args.out, "w") as fh:
        json.dump(model, fh)
    print("wrote {} ({:.0f} KB) in {:.0f}s".format(os.path.relpath(args.out, ROOT),
                                                    os.path.getsize(args.out) / 1024, time.time() - t0))


if __name__ == "__main__":
    main()
