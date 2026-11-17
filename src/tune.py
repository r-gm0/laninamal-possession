"""Stage 5: fit HMM / emission hyper-parameters on OTHER matches, never the challenge one.

    python -m src.tune [--rounds 2] [--out experiments/tuned.yaml]

Coordinate search: one parameter at a time over a small grid (config 'tuning.grid'), keeping a
change only if mean spellF1 over the external games improves by more than 'tuning.min_gain'
(prefer the current/default value on ties -> fewer, safer moves). Deterministic.
Then reports leave-one-game-out: tune on all games but one, score the held-out one -- the honest
estimate of how much tuning transfers. Writes the tuned config to --out; it is NOT applied to
src/config.yaml automatically.
"""

import argparse
import copy
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import run, validate  # noqa: E402


def get(cfg, path):
    for k in path.split(".")[:-1]:
        cfg = cfg[k]
    return cfg[path.split(".")[-1]]


def put(cfg, path, value):
    c = copy.deepcopy(cfg)
    d = c
    for k in path.split(".")[:-1]:
        d = d[k]
    d[path.split(".")[-1]] = value
    return c


def coordinate_search(cfg, prepared, grid, rounds, min_gain, verbose=True):
    best = cfg
    best_score = validate.score(best, prepared, verbose=False)[0]["spell"]
    if verbose:
        print("start: mean spellF1 {:.4f}".format(best_score))
    for r in range(rounds):
        changed = False
        for path in sorted(grid):
            cur = get(best, path)
            for v in grid[path]:
                if v == cur:
                    continue
                cand = put(best, path, v)
                s = validate.score(cand, prepared, verbose=False)[0]["spell"]
                if s > best_score + min_gain:
                    best, best_score, changed = cand, s, True
                    if verbose:
                        print("  round {} {} = {} -> mean spellF1 {:.4f}".format(r + 1, path, v, s))
        if not changed:
            break
    return best, best_score


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=os.path.join(ROOT, "src", "config.yaml"))
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(ROOT, "experiments", "tuned.yaml"))
    ap.add_argument("--no-logo", action="store_true", help="skip leave-one-game-out")
    args = ap.parse_args()

    cfg = run.load_config(args.config)
    cfg["emission"]["mode"] = "rules"   # this search fits the rule parameters (logs E1-E16)
    grid, min_gain = cfg["tuning"]["grid"], cfg["tuning"]["min_gain"]
    prepared = validate.prepare(cfg)

    print("\n== default config")
    validate.score(cfg, prepared, with_baseline=True)
    print("\n== coordinate search on all external games")
    tuned, _ = coordinate_search(cfg, prepared, grid, args.rounds, min_gain)
    validate.score(tuned, prepared)
    with open(args.out, "w") as fh:
        yaml.safe_dump(tuned, fh, sort_keys=False)
    print("tuned config written to {} (not applied)".format(args.out))

    if not args.no_logo and len(prepared) > 1:
        print("\n== leave-one-game-out (held-out spellF1: default vs tuned-without-it)")
        for i, held in enumerate(prepared):
            rest = prepared[:i] + prepared[i + 1:]
            t, _ = coordinate_search(cfg, rest, grid, args.rounds, min_gain, verbose=False)
            d = validate.score(cfg, [held], verbose=False)[0]["spell"]
            s = validate.score(t, [held], verbose=False)[0]["spell"]
            print("  {:<14} default {:.3f}  tuned {:.3f}  ({:+.3f})".format(held[0], d, s, s - d))


if __name__ == "__main__":
    main()
