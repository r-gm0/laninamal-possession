"""Stage 5 plumbing test on SYNTHETIC files written in the Metrica sample-data format.

    python -m tests.test_external

Writes a fake Sample_Game_9 (CSV tracking + events) to a temp dir, then runs the real loader,
the event->possession reference, the degradation and the validation scoring on it.
"""

import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks  # noqa: E402
from src import run, validate  # noqa: E402
from src.external import metrica, reference  # noqa: E402
from tests.test_synthetic import synthetic_match  # noqa: E402


def write_fake_game(base, name="Sample_Game_9"):
    atd, truth, _, _ = synthetic_match(seed=3)
    folder = os.path.join(base, name)
    os.makedirs(folder)
    blocks = half_video_blocks(GAME)
    rows = np.concatenate([np.arange(a, b + 1) for _, a, b in blocks])
    period = np.concatenate([np.full(b - a + 1, p) for p, a, b in blocks])
    ball = atd["ball"].astype(np.float64)
    for side, tm in (("Home", 0), ("Away", 1)):
        cols = np.flatnonzero(atd["team"] == tm)[:14]
        xy = atd["players"][rows][:, cols].reshape(len(rows), -1)
        data = np.column_stack([period, rows + 1, (rows + 1) / 25.0, xy, ball[rows]])
        with open(os.path.join(folder, "{}_RawTrackingData_{}_Team.csv".format(name, side)), "w") as fh:
            fh.write(",,," + ",".join([side, ""] * (len(cols) + 1)) + "\n")
            fh.write(",,," + ",".join(x for i in range(len(cols))
                                      for x in (str(i + 1), "")) + ",,\n")
            fh.write("Period,Frame,Time [s]," + ",".join(
                x for i in range(len(cols)) for x in ("Player{}".format(i + 1), "")) + ",Ball,\n")
            np.savetxt(fh, data, delimiter=",", fmt="%.6g")
    # events consistent with the synthetic truth
    ev = []
    for p, a, b in blocks:
        w = truth[a:b + 1]
        brk = np.flatnonzero(w[1:] != w[:-1]) + 1
        bounds = np.concatenate(([0], brk, [len(w)]))
        prev = "D"
        for s, e in zip(bounds[:-1], bounds[1:]):
            st, f0, f1 = w[s], a + s, a + e - 1
            if st == "D":
                if s > 0:
                    ev.append(("Home" if prev == "H" else "Away", "BALL OUT", "", p, f0, f0))
            else:
                team = "Home" if st == "H" else "Away"
                ev.append((team, "SET PIECE" if prev == "D" else "RECOVERY", "", p, f0, f0))
                for f in range(f0 + 50, f1, 50):
                    ev.append((team, "PASS", "", p, f, f + 10))
            prev = st
    with open(os.path.join(folder, name + "_RawEventsData.csv"), "w") as fh:
        fh.write("Team,Type,Subtype,Period,Start Frame,Start Time [s],End Frame,End Time [s],"
                 "From,To,Start X,Start Y,End X,End Y\n")
        for t, ty, su, p, f0, f1 in ev:
            fh.write("{},{},{},{},{},{},{},{},P1,P2,0.5,0.5,0.5,0.5\n".format(
                t, ty, su, p, f0 + 1, (f0 + 1) / 25, f1 + 1, (f1 + 1) / 25))
    return truth


def main():
    cfg = run.load_config(os.path.join(ROOT, "src", "config.yaml"))
    with tempfile.TemporaryDirectory() as base:
        truth = write_fake_game(base)
        assert metrica.available_games(base) == ["Sample_Game_9"]
        atd, events, periods = metrica.load_game("Sample_Game_9", base)
        ref = reference.possession_labels(events, periods, len(atd["ball"]))
        n = min(len(ref), len(truth))
        graded = truth[:n] != "X"
        agree = np.mean(ref[:n][graded] == truth[:n][graded])
        print("loader+reference: {} events, periods {}, reference agrees with truth on {:.1%}"
              .format(len(events), [(p, a, b) for p, a, b in periods], agree))
        assert agree > 0.99, agree
        prepared = validate.prepare(cfg, ["Sample_Game_9"], base)
        validate.score(cfg, prepared, with_baseline=True)


if __name__ == "__main__":
    main()
