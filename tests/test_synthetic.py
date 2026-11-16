"""Sanity tests on SYNTHETIC data only (no match data involved).

    python -m tests.test_synthetic

1. Viterbi respects minimum durations and recovers a clean sequence.
2. The raw-ATD parser reads a tiny generated EPTS file correctly.
3. End-to-end: a synthetic match (possessions, passes, brief touches by the opponent, ball
   missing ~41% in bursts, dead balls) goes through features -> emissions -> Viterbi and is
   scored with the official metric library against its own synthetic truth.
"""

import copy
import os
import sys
import tempfile

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from games import GAMES, GAME, half_video_blocks, source_frames  # noqa: E402
from src import atd as atd_mod  # noqa: E402
from src import run, viterbi  # noqa: E402

L, W = GAMES[GAME]["pitch"]


def test_viterbi():
    rng = np.random.default_rng(0)
    truth = np.repeat([2, 0, 1, 0, 2, 1], [40, 120, 90, 150, 30, 100])
    E = np.full((len(truth), 3), -2.0)
    E[np.arange(len(truth)), truth] = 0.0
    E += rng.normal(0, 1.5, E.shape)
    E[200:205, :] = [-5, 0, -5]          # a 5-frame "touch" by A inside an A run? (truth 1 here)
    E[300:306, :] = [-5, 0, -5]          # a 6-frame touch by A inside an H run -> must be ignored
    path = viterbi.decode(E, [50, 50, 25], np.array([[0, 8, 6], [8, 0, 6], [6, 6, 0]]),
                          [-10, -10, 0])
    runs = np.split(path, np.flatnonzero(np.diff(path)) + 1)
    assert all(len(r) >= 25 for r in runs[:-1]), [len(r) for r in runs]
    acc = np.mean(path == truth)
    assert acc > 0.9, acc
    print("viterbi: ok  accuracy {:.3f}  runs {}".format(acc, len(runs)))


def test_parser():
    n_slots, n = 46, 5
    team = np.array([0] * 23 + [1] * 23)
    players = "".join('<Player id="p{0}" teamId="{1}"><ShirtNumber>{0}</ShirtNumber></Player>'
                      .format(i, "t1" if t == 0 else "t2") for i, t in enumerate(team))
    pcs = "".join('<PlayerChannel id="p{0}_{1}" playerId="p{0}" channelId="{1}"/>'.format(i, c)
                  for i in range(n_slots) for c in ("x", "y", "speed"))
    refs = "".join('<SplitRegister separator=",">' + "".join(
        '<PlayerChannelRef playerChannelId="p{}_{}"/>'.format(i, c) for c in ("x", "y", "speed"))
        + "</SplitRegister>" for i in range(n_slots))
    xml = ('<main><Metadata><Teams><Team id="t1"><Name>H</Name></Team><Team id="t2"><Name>A'
           '</Name></Team></Teams><Players>{}</Players><PlayerChannels>{}</PlayerChannels>'
           '</Metadata><DataFormatSpecifications><DataFormatSpecification separator=":">'
           '<StringRegister name="frameCount"/><SplitRegister separator=";">{}</SplitRegister>'
           '<SplitRegister separator=","><BallChannelRef channelId="x"/>'
           '<BallChannelRef channelId="y"/></SplitRegister></DataFormatSpecification>'
           '</DataFormatSpecifications></main>').format(players, pcs, refs)
    rng = np.random.default_rng(1)
    P = rng.random((n, n_slots, 3))
    B = rng.random((n, 2))
    lines = []
    for f in range(n):
        grp = ";".join("NaN,NaN,NaN" if (f == 2 and s == 3) else
                       "{:.5f},{:.5f},{:.3f}".format(*P[f, s]) for s in range(n_slots))
        ball = "NaN,NaN" if f == 1 else "{:.5f},{:.5f}".format(*B[f])
        lines.append("{}:{}:{}\n".format(f + 1, grp, ball))
    with tempfile.TemporaryDirectory() as d:
        mp, rp = os.path.join(d, "m.xml"), os.path.join(d, "r.txt")
        open(mp, "w").write(xml)
        open(rp, "w").write("".join(lines))
        t, pch, bch, names = atd_mod.read_meta(mp)
        ball, pl, sp, bad = atd_mod.parse_raw(rp, n, len(t), pch, bch)
    assert names == ['H', 'A'] and (t == team).all() and pch == ["x", "y", "speed"] and bch == ["x", "y"] and bad == 0
    assert np.isnan(ball[1]).all() and np.allclose(ball[0], B[0], atol=1e-4)
    assert np.isnan(pl[2, 3]).all() and np.allclose(pl[4, 10], P[4, 10, :2], atol=1e-4)
    print("parser: ok")


TAG_NAME = {"H": "Home FC", "A": "Away Utd"}


def add_phase_tags(events, rng, a, b, state, prev_state):
    """Fake Smart Tagging phases for one possession: jittered, partial, sometimes wrong team."""
    other = "A" if state == "H" else "H"
    def tag(code, team, s0, e0):
        s0, e0 = int(s0), int(e0)
        if e0 > s0:
            events.append({"code": code, "start_frame": s0, "end_frame": e0, "start_s": s0 / 25,
                           "end_s": e0 / 25, "tags": {"Team": TAG_NAME[team]}})
    if rng.random() < 0.6:
        team = state if rng.random() > 0.1 else other
        s0 = a + rng.uniform(0, 0.4) * (b - a) + rng.normal(0, 40)
        tag(str(rng.choice(["BUILD UP", "PROGRESSION", "CREATING CHANCES"])), team, s0,
            s0 + rng.uniform(0.3, 0.6) * (b - a))
    if rng.random() < 0.4:
        team = other if rng.random() > 0.1 else state
        s0 = a + rng.uniform(0, 0.5) * (b - a) + rng.normal(0, 40)
        tag("DEFENDING IN MIDDLE THIRD", team, s0, s0 + rng.uniform(0.2, 0.5) * (b - a))
    if prev_state == other and rng.random() < 0.5:
        s0 = a + rng.normal(0, 37)
        tag("ATTACKING TRANSITION", state, s0, s0 + 150)


def synthetic_match(seed=0):
    """A crude but structured fake match on the real frame layout."""
    rng = np.random.default_rng(seed)
    n = source_frames("tactical")
    team = np.array([0] * 11 + [1] * 11 + [0] * 12 + [1] * 12, dtype=np.int8)
    ball = np.full((n, 2), np.nan)
    players = np.full((n, 46, 2), np.nan)
    truth = np.full(n, "X", dtype="<U1")
    events = []
    for _, first, last in half_video_blocks(GAME):
        t = first
        state = "D"
        prev_state = "D"
        while t <= last:
            if state == "D":
                dur = int(rng.integers(75, 300))
            else:
                dur = int(rng.integers(100, 700))
            end = min(last + 1, t + dur)
            truth[t:end] = state
            base = rng.random((22, 2)) * [L, W]
            step = 0.04 * rng.uniform(0.4, 1.0) if state == "D" else 0.07 * rng.uniform(0.6, 1.4)
            drift = np.cumsum(rng.normal(0, step, (end - t, 22, 2)), axis=0)
            pos = base + drift
            players[t:end, :22] = pos
            if state == "D":
                u = rng.random()
                if u < 0.4:              # ball out of play beyond a line
                    ball[t:end] = [L + 2.0, rng.random() * W]
                elif u < 0.7:            # ball placed still, in bounds, for the restart
                    ball[t:end] = rng.random(2) * [L, W]
                # else: ball not seen during the stoppage
                if end <= last and t > first and rng.random() < 0.8:  # tagged restart, jittered
                    s0 = end - 25 + int(rng.normal(0, 50))
                    events.append({"code": "SET PIECES", "start_frame": s0, "end_frame": s0 + 100,
                                   "start_s": s0 / 25, "end_s": (s0 + 100) / 25,
                                   "tags": {"Type": str(rng.choice(["Throw-in", "Goal Kick", "Free Kick", "Corner Kick"])),
                                            "Team": "N/A"}})
            else:
                tm = 0 if state == "H" else 1
                mates = np.flatnonzero(team[:22] == tm)
                opp = np.flatnonzero(team[:22] != tm)
                f = t
                carrier = rng.choice(mates)
                while f < end:
                    hold = int(rng.integers(25, 75))
                    seg = slice(f, min(end, f + hold))
                    ball[seg] = pos[seg.start - t:seg.stop - t, carrier] + rng.normal(0, 0.4, 2)
                    # nearest opponent 2-6 m away
                    o = rng.choice(opp)
                    off = rng.normal(0, 1, 2)
                    off = off / np.linalg.norm(off) * rng.uniform(2, 6)
                    players[seg, o] = ball[seg] + off
                    if rng.random() < 0.15:  # brief opponent touch (not a change)
                        k = min(seg.stop, seg.start + 6)
                        players[seg.start:k, o] = ball[seg.start:k] + 0.3
                    f = seg.stop
                    nxt = rng.choice(mates)
                    flight = min(end, f + 12)
                    if flight > f:  # pass in flight
                        a = pos[min(f, end - 1) - t, carrier]
                        b = pos[min(flight, end - 1) - t, nxt]
                        ball[f:flight] = a + np.linspace(0, 1, flight - f)[:, None] * (b - a)
                    f = flight
                    carrier = nxt
            if state in ("H", "A"):
                add_phase_tags(events, rng, t, end, state, prev_state)
            prev_state = state
            t = end
            state = "D" if state != "D" and rng.random() < 0.4 else \
                rng.choice(["H", "A"]) if state == "D" else ("A" if state == "H" else "H")
    # drop the ball in bursts (~41% of frames)
    drop = np.zeros(n, dtype=bool)
    t = 0
    while t < n:
        t += int(rng.exponential(35))
        k = int(rng.exponential(25))
        drop[t:t + k] = True
        t += k
    ball[drop] = np.nan
    atd = {"team_names": np.array(["Home Football Club", "Away United"]),
           "ball": (ball / [L, W]).astype(np.float32),
           "players": (players / [L, W]).astype(np.float32),
           "pspeed": np.full((n, 46), np.nan, np.float32), "team": team}
    return atd, truth, float(drop.mean()), events


def stage2_config(cfg):
    """Same config with the Stage 3 team-phase prior and transition hints off (ablation)."""
    c = copy.deepcopy(cfg)
    c["tagging"]["use_team_phases"] = False
    c["tagging"]["use_transition_hints"] = False
    return c


def stage1_config(cfg):
    """Same config with every Stage 2 dead-ball term (and Stage 3) switched off (ablation)."""
    c = stage2_config(cfg)
    for k in list(c["dead"]):
        if k.startswith("w_"):
            c["dead"][k] = 0.0
    c["dead"]["use_smart_tagging"] = False
    return c


def score(name, labels, truth, drop):
    r = pm.evaluate(pm.label_codes(truth), pm.label_codes(labels))
    s = run.summarise(labels)
    print("{:<8} (synthetic, ball dropped {:.0%}): spellF1 {:.3f} macroF1 {:.3f} D-F1 {:.3f} "
          "poss {} (truth {}) mean dur {:.1f}s".format(
              name, drop, r["spell_f1_mean"], r["macro_f1"], r["per_class"]["D"]["f1"],
              s["n_possessions"], r["n_gt_spells"], s["mean_dur_s"]))


def test_end_to_end():
    cfg = run.load_config(os.path.join(ROOT, "src", "config.yaml"))
    atd, truth, drop, events = synthetic_match()
    labels, _ = run.predict(atd, cfg, events=events)
    labels2, _ = run.predict(atd, cfg, events=events)
    assert (labels == labels2).all(), "non-deterministic"
    score("stage1", run.predict(atd, stage1_config(cfg))[0], truth, drop)
    score("stage2", run.predict(atd, stage2_config(cfg), events=events)[0], truth, drop)
    score("stage3", labels, truth, drop)


if __name__ == "__main__":
    test_viterbi()
    test_parser()
    test_end_to_end()
