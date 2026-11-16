"""Stage 4: where to write X (abstain).

How the scorer treats X (evaluation/possession_metrics.py, fill_abstentions):
  * spellF1 (ranking) forward-fills X with the last non-X label of the half. So an X inside a
    segment, preceded by a labelled frame of that same segment, changes nothing; an X right after
    a change would drag the old label across it. => X can never raise spellF1, only lower it.
  * macroF1 (tie-break) counts X as a miss of the true class, but unlike a wrong guess it adds
    no false positive to another class. => X helps macroF1 only where a guess is likely wrong.

Policy: X only on long stretches with NO evidence at all (no ball near anyone, no tag), strictly
inside a decoded segment and never on its first frame. Every possible change of possession keeps
an explicit guess. The invariant "forward-filled output == decoded output" is asserted, so the
X policy provably leaves spellF1 untouched. Frames outside the graded halves are X regardless.
"""

import numpy as np


def no_evidence(E, eps=1e-6):
    """Frames whose emission row carries no information (all states scored equally)."""
    return np.ptp(E, axis=1) < eps


def apply(labels, E, blocks, cfg):
    """labels: decoded 'H'/'A'/'D'/'X' array. E: emissions (N, 3). blocks: graded halves."""
    xc = cfg["xpolicy"]
    out = labels.copy()
    if not xc.get("enabled", False):
        return out
    empty = no_evidence(E)
    min_run = int(xc["min_no_evidence_frames"])
    classes = set(xc.get("classes", ["H", "A"]))
    for _, first, last in blocks:
        w = labels[first:last + 1]
        brk = np.flatnonzero(w[1:] != w[:-1]) + 1
        bounds = np.concatenate(([0], brk, [len(w)]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            if w[a] not in classes:
                continue
            seg = empty[first + a + 1:first + b]          # never the segment's first frame
            e = np.diff(np.concatenate(([0], seg.astype(np.int8), [0])))
            for s, t in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
                if t - s >= min_run:
                    out[first + a + 1 + s:first + a + 1 + t] = "X"
    _check_fill_invariant(labels, out, blocks)
    return out


def _check_fill_invariant(decoded, with_x, blocks):
    """Forward-fill X exactly as the scorer does; it must give back the decoded labels."""
    for _, first, last in blocks:
        w = with_x[first:last + 1]
        idx = np.where(w != "X", np.arange(len(w)), -1)
        idx = np.maximum.accumulate(idx)
        known = np.flatnonzero(w != "X")
        idx[idx < 0] = known[0] if len(known) else 0
        if not np.array_equal(w[idx], decoded[first:last + 1]):
            raise AssertionError("X policy would change spellF1 (fill invariant broken)")
