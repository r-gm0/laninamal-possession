# Ball possession by duration-constrained Viterbi decoding

*More than a Hack 2027, Phase 1. Technical document.*

> No number in this document comes from the challenge's ground truth, except the explicitly
> labelled public-window pipeline check in §6. Every parameter was chosen on other matches
> (§5) or set from the wording of CHALLENGE.md.

## 1. Approach and why

The ranking metric (spellF1) scores **whole possessions** matched by temporal IoU. A per-frame
classifier, even an accurate one, flickers and invents many short possessions: the starter
baseline at ±0 frames has a good macroF1 but a spellF1 of 0.04. Possession is a *state* that
persists, and changes only on sustained control. So we model it as a state sequence:

1. **Per-frame evidence (emissions)** from the automatic tracking: who could be controlling the
   ball, and whether the ball looks dead.
2. **A hidden-state model** with three states, H (home), A (away) and D (dead). We decode the
   single most likely sequence with Viterbi, under two constraints that encode "control must
   be sustained":
   * a **minimum duration** per state (3 s for a possession, 1 s for a dead ball). This is
     implemented exactly by expanding each state into a chain of sub-states (an
     explicit-duration HMM), so a short touch can never become a possession;
   * a **penalty (in nats) for each change of state**, which sets how much accumulated evidence
     a change of possession needs.

We chose this over a learned sequence model (e.g. a recurrent network) because there is no
labelled training data for the provided match, the external data is small (3 matches), and
every parameter here has a football meaning that can be checked and explained.

## 2. Evidence used

All distances in metres (pitch 105.3 × 68.0 m, from `games.py`).

| feature | use |
|---|---|
| distance from the ball to the nearest player of each team | main H vs A evidence: logit = (d_away − d_home)/s + β·(n_home − n_away), clipped to ±0.75 |
| players of each team within 5 m of the ball | local numerical superiority (β term) |
| distance of the nearest player to the ball | evidence weight decays when nobody is within 1 m (loose ball / ball in the air) |
| ball speed (from positions) | evidence weight decays above 8 m/s (a pass or shot is not control); a fast ball counts against D |
| ball outside the pitch lines | strong D evidence |
| players walking (mean speed < 2 m/s) while nobody controls the ball | weak D evidence (restart being prepared) |
| ball still while players walk | D evidence |
| Smart Tagging SET PIECES | soft D prior in a window around each tagged restart |
| Smart Tagging phases with a Team | soft H/A prior (Build Up, Progression… = that team has the ball; Defending … = the other team) |
| Smart Tagging transitions | make a direct H↔A change cheaper near the tag |

The clip at ±0.75 matters: each frame alone is weak evidence, so a change of possession is
decided by evidence accumulated over seconds, not by one frame where an opponent happens to
be closest.

**Missing ball (41 % of graded frames).** Gaps of up to 12 frames are interpolated linearly.
Longer gaps give a *neutral* emission (zero evidence for every state), so the decoded state is
carried by the transition model: the possession continues through an occlusion. **A missing
ball is never taken as evidence of a dead ball.**

## 3. Possession rules implemented

* **Change of possession** only when the evidence for the other team, summed over at least the
  minimum duration (3 s, CHALLENGE.md: "a player keeps the ball for a few seconds"), outweighs
  the switch penalty. A touch, a deflection or a duel won in the air without follow-up does
  not last long enough to pay for the switch.
* **Loose ball / contested ball**: the evidence is weak (nobody close, or both teams close), so
  the state stays with the last team, as the challenge definition asks.
* **Restarts**: a dead ball ends when the tracking shows the ball moving or controlled again.
  The Smart Tagging set piece only says roughly when (its intervals are approximate, ±1–3 s);
  the tracking places the frame. Team = N/A (every throw-in) gives no team evidence: the taker
  is found from the tracking.
* **Start of each half**: the ball is dead before the kick-off. This is a general rule (the
  initial state prior favours D), not a frame number.
* **X (abstain)**: written only on frames outside the two graded halves. A within-possession X
  policy is implemented (`src/xpolicy.py`) with a built-in check that it cannot change spellF1,
  because the scorer fills X with the previous label. It only trades macroF1, and it lowered
  macroF1 in our tests, so it is off.

## 4. Assets used and dropped

| asset | status |
|---|---|
| tactical ATD (players + ball) | **used**: all tracking evidence |
| tactical Smart Tagging | **used** as soft evidence (never copied as a label) |
| homography | not used: the ATD is already in pitch coordinates |
| tactical / broadcast video | not used (data terms forbid sending frames to cloud services; no local detector was built in time) |
| broadcast ATD | not used (not synchronised to the tactical timeline) |
| D evidence "long ball absence + slow players" | implemented, **off**: it hurt on the external matches (log E13) |
| tracking-only restart detector (`src/restart.py`: fast ball after a calm stretch → cheaper D→H/A) | implemented, **off**: on the provided match only 8–9 % of its detections fall within 3 s of a Smart Tagging set piece; the ball is usually missing at restarts (log E15) |

## 5. Training and validation data

**None of it comes from the provided match.** The 5 public minutes were used only to check that
the pipeline is correct (alignment, file format, label meaning), never to train or choose
parameters.

* **Synthetic matches** generated by our own code (`tests/test_synthetic.py`), used only to
  test the software. Synthetic results are not evidence of real performance.
* **Metrica Sports open sample data** (github.com/metrica-sports/sample-data, used with
  acknowledgement as its README asks): 3 professional matches with tracking and
  **human-annotated event data** (annotated by Metrica, not by us).
  * **Reference possession from events** (`src/external/reference.py`): a team gains possession
    with a controlled action; stoppages (ball out, foul, shot out or goal) give D until the
    next set piece. Following CHALLENGE.md ("a simple touch without control does not count"),
    a team run that follows the other team's possession is kept only if it contains a pass,
    carry, shot or set piece, or lasts ≥ 3 s; otherwise possession stays with the last team.
  * **Degradation to look like the challenge ATD**: 41 % of ball frames dropped in bursts (more
    often in crowds), position noise, identity switches, player dropouts. The noise is
    temporally correlated (0.15 m, 50-frame smoothing). It was calibrated so the speed
    distributions match the challenge ATD's *aggregate* speed percentiles (still ball 5 % vs
    5 %; median mean player speed 1.68 vs 1.48 m/s); this uses tracking statistics only, no
    labels. Metrica's clean tracking drops the ball while it is out of play, which would leak
    "missing = dead"; those gaps are first filled by straight lines to the restart spot.
  * **Fitting**: coordinate search over a small grid, a change kept only if mean spellF1
    improves by > 0.01, with leave-one-game-out reporting (`src/tune.py`).
* Caveats: the challenge ground truth comes from corrected *tracking* ownership, while our
  external reference comes from *events*; the two definitions are close but not identical.
  Game 3 annotates CARRY events and games 1–2 do not, so game 3's reference stays finer.

Building this benchmark honestly mattered more than the tuning. Three bugs in it were found
and fixed before any parameter was trusted (log E7, E9, E12): an EPTS loader that lost all
players in game 3; "ball missing" leaking "dead ball"; and i.i.d. noise that made players look
like they ran at 6 m/s, which hid every slow-play cue.

## 6. Results

External matches (Metrica 1–3, degraded, final benchmark), mean of three games:

| | spellF1 | macroF1 | possessions per game |
|---|---|---|---|
| reference (from events) | | | 238 / 225 / 379 |
| baseline smooth50 (same degraded data) | 0.236 | 0.427 | ~650 |
| ours, before fitting (default config) | 0.432 | 0.589 | ~260 |
| **ours, submitted config** | **0.466** | **0.668** | ~240 |

Leave-one-game-out for the main fitting step (log E14): held-out spellF1 improved on all three
games (+0.035, +0.051, +0.012). Differences below 0.04 spellF1 are treated as noise.

**One deliberate deviation from the benchmark optimum** (log E16): weighting "players walking"
at 0.2 scored best externally (0.480) but predicted dead ball on 54 % of the provided match's
graded frames (typical football: 35–40 %) and also over-predicted D externally (38 % vs 23 %).
We submit 0.1 (0.466, within noise of the optimum).

Provided match (no score available): 236 possessions, mean 13.4 s, median 10.9 s, none under
2 s; dead ball on 50 % of graded frames in 112 stretches.

*Pipeline check only, not used for any choice*: on the 5 public minutes the decoded labels
align with the ground truth's format and meaning (16 possessions predicted for 17 true).

## 7. Limitations and what we would do with more time

* **Dead-ball share is high** (50 % of the provided match). D evidence comes mostly from
  players walking and Smart Tagging set pieces; a walking team in a slow build-up can be
  mistaken for a stoppage.
* All tracking evidence is geometric. A player running alongside the ball looks the same as a
  player dribbling it, and aerial balls are invisible to the distance features.
* Without the ball, possession is inferred only from continuity: a change of possession inside
  a long occlusion is detected late, at the first frame the ball reappears.
* Hyper-parameters are fitted on 3 external games with a reference derived from events.
* Ball re-detection from the video (run locally), and a learned emission model trained on more
  external matches, are the two most promising extensions.

## 8. Reproduction

```bash
py -3.12 -m venv .venv && .venv/Scripts/activate        # Python 3.12
pip install -r requirements.txt
# challenge data unpacked in data/ or anywhere with MTAH_DATA_DIR pointing to it (see DATA.md)
python -m src.run --out submissions/<team_name>.csv
python evaluation/validate_submission.py submissions/<team_name>.csv
```

The pipeline is deterministic: running it twice gives byte-identical files (`src.run` prints
the SHA-256). All parameters are in `src/config.yaml`; set `data.cache_dir` to a writable
folder. External validation: `python -m src.validate` and `python -m src.tune`, with the
Metrica sample data in `data_external/metrica/`. Tests on synthetic data:
`python -m tests.test_synthetic` and `python -m tests.test_external`. Every experiment is in
`experiments/log.md`.
