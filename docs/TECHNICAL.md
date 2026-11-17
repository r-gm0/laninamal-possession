# Ball possession: learned evidence + duration-constrained Viterbi

*More than a Hack 2027, Phase 1. Team laninamal. Technical document.*

> No number in this document comes from the challenge's ground truth, except the explicitly
> labelled public-window pipeline check in §6. Every parameter and model was fitted on other
> matches (§5); statistics of the provided match used anywhere are *unlabelled aggregates of the
> tracking* (gap lengths, speed percentiles), never labels.

## 1. Approach and why

The ranking metric (spellF1) scores **whole possessions** matched by temporal IoU. A per-frame
classifier, even an accurate one, flickers and invents many short possessions: the starter
baseline at ±0 frames has a good macroF1 but a spellF1 of 0.04. Possession is a *state* that
persists and changes only on sustained control. So we model it as a state sequence:

1. **Per-frame evidence (emissions).** About 40 features from the automatic tracking feed two
   gradient-boosted tree ensembles (300 trees of depth 4 each), trained on external matches only.
   They output, per frame, the log-odds that *the ball is dead* and that *home rather than away
   controls it*. Smart Tagging adds soft priors on top.
2. **A hidden-state model** with three states, H (home), A (away) and D (dead). We decode the
   single most likely sequence with Viterbi under two constraints that encode "control must be
   sustained":
    * a **minimum duration** per state (3 s for a possession, 1 s for a dead ball), implemented
      exactly by expanding each state into a chain of sub-states (an explicit-duration HMM), so
      a short touch can never become a possession;
    * a **penalty (in nats) for each change of state**, which sets how much accumulated evidence
      a change of possession needs.

The decoder is our own; the feature set of the tree models was designed by our teammate
(GitHub `fermartinezm`, in a parallel pipeline, see §4) and ported into `src/learned.py`. What we changed
is the training data (§5): the trees are trained on external matches degraded to reproduce how
the challenge tracking actually loses the ball, which removed a leak that made a model trained
on clean data predict "dead ball" whenever the ball was missing.

We kept a learned model small and its inputs interpretable because the external labelled data is
small (3 matches): the per-frame trees capture interactions between cues that hand-written rules
missed, while the duration constraints and switch penalties stay explicit and explainable.

## 2. Evidence used

All distances in metres, from the tactical ATD (pitch 105.3 × 68.0 m, from `games.py`).

**Dead-ball model (27 features).**

* Team activity: mean player speed over 0.5 / 2 / 6 / 15 s, its trend, past vs future 3 s; share
  of players running.
* Ball visibility: seconds since / until the ball was last / next seen; share of frames with the
  ball seen over 2 / 6 / 10 s.
* Ball outside the pitch, and time since / until it was out.
* Ball speed and stillness.
* Ball position: distance to the goal line and to the touchline.
* Players near the ball (within 3 and 10 m) and the nearest-player distance.

**Team model (17 features, home minus away, so swapping the teams flips the sign).**

* Nearest-player advantage now and over the previous / next 1 s and 3 s, also restricted to a
  slow ball.
* Touches (a sharp change of ball speed next to a player), now and over ±2 s.
* Players of each team within 5 and 10 m.
* Which team stands goal-side of the ball.
* How fast each team closes in on the ball.
* Whether the ball travels towards a home or an away player.

The team score is made exactly symmetric, (f(x) − f(−x)) / 2, and the training set is augmented
with the mirrored frames.

**Smart Tagging (soft, never copied as a label):**

| tag | use |
|---|---|
| SET PIECES | D prior in a window around each tagged restart (its timing is approximate, ±1–3 s; the tracking places the frame) |
| phases with a Team (Build Up, Progression … / Defending …) | weak H/A prior |
| transitions | make a direct H↔A change cheaper near the tag |

**Missing ball (41 % of graded frames).** In the challenge ATD the ball is lost in about 150
long gaps (median about 11 s), so many possessions happen with no ball at all. The team model
then works from the players only: shape, goal-side, activity and the context features just
before and after the gap. The training benchmark reproduces those gaps at random times, so **a
missing ball is not learned as evidence of a dead ball** (§5).

## 3. Possession rules implemented

* **Change of possession** only when evidence for the other team, summed over at least the
  minimum duration (3 s, CHALLENGE.md: "a player keeps the ball for a few seconds"), outweighs
  the switch penalty. A touch, a deflection or a duel won in the air without follow-up does not
  last long enough to pay for the switch.
* **Loose / contested ball.** Weak or balanced evidence does not pay for a switch, so the state
  stays with the last team, as the challenge definition asks.
* **Restarts.** A dead ball ends when the evidence says play resumed. Smart Tagging only says
  roughly when; the tracking decides. Team = N/A (every throw-in) gives no team evidence: the
  taker is found from the tracking.
* **Start of each half.** The ball is dead before the kick-off. This is a general rule (the
  initial state prior favours D), not a frame number.
* **X (abstain)** is written only on frames outside the two graded halves. A within-possession X
  policy exists (`src/xpolicy.py`), with a check that it cannot change spellF1. It only trades
  macroF1, lowered it in our tests, and is off.

## 4. Assets used and dropped

| asset | status |
|---|---|
| tactical ATD (players + ball) | **used**: all tracking evidence |
| tactical Smart Tagging | **used** as soft evidence |
| homography | not used: the ATD is already in pitch coordinates |
| tactical / broadcast video | not used: the data terms forbid sending frames to cloud services, and no local detector was built in time |
| broadcast ATD | not used: not synchronised to the tactical timeline |
| hand-written rule emissions (`emission.mode: rules`) | **replaced** by the learned evidence (log E17–E19). Kept as the interpretable baseline: with `mode: rules` the pipeline regenerates our earlier submission byte for byte |
| tracking-only restart detector (`src/restart.py`) | implemented, **off**: only 8–9 % of its detections fall within 3 s of a Smart Tagging set piece (log E15) |
| teammate's pipeline as a whole (team laninamal, `fermartinezm/laninamal-mtah2027`) | compared head to head on the same external games (log E17). Its feature set was **adopted**. Its training data was not: Metrica's clean tracking loses the ball only when it is out of play, and the model trained on it predicted D on 47 % of frames under realistic ball loss |

## 5. Training and validation data

**None of it comes from the provided match.** The 5 public minutes were used only to check that
the pipeline is correct (alignment, file format, label meaning), never to train or choose
anything.

* **Metrica Sports open sample data** (github.com/metrica-sports/sample-data, used with
    acknowledgement as its README asks): 3 professional matches with tracking and
    **human-annotated event data** (annotated by Metrica, not by us). About 140k graded frames per
    match.
    * **Reference possession from events** (`src/external/reference.py`). A team gains possession
        with a controlled action. Stoppages (ball out, foul, shot out or goal) give D until the next
        set piece. Following CHALLENGE.md ("a simple touch without control does not count"), a team
        run after the other team's possession is kept only if it contains a pass, carry, shot or set
        piece, or lasts at least 3 s; otherwise possession stays with the last team.
    * **Degradation to look like the challenge ATD.** All targets are aggregates of the challenge
        tracking, with no labels:
        * player and ball position noise, temporally correlated (0.15 m, 50-frame smoothing), so
          player speeds match the ATD (median 1.68 vs 1.48 m/s);
        * identity switches and player dropouts;
        * i.i.d. ball jitter of 0.08 m, so the ball looks still on 4 % of frames (ATD: 5 %);
        * ball lost on 41 % of graded frames in **long gaps**, drawn from the deciles of the ATD's own
          gap lengths, non-overlapping and placed at random (log E18). Metrica's out-of-play ball
          gaps are filled first, so "missing" cannot leak "dead".
* **Training** (`src/train_learned.py`, scikit-learn 1.9.1, every 4th graded frame):
    * leave-one-game-out: trees fitted on two games give out-of-fold log-odds on the third;
    * the emission scales (dead 0.2, team 0.1) and a dead-ball bias (0.12 nats, log E19) are chosen
        on those out-of-fold predictions only;
    * the final trees are fitted on all three games and exported to `src/learned_model.json`,
        which is evaluated with numpy alone;
    * the HMM durations and penalties are those fitted earlier for the rule model (log E10–E14).
* **Synthetic matches** (`tests/test_synthetic.py`) are used only to test the software.
* Caveats:
    * The challenge ground truth comes from corrected *tracking* ownership, while our reference
        comes from *events*; the two definitions are close but not identical.
    * Game 3 annotates CARRY events and games 1–2 do not, so game 3's reference stays finer.

Building the benchmark honestly mattered more than the model. Five problems in it were found and
fixed before any result was trusted (log E7, E9, E12, E17, E18):

* an EPTS loader that lost all players in game 3;
* a missing ball leaking "dead ball";
* i.i.d. noise that made players look like they ran at 6 m/s;
* short ball dropouts where the real tracker loses the ball for many seconds;
* a ball that never jittered.

## 6. Results

External matches (Metrica 1–3, final benchmark, held-out game in every row except the
baseline):

| | spellF1 per game | **spellF1** | macroF1 | dead-ball share (ref. 23 %) |
|---|---|---|---|---|
| baseline smooth50 (same degraded data) | 0.219 / 0.218 / 0.194 | 0.210 | 0.459 | — |
| rule evidence (our previous submission) | 0.353 / 0.381 / 0.355 | 0.363 | 0.585 | — |
| **learned evidence (submitted)** | **0.522 / 0.503 / 0.407** | **0.478** | 0.702† | 24 % |

† Measured before the dead-ball bias was added (log E18).

The learned evidence beats the rules on every held-out game (+0.17, +0.12, +0.05). Differences
below 0.04 spellF1 are treated as noise.

**Provided match** (no score available):

* 258 possessions, mean 17.3 s, median 12.8 s, none under 2 s;
* dead ball on 30 % of graded frames, in 108 stretches;
* Barcelona with 62 % of live-ball time.

As an unlabelled plausibility check we required the dead-ball share to fall between 25 % and
45 %. A first learned model trained on short ball dropouts gave 7.5 % (log E17); that exposed the
gap-length mismatch fixed in E18. The rule model gave 50 %.

*Pipeline check only, not used for any choice*: on the 5 public minutes the decoded labels align
with the ground truth's format and meaning (18 possessions predicted for 17 true).

## 7. Limitations and what we would do with more time

* **Three training matches.** The trees see about 140k graded frames per match from one
  provider; game 3, the one in the challenge's EPTS format, gains least (+0.05).
* **Event-derived reference.** The moment a possession changes in the challenge ground truth may
  follow a slightly different convention.
* **Long ball gaps** (median about 11 s) leave possession to player-only cues: a change of
  possession inside a gap is detected late or missed.
* **Geometric evidence.** A player running alongside the ball looks like one dribbling it, and
  aerial balls are invisible to the distance features.
* **Decoder not re-fitted.** The HMM durations and penalties were fitted for the rule model;
  re-fitting them for the learned evidence is the obvious next step.
* **Further extensions:** local ball re-detection from the video, and more external matches from
  tactical cameras.

## 8. Reproduction

```bash
py -3.12 -m venv .venv && .venv/Scripts/activate        # Python 3.12
pip install -r requirements.txt                          # numpy, pandas, pyyaml (pinned)
# challenge data unpacked in data/ or anywhere with MTAH_DATA_DIR pointing to it (see DATA.md)
python -m src.run --out submissions/laninamal.csv        # uses the committed src/learned_model.json
python evaluation/validate_submission.py submissions/laninamal.csv
```

* **Determinism.** Two runs give byte-identical files; `src.run` prints the SHA-256.
* **Parameters.** All of them are in `src/config.yaml`. `emission.mode: rules` reproduces the
  previous rule-based submission.
* **Other checks:** external validation with `python -m src.validate`; tests on synthetic data
  with `python -m tests.test_synthetic` and `python -m tests.test_external`.
* **Experiment log.** Every experiment is in `experiments/log.md`.

**Retraining the model** needs the Metrica sample data in `data_external/metrica/`. It rewrites
`src/learned_model.json` and prints the held-out scores:

```bash
pip install -r requirements-train.txt
python -m src.train_learned
```
