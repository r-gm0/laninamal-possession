"""Loader for Metrica Sports' open sample data (other matches, NOT the challenge match).

Source: https://github.com/metrica-sports/sample-data  (downloaded by the team, not by this
code). Expected layout, keeping the repository's own folder and file names:

    data_external/metrica/
        Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv
        Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv
        Sample_Game_1/Sample_Game_1_RawEventsData.csv
        Sample_Game_2/...  (same names with _2)
        Sample_Game_3/Sample_Game_3_metadata.xml, Sample_Game_3_tracking.txt, Sample_Game_3_events.json

Games 1-2: CSV tracking (3 header lines: team, jersey, "Period,Frame,Time [s],PlayerN,,...,Ball,"),
coordinates normalised 0-1, 25 fps, frames counted from 1. Events CSV with columns
Team,Type,Subtype,Period,Start Frame,Start Time [s],End Frame,End Time [s],From,To,Start X,...
Game 3: FIFA/EPTS tracking (same format as the challenge ATD) + events JSON.

Everything returned uses the challenge ATD conventions so src/ can run on it unchanged:
    atd = {ball (N,2), players (N,S,2), pspeed (N,S) NaN, team (S,), team_names (2,)}
    events = [{team: 0/1/None, type, subtype, period, start, end}]  frames 0-based
    periods = [(period, first_frame, last_frame)]  0-based, inclusive
"""

import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DIR = os.path.join(ROOT, "data_external", "metrica")
HOME, AWAY = 0, 1


def available_games(base=DEFAULT_DIR):
    return sorted(os.path.basename(d) for d in glob.glob(os.path.join(base, "Sample_Game_*"))
                  if os.path.isdir(d))


# -- games 1 and 2: CSV ------------------------------------------------------------------------

def _read_tracking_csv(path):
    """(period (N,), frame (N,), xy (N, P, 2), ball (N, 2)) from one team's CSV."""
    import pandas as pd
    with open(path) as fh:
        fh.readline(); fh.readline()
        header = fh.readline().rstrip("\n").split(",")
    df = pd.read_csv(path, skiprows=3, header=None)
    vals = df.to_numpy(dtype=np.float64)
    period, frame = vals[:, 0].astype(int), vals[:, 1].astype(int)
    names = [h for h in header[3:] if h]
    if not names or names[-1].lower() != "ball":
        raise ValueError("{}: last column pair is not Ball".format(path))
    xy = vals[:, 3:3 + 2 * len(names)].reshape(len(vals), len(names), 2)
    return period, frame, xy[:, :-1], xy[:, -1]


def _load_csv_game(folder, name):
    p_home = os.path.join(folder, name + "_RawTrackingData_Home_Team.csv")
    p_away = os.path.join(folder, name + "_RawTrackingData_Away_Team.csv")
    p_ev = os.path.join(folder, name + "_RawEventsData.csv")
    per, frame, home, ball = _read_tracking_csv(p_home)
    per2, frame2, away, ball2 = _read_tracking_csv(p_away)
    if not np.array_equal(frame, frame2):
        raise ValueError("home/away tracking frames differ in " + folder)
    n = int(frame.max())
    idx = frame - 1
    S = home.shape[1] + away.shape[1]
    players = np.full((n, S, 2), np.nan, dtype=np.float32)
    players[idx, :home.shape[1]] = home
    players[idx, home.shape[1]:] = away
    b = np.full((n, 2), np.nan, dtype=np.float32)
    b[idx] = np.where(np.isnan(ball), ball2, ball)
    team = np.array([HOME] * home.shape[1] + [AWAY] * away.shape[1], dtype=np.int8)
    periods = _periods(per, idx)

    import pandas as pd
    ev = pd.read_csv(p_ev)
    events = []
    for r in ev.itertuples(index=False):
        t = str(getattr(r, "Team")).strip().lower()
        events.append({
            "team": HOME if t == "home" else AWAY if t == "away" else None,
            "type": str(r.Type).strip().upper(),
            "subtype": "" if pd.isna(r.Subtype) else str(r.Subtype).strip().upper(),
            "period": int(r.Period),
            "start": int(r[4]) - 1, "end": int(r[6]) - 1,
        })
    atd = {"ball": b, "players": players, "pspeed": np.full((n, S), np.nan, np.float32),
           "team": team, "team_names": np.array(["Home", "Away"])}
    return atd, events, periods


def _periods(per, idx):
    out = []
    for p in sorted(set(per.tolist())):
        f = idx[per == p]
        out.append((int(p), int(f.min()), int(f.max())))
    return out


# -- game 3: EPTS + JSON -----------------------------------------------------------------------

def _load_epts_game(folder, name):
    from src import atd as atd_mod
    meta = os.path.join(folder, name + "_metadata.xml")
    raw = os.path.join(folder, name + "_tracking.txt")
    team, pch, bch, names = atd_mod.read_meta(meta)
    with open(raw) as fh:
        n = sum(1 for _ in fh)
    first_id = None
    with open(raw) as fh:
        first_id = int(fh.readline().split(":")[0])
    # parse_raw maps frame_id -> frame_id - 1; shift so the file starts at index 0
    segs = atd_mod.read_segments(meta)      # players on the pitch change with substitutions
    ball, players, pspeed, n_bad = atd_mod.parse_raw(raw, n + first_id, len(team), pch, bch, segs)
    if n_bad > 0.01 * n:
        raise ValueError("{}: {} of {} tracking lines unparsed".format(name, n_bad, n))
    ball, players, pspeed = ball[first_id - 1:], players[first_id - 1:], pspeed[first_id - 1:]
    with open(os.path.join(folder, name + "_events.json")) as fh:
        data = json.load(fh)
    rows = data["data"] if isinstance(data, dict) else data
    import difflib

    def side(team_name):
        if not team_name:
            return None
        sc = [difflib.SequenceMatcher(None, team_name.lower(), x.lower()).ratio() for x in names]
        return int(np.argmax(sc))

    events = []
    for r in rows:
        tname = (r.get("team") or {}).get("name")
        subs = r.get("subtypes")
        if isinstance(subs, list):
            sub = "-".join(str((s or {}).get("name", "")) for s in subs)
        elif isinstance(subs, dict):
            sub = str(subs.get("name", ""))
        else:
            sub = ""
        events.append({
            "team": side(tname), "type": str(r["type"]["name"]).strip().upper(),
            "subtype": sub.strip().upper(), "period": int(r.get("period") or 1),
            "start": int(r["start"]["frame"]) - first_id,
            "end": int((r.get("end") or r["start"])["frame"]) - first_id,
        })
    # Periods from the events (tracking export has no period column): span of each period's
    # events, widened to the tracking frames between them.
    periods = []
    for p in sorted({e["period"] for e in events}):
        fr = [e["start"] for e in events if e["period"] == p] + \
             [e["end"] for e in events if e["period"] == p]
        periods.append((p, max(0, min(fr)), min(len(ball) - 1, max(fr))))
    atd = {"ball": ball, "players": players, "pspeed": pspeed, "team": team,
           "team_names": np.array(names)}
    return atd, events, periods


def load_game(name, base=DEFAULT_DIR):
    folder = os.path.join(base, name)
    if os.path.isfile(os.path.join(folder, name + "_RawEventsData.csv")):
        return _load_csv_game(folder, name)
    if os.path.isfile(os.path.join(folder, name + "_events.json")):
        return _load_epts_game(folder, name)
    raise IOError("no recognised Metrica sample files in " + folder)
