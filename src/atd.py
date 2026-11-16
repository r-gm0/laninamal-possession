"""Read the raw FIFA/EPTS ATD into numpy arrays, with an on-disk cache.

Output (all indexed by VIDEO frame, i.e. frame_id - 1):
    ball      (N, 2) float32  normalised 0-1, NaN where the tracker lost the ball
    players   (N, S, 2) float32 normalised 0-1, NaN where a slot is empty
    pspeed    (N, S) float32  player speed channel if the export has one (m/s), else NaN
    team      (S,) int8       0 = home, 1 = away, per track slot
    team_names (2,) str       club names (home, away) from the metadata

The cache is frame-level derived data from the match: it lives in cache/ which is gitignored.
Nothing here prints values -- only shapes, channel names and aggregate counts.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, require, source_frames  # noqa: E402

HOME, AWAY = 0, 1
CACHE_VERSION = 2


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def read_meta(meta_path):
    """Team of each slot, player channel names and ball channel names, from the metadata XML."""
    root = ET.parse(meta_path).getroot()
    teams = [t for t in root.iter() if _local(t.tag) == "Team"]
    team_ids = [t.get("id") for t in teams]
    names = []
    for t in teams:
        nm = next((c.text for c in t if _local(c.tag) == "Name"), None)
        names.append((nm or "").strip())
    if len(team_ids) != 2:
        raise ValueError("expected 2 teams in metadata, found {}".format(len(team_ids)))
    players = [p for p in root.iter() if _local(p.tag) == "Player" and p.get("teamId")]
    team = np.array([HOME if p.get("teamId") == team_ids[0] else AWAY for p in players],
                    dtype=np.int8)

    # PlayerChannel id -> channel id (e.g. "player1_x" -> "x")
    pc_map = {pc.get("id"): pc.get("channelId")
              for pc in root.iter() if _local(pc.tag) == "PlayerChannel"}
    spec = next((s for s in root.iter() if _local(s.tag) == "DataFormatSpecification"), None)
    player_channels, ball_channels = ["x", "y", "speed"], ["x", "y"]
    if spec is not None:
        ball_channels = [b.get("channelId") for b in spec.iter()
                         if _local(b.tag) == "BallChannelRef"] or ball_channels
        for reg in spec.iter():
            refs = [c for c in reg if _local(c.tag) == "PlayerChannelRef"]
            if refs:  # the first per-player register defines the group layout
                player_channels = [pc_map.get(r.get("playerChannelId"), "?") for r in refs]
                break
    return team, player_channels, ball_channels, names


def read_segments(meta_path):
    """[(start_frame_id, end_frame_id, slot indices in line order)] -- one per
    DataFormatSpecification. EPTS files with substitutions (e.g. Metrica Sample_Game_3) list
    only the players on the pitch in each frame range; slots index the metadata Player list."""
    root = ET.parse(meta_path).getroot()
    players = [p.get("id") for p in root.iter() if _local(p.tag) == "Player" and p.get("teamId")]
    slot_of = {pid: i for i, pid in enumerate(players)}
    pc_player = {pc.get("id"): pc.get("playerId")
                 for pc in root.iter() if _local(pc.tag) == "PlayerChannel"}
    segs = []
    for spec in (s for s in root.iter() if _local(s.tag) == "DataFormatSpecification"):
        order = []
        for r in spec.iter():
            if _local(r.tag) == "PlayerChannelRef":
                pid = pc_player.get(r.get("playerChannelId"))
                if pid in slot_of and (not order or order[-1] != slot_of[pid]):
                    order.append(slot_of[pid])
        segs.append((int(spec.get("startFrame", 1)), int(spec.get("endFrame", 10 ** 9)),
                     np.array(order, dtype=np.int64)))
    return sorted(segs, key=lambda s: s[0])


def _index_of(channels, *names):
    for name in names:
        for i, c in enumerate(channels):
            if c and c.lower().endswith(name):
                return i
    return None


def parse_raw(raw_path, n_frames, n_slots, player_channels, ball_channels, segments=None):
    """Stream the raw text once. Each line: '<frame_id>:<p1>;<p2>;...:<ball>'.
    segments (from read_segments): per frame range, which slots the line lists; None = all."""
    npc, nbc = len(player_channels), len(ball_channels)
    ix, iy = _index_of(player_channels, "x"), _index_of(player_channels, "y")
    isp = _index_of(player_channels, "speed")
    bx, by = _index_of(ball_channels, "x"), _index_of(ball_channels, "y")
    if None in (ix, iy, bx, by):
        raise ValueError("cannot find x/y channels in the ATD data format specification")

    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    pspeed = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    n_bad = 0
    seg_starts = np.array([s[0] for s in segments]) if segments is not None else None
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) < 3:
                n_bad += 1
                continue
            f = int(parts[0]) - 1
            if not 0 <= f < n_frames:
                n_bad += 1
                continue
            vals = np.array(parts[1].replace(";", ",").split(","), dtype=np.float64)
            if segments is not None:
                k = int(np.searchsorted(seg_starts, f + 1, side="right")) - 1
                slots = segments[max(k, 0)][2]
                if vals.size == len(slots) * npc:
                    grid = vals.reshape(len(slots), npc)
                    players[f, slots, 0] = grid[:, ix]
                    players[f, slots, 1] = grid[:, iy]
                    if isp is not None:
                        pspeed[f, slots] = grid[:, isp]
                else:
                    n_bad += 1
            elif vals.size == n_slots * npc:
                grid = vals.reshape(n_slots, npc)
                players[f, :, 0] = grid[:, ix]
                players[f, :, 1] = grid[:, iy]
                if isp is not None:
                    pspeed[f] = grid[:, isp]
            else:
                n_bad += 1
            b = parts[2].split(",")
            if len(b) >= max(bx, by) + 1:
                ball[f, 0] = float(b[bx])
                ball[f, 1] = float(b[by])
    return ball, players, pspeed, n_bad


def load(cache_dir="cache", game=GAME, data_dir=None, verbose=True):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    stat = os.stat(raw_path)
    key = "{}_{}_{}_v{}".format(game, stat.st_size, int(stat.st_mtime), CACHE_VERSION)
    cache_path = os.path.join(ROOT, cache_dir, "atd_{}.npz".format(key))
    if os.path.isfile(cache_path):
        z = np.load(cache_path)
        out = {k: z[k] for k in z.files}
    else:
        team, pch, bch, names = read_meta(meta_path)
        if verbose:
            print("ATD schema: {} slots (home {}, away {}); player channels {}; ball channels {}"
                  .format(len(team), int((team == HOME).sum()), int((team == AWAY).sum()),
                          pch, bch))
        ball, players, pspeed, n_bad = parse_raw(raw_path, n_frames, len(team), pch, bch)
        out = {"ball": ball, "players": players, "pspeed": pspeed, "team": team,
               "team_names": np.array(names)}
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, **out)
        if verbose and n_bad:
            print("  {} malformed lines skipped".format(n_bad))
    if verbose:
        print("ATD loaded: ball {} players {} ; ball present on {:.1f}% of frames"
              .format(out["ball"].shape, out["players"].shape,
                      100.0 * np.mean(~np.isnan(out["ball"][:, 0]))))
    return out
