"""Viterbi decoding with a minimum duration per state (explicit-duration HMM, flat layout).

Each state k is expanded into a chain of m_k sub-states: entering k puts you on sub-state 0, and
you must walk the chain (one step per frame) before reaching sub-state m_k-1, the only one from
which you may stay or leave. So every run of state k lasts at least m_k frames, except a run cut
by the end of the half. Switching k -> j costs penalty[k, j] nats.

Cost: O(N * sum(m_k)) with tiny numpy ops; one half (~80k frames) decodes in a few seconds.
"""

import numpy as np

NEG = -1e18


def decode(E, min_dur, penalty, init_logprior, switch_bonus=None, bonus_mask=None):
    """E: (N, K) log-emission scores. min_dur: K ints >= 1. penalty: (K, K) nats, diag ignored.
    init_logprior: (K,). Optional switch_bonus (N,) nats subtracted from the penalty of the
    transitions selected by bonus_mask (K, K) bool, for a transition entering at frame t.
    Returns int array (N,) of states."""
    E = np.asarray(E, dtype=np.float64)
    n, K = E.shape
    m = [max(1, int(x)) for x in min_dur]
    off = np.concatenate(([0], np.cumsum(m)))
    S = int(off[-1])
    first = off[:-1].astype(np.int64)            # sub-state 0 of each state
    last = (off[1:] - 1).astype(np.int64)        # final sub-state of each state
    state_of = np.repeat(np.arange(K), m)
    trans = -np.asarray(penalty, dtype=np.float64)
    np.fill_diagonal(trans, NEG)                 # self-moves go through the chain, not here

    # shift source: sub-state j>0 comes from j-1; sub-state 0 has no chain predecessor (sentinel S)
    src = np.arange(S) - 1
    src[first] = S
    src = np.append(src, S)                      # the sentinel maps to itself
    multi = np.array([mk >= 2 for mk in m])

    V = np.full(S + 1, NEG)
    V[first] = np.asarray(init_logprior, dtype=np.float64) + E[0]
    bp_first = np.zeros((n, K), dtype=np.int32)  # flat predecessor of sub-state 0
    bp_last = np.zeros((n, K), dtype=np.int32)   # flat predecessor of the last sub-state

    # switch_bonus: (N,) with bonus_mask (K, K), or a full (N, K, K) per-frame bonus matrix
    full = switch_bonus is not None and np.ndim(switch_bonus) == 3
    if full:
        active = np.any(switch_bonus != 0.0, axis=(1, 2))
    for t in range(1, n):
        new = V[src]                                       # chain step
        if full and active[t]:
            cand = V[last][:, None] + trans + switch_bonus[t]
        elif not full and switch_bonus is not None and switch_bonus[t] != 0.0:
            cand = V[last][:, None] + trans + switch_bonus[t] * bonus_mask
        else:
            cand = V[last][:, None] + trans                # (from, to)
        ent_from = np.argmax(cand, axis=0)
        ent = cand[ent_from, np.arange(K)]
        ent_src = last[ent_from]
        # last sub-state: stay (from itself) vs arrive from the previous sub-state
        stay = V[last]
        arrive = np.where(multi, V[np.maximum(last - 1, 0)], NEG)
        take_stay = stay >= arrive
        last_val = np.where(take_stay, stay, arrive)
        last_src = np.where(take_stay, last, last - 1)
        # single-sub-state states: sub-state 0 == last; compete stay vs entry
        single_val = np.where(stay >= ent, stay, ent)
        single_src = np.where(stay >= ent, last, ent_src)
        new_first = np.where(multi, ent, single_val)
        new_first_src = np.where(multi, ent_src, single_src)
        new[last] = np.where(multi, last_val, new[last])
        new[first] = new_first
        bp_first[t] = new_first_src
        bp_last[t] = np.where(multi, last_src, new_first_src)
        new[:S] += E[t, state_of]
        new[S] = NEG
        V = new

    s = int(np.argmax(V[:S]))
    path = np.empty(n, dtype=np.int8)
    is_first = np.zeros(S, dtype=bool)
    is_first[first] = True
    is_last = np.zeros(S, dtype=bool)
    is_last[last] = True
    for t in range(n - 1, -1, -1):
        k = state_of[s]
        path[t] = k
        if t == 0:
            break
        if is_first[s]:
            s = int(bp_first[t, k])
        elif is_last[s]:
            s = int(bp_last[t, k])
        else:
            s = s - 1
    return path
