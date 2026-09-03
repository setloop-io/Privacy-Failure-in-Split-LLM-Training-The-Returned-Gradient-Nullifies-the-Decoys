#!/usr/bin/env python3
"""Pure-python DTW / edit-distance realignment core (TORCH-FREE).

Used by the alignment-search (E-R8) attacker: the jittered-scaffold defense
(E-R3 arm b) inserts random pads and permutes template slots, so the
attacker's assumed position alignment is wrong. This module realigns a
captured boundary-activation sequence against candidate prefix sequences
before any W solve.

Rows are passed as plain Python sequences of floats, so the whole module is
testable (and pinned by frozen fixtures) without torch. Torch callers
convert with `.tolist()`.

Two primitives:
  * dtw_distance(a, b, window=None) — Sakoe-Chiba banded dynamic-time-warping
    with per-row Euclidean local cost; returns (normalized_cost, path).
  * edit_alignment(a, b) — classic Levenshtein-style alignment (match /
    insert / delete with continuous match cost), returns aligned index
    pairs. Used when the jitter is dominated by INSERTED pads (the actual
    E-R3 jitter mechanism), where DTW's many-to-one mapping is the wrong
    bias.
"""

import math


def _row_dist(x, y):
    return math.sqrt(sum((xi - yi) ** 2 for xi, yi in zip(x, y)))


def dtw_distance(a, b, window=None):
    """Banded DTW. a, b: sequences of equal-length float rows.
    Returns (cost / path_len, path) with path a list of (i, j) pairs
    monotone from (0,0) to (n-1, m-1). window=None means full band."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf"), []
    if window is None:
        window = max(n, m)
    window = max(window, abs(n - m))
    inf = float("inf")
    prev = [inf] * (m + 1)
    prev[0] = 0.0
    parent = {}  # (i,j) -> (pi,pj)
    for i in range(n):
        cur = [inf] * (m + 1)
        j0 = max(0, i - window)
        j1 = min(m - 1, i + window)
        for j in range(j0, j1 + 1):
            c = _row_dist(a[i], b[j])
            best, arg = prev[j + 1], (i - 1, j)      # up (insertion in b)
            if prev[j] < best:
                best, arg = prev[j], (i - 1, j - 1)  # diagonal
            if cur[j] < best:
                best, arg = cur[j], (i, j - 1)       # left
            cur[j + 1] = c + best
            parent[(i, j)] = arg
        prev = cur
    path = []
    node = (n - 1, m - 1)
    if prev[m] == inf:
        return inf, []
    while True:
        path.append(node)
        if node == (0, 0):
            break
        node = parent[node]
    path.reverse()
    return prev[m] / len(path), path


def edit_alignment(a, b, gap_cost=None):
    """Levenshtein-style global alignment with continuous match cost.
    Returns list of (i, j) matched pairs (deletions of b-rows and insertions
    in a are skipped, not paired). gap_cost defaults to the median pairwise
    row distance of a small sample — a scale-free default."""
    n, m = len(a), len(b)
    if gap_cost is None:
        ds = []
        for i in range(0, n, max(1, n // 8)):
            for j in range(0, m, max(1, m // 8)):
                ds.append(_row_dist(a[i], b[j]))
        ds.sort()
        gap_cost = ds[len(ds) // 2] if ds else 1.0
    inf = float("inf")
    dp = [[inf] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    move = {}
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            best, arg = inf, None
            if i > 0 and j > 0:
                c = dp[i - 1][j - 1] + _row_dist(a[i - 1], b[j - 1])
                if c < best:
                    best, arg = c, "match"
            if i > 0 and dp[i - 1][j] + gap_cost < best:
                best, arg = dp[i - 1][j] + gap_cost, "del_a"
            if j > 0 and dp[i][j - 1] + gap_cost < best:
                best, arg = dp[i][j - 1] + gap_cost, "del_b"
            dp[i][j] = best
            move[(i, j)] = arg
    pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        mv = move.get((i, j))
        if mv == "match":
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif mv == "del_a":
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


# Frozen fixtures for --self-test (torch-free).
FIXTURE_DTW = {
    # identical sequences -> cost 0, identity path
    "identity_cost": 0.0,
    # 'shifted' is the base sequence with one row dropped and one inserted:
    # DTW must still find a cheap path (cost well below the independent-row
    # baseline computed below).
    "base": [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
    "shifted": [[0.0, 0.0], [9.0, 9.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0],
                [4.0, 0.0]],
}
FIXTURE_EDIT_PAIRS = [(0, 0), (1, 2), (2, 3), (3, 4), (4, 5)]


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    base, shifted = FIXTURE_DTW["base"], FIXTURE_DTW["shifted"]
    print("DTW core (pure python):")
    c0, p0 = dtw_distance(base, base)
    check("identity sequence -> cost 0", abs(c0) < 1e-12)
    check("identity path is the diagonal",
          p0 == [(i, i) for i in range(len(base))])
    c1, p1 = dtw_distance(base, shifted)
    diag = sum(_row_dist(base[min(i, len(base) - 1)],
                         shifted[i]) for i in range(len(shifted))
               ) / len(shifted)
    check(f"pad-inserted sequence realigns cheaper than the diagonal "
          f"({c1:.3f} < {diag:.3f})", c1 < diag)
    check("true rows map to their shifted positions "
          "[(1,2),(2,3),(3,4),(4,5)] in the path",
          all(p in p1 for p in [(1, 2), (2, 3), (3, 4), (4, 5)]))
    check("path is monotone and spans both sequences",
          p1[0] == (0, 0) and p1[-1] == (len(base) - 1, len(shifted) - 1)
          and all(b0 <= b1 and a0 <= a1
                  for (a0, b0), (a1, b1) in zip(p1, p1[1:])))
    cw, _ = dtw_distance(base, shifted, window=1)
    check("windowed DTW runs and stays finite-or-inf consistently",
          cw == float("inf") or cw >= 0.0)
    check("empty input returns (inf, [])",
          dtw_distance([], base) == (float("inf"), []))

    print("edit alignment (insert/delete model):")
    pairs = edit_alignment(base, shifted, gap_cost=2.0)
    check("inserted pad row is skipped, all 5 true rows matched",
          pairs == FIXTURE_EDIT_PAIRS)
    check("identity aligns to the diagonal",
          edit_alignment(base, base, gap_cost=2.0)
          == [(i, i) for i in range(len(base))])

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
