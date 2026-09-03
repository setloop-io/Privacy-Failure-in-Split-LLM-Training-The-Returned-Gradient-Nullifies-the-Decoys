#!/usr/bin/env python3
"""E8 robustness battery — honest bounds on the secret-W boundary defense.

e8_obfuscation.py showed that obfuscating boundary activations with a secret
orthogonal W collapses the trained MLP attacker (37-80% -> 0.00% top-1) at
~zero utility/overhead cost. That measurement alone overstates the defense:
this script stress-tests the four ways a rational attacker responds.

THREAT-MODEL FRAMING (read before citing any number here):
  - PASSIVE OBSERVER (attacks 1, 2, 4): the cloud sees only h' = h @ W
    vectors. It has public text and the public base model (so it can train
    the plain-h decoder and compute public activation statistics), but NO
    (h', token) labels for victim traffic. Attacks 1a/1b are LABEL-FREE.
  - ORACLE / CHOSEN-PLAINTEXT (attack 3): a strictly stronger attacker who
    can obtain K (activation, known-token) pairs at the boundary — e.g. by
    slipping chosen single-token prompts through the local node. This is
    NOT the default E8 threat model; the resulting recovery-vs-K curve is
    the honest security parameter of the defense (with H=1024, full W
    recovery needs ~H well-spread pairs; we verify empirically).
  - Attack 4 (partial leak) parametrizes the breach case between the two:
    a fraction of W's rows has leaked (insider), the rest stays secret.

Attacks:
  1a. Label-free PCA/whitening of h', nearest-token decode against whitened
      public per-token mean activations. Fails in theory (whitened h' still
      differs from whitened h by the unknown rotation) — verify empirically.
  1b. Label-free second-moment alignment: match mean+covariance of h' to
      the public h distribution via Q = cov(h')^{-1/2} cov(h)^{1/2}
      (symmetric sqrt), then run the TRAINED decoder on (h'-mu')Q + mu.
      Fails in theory up to spectrum degeneracies: the label-free problem
      is unidentifiable — Q = cov(h')^{-1/2} R cov(h)^{1/2} is a valid
      second-moment match for ANY orthogonal R. Verify empirically.
  2.  Session rotation: W re-seeded per session (S=3); the attacker pools
      h' across sessions. Quantify marginal information per session:
      per-session and pooled top-1 should stay at the random baseline.
  3.  Oracle K pairs: exact position-0 boundary activations of K chosen
      tokens (single-token prompts are context-free, so the attacker's
      public h(t) is EXACT). Solve W_hat = argmin ||H W - H'||_F by least
      squares, orthogonalize (polar factor), decode victim h' @ W_hat^T.
      Sweep K in {10, 100, 1000, 5000}; report the recovery-vs-K curve and
      the K where top-1 crosses 50% (the honest security parameter),
      interpolated linearly in K between the bracketing measured points
      (`oracle_k50`), not read off the sampled grid.
  4.  Partial leak: 50% / 90% of W's rows leaked, remainder replaced by a
      random orthogonal completion of the leaked row space. Decoding
      h' @ W_tilde^T then recovers the leaked coordinates of h EXACTLY and
      garbage elsewhere — measures graceful vs catastrophic degradation.

Metrics: top-1/top-5 token recovery, mean +/- std over --seeds (decoder
init/shuffle), per depth in {1, 4, 8}. JSON + training_status writes.

Usage:
    python e8_robustness.py --help        # works without torch
    python e8_robustness.py --self-test   # works without torch
    python e8_robustness.py --toy --quick # CPU machinery check
    python e8_robustness.py --model <hf-model> --corpus-file <docs.txt> --output e8r.json
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter

# Guarded heavy imports: `--help` must work on torch-less hosts.
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Unguarded (as in e8_obfuscation.py): add_numerics_args runs while the
# parser is built, so `--help`/`--self-test` must not need the heavy import.
from trained_inversion import (add_numerics_args, apply_numerics,  # noqa: E402
                               tensor_sha256)
try:
    from split_trainer import (TEXT_SAMPLES, _write_training_status,
                               build_modules, make_layer_kwargs,
                               run_layer_stack)
    from trained_inversion import (collect_base_pairs, evaluate_decoder,
                                   make_provenance, mean_std, seed_all,
                                   split_at, train_decoder)
    from e8_obfuscation import make_secret
except ImportError:  # pragma: no cover - torch-less host
    TEXT_SAMPLES = []
    _write_training_status = lambda **k: None
    build_modules = make_layer_kwargs = run_layer_stack = None
    collect_base_pairs = evaluate_decoder = mean_std = seed_all = None
    split_at = train_decoder = make_secret = make_provenance = None


# Pure crossing-K logic (no torch) — CPU-verifiable via --self-test.
#
# The oracle "security parameter" is the K at which top-1 recovery crosses 50%.
# Reading off the first *sampled* K with top1 >= 50 gives a grid point, not a
# crossing: on the default --k-list the 100 -> 1000 jump turned a 49.55%
# reading at K=100 into a "crosses at K=1000", ~10x too generous.

K50_THRESHOLD_PCT = 50.0

# Guard against interpolating a single confident number across a decade-wide
# bracket. A point estimate can never be sharper than the bracket it sits in,
# and these recovery curves are visibly concave over decade scales, so a 10x
# bracket cannot support a linear reading. 3x admits every sub-grid bracket
# measured so far (1.29x, 1.11x, 1.29x) and rejects the default grid's 10x.
# k50_bracket is emitted either way, so the reader always sees the resolution.
K50_MAX_BRACKET_RATIO = 3.0


def setting_to_k(setting):
    """Parse an oracle summary setting, 'K=1000' -> 1000; None if not one.

    Uses removeprefix, NOT lstrip. `"KK=5".lstrip("K=")` strips a character
    SET and yields "5"; removeprefix leaves it alone, as it must.
    """
    rest = setting.removeprefix("K=")
    if rest == setting or not rest.isdigit():
        return None
    return int(rest)


def crossing_k(curve, threshold=K50_THRESHOLD_PCT,
               max_bracket_ratio=K50_MAX_BRACKET_RATIO):
    """First upward crossing of `threshold`, linearly interpolated in K.

    curve: iterable of {"K": int, "top1_mean": float}, any order.

    Takes the FIRST crossing from below (the curves are not monotone: the
    seed-42 sub-grid dips 44.14 -> 43.75 from K=60 to K=70), interpolates
    linearly in K -- not in log K, which lands 0.13-0.27 low on the committed
    brackets (87.42 / 94.16 against 87.68 / 94.29) -- and declines rather than
    guessing when the bracket is coarser than `max_bracket_ratio`.

    Returns a dict of new artifact fields. `k50_interpolated` is an explicit
    None whenever no number is defensible; `k50_method` says which case:
      "linear_in_K"        crossing bracketed and interpolated
      "bracket_too_coarse" crossing bracketed, gap too wide to interpolate
      "crossed_at_or_below_min_k"  already >= threshold at the smallest K
                           sampled, so the crossing is below the grid
      "not_bracketed"      curve never reaches `threshold` at any sampled K
    """
    out = {"k50_interpolated": None, "k50_bracket": None,
           "k50_bracket_ratio": None, "k50_method": "not_bracketed",
           "k50_threshold_pct": threshold, "k50_max_sampled_k": None,
           "k50_n_points": 0}
    # Non-finite means the cell failed; dropping it is honest, whereas leaving
    # it in silently kills the bracket on BOTH sides (every comparison against
    # NaN is False) and yields a confident "never crosses".
    pts = sorted((int(p["K"]), float(p["top1_mean"])) for p in curve
                 if math.isfinite(float(p["top1_mean"])))
    if not pts:
        return out
    out["k50_max_sampled_k"] = pts[-1][0]
    out["k50_n_points"] = len(pts)
    if pts[0][1] >= threshold:
        # Already over the line at the smallest K sampled. This is NOT
        # "never crosses": the crossing exists, it is just below the grid.
        # Reporting it as not_bracketed would understate a real break.
        out["k50_method"] = "crossed_at_or_below_min_k"
        out["k50_bracket"] = [None, pts[0][0]]
        return out
    for (k_lo, y_lo), (k_hi, y_hi) in zip(pts, pts[1:]):
        if not (y_lo < threshold <= y_hi):
            continue
        out["k50_bracket"] = [k_lo, k_hi]
        # A non-positive lower K has no meaningful ratio; treat as unusable
        # rather than storing inf, which json.dump writes as bare `Infinity`.
        ratio = round(k_hi / k_lo, 4) if k_lo > 0 else None
        out["k50_bracket_ratio"] = ratio
        if ratio is None or ratio > max_bracket_ratio:
            out["k50_method"] = "bracket_too_coarse"
            return out
        frac = (threshold - y_lo) / (y_hi - y_lo)
        out["k50_interpolated"] = round(k_lo + frac * (k_hi - k_lo), 4)
        out["k50_method"] = "linear_in_K"
        return out
    return out


def oracle_k50_from_rows(rows, threshold=K50_THRESHOLD_PCT,
                         max_bracket_ratio=K50_MAX_BRACKET_RATIO):
    """Per-depth crossing dicts from `summary`-shaped oracle rows.

    Pure and keyed exactly as emitted, so --self-test covers the assembly that
    produces the artifact field -- not just the interpolator underneath it.
    Uses K_effective (the pairs actually used; the pool caps the requested K),
    falling back to parsing the setting for artifacts that predate that field.
    """
    by_depth = {}
    for r in rows:
        if r.get("attack") != "3_oracle_pairs":
            continue
        k = r.get("K_effective")
        k = setting_to_k(r["setting"]) if k is None else int(k)
        assert k is not None, f"oracle row with unparseable setting: {r}"
        by_depth.setdefault(int(r["depth"]), []).append(
            {"K": k, "top1_mean": r["top1_mean"]})
    return {str(d): crossing_k(c, threshold, max_bracket_ratio)
            for d, c in sorted(by_depth.items())}


def quoted_crossing(oracle_k50):
    """(depth, crossing) for the depth the verdict quotes: the SHALLOWEST run.

    Shallowest because it is the strongest attack surface and the depth
    COMPARISON_MATRIX section 5b.1 reports. Deeper splits cross far later --
    on the seed-123 sub-grid, depth 1 crosses at 87.4 but depth 4 at 352.3 --
    so quoting the wrong one overstates the defense by ~4x.
    """
    if not oracle_k50:
        return None, None
    depth = min(oracle_k50, key=int)
    return int(depth), oracle_k50[depth]


def oracle_verdict_clause(cross, depth):
    """The ORACLE sentence of the `verdict` string, from a crossing_k dict.

    Pure so --self-test can prove it: the verdict is the ONLY surface the
    crossing-K ever escaped on (k50 was a local, never an artifact field), so
    it is the surface that has to stop asserting a grid point.

    Everything it states is read back out of `cross` -- threshold, bracket and
    largest K actually sampled -- so the sentence cannot drift from the field
    it describes. The interpolated K is printed to 1 dp: the artifact keeps
    full resolution, but the bracket is tens of K wide and the prose should
    not imply four decimals of confidence.
    """
    method = (cross or {}).get("k50_method")
    thr = f"{(cross or {}).get('k50_threshold_pct', K50_THRESHOLD_PCT):g}"
    if method == "linear_in_K":
        lo, hi = cross["k50_bracket"]
        return (f"ORACLE: top-1 crosses {thr}% at K={cross['k50_interpolated']:.1f} "
                f"labeled pairs (interpolated linearly in K between measured "
                f"K={lo} and K={hi}, depth {depth}) ")
    if method == "bracket_too_coarse":
        lo, hi = cross["k50_bracket"]
        return (f"ORACLE: top-1 crosses {thr}% somewhere above K={lo} and at or "
                f"below K={hi} (depth {depth}); the sampled grid is "
                f"{cross['k50_bracket_ratio']}x wide there, too coarse to "
                f"interpolate — refine --k-list before quoting a crossing K ")
    if method == "crossed_at_or_below_min_k":
        return (f"ORACLE: top-1 is already at or above {thr}% at the smallest K "
                f"sampled (K={cross['k50_bracket'][1]}, depth {depth}), so the "
                f"crossing is at or below it — extend --k-list downward to "
                f"resolve it ")
    if method == "not_bracketed":
        return (f"ORACLE: top-1 never reaches {thr}% at any sampled K "
                f"(max sampled K={cross.get('k50_max_sampled_k')}, "
                f"depth {depth}) ")
    return "ORACLE: not measured in this run "


# Label-free math helpers (fp64 for numerical stability in the decompositions)
def sym_sqrt(cov, inv=False, eps=1e-8):
    """Symmetric (inverse) square root of a covariance matrix."""
    evals, evecs = torch.linalg.eigh(cov.double())
    evals = evals.clamp_min(eps)
    s = evals.rsqrt() if inv else evals.sqrt()
    return (evecs * s) @ evecs.T


def polar(m):
    """Orthogonal polar factor of m (nearest orthogonal matrix)."""
    u, _, vt = torch.linalg.svd(m.double())
    return u @ vt


def cov_of(x):
    x = x.double()
    x = x - x.mean(0, keepdim=True)
    return x.T @ x / max(1, x.shape[0] - 1)


def boundary_acts(embed, head, rotary, ids_list, args):
    """Per-position boundary activations h* for a list of token-id blocks."""
    hs = []
    with torch.no_grad():
        for ids in ids_list:
            x = ids[:-1].unsqueeze(0).to(args.device)
            position_ids = torch.arange(x.shape[1],
                                        device=args.device).unsqueeze(0)
            hidden = embed(x)
            lk = make_layer_kwargs(rotary, hidden, position_ids, args)
            hs.append(run_layer_stack(head, hidden, lk)[0].float().cpu())
    return torch.cat(hs)


def position0_acts(embed, head, rotary, token_ids, args, batch=512):
    """Exact context-free boundary activation of single-token prompts
    (batched as length-1 sequences). Returns [len(token_ids), H] fp32."""
    outs = []
    with torch.no_grad():
        for i in range(0, len(token_ids), batch):
            ids = torch.tensor(token_ids[i:i + batch], dtype=torch.long,
                               device=args.device).unsqueeze(1)  # [B, 1]
            position_ids = torch.zeros_like(ids)
            hidden = embed(ids)
            lk = make_layer_kwargs(rotary, hidden, position_ids, args)
            h = run_layer_stack(head, hidden, lk)
            outs.append(h[:, 0].float().cpu())
    return torch.cat(outs)


def frequent_tokens(encode, docs, args, k):
    """Top-k most frequent token ids in the given documents."""
    c = Counter()
    for doc in docs:
        for b in encode([doc], args.seq_len):
            c.update(b.tolist())
    return [t for t, _ in c.most_common(k)]


def token_id_hash(token_ids):
    """Stable hash of an ordered probe-token selection."""
    payload = ",".join(str(int(t)) for t in token_ids).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def nested_repetition_stats(repetitions, metric):
    """Variance decomposition for probe repeats containing decoder repeats.

    `between_probe_std` is the sample SD of per-probe means. The within term is
    the square root of the mean sample variance across decoder seeds. Keeping
    both avoids presenting decoder-initialisation jitter as full attack
    repeatability.
    """
    rows = [[float(v) for v in r[metric]] for r in repetitions]
    means = [sum(row) / len(row) for row in rows]

    def sample_var(xs):
        if len(xs) < 2:
            return 0.0
        mu = sum(xs) / len(xs)
        return sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)

    grand = sum(means) / len(means)
    return {
        "grand_mean": round(grand, 4),
        "between_probe_std": round(math.sqrt(sample_var(means)), 4),
        "within_decoder_std": round(
            math.sqrt(sum(sample_var(row) for row in rows) / len(rows)), 4),
        "n_probe_repetitions": len(rows),
        "n_decoder_seeds_per_probe": len(rows[0]),
    }


def attack_1a_whiten_nn(h_prime, ref_h, ref_tok, victim_tok, vocab_size):
    """Label-free: whiten h' with its OWN stats, whiten public per-token
    means with THEIR own stats, nearest-mean decode. Seed-independent."""
    mu_p = h_prime.double().mean(0, keepdim=True)
    w_p = sym_sqrt(cov_of(h_prime), inv=True)
    mu_r = ref_h.double().mean(0, keepdim=True)
    w_r = sym_sqrt(cov_of(ref_h), inv=True)
    hp = (h_prime.double() - mu_p) @ w_p
    # per-token reference means (tokens present in the public pool)
    refs, labels = [], []
    for t in ref_tok.unique():
        m = (ref_tok == t)
        refs.append(((ref_h[m].double() - mu_r) @ w_r).mean(0))
        labels.append(t)
    refs = torch.stack(refs)
    labels = torch.stack(labels)
    # cosine nearest neighbour (memory-safe: normalized matmul, not cdist —
    # a full fp64 distance matrix over ~20K refs would be tens of GB)
    hp_n = torch.nn.functional.normalize(hp.float(), dim=-1)
    refs_n = torch.nn.functional.normalize(refs.float(), dim=-1)
    pred = labels[(hp_n @ refs_n.T).argmax(1)]
    top1 = (pred == victim_tok).float().mean().item() * 100
    coverage = torch.isin(victim_tok, labels).float().mean().item() * 100
    return round(top1, 2), round(coverage, 2)


def attack_1b_moment_align(h_prime, ref_h):
    """Label-free second-moment alignment. Returns the Q to apply to h'."""
    mu_p = h_prime.double().mean(0, keepdim=True)
    mu_r = ref_h.double().mean(0, keepdim=True)
    q = sym_sqrt(cov_of(h_prime), inv=True) @ sym_sqrt(cov_of(ref_h))
    return mu_p, mu_r, q


# Frozen fixtures for --self-test: the depth-1 `3_oracle_pairs` top-1 curves
# as committed in paper-data/results-h100-1/v2/privacy/ at 015785b, copied
# here on purpose — those artifacts will be regenerated, and a gate pinned to
# files expected to change gets rewritten to pass; the arithmetic pinned here
# does not change.
FIXTURE_KVOID_SEED99 = [  # ..._kvoid_seed99.json  -> K50 = 87.7
    (10, 21.55), (30, 29.6867), (50, 37.37), (70, 44.53), (90, 50.7167),
    (100, 52.3433), (200, 62.1733), (300, 66.54), (500, 68.2933),
    (700, 68.6833), (1000, 68.62)]
FIXTURE_KVOID_SUB100_SEED42 = [  # ..._kvoid_sub100_seed42.json -> K50 = 94.3
    (10, 21.2867), (20, 25.13), (30, 26.6933), (40, 34.3767), (50, 38.4133),
    (60, 44.14), (70, 43.75), (80, 48.1133), (90, 49.61), (100, 50.52)]
FIXTURE_KVOID_SEED123 = [  # ..._kvoid_seed123.json -> K50 = 87.4
    (10, 19.4633), (30, 29.2333), (50, 37.89), (70, 43.1633), (90, 51.04),
    (100, 52.3433), (200, 61.9133), (300, 67.06), (500, 69.14),
    (700, 69.5933), (1000, 69.7267), (1024, 69.66), (2000, 69.66),
    (5000, 69.66)]
FIXTURE_SEED123_DEPTH4 = [  # ..._kvoid_seed123.json depth 4 -> K50 = 352.3
    (10, 2.28), (30, 14.91), (50, 14.3867), (70, 14.4533), (90, 31.5133),
    (100, 32.62), (200, 40.1667), (300, 45.9633), (500, 61.3933),
    (700, 66.5367), (1000, 66.9933), (1024, 67.0567), (2000, 67.0567),
    (5000, 67.0567)]
FIXTURE_DEFAULT_GRID = [  # ..._rep2.json depth 1 on the argparse-default grid
    (10, 21.55), (100, 49.5467), (1000, 68.5533), (1024, 68.5533),
    (5000, 68.5533)]


def _curve(pairs):
    return [{"K": k, "top1_mean": t} for k, t in pairs]


def _log_k_crossing(pairs, threshold=K50_THRESHOLD_PCT):
    """Reference log-K interpolation — the wrong method, kept as a foil."""
    pts = sorted(pairs)
    for (k_lo, y_lo), (k_hi, y_hi) in zip(pts, pts[1:]):
        if y_lo < threshold <= y_hi:
            frac = (threshold - y_lo) / (y_hi - y_lo)
            return math.exp(math.log(k_lo) + frac * (math.log(k_hi)
                                                     - math.log(k_lo)))
    return None


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and cond

    print("D2/AC1 interpolated crossing-K reproduces the committed sub-grids:")
    r99 = crossing_k(_curve(FIXTURE_KVOID_SEED99))
    check("seed99: K50 = 87.7 +- 0.1 (got "
          f"{r99['k50_interpolated']})",
          r99["k50_interpolated"] is not None
          and abs(r99["k50_interpolated"] - 87.7) <= 0.1)
    check("seed99: bracket is the measured pair (70, 90), method linear_in_K",
          r99["k50_bracket"] == [70, 90] and r99["k50_method"] == "linear_in_K")
    r42 = crossing_k(_curve(FIXTURE_KVOID_SUB100_SEED42))
    check("seed42: K50 = 94.3 +- 0.1 (got "
          f"{r42['k50_interpolated']})",
          r42["k50_interpolated"] is not None
          and abs(r42["k50_interpolated"] - 94.3) <= 0.1)
    check("seed42: bracket is the measured pair (90, 100)",
          r42["k50_bracket"] == [90, 100])
    r123 = crossing_k(_curve(FIXTURE_KVOID_SEED123))
    check("seed123: K50 = 87.4 +- 0.1 (got "
          f"{r123['k50_interpolated']})",
          r123["k50_interpolated"] is not None
          and abs(r123["k50_interpolated"] - 87.4) <= 0.1)

    print("D2/AC1 the +-0.1 tolerance is what pins LINEAR in K, not log K:")
    l99, l42 = (_log_k_crossing(FIXTURE_KVOID_SEED99),
                _log_k_crossing(FIXTURE_KVOID_SUB100_SEED42))
    check(f"log-K would give {l99:.4f} / {l42:.4f}, both >0.1 off 87.7 / 94.3",
          abs(l99 - 87.7) > 0.1 and abs(l42 - 94.3) > 0.1)

    print("D2/AC2 a curve entirely below the threshold returns explicit null:")
    below = crossing_k(_curve([(10, 5.0), (100, 20.0), (1000, 45.0)]))
    check("k50_interpolated is None (not a grid value, not a string)",
          below["k50_interpolated"] is None)
    check("method records that the curve does not bracket the threshold",
          below["k50_method"] == "not_bracketed" and below["k50_bracket"] is None)

    print("D2/AC3 the argparse-default grid does NOT report K=1000:")
    grid = crossing_k(_curve(FIXTURE_DEFAULT_GRID))
    check("k50_interpolated is not 1000 (that was the published defect)",
          grid["k50_interpolated"] != 1000)
    check("declined as bracket_too_coarse; bracket (100, 1000) still reported",
          grid["k50_method"] == "bracket_too_coarse"
          and grid["k50_bracket"] == [100, 1000]
          and grid["k50_interpolated"] is None)
    check("the 10x gap is what triggers it (ratio 10.0 > 3.0)",
          grid["k50_bracket_ratio"] == 10.0)

    print("D2 first crossing from below (curves are not monotone):")
    dip = crossing_k(_curve([(10, 20.0), (20, 55.0), (30, 40.0), (40, 60.0)]))
    check("takes the FIRST crossing (10, 20), not the later (30, 40)",
          dip["k50_bracket"] == [10, 20]
          and abs(dip["k50_interpolated"] - 18.5714) <= 1e-4)
    check("a single point below the threshold cannot bracket anything",
          crossing_k(_curve([(100, 20.0)]))["k50_method"] == "not_bracketed")
    check("a single point above it puts the crossing below the grid",
          crossing_k(_curve([(100, 80.0)]))["k50_method"]
          == "crossed_at_or_below_min_k")
    check("an empty curve is handled, not raised",
          crossing_k([])["k50_method"] == "not_bracketed")
    edge = crossing_k(_curve([(10, 40.0), (20, 50.0)]))
    check("y_hi exactly at the threshold interpolates to k_hi",
          edge["k50_interpolated"] == 20.0)
    check("a bracket at exactly the ratio limit is admitted, not rejected",
          crossing_k(_curve([(100, 40.0), (300, 60.0)]))["k50_method"]
          == "linear_in_K")
    check("just past the ratio limit is rejected",
          crossing_k(_curve([(100, 40.0), (301, 60.0)]))["k50_method"]
          == "bracket_too_coarse")
    check("a non-positive lower K yields null ratio, not JSON-invalid Infinity",
          crossing_k(_curve([(0, 40.0), (10, 60.0)]))["k50_bracket_ratio"] is None)
    nan = crossing_k(_curve([(10, 20.0), (90, float("nan")), (100, 60.0)]))
    check("a NaN cell is dropped, not left to silently kill the bracket",
          nan["k50_n_points"] == 2 and nan["k50_bracket"] == [10, 100])

    print("D2 a crossing BELOW the sampled grid is not 'never crosses':")
    # ..._kvoid.json is exactly this shape: k_list starts at 100 and depth-1
    # top-1 is already 51.56 there, rising to 69.92 — calling that "never
    # crosses 50%" would understate a real break as a defense success.
    low = crossing_k(_curve([(100, 51.56), (200, 61.46), (1000, 69.9233)]))
    check("classified crossed_at_or_below_min_k, not not_bracketed",
          low["k50_method"] == "crossed_at_or_below_min_k")
    check("bracket is open below and names the smallest sampled K",
          low["k50_bracket"] == [None, 100])
    check("still no fabricated number",
          low["k50_interpolated"] is None)
    v_low = oracle_verdict_clause(low, 1)
    check("verdict says the crossing is at or below K=100, not that it never "
          "happened",
          "at or below" in v_low and "K=100" in v_low
          and "never reaches" not in v_low)

    print("D2/AC4 the verdict prose stops asserting a grid point:")
    v_grid = oracle_verdict_clause(grid, 1)
    check("default grid: verdict does NOT say 'crosses 50% at K=1000'",
          "crosses 50% at K=1000" not in v_grid)
    check("default grid: verdict states the bracket and why it declined",
          "above K=100" in v_grid and "below K=1000" in v_grid
          and "too coarse to interpolate" in v_grid)
    v99 = oracle_verdict_clause(r99, 1)
    check("seed99: verdict quotes the interpolated K, not a sampled one",
          "crosses 50% at K=87.7 " in v99
          and "between measured K=70 and K=90" in v99)
    check("seed99: no sampled K is asserted as the crossing",
          not any(f"crosses 50% at K={k}" in v99
                  for k, _ in FIXTURE_KVOID_SEED99))
    v_none = oracle_verdict_clause(below, 1)
    check("below-threshold curve: verdict asserts no crossing at all",
          "never reaches 50%" in v_none and "crosses 50% at K=" not in v_none)
    check("...and quotes the largest K actually SAMPLED, not the requested max",
          "max sampled K=1000" in v_none)
    check("absent oracle sweep degrades to 'not measured', not a number",
          oracle_verdict_clause(None, 1) == "ORACLE: not measured in this run ")
    check("default threshold still reads '50%', not '50.0%'",
          "crosses 50% at K=" in v99)
    r90 = crossing_k(_curve(FIXTURE_KVOID_SEED99), threshold=60.0)
    check("a non-default threshold is reflected in the prose, not hardcoded",
          "crosses 60% at K=" in oracle_verdict_clause(r90, 1)
          and r90["k50_threshold_pct"] == 60.0)

    print("D2 the artifact-emission wiring (previously untested):")
    rows = ([{"attack": "3_oracle_pairs", "depth": 1, "setting": f"K={k}",
              "K_effective": k, "top1_mean": t}
             for k, t in FIXTURE_KVOID_SEED123]
            + [{"attack": "3_oracle_pairs", "depth": 4, "setting": f"K={k}",
                "K_effective": k, "top1_mean": t}
               for k, t in FIXTURE_SEED123_DEPTH4]
            + [{"attack": "0_baseline_no_defense", "depth": 1,
                "setting": "plain_h", "top1_mean": 99.0}])
    ok50 = oracle_k50_from_rows(rows)
    check("keys are the depths present, as strings, oracle rows only",
          sorted(ok50) == ["1", "4"])
    check("depth 1 reproduces the seed-123 crossing",
          abs(ok50["1"]["k50_interpolated"] - 87.4) <= 0.1)
    check("depth 4 crosses far later, so the two are not interchangeable",
          ok50["4"]["k50_interpolated"] > 300)
    d, q = quoted_crossing(ok50)
    check("the verdict quotes the SHALLOWEST depth (strongest attack surface)",
          d == 1 and q is ok50["1"])
    check("...which matters: quoting depth 4 instead would publish "
          f"{ok50['4']['k50_interpolated']:.0f} rather than "
          f"{ok50['1']['k50_interpolated']:.0f}",
          ok50["4"]["k50_interpolated"] > 3 * ok50["1"]["k50_interpolated"])
    check("the emitted clause is built from the quoted depth",
          oracle_verdict_clause(q, d) == oracle_verdict_clause(ok50["1"], 1))
    check("K_effective is what indexes the curve, not the requested K",
          oracle_k50_from_rows(
              [{"attack": "3_oracle_pairs", "depth": 1, "setting": "K=5000",
                "K_effective": 40, "top1_mean": 40.0},
               {"attack": "3_oracle_pairs", "depth": 1, "setting": "K=9000",
                "K_effective": 60, "top1_mean": 60.0}])["1"]["k50_bracket"]
          == [40, 60])
    check("a row without K_effective falls back to parsing its setting",
          oracle_k50_from_rows(
              [{"attack": "3_oracle_pairs", "depth": 1, "setting": "K=40",
                "top1_mean": 40.0},
               {"attack": "3_oracle_pairs", "depth": 1, "setting": "K=60",
                "top1_mean": 60.0}])["1"]["k50_bracket"] == [40, 60])
    check("no oracle rows -> no field entries, and 'not measured' prose",
          oracle_k50_from_rows([]) == {}
          and quoted_crossing({}) == (None, None))
    check("the whole field JSON-round-trips with real nulls",
          json.loads(json.dumps({"oracle_k50": ok50}))["oracle_k50"] == ok50)

    print("D2/AC-negative removeprefix, not lstrip:")
    check("'K=1000' -> 1000", setting_to_k("K=1000") == 1000)
    check("'K=10' -> 10", setting_to_k("K=10") == 10)
    check("'KK=5' is NOT mangled to 5 (lstrip gives '5'; we give None)",
          "KK=5".lstrip("K=") == "5" and setting_to_k("KK=5") is None)
    check("non-oracle settings are rejected, not coerced",
          setting_to_k("plain_h") is None and setting_to_k("h@W") is None
          and setting_to_k("rows_leaked=50%") is None
          and setting_to_k("3_sessions") is None)

    print("D3 nested probe/decoder repetition accounting:")
    reps = [
        {"top1": [10.0, 12.0, 11.0]},
        {"top1": [20.0, 22.0, 21.0]},
        {"top1": [30.0, 32.0, 31.0]},
    ]
    ns = nested_repetition_stats(reps, "top1")
    check("grand mean uses every nested observation", ns["grand_mean"] == 21.0)
    check("between-probe SD uses probe means", ns["between_probe_std"] == 10.0)
    check("within-decoder SD is kept separate", ns["within_decoder_std"] == 1.0)
    check("repetition counts are explicit",
          ns["n_probe_repetitions"] == 3
          and ns["n_decoder_seeds_per_probe"] == 3)
    check("token hashes are ordered, stable, and selection-sensitive",
          token_id_hash([1, 2, 3]) == token_id_hash([1, 2, 3])
          and token_id_hash([1, 2, 3]) != token_id_hash([3, 2, 1]))

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def eval_with_decoder(decoder, feats, victim_tok, device):
    return evaluate_decoder(decoder, feats.float(), victim_tok, device)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.expanduser(
        "~/experiments/models/qwen3-0.6b"), help="HF model path (ignored with --toy)")
    ap.add_argument("--toy", action="store_true",
                    help="tiny random built-in model (CPU machinery check only; "
                         "depths clamp to the toy's 4 layers, K caps at vocab 128)")
    ap.add_argument("--corpus-file", default=None,
                    help="public attack-training text, one document per line; "
                         "the LAST --victim-docs documents are victim eval docs")
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--victim-docs", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--max-pairs", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--k-list", type=int, nargs="+", default=[10, 100, 1000, 5000],
                    help="oracle (chosen-plaintext) pair counts to sweep")
    ap.add_argument("--d3-repetitions", type=int, default=0,
                    help="independent depth-1 K=100 probe/solve repetitions; "
                         "0 disables D3, otherwise must be at least 3")
    ap.add_argument("--d3-probe-pool-size", type=int, default=1000,
                    help="number of frequent public tokens from which each "
                         "D3 K=100 probe set is sampled without replacement")
    ap.add_argument("--leak-fracs", type=float, nargs="+", default=[0.5, 0.9],
                    help="fractions of W's rows leaked (attack 4)")
    ap.add_argument("--sessions", type=int, default=3,
                    help="number of rotated-W sessions (attack 2)")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device",
                    default="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    ap.add_argument("--attn-impl", choices=["sdpa", "eager"], default="sdpa")
    ap.add_argument("--seed", type=int, default=42)
    add_numerics_args(ap)
    ap.add_argument("--quick", action="store_true",
                    help="depth 1, 1 seed, K in {10,100}, 2 sessions, 2 victim "
                         "docs, 5 epochs, seq 16")
    ap.add_argument("--output", default="e8_robustness.json")
    ap.add_argument("--self-test", action="store_true",
                    help="pure-python crossing-K logic checks; no torch needed")
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(self_test())

    if args.d3_repetitions not in (0,) and args.d3_repetitions < 3:
        ap.error("--d3-repetitions must be 0 or at least 3")
    if args.d3_repetitions and len(args.seeds) < 3:
        ap.error("D3 requires at least three --seeds for nested decoder repeats")
    if args.d3_repetitions and 1 not in args.depths:
        ap.error("D3 requires depth 1 in --depths")
    if args.d3_repetitions and args.d3_probe_pool_size < 100:
        ap.error("--d3-probe-pool-size must be at least 100")

    if torch is None or build_modules is None:
        ap.error("torch/transformers not installed; install them or run --help only")

    numerics = apply_numerics(args)
    print(f"[numerics] {numerics}")

    if args.quick:
        args.depths = [1]
        args.seeds = [0]
        args.k_list = [10, 100]
        args.sessions = 2
        args.victim_docs = 2
        args.epochs = 5
        args.seq_len = 16
        args.max_pairs = 2000

    seed_all(args.seed)
    _write_training_status(state="running", task="e8_robustness",
                           depths=args.depths, k_list=args.k_list,
                           started=time.strftime("%Y-%m-%dT%H:%M:%S%z"))

    embed, layers, norm, lm_head, rotary, encode = build_modules(args)
    n_layers = len(layers)
    vocab_size = lm_head.weight.shape[0] if not args.toy else embed.weight.shape[0]
    hidden_dim = embed.weight.shape[1]
    random_top1 = round(100.0 / vocab_size, 4)
    print(f"[model] {'toy' if args.toy else args.model}: {n_layers} layers, "
          f"hidden={hidden_dim}, vocab={vocab_size}, device={args.device}")

    # victim docs from the END of the corpus
    # --corpus-file REPLACES TEXT_SAMPLES (no mixing, so results are
    # attributable to one source).
    if args.corpus_file:
        corpus_source = "corpus_file"
        with open(args.corpus_file) as f:
            docs = [l.strip() for l in f if len(l.strip()) > 500]  # real docs only — wikitext short lines are formatting artifacts
    else:
        corpus_source = "TEXT_SAMPLES"
        docs = list(TEXT_SAMPLES)
    if len(docs) < args.victim_docs + 4:
        raise ValueError(f"corpus too small: {len(docs)} docs")
    victim_docs = docs[-args.victim_docs:]
    attack_docs = docs[:-args.victim_docs]
    provenance = make_provenance(
        args.corpus_file, corpus_source, len(docs),
        range(len(docs) - args.victim_docs, len(docs)), model_path=getattr(args, 'model', None))
    n_val = max(1, int(round(args.val_frac * len(attack_docs))))
    val_docs, train_docs_pool = attack_docs[:n_val], attack_docs[n_val:]
    print(f"[data] {len(train_docs_pool)} attack-train, {n_val} attack-val, "
          f"{len(victim_docs)} victim (held out from end)")

    victim_ids, victim_tokens = [], []
    for doc in victim_docs:
        b = encode([doc], args.seq_len)
        if b:
            victim_ids.append(b[0])
            victim_tokens.append(b[0][:-1])
    if not victim_ids:
        raise ValueError("no victim doc long enough to yield a block")
    victim_tok = torch.cat(victim_tokens)

    # oracle token pool: top max(K) frequent tokens in the PUBLIC pool
    k_max = max(max(args.k_list),
                args.d3_probe_pool_size if args.d3_repetitions else 0)
    pool_tokens = frequent_tokens(encode, train_docs_pool, args, k_max)
    print(f"[oracle] token pool: {len(pool_tokens)} distinct public tokens "
          f"(need {k_max})")
    # max(k_list) SIZES this pool, so it is a result-determining input, not a
    # reporting grid: runs with different max(k_list) drew 5000/1000/100-token
    # pools and are not comparable. Hashing the pool makes that detectable
    # instead of inferable.
    pool_sha256 = tensor_sha256(torch.tensor(pool_tokens, dtype=torch.long))
    print(f"[oracle] pool sha256: {pool_sha256}")

    results, summary, d3_oracle_repeatability = [], [], []
    h0_sha256 = {}

    def record(attack, depth, setting, tops1, tops5, extra=None):
        m1, s1 = mean_std(tops1)
        m5, s5 = mean_std(tops5)
        entry = {"attack": attack, "depth": depth, "setting": setting,
                 "top1_mean": m1, "top1_std": s1, "top5_mean": m5,
                 "top5_std": s5, "n_seeds": len(tops1)}
        if extra:
            entry.update(extra)
        summary.append(entry)
        print(f"[eval] depth={depth} {attack}/{setting}: "
              f"top-1={m1:.2f}+-{s1:.2f}% top-5={m5:.2f}+-{s5:.2f}%")

    for depth in args.depths:
        head, middle, tail, sa, ra = split_at(layers, depth, n_layers)
        if sa != depth:
            print(f"[split] depth {depth} clamped to sa={sa} ({n_layers} layers)")
        W = make_secret(hidden_dim, args.seed)  # fp32 CPU; the defense secret

        # base pairs + victim boundary (plain h; attacker sees h@W)
        t0 = time.time()
        tr_h, _, tr_tok = collect_base_pairs(
            embed, head, middle, tail, norm, lm_head, rotary, encode,
            train_docs_pool, args, with_grad=False)
        va_h, _, va_tok = collect_base_pairs(
            embed, head, middle, tail, norm, lm_head, rotary, encode,
            val_docs, args, with_grad=False)
        h_star = boundary_acts(embed, head, rotary, victim_ids, args)
        h_prime = h_star @ W
        print(f"[collect] depth={sa}: train={tr_h.shape[0]} val={va_h.shape[0]} "
              f"victim-positions={h_star.shape[0]} ({time.time() - t0:.1f}s)")

        decoders = {}
        for seed in args.seeds:
            seed_all(args.seed + seed)
            decoders[seed] = train_decoder(
                tr_h, tr_tok, va_h, va_tok, tr_h.shape[1], vocab_size, args,
                f"e8r_d{sa}_seed{seed}")
        _write_training_status(state="running", phase="attacks", depth=sa)

        # sanity: undefended baseline (decoder on plain h*)
        record("0_baseline_no_defense", sa, "plain_h",
               *zip(*[eval_with_decoder(decoders[s], h_star, victim_tok,
                                        args.device) for s in args.seeds]))
        # and the E8 headline: decoder on h' (single static W)
        record("0_obfuscated_static_W", sa, "h@W",
               *zip(*[eval_with_decoder(decoders[s], h_prime, victim_tok,
                                        args.device) for s in args.seeds]))

        # Attack 1a: whiten + nearest public token mean (label-free)
        top1_1a, coverage = attack_1a_whiten_nn(h_prime, tr_h, tr_tok,
                                                victim_tok, vocab_size)
        record("1a_whiten_nn", sa, "label-free",
               [top1_1a], [top1_1a],
               extra={"ref_token_coverage_pct": coverage,
                      "note": "seed-independent; top5==top1 (NN decode)"})

        # Attack 1b: second-moment alignment + trained decoder
        mu_p, mu_r, q = attack_1b_moment_align(h_prime, tr_h)
        h_aligned = ((h_prime.double() - mu_p) @ q + mu_r).float()
        record("1b_moment_align", sa, "label-free",
               *zip(*[eval_with_decoder(decoders[s], h_aligned, victim_tok,
                                        args.device) for s in args.seeds]))

        # Attack 2: session rotation, pooled label-free alignment
        per_session_t1 = []
        aligned_sessions = []
        sess_docs = [victim_ids[i::args.sessions] for i in range(args.sessions)]
        sess_toks = [victim_tokens[i::args.sessions] for i in range(args.sessions)]
        for si in range(args.sessions):
            W_s = make_secret(hidden_dim, args.seed + 1000 + si)
            h_s = boundary_acts(embed, head, rotary, sess_docs[si], args)
            hp_s = h_s @ W_s
            tok_s = torch.cat(sess_toks[si])
            # per-session label-free alignment + decoder (first seed)
            mu_s, _, q_s = attack_1b_moment_align(hp_s, tr_h)
            ha_s = ((hp_s.double() - mu_s) @ q_s + mu_r).float()
            t1, _ = eval_with_decoder(decoders[args.seeds[0]], ha_s, tok_s,
                                      args.device)
            per_session_t1.append(t1)
            aligned_sessions.append(hp_s)  # raw h' pooling below
        # pooled: concat raw h' across sessions, ONE alignment+decode
        hp_pool = torch.cat(aligned_sessions)
        tok_pool = torch.cat([torch.cat(sess_toks[si])
                              for si in range(args.sessions)])
        mu_pool, _, q_pool = attack_1b_moment_align(hp_pool, tr_h)
        ha_pool = ((hp_pool.double() - mu_pool) @ q_pool + mu_r).float()
        pool_t1, pool_t5 = eval_with_decoder(decoders[args.seeds[0]], ha_pool,
                                             tok_pool, args.device)
        record("2_session_rotation", sa, f"{args.sessions}_sessions",
               [pool_t1], [pool_t5],
               extra={"per_session_top1": per_session_t1,
                      "marginal_info_per_session": (
                          "none: pooled ~= per-session ~= random"
                          if pool_t1 < 5 * random_top1 else
                          f"pooled {pool_t1}% vs per-session {per_session_t1}")})

        # Attack 3: oracle K pairs (chosen plaintext), W by lstsq
        primary_pool_tokens = pool_tokens[:max(args.k_list)]
        h0 = position0_acts(embed, head, rotary, primary_pool_tokens, args)
        # h0 is the only GPU-computed input to the oracle attack (h0_prime,
        # lstsq and polar are all CPU/fp64 below), so it is where forward-pass
        # nondeterminism could enter and re-randomize the rank-deficient
        # null-space completion. Hashing it makes "was the input identical?"
        # answerable across runs instead of assumed.
        h0_sha256[str(sa)] = tensor_sha256(h0)
        print(f"[oracle] depth={sa} h0 sha256: {h0_sha256[str(sa)]}")
        h0_prime = h0 @ W
        for K in args.k_list:
            K_eff = min(K, len(primary_pool_tokens))
            # W_hat = argmin ||H W - H'||_F  (exact pairs, context-free)
            sol = torch.linalg.lstsq(h0[:K_eff].double(),
                                     h0_prime[:K_eff].double())
            w_hat = polar(sol.solution)  # nearest orthogonal matrix
            w_err = ((w_hat - W.double()).norm()
                     / W.double().norm()).item()
            h_rec = (h_prime.double() @ w_hat.T).float()
            t1s, t5s = zip(*[eval_with_decoder(decoders[s], h_rec, victim_tok,
                                               args.device)
                             for s in args.seeds])
            record("3_oracle_pairs", sa, f"K={K}", list(t1s), list(t5s),
                   extra={"K_effective": K_eff,
                          "W_recovery_rel_err": round(w_err, 4)})

        # D3: independent probe selection + solve repetitions. Additive: the
        # primary top-frequency K curve keeps its convention; D3 estimates the
        # variation that convention hides and reports it separately.
        if args.d3_repetitions and sa == 1:
            candidate_tokens = pool_tokens[:args.d3_probe_pool_size]
            if len(candidate_tokens) < 100:
                raise ValueError("D3 needs at least 100 distinct public tokens")
            probe_reps = []
            for ri in range(args.d3_repetitions):
                probe_seed = args.seed + 10000 + ri
                g = torch.Generator().manual_seed(probe_seed)
                selected_idx = torch.randperm(len(candidate_tokens), generator=g)[:100]
                selected = [candidate_tokens[int(i)] for i in selected_idx]
                probe_h = position0_acts(embed, head, rotary, selected, args)
                probe_hp = probe_h @ W
                sol = torch.linalg.lstsq(probe_h.double(), probe_hp.double())
                probe_w_hat = polar(sol.solution)
                probe_w_err = ((probe_w_hat - W.double()).norm()
                               / W.double().norm()).item()
                probe_rec = (h_prime.double() @ probe_w_hat.T).float()
                scores = [eval_with_decoder(decoders[s], probe_rec, victim_tok,
                                            args.device) for s in args.seeds]
                rep = {
                    "probe_repetition": ri,
                    "probe_seed": probe_seed,
                    "selected_token_ids_sha256": token_id_hash(selected),
                    "K_effective": len(selected),
                    "W_recovery_rel_err": round(probe_w_err, 6),
                    "decoder_seeds": list(args.seeds),
                    "top1": [round(x[0], 4) for x in scores],
                    "top5": [round(x[1], 4) for x in scores],
                }
                probe_reps.append(rep)
                print(f"[d3] repeat={ri} probe_seed={probe_seed} "
                      f"token_hash={rep['selected_token_ids_sha256'][:12]} "
                      f"W_err={probe_w_err:.4f}")
            d3_entry = {
                "attack": "3_oracle_pairs_independent_probes",
                "depth": sa,
                "setting": "K=100",
                "probe_selection": "uniform_without_replacement_from_frequent_pool",
                "probe_pool_size_requested": args.d3_probe_pool_size,
                "probe_pool_size_effective": len(candidate_tokens),
                "probe_repetitions": probe_reps,
                "top1_variance_decomposition": nested_repetition_stats(
                    probe_reps, "top1"),
                "top5_variance_decomposition": nested_repetition_stats(
                    probe_reps, "top5"),
            }
            d3_oracle_repeatability.append(d3_entry)

        # Attack 4: partial row leak
        for frac in args.leak_fracs:
            n_leak = int(round(frac * hidden_dim))
            g = torch.Generator().manual_seed(args.seed + 7)
            leaked = torch.randperm(hidden_dim, generator=g)[:n_leak]
            mask = torch.zeros(hidden_dim, dtype=torch.bool)
            mask[leaked] = True
            # W_tilde: true rows where leaked; orthonormal completion elsewhere
            w_tilde = torch.empty(hidden_dim, hidden_dim, dtype=torch.double)
            w_tilde[mask] = W.double()[mask]
            comp, _ = torch.linalg.qr(torch.randn(hidden_dim, hidden_dim,
                                                  generator=g).double())
            # Gram-Schmidt the random rows against the leaked rows
            basis = []
            for i in range(hidden_dim):
                v = comp[:, i].clone()
                for b in basis:
                    v -= (v @ b) * b
                v -= W.double()[mask].T @ (W.double()[mask] @ v)
                n = v.norm()
                if n > 1e-6:
                    basis.append(v / n)
                if len(basis) == hidden_dim - n_leak:
                    break
            comp_basis = torch.stack(basis, dim=0)
            w_tilde[~mask] = comp_basis
            h_rec = (h_prime.double() @ w_tilde.T).float()
            t1s, t5s = zip(*[eval_with_decoder(decoders[s], h_rec, victim_tok,
                                               args.device)
                             for s in args.seeds])
            record("4_partial_leak", sa, f"rows_leaked={frac:.0%}",
                   list(t1s), list(t5s))

    # verdict
    def get(attack, setting_prefix, depth=None):
        for e in summary:
            if e["attack"] == attack and e["setting"].startswith(setting_prefix):
                if depth is None or e["depth"] == depth:
                    return e
        return None

    labelfree_max = max(e["top1_mean"] for e in summary
                        if e["attack"].startswith(("1a", "1b", "2_")))
    # Per depth, off the recorded summary rows; both steps are pure functions
    # so --self-test covers them.
    oracle_k50 = oracle_k50_from_rows(summary)
    for d, c in oracle_k50.items():
        print(f"[oracle] depth={d} K50: {c}")
    quoted_depth, quoted = quoted_crossing(oracle_k50)
    oracle = oracle_verdict_clause(quoted, quoted_depth)
    leak90 = get("4_partial_leak", f"rows_leaked={args.leak_fracs[-1]:.0%}")
    verdict = (f"PASSIVE: label-free attacks (whiten-NN, moment alignment, "
               f"session pooling) recover at most {labelfree_max:.2f}% top-1 "
               f"(random {random_top1}%) -> no label-free route past W. "
               + oracle +
               f"(H={hidden_dim}) -> chosen-plaintext queries are the real "
               f"security parameter; rotate W per session to cap an oracle's "
               f"accumulation. PARTIAL LEAK: {args.leak_fracs[-1]:.0%} of rows "
               f"leaked -> {leak90['top1_mean'] if leak90 else '?'}% top-1.")

    out = {
        # measurement_kind is part of the evidence contract; without it an
        # artifact cannot be classified measured-vs-simulated automatically.
        "config": {"measurement_kind": "measured",
                   "model": "toy" if args.toy else args.model,
                   "n_layers": n_layers, "depths": args.depths,
                   "seeds": args.seeds, "victim_docs": args.victim_docs,
                   "seq_len": args.seq_len, "max_pairs": args.max_pairs,
                   "epochs": args.epochs, "k_list": args.k_list,
                   "d3_repetitions": args.d3_repetitions,
                   "d3_probe_pool_size": args.d3_probe_pool_size,
                   "leak_fracs": args.leak_fracs, "sessions": args.sessions,
                   "W_seed": args.seed, "dtype": args.dtype,
                   "device": args.device, "quick": args.quick,
                   "val_frac": args.val_frac, "batch_size": args.batch_size,
                   "attn_impl": args.attn_impl,
                   "matmul_precision_requested": args.matmul_precision,
                   **numerics},
        "threat_model": "attacks 1/2/4: PASSIVE observer of h'=h@W with public "
                        "text+model but no victim labels. attack 3: ORACLE / "
                        "chosen-plaintext (K exact position-0 pairs) — strictly "
                        "stronger than the E8 default; the K-curve is the "
                        "honest security parameter, not a claim about the "
                        "default model.",
        "evidence_status": "supporting",
        "known_limitations": [
            "attack 3 is chosen-plaintext: its K-curve is a stronger-attacker "
            "bound, not a measurement of the passive threat model attacks "
            "1/2/4 assume.",
            "oracle_k50 is interpolated between grid points in k_list; it is "
            "not a measured crossing and moves with the grid's resolution.",
            "d3_oracle_repeatability is empty unless --d3-repetitions is "
            "passed, so for a default run the reported top1_std covers decoder "
            "seeds only and not probe selection.",
        ],
        "random_baseline_top1_pct": random_top1,
        "provenance": {**provenance, "oracle_pool_sha256": pool_sha256,
                       "oracle_h0_sha256": h0_sha256},
        # The interpolated oracle crossing-K per depth; the verdict prose
        # reports a grid point. Keyed by the (possibly clamped) split depth.
        "oracle_k50": oracle_k50,
        # Additive D3 field. Empty for default runs; populated only
        # when --d3-repetitions is explicitly requested.
        "d3_oracle_repeatability": d3_oracle_repeatability,
        "verdict": verdict,
        "summary": summary,
        "results": results,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    _write_training_status(state="done", result_file=args.output)
    print(f"\nVerdict: {verdict}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
