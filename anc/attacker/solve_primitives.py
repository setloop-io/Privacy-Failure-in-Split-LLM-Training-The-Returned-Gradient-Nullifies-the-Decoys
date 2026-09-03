#!/usr/bin/env python3
"""Shared solve primitives for every attack in the framework.

One module, one place for the attack math so the per-attack modules cannot
drift apart. Torch section (hardened core of split-training/
rotation_lifetime.py, byte-for-byte semantics):
  * solve_w — polar(lstsq) orthogonal solve with contiguous-fp64
    discipline; polar factor falls back from SVD (gesdd) to eigh of
    sol^T sol on non-convergence. Never raises: failures return
    (None, "error: <msg>") so the caller journals the cell and continues.
  * recovery_with_what, h_wire, w_rel_err — decode-side helpers.
Pure-python section (torch-free, pinned by --self-test fixtures):
  ratchet_seed, stagger_rotating_blocks, partition_pairs, crossing_k.
upstream() exposes the split-training/ originals for parity cross-checks
when importable; the framework does NOT require that path.

Guarded heavy import: --help/--self-test must work torch-less (repo rule).
"""

import hashlib
import math
import os
import sys

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

K50_THRESHOLD_PCT = 50.0
RANDOM_REL_ERR = math.sqrt(2.0)  # E||A-B||_F/||B||_F for independent orthogonals


# Pure-python helpers (torch-free; frozen fixtures in self_test()).
def ratchet_seed(master_seed, epoch):
    """One-way per-epoch key derivation: sha256(f"{master}:{epoch}")[:8]
    little-endian — byte-identical to rotation_lifetime.ratchet_seed and
    er_ratchet.epoch_seed so cross-framework comparisons stay valid."""
    return int.from_bytes(hashlib.sha256(
        f"{master_seed}:{epoch}".encode()).digest()[:8], "little")


def stagger_rotating_blocks(epoch, n_blocks, period):
    """Blocks rotating at `epoch` under the staggered schedule: block i
    rotates when (epoch + i*(period//n_blocks)) % period == 0."""
    step = period // n_blocks
    return [i for i in range(n_blocks) if (epoch + i * step) % period == 0]


def partition_pairs(n_pairs, n_epochs):
    """Split n_pairs as evenly as possible across n_epochs (first r get +1)."""
    base, rem = divmod(n_pairs, n_epochs)
    return [base + (1 if e < rem else 0) for e in range(n_epochs)]


def crossing_k(curve, threshold=K50_THRESHOLD_PCT):
    """Linear-in-K interpolation of where a (K, top1_mean) curve crosses
    `threshold` (e8_robustness.crossing_k semantics, minimal form).

    curve: iterable of {"K": int, "top1_mean": float} sorted by K.
    Returns {"k50_interpolated": float|None, "k50_bracket": [Klo, Khi]|None,
             "k50_method": "linear_in_K"|"not_bracketed"}."""
    pts = sorted(curve, key=lambda r: r["K"])
    for lo, hi in zip(pts, pts[1:]):
        y0, y1 = lo["top1_mean"], hi["top1_mean"]
        if (y0 - threshold) * (y1 - threshold) <= 0 and y0 != y1:
            k = lo["K"] + (threshold - y0) / (y1 - y0) * (hi["K"] - lo["K"])
            return {"k50_interpolated": k,
                    "k50_bracket": [lo["K"], hi["K"]],
                    "k50_method": "linear_in_K"}
    return {"k50_interpolated": None, "k50_bracket": None,
            "k50_method": "not_bracketed"}


# Frozen fixtures (identical values to rotation_lifetime.py's, by design).
FIXTURE_RATCHET_12345 = [10901005920735059415, 12207851219204689068,
                         14158799805508211343, 5173715490049009419]
FIXTURE_STAGGER_S4_P8 = {0: [0], 1: [], 2: [3], 3: [], 4: [2], 5: [],
                         6: [1], 7: [], 8: [0], 9: [], 10: [3], 11: [],
                         12: [2], 13: [], 14: [1], 15: []}
FIXTURE_CROSSING_CURVE = [(64, 12.0), (256, 38.0), (512, 61.0), (4096, 66.0)]


# Torch section.
def _polar_eigh(m, eps=1e-12):
    """Polar factor via eigendecomposition of m^T m: polar = m (m^T m)^{-1/2}.
    Fallback for LAPACK gesdd non-convergence in torch.linalg.svd; eigh uses
    syevd, an independent LAPACK code path. Tiny eigenvalues are floored."""
    evals, evecs = torch.linalg.eigh(m.double().T @ m.double())
    evals = evals.clamp_min(eps)
    return m.double() @ (evecs * evals.rsqrt()) @ evecs.T


def polar(m):
    """Nearest orthogonal factor of m (SVD). Raises RuntimeError on
    non-convergence — solve_w catches and falls back to _polar_eigh."""
    u, _, vh = torch.linalg.svd(m.double(), full_matrices=False)
    return u @ vh


def solve_w(h_pairs, hw_pairs):
    """W_hat = polar factor of the least-squares solution (e8_robustness
    attack 3), returned as (w_hat, solver_tag).

    NEVER raises: a linalg failure returns (None, "error: <msg>") so the
    caller journals the cell as failed and continues to the next one. One
    ill-conditioned cell must not kill a multi-hour run.

    Contiguous-fp64 discipline: BOTH operands are
    materialized as contiguous fp64 copies before the solve — .double() is
    a NO-OP on an already-fp64 input, so a strided column slice would hand
    LAPACK a non-contiguous view into a large live allocation (the GB10
    memory-stomp class of failure)."""
    try:
        sol = torch.linalg.lstsq(h_pairs.double().contiguous(),
                                 hw_pairs.double().contiguous()).solution
        try:
            return polar(sol), "lstsq+svd"
        except RuntimeError:
            print("[solve_w] polar SVD failed; retrying polar via eigh "
                  "of sol^T sol")
            return _polar_eigh(sol), "lstsq+eigh_polar_fallback"
    except RuntimeError as e:
        return None, "error: " + str(e).splitlines()[0]


def h_wire(h, w):
    """The wire tensor: h @ W computed in fp64 (mirrors the fp32 fp-exact
    seam of the deployed defense closely enough for rel-err purposes)."""
    return h.double() @ w.double()


def w_rel_err(a, b):
    """Relative Frobenius error ||a-b||/||b||; ~= sqrt(2) for unrelated Ws."""
    return round(((a - b).norm() / b.norm()).item(), 6)


def recovery_with_what(evaluate_decoder, decoder, h_wire_t, w_hat,
                       victim_tok, device):
    """Decode wire features de-obfuscated with W_hat; top-1 %.
    evaluate_decoder is injected (trained_inversion.evaluate_decoder) so
    this module never imports the training stack itself."""
    h_rec = (h_wire_t.double() @ w_hat.double().T).float()
    return evaluate_decoder(decoder, h_rec, victim_tok, device)[0]


def nearest_neighbor_top1(h_rec, ref_h, ref_tok, victim_tok, chunk=4096):
    """Label-free decode used by several attacks: nearest reference
    activation per row (L2), accuracy against victim_tok, in %.
    Torch-only. Self-contained so label-free attacks do not need a trained
    decoder."""
    correct = 0
    total = h_rec.shape[0]
    ref = ref_h.float()
    ref_norm2 = (ref * ref).sum(1)
    for i in range(0, total, chunk):
        x = h_rec[i:i + chunk].float()
        d2 = (x * x).sum(1, keepdim=True) - 2 * x @ ref.T + ref_norm2
        nn = d2.argmin(1)
        correct += int((ref_tok[nn] == victim_tok[i:i + nn.shape[0]]).sum())
    return round(100.0 * correct / max(1, total), 4)


def ratchet_secret(hidden_dim, master_seed, epoch, make_secret=None):
    """Per-epoch W from the ratchet chain (fp32 CPU). make_secret is the
    e8_obfuscation.make_secret construction; when not injected, a local
    QR-sign equivalent is used so the framework stays self-contained."""
    if make_secret is not None:
        return make_secret(hidden_dim, ratchet_seed(master_seed, epoch))
    g = torch.Generator().manual_seed(ratchet_seed(master_seed, epoch))
    a = torch.randn(hidden_dim, hidden_dim, generator=g, dtype=torch.float32)
    q, r = torch.linalg.qr(a)
    return q * torch.sign(torch.diagonal(r))


def block_diag_secret(hidden_dim, n_blocks, seed):
    """Block-diagonal orthogonal W: independent seeded QR per diagonal block."""
    assert hidden_dim % n_blocks == 0
    hb = hidden_dim // n_blocks
    w = torch.zeros(hidden_dim, hidden_dim)
    for i in range(n_blocks):
        w[i * hb:(i + 1) * hb, i * hb:(i + 1) * hb] = ratchet_secret(
            hb, seed, i)
    return w


def upstream():
    """Best-effort import of the original functions from split-training/
    for parity cross-checks. Returns a dict of what was found ({} when the
    path is unavailable). The framework NEVER requires this to succeed."""
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, "..", "split-training")
    out = {}
    if os.path.isdir(cand) and cand not in sys.path:
        sys.path.insert(0, cand)
    try:
        import rotation_lifetime as rl
        out["rotation_lifetime.solve_w"] = rl.solve_w
        out["rotation_lifetime.ratchet_seed"] = rl.ratchet_seed
        out["rotation_lifetime.partition_pairs"] = rl.partition_pairs
    except Exception:
        pass
    try:
        import e8_robustness as e8
        out["e8_robustness.polar"] = e8.polar
        out["e8_robustness.crossing_k"] = e8.crossing_k
    except Exception:
        pass
    return out


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    print("solve_primitives: ratchet derivation pins the frozen sha256 chain:")
    got = [ratchet_seed(12345, t) for t in range(4)]
    check(f"epochs 0..3 == {FIXTURE_RATCHET_12345}", got == FIXTURE_RATCHET_12345)
    check("different master -> different chain",
          ratchet_seed(12346, 0) != got[0])

    print("stagger schedule (s=4, period=8):")
    got = {e: stagger_rotating_blocks(e, 4, 8) for e in range(16)}
    check("full 16-epoch map matches the frozen fixture",
          got == FIXTURE_STAGGER_S4_P8)
    check("no 2-epoch window sees all 4 blocks rotate",
          all(len(set(got[e]) | set(got[e + 1])) < 4 for e in range(15)))

    print("partition arithmetic:")
    check("250 over 4 = [63,63,62,62]",
          partition_pairs(250, 4) == [63, 63, 62, 62])
    check("partition always sums to the request",
          sum(partition_pairs(4096, 4)) == 4096)

    print("crossing interpolation on the frozen synthetic curve:")
    curve = [{"K": k, "top1_mean": t} for k, t in FIXTURE_CROSSING_CURVE]
    cross = crossing_k(curve, threshold=K50_THRESHOLD_PCT)
    check("K50 interpolates linearly between 256 and 512",
          cross["k50_method"] == "linear_in_K"
          and cross["k50_bracket"] == [256, 512])
    check("K50 = 389.5652 +- 0.01",
          cross["k50_interpolated"] is not None
          and abs(cross["k50_interpolated"] - 389.5652) <= 0.01)
    flat = crossing_k([{"K": 64, "top1_mean": 3.0},
                       {"K": 4096, "top1_mean": 3.0}])
    check("a curve in the label-free band never brackets 50%",
          flat["k50_method"] == "not_bracketed")

    if torch is not None:
        print("solve hardening / polar fallback (torch present):")
        g = torch.Generator().manual_seed(0)
        w_true = torch.linalg.qr(torch.randn(16, 16, generator=g))[0]
        h = torch.randn(64, 16, generator=g)
        h[1] = h[0] * (1 + 1e-14)
        w_hat, solver = solve_w(h, h @ w_true)
        check("ill-conditioned solve returns a result, not a raise",
              w_hat is not None)
        check("solver tag is recorded", isinstance(solver, str))
        if w_hat is not None:
            ident = w_hat.double().T @ w_hat.double()
            check("W_hat stays ~orthogonal",
                  bool((ident - torch.eye(16, dtype=torch.float64))
                       .abs().max() < 1e-6))
        q = _polar_eigh(torch.randn(16, 16, generator=g).double())
        check("eigh polar fallback yields an orthogonal factor",
              bool((q.T @ q - torch.eye(16, dtype=torch.float64))
                   .abs().max() < 1e-8))
        w_bad, info = solve_w(torch.randn(8, 4), torch.randn(9, 4))
        check("a failing solve returns (None, 'error: ...'), never raises",
              w_bad is None and isinstance(info, str)
              and info.startswith("error:"))
        w1 = ratchet_secret(16, 12345, 0)
        check("ratchet_secret is orthogonal",
              bool(((w1.double() @ w1.double().T)
                    - torch.eye(16, dtype=torch.float64)).abs().max() < 1e-4))
        check("block_diag_secret is orthogonal",
              bool(((lambda b: b @ b.T)(block_diag_secret(16, 4, 7).double())
                    - torch.eye(16, dtype=torch.float64)).abs().max() < 1e-4))
        up = upstream()
        if "rotation_lifetime.ratchet_seed" in up:
            check("parity with rotation_lifetime.ratchet_seed",
                  all(up["rotation_lifetime.ratchet_seed"](12345, t) == v
                      for t, v in enumerate(FIXTURE_RATCHET_12345)))
    else:
        print("torch section SKIPPED (torch not installed)")

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
