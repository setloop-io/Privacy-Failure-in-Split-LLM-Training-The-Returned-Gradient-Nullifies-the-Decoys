#!/usr/bin/env python3
"""Pure-torch FastICA (label-free blind source separation).

The repo venvs do not ship sklearn, so the framework carries its own
FastICA: symmetric decorrelation with the tanh negentropy contrast
(Hyvarinen fixed-point), after PCA whitening.

Used by the ica-bss attacker: within-epoch wire captures h' = h @ W_t are
a rotated (orthogonal = independent-component-preserving) mixture of the
boundary activation coordinates; ICA asks whether the unidentifiability
that defeats second-moment attacks (e8_robustness attack 1b) survives a
HIGHER-ORDER algorithm.

Guarded heavy import: --help / --self-test work torch-less.
"""

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def whiten(x, n_components=None, eps=1e-10):
    """PCA whitening. x: [n, d] rows of observations.
    Returns (z [n, k], whiten_mat [d, k], dewhiten [k, d], mean [d])."""
    mu = x.mean(0)
    xc = (x - mu).double()
    cov = (xc.T @ xc) / max(1, xc.shape[0] - 1)
    evals, evecs = torch.linalg.eigh(cov)
    order = torch.argsort(evals, descending=True)
    k = n_components or x.shape[1]
    evals, evecs = evals[order][:k], evecs[:, order][:, :k]
    scale = (evals + eps).rsqrt()
    wmat = evecs * scale            # [d, k]
    z = xc @ wmat
    return z, wmat, (evecs * (evals + eps).sqrt()).T, mu


def fastica(x, n_components=None, max_iter=200, tol=1e-6, seed=0):
    """Symmetric FastICA with tanh (logcosh) negentropy contrast.

    x: [n, d] observation rows (e.g. wire activations h').
    Returns {"sources": [n, k], "unmixing": [k, d] applied to CENTERED x,
             "mean": [d], "n_iter": int, "converged": bool}.
    Never raises on non-convergence: converged=False is reported so the
    caller journals the cell (rotation_lifetime-style hardening)."""
    z, wmat, _, mu = whiten(x, n_components)
    n, k = z.shape
    g = torch.Generator().manual_seed(seed)
    B = torch.linalg.qr(torch.randn(k, k, generator=g,
                                    dtype=torch.float64))[0]
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        Y = z @ B                       # [n, k]
        G = torch.tanh(Y)
        Gp = 1.0 - G * G
        B_new = (z.T @ G) / n - B * Gp.mean(0)
        # symmetric orthogonalization
        q, r = torch.linalg.qr(B_new)
        B_new = q * torch.sign(torch.diagonal(r))
        lim = ((B_new * B).sum(0).abs() - 1).abs().max().item()
        B = B_new
        if lim < tol:
            converged = True
            break
    sources = z @ B
    unmixing = wmat @ B                 # [d, k]: s = (x - mu) @ unmixing
    return {"sources": sources, "unmixing": unmixing, "mean": mu,
            "n_iter": it, "converged": converged}


def match_components(s_hat, s_true):
    """Greedy |corr| matching of recovered to true components.
    Returns (mean_abs_corr_of_matched, matched_pairs [(i,j,corr),...]).
    ICA sign/permutation ambiguity is absorbed by the absolute value."""
    a = (s_hat - s_hat.mean(0)).double()
    b = (s_true - s_true.mean(0)).double()
    a = a / a.norm(dim=0).clamp_min(1e-12)
    b = b / b.norm(dim=0).clamp_min(1e-12)
    c = (a.T @ b).abs()                 # [k_hat, k_true]
    used_j, pairs = set(), []
    order = torch.argsort(c.flatten(), descending=True)
    for flat in order.tolist():
        i, j = divmod(flat, c.shape[1])
        if i in (p[0] for p in pairs) or j in used_j:
            continue
        used_j.add(j)
        pairs.append((i, j, round(c[i, j].item(), 6)))
        if len(pairs) == min(c.shape):
            break
    mean_c = sum(p[2] for p in pairs) / max(1, len(pairs))
    return round(mean_c, 6), pairs


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    if torch is None:
        print("FastICA checks SKIPPED (torch not installed)")
        print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
        return 0 if ok else 1

    print("FastICA on a frozen synthetic mixture (torch present):")
    g = torch.Generator().manual_seed(1234)
    n, k = 4000, 3
    # non-Gaussian sources (ICA needs higher-order signal): laplace-ish,
    # uniform, bimodal
    s = torch.cat([
        (torch.rand(n, 1, generator=g) - torch.rand(n, 1, generator=g)),
        torch.rand(n, 1, generator=g) - 0.5,
        torch.sign(torch.randn(n, 1, generator=g)) * torch.rand(n, 1, generator=g),
    ], dim=1).double()
    mix = torch.linalg.qr(torch.randn(k, k, generator=g,
                                      dtype=torch.float64))[0]
    x = s @ mix.T
    out = fastica(x, n_components=k, seed=7)
    check("converges on a clean mixture", out["converged"])
    mean_c, pairs = match_components(out["sources"], s)
    check(f"mean |corr| of matched components >= 0.95 (got {mean_c})",
          mean_c >= 0.95)
    out2 = fastica(x, n_components=k, seed=7)
    check("deterministic for a fixed seed",
          torch.allclose(out["sources"], out2["sources"]))
    out3 = fastica(x, n_components=k, seed=8)
    check("a different seed recovers the same subspace (|corr| >= 0.9)",
          match_components(out3["sources"], s)[0] >= 0.9)

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
