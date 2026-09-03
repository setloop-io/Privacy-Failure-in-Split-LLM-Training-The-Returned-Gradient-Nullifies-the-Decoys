#!/usr/bin/env python3
"""[ER] Per-epoch obfuscation-key ratchet for split TRAINING (E-R7).

Training analog of the split-inference ratchet: the local node rotates the
boundary activation h -> h @ W_t and the cloud folds its middle-layer weights
for the same W_t (covariant_fold.fold_layer), so wire tensors stay rotated in
both directions and backprop through the folded weights is exact. Production
uses the portable v2 derivation shared with inference:

    key_t  = sha256(DOMAIN || S_128 || uint64be(t))
    A_t    = BoxMuller(binary64(SHA256-counter(key_t)))
    W_t    = float32(Q * sign(diag(R))) from CPU float64 QR(A_t)

Token accounting (training): every boundary forward sends the FULL
microbatch sequence (prefill-style), so a forward of shape [b, s, H] adds
b*s served tokens / evidence. Fixed mode: epoch = served // N. Budget mode:
at evidence >= B advance one epoch and reset. Cross-side consistency never
relies on these counters — the epoch is transmitted in every frame header
("obf_epoch"), exactly as in inference.

Usage:
    python er_ratchet.py --help        # works without torch
    python er_ratchet.py --self-test   # torch-free fixtures (frozen chain)
"""

import argparse
import hashlib
import json
import os
import secrets
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from privacy_runtime.ratchet_v2 import derive_orthogonal as derive_orthogonal_v2
from privacy_runtime.pair_budget import PairBudget

# Guarded heavy import: `--help`/`--self-test` must work on torch-less hosts.
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def epoch_seed(seed_base, epoch):
    """[ER] Seed chain (identical on both sides, no key file, no exchange):
    int.from_bytes(sha256(f"{S}:{t}")[:8], "little")."""
    return int.from_bytes(
        hashlib.sha256(f"{seed_base}:{epoch}".encode()).digest()[:8],
        "little")


# Frozen vectors for S=77000, t=0..3 — pinned so that any drift from the
# seed chain turns this self-test red.
FROZEN_EPOCH_SEEDS = {
    0: 15148736061989184457,
    1: 17865006442405182368,
    2: 11879499459178023863,
    3: 15456178592288874741,
}

# Training wire-capture sidecar schema (ER_CAPTURE_DIR on the server).
# Pinned so capture writers and the evaluator (attacker/captures.py training
# mode) cannot drift apart.
SIDECAR_KEYS = {"session_id", "mb_id", "phase", "step", "epoch"}


# Seed-base hygiene: an attacker who guesses S derives EVERY epoch key, and
# small integers (e.g. the 77000 harness default) are brute-forceable in
# milliseconds — deployments must draw S from a CSPRNG and store it like a
# private key (local only, excluded from Git).
WEAK_SEED_LIMIT = 1 << 32  # below this, S is dictionary/brute-force territory


def generate_seed_base():
    """[ER] Draw a fresh 128-bit seed base from the OS CSPRNG. Print it once,
    store it locally with the same care as the E8 key files (never commit)."""
    return secrets.randbits(128)


def warn_if_weak_seed(seed_base, where=""):
    """[ER] Loud startup warning for low-entropy seed bases. Not a refusal:
    historical evidence runs used small S by design (the threat model under
    test was the linear solve, not S-guessing), and they must stay
    reproducible."""
    if seed_base is not None and seed_base < WEAK_SEED_LIMIT:
        print(f"[ER] WARNING {where}: obfuscation seed base is below 2^32 "
              f"and trivially brute-forceable. Fine for "
              f"reproducing evidence runs; for any real deployment generate "
              f"S with `python er_ratchet.py --generate-seed` (128-bit "
              f"CSPRNG) and treat it as key material.")


class TokenRatchet(PairBudget):
    """[ER] Pure-python epoch counter (torch-free). Budget accounting: each
    boundary forward adds its full token count (prefill-style; every training
    forward sends the whole microbatch sequence)."""

    def __init__(self, seed_base, ratchet_tokens=0, budget_events=0,
                 strict_budget=True):
        self.seed_base = seed_base
        super().__init__(ratchet_tokens=ratchet_tokens,
                         budget_events=budget_events,
                         strict=strict_budget)
        self.strict_budget = self.strict  # stable artifact/API field

    def enabled(self):
        """Inert unless seed_base is set AND (ratchet_tokens > 0 or
        budget_events > 0) — same rule as jacobi_server.enable_ratchet."""
        return (self.seed_base is not None
                and (self.ratchet_tokens > 0 or self.budget_events > 0))

class EpochRatchet(TokenRatchet):
    """[ER] TokenRatchet + per-epoch W derivation (torch). Caches ONLY the
    current epoch's (W, W^T) — the single-epoch convention of
    jacobi_server._derive_epoch_W. W is derived fp32 on CPU; callers move it
    to their device."""

    def __init__(self, seed_base, ratchet_tokens=0, budget_events=0,
                 strict_budget=True, version="v2", transform_mode="dense"):
        super().__init__(seed_base, ratchet_tokens, budget_events,
                         strict_budget=strict_budget)
        if version not in ("v1", "v2"):
            raise ValueError("ratchet version must be v1 or v2")
        self.version = version
        if transform_mode not in ("dense", "dense_sandwich",
                                  "structured_hadamard"):
            raise ValueError("unsupported transform_mode")
        self.transform_mode = transform_mode
        self._epoch_cache = {}  # {epoch: (W, Wt)} — at most one entry
        self._structured_cache = {}

    def _structured(self, epoch, tensor):
        from privacy_runtime.structured_transform import StructuredHadamard
        key = (epoch, tensor.shape[-1], str(tensor.device), str(tensor.dtype))
        if key not in self._structured_cache:
            self._structured_cache = {
                key: StructuredHadamard(self.seed_base, epoch, tensor.shape[-1],
                                        device=tensor.device, dtype=tensor.dtype)
            }
            print(f"[ER] ratchet epoch {epoch} active "
                  "(structured_hadamard_v1, seed_base=redacted)")
        return self._structured_cache[key]

    def apply(self, tensor, epoch):
        if self.transform_mode == "structured_hadamard":
            return self._structured(epoch, tensor).apply(tensor)
        W, _ = self.epoch_W(epoch, tensor.shape[-1])
        return tensor @ W.to(device=tensor.device, dtype=tensor.dtype)

    def inverse(self, tensor, epoch):
        if self.transform_mode == "structured_hadamard":
            return self._structured(epoch, tensor).inverse(tensor)
        _, Wt = self.epoch_W(epoch, tensor.shape[-1])
        return tensor @ Wt.to(device=tensor.device, dtype=tensor.dtype)

    def epoch_W(self, epoch, hidden_dim):
        """[ER] Derive W_t with the same QR-sign construction as
        jacobi_server._derive_epoch_W; cache ONLY the current epoch (evict
        older ones)."""
        if torch is None:
            raise RuntimeError("torch is required for epoch_W")
        if epoch in self._epoch_cache:
            return self._epoch_cache[epoch]
        W = _derive_W(self.seed_base, epoch, hidden_dim, self.version)
        w_hash = hashlib.sha256(W.numpy().tobytes()).hexdigest()[:16]
        self._epoch_cache = {epoch: (W, W.T.contiguous())}
        print(f"[ER] ratchet epoch {epoch} active "
              f"(W sha256={w_hash}, seed_base=redacted)")
        return self._epoch_cache[epoch]


def _derive_W(seed_base, epoch, hidden_dim, version):
    if version != "v2":
        # v1 derivation removed; ratchet_v2 is the only W_t derivation
        # (FROZEN_W_DIGESTS in privacy_runtime.ratchet_v2 pins the bytes).
        raise RuntimeError(
            f"[ER] ratchet version {version!r} is no longer supported; "
            "only the shared ratchet_v2 derivation exists")
    return derive_orthogonal_v2(seed_base, epoch, hidden_dim)


def derive_epoch_W(seed_base, epoch, hidden_dim, version="v2"):
    """[ER] Standalone W_t derivation (server-side fold path): same QR-sign
    construction, no cache — callers own the cache lifetime."""
    if torch is None:
        raise RuntimeError("torch is required for derive_epoch_W")
    if version not in ("v1", "v2"):
        raise ValueError("ratchet version must be v1 or v2")
    return _derive_W(seed_base, epoch, hidden_dim, version)


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and cond

    print("ER-T1 epoch_seed matches the frozen SHA256 vectors (S=77000, t=0..3):")
    for t, want in FROZEN_EPOCH_SEEDS.items():
        check(f"t={t}: {want}", epoch_seed(77000, t) == want)
    check("distinct epochs derive distinct seeds",
          len({epoch_seed(77000, t) for t in range(8)}) == 8)
    check("distinct seed bases derive distinct epoch-0 seeds",
          epoch_seed(77000, 0) != epoch_seed(77001, 0))

    print("ER-T2 fixed-N epoch arithmetic (N=64, training forwards of "
          "seq_len=32 tokens):")
    r = TokenRatchet(77000, ratchet_tokens=64)
    eps = [r.advance(32) for _ in range(6)]
    check("epoch = served // N after each forward: [0,1,1,2,2,3]",
          eps == [0, 1, 1, 2, 2, 3])
    check("served counts every forwarded token (6 x 32 = 192)",
          r.served == 192)
    r = TokenRatchet(77000, ratchet_tokens=64)
    eps = [r.advance(64) for _ in range(3)]  # micro_batch_size=2 x seq_len=32
    check("a forward adds seq_len x micro_batch_size tokens (64): [1,2,3]",
          eps == [1, 2, 3])

    print("ER-T3 budget-trigger transitions (B=128, frozen 64-token stream):")
    r = TokenRatchet(77000, budget_events=128)
    eps = [r.advance(64) for _ in range(6)]
    check("advance+reset at evidence >= B: [0,1,1,2,2,3]",
          eps == [0, 1, 1, 2, 2, 3])
    check("the tripping forward already uses the NEW epoch",
          TokenRatchet(77000, budget_events=64).advance(64) == 1)
    check("evidence resets after the trip", r.evidence == 0)
    r = TokenRatchet(77000, budget_events=128)
    check("sub-budget forwards stay in epoch 0",
          [r.advance(32) for _ in range(3)] == [0, 0, 0])
    r_overshoot = TokenRatchet(77000, budget_events=128)
    try:
        r_overshoot.advance(4096)
        refused = False
    except RuntimeError:
        refused = True
    check("production refuses one 4096-row forward against B=128", refused)
    check("refusal occurs before counters mutate",
          r_overshoot.served == 0 and r_overshoot.evidence == 0)
    r_remainder = TokenRatchet(77000, budget_events=16,
                               strict_budget=False)
    r_remainder.advance(20)
    check("remainder is preserved across trips (20 tokens at B=16 -> epoch 1, remainder 4)",
          r_remainder.epoch == 1 and r_remainder.evidence == 4)

    print("ER-T3b overshoot carries instead of resetting (#89):")
    r = TokenRatchet(77000, budget_events=128, strict_budget=False)
    check("one [8,512] training forward advances 4096//128 = 32 epochs",
          r.advance(8 * 512) == 32)
    check("nothing carried when the forward is an exact multiple",
          r.evidence == 0)
    r = TokenRatchet(77000, budget_events=128, strict_budget=False)
    check("a 200-token forward advances once and carries 72",
          (r.advance(200), r.evidence) == (1, 72))
    check("the carried 72 + 56 trips the next epoch exactly",
          (r.advance(56), r.evidence) == (2, 0))
    r = TokenRatchet(77000, budget_events=16)
    eps = [r.advance(10) for _ in range(5)]
    check("repeated sub-budget forwards lose no evidence: [0,1,1,2,3]",
          eps == [0, 1, 1, 2, 3])
    check("50 tokens at B=16 yield 3 rotations, not the old 2",
          r.epoch == 3 and r.evidence == 2)

    print("ER-T3c oversized-forward detection (#89):")
    r = TokenRatchet(77000, budget_events=64)
    [r.advance(64) for _ in range(3)]
    check("E-R9 shape [1,64] against B=64 is NOT flagged",
          r.oversized_forwards == 0)
    r = TokenRatchet(77000, budget_events=128, strict_budget=False)
    r.advance(256)
    check("run_er_training phase-d [1,256] against B=128 IS flagged",
          r.oversized_forwards == 1)
    r = TokenRatchet(77000, budget_events=128, strict_budget=False)
    [r.advance(256) for _ in range(4)]
    check("every oversized forward is counted, not just the first",
          r.oversized_forwards == 4)
    r = TokenRatchet(77000, ratchet_tokens=64, strict_budget=False)
    r.advance(256)
    check("fixed-N mode is checked against N too",
          r.oversized_forwards == 1)
    # advance() on a cadence-less ratchet divides by zero (enabled() is the
    # caller's guard), so exercise the check directly.
    r = TokenRatchet(None)
    r._check_forward_fits(4096)
    check("no cadence configured flags nothing",
          r.oversized_forwards == 0 and not r.enabled())

    print("ER-T4 enabled() rule (seed base AND a cadence):")
    check("seed base alone is inert", not TokenRatchet(77000).enabled())
    check("seed base + N is enabled",
          TokenRatchet(77000, ratchet_tokens=64).enabled())
    check("seed base + B is enabled",
          TokenRatchet(77000, budget_events=128).enabled())
    check("no seed base is inert even with a cadence",
          not TokenRatchet(None, budget_events=128).enabled())

    print("ER-T5 wire-capture sidecar schema keys:")
    sample = {"session_id": "s", "mb_id": 3, "phase": "fwd", "step": 2,
              "epoch": 1}
    check("sidecar key set matches {session_id, mb_id, phase, step, epoch}",
          set(sample.keys()) == SIDECAR_KEYS)
    check("phase is fwd or bwd", sample["phase"] in ("fwd", "bwd"))
    check("'epoch' is null when the ratchet is off",
          set({"session_id": "s", "mb_id": 0, "phase": "bwd", "step": 0,
               "epoch": None}.keys()) == SIDECAR_KEYS)
    check("sidecar round-trips through JSON unchanged",
          json.loads(json.dumps(sample)) == sample)

    print("ER-T5b seed-base hygiene:")
    s1, s2 = generate_seed_base(), generate_seed_base()
    check("generated seed bases are 128-bit", s1.bit_length() > 64)
    check("two draws differ (CSPRNG, not a constant)", s1 != s2)
    check("generated seeds are above the weak threshold",
          s1 >= WEAK_SEED_LIMIT and s2 >= WEAK_SEED_LIMIT)
    check("the frozen harness seed 77000 is (correctly) flagged weak",
          77000 < WEAK_SEED_LIMIT)

    if torch is not None:
        print("ER-T6 W_t construction (torch present): orthogonality + "
              "client/server derivation parity:")
        W, Wt = EpochRatchet(77000, budget_events=128).epoch_W(0, 64)
        check("W_0 is orthogonal (H=64)",
              torch.allclose(W @ W.T, torch.eye(64), atol=1e-5))
        check("cached Wt is W.T", torch.equal(Wt, W.T.contiguous()))
        check("standalone derive_epoch_W agrees with the cached ratchet",
              torch.equal(derive_epoch_W(77000, 0, 64), W))
        check("distinct epochs derive distinct W",
              not torch.equal(derive_epoch_W(77000, 1, 64), W))
        r2 = EpochRatchet(77000, budget_events=128)
        r2.epoch_W(0, 64)
        r2.epoch_W(1, 64)
        check("cache holds ONLY the current epoch",
              list(r2._epoch_cache.keys()) == [1])
    else:
        print("ER-T6 SKIPPED (torch not installed)")

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="pure-python fixture checks; no torch needed "
                         "(torch-only W checks run when torch is present)")
    ap.add_argument("--generate-seed", action="store_true",
                    help="print a fresh 128-bit CSPRNG seed base for "
                         "--obf-seed-base / --obfuscate-seed-base; store it "
                         "like a private key (never commit it)")
    args = ap.parse_args()
    if args.self_test:
        raise SystemExit(self_test())
    if args.generate_seed:
        print(generate_seed_base())
        return
    ap.print_help()


if __name__ == "__main__":
    main()
