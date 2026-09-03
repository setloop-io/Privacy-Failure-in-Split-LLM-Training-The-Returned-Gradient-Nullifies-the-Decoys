#!/usr/bin/env python3
"""Leakage-injection dose-response calibration of the latent-probe privacy gate.

The campaign's privacy gate is the frozen nine-arm `latent-probe` attacker
scored against a label-free majority control. The bundle forensics analysis
(bin/deleg6040_bundle_forensics.py) showed the
gate has no FALSE POSITIVES (a shuffled-label control reads clean) and that
every arm which beats the control is a near-constant predictor. Nobody has ever
measured the gate's FALSE NEGATIVE rate: does it detect token information that
really is present in the released frame?

This tool answers that by injecting a known, controlled amount of true token
information into a retained bundle's released rows and re-scoring the frozen
attacker unmodified.

THE INJECTION
-------------
A fixed codebook assigns every token value one Haar-uniform unit direction
c_t in R^D (D=64), drawn from the v2 CSPRNG stream. For a selected released
row x with token t the row is rotated toward its codeword and renormalised:

    x' = ||x|| * normalise( sqrt(1-a^2) * x/||x||  +  a * c_t )

Two knobs, both exactly quantifiable:

  coverage rho  the fraction of released rows that carry the injected signal.
                Rows are the first round(rho*N) of one fixed CSPRNG
                permutation, so the injected sets are NESTED across rho.
  amplitude a   the codeword's share of the row's unit direction. a^2 is the
                injected power fraction, so the token-correlated SNR is
                exactly a^2/(1-a^2). At a=1 the row IS the codeword scaled to
                its original norm: the token is recoverable from that row
                exactly, with zero residual.

The blend is norm-preserving by construction, so the released frame's row-norm
distribution is untouched and no arm can win on a magnitude artefact.

TWO FRAMES OF REFERENCE
-----------------------
This cell releases with `--secret-wire-rotation` and no token gauge
(`secret_token_gauge=False`, `wire_quant=none`), so each released frame is
`(latent ++ chaff)[permutation] @ R_frame` for a fresh Haar R_frame drawn
inside the trusted node and never released.

  --mode coordinate  c_t is fixed in RELEASED coordinates, identical in every
                     frame. This models leakage that bypasses or survives the
                     wire rotation (absent, stale or reused rotation, or a
                     side channel). It is the EASIEST possible leak: the
                     coordinate arms see a single consistent linear code.

  --mode invariant   c_t is carried through a fresh per-frame Haar rotation,
                     exactly as `release()` would carry a direction injected
                     into the trusted latent before rotation. Because the
                     codebook is Haar-uniform and independent of the real
                     rotation, {c_t @ R_frame} is equal in law to a fresh Haar
                     dictionary independent of the released frame, so this is
                     a faithful simulation of a leaking encoder and not an
                     approximation of one. Only the rotation-invariant
                     structure survives: within a frame, rows sharing a token
                     move toward the same direction.

WHAT IS MEASURED
----------------
Every point runs the frozen attacker unmodified (`python3 -m attacker --attack
latent-probe`) on the injected bundle and re-derives the gate from raw counts
with `deleg6040_gate_recalibrate.recalibrate`, so the excess, the statistical
floor, the binomial z and both gate verdicts are computed exactly as the
campaign computes them. Alongside, the tool reports how recoverable the
injected token actually is from the released row, by nearest-codeword decoding
(with and without knowledge of the per-frame rotation), so a null result can
never be confused with a leak that was never there.

Labels are never touched, so `label_free_majority_pct` is invariant across the
whole sweep and every movement in the excess is attributable to the arms.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "bin"))
sys.path.insert(0, str(REPO_ROOT))

from deleg6040_bundle_forensics import concentration, run_attacker  # noqa: E402
from deleg6040_gate_recalibrate import recalibrate  # noqa: E402
from deleg6040_verify_stats import binomial_z  # noqa: E402

SCHEMA = "dtraining.deleg6040.gate_sensitivity.v1"
MODES = ("coordinate", "invariant")
FAMILY = {"coordinate_plus_invariants": "coordinate",
          "invariant_only": "invariant", "invariant_graph": "invariant"}


@dataclass(frozen=True)
class Spec:
    """One dose: injection frame of reference, coverage and amplitude."""
    mode: str
    coverage: float
    amplitude: float
    seed: int

    def tag(self) -> str:
        return f"{self.mode}_rho{self.coverage:g}_a{self.amplitude:g}"

    def validate(self) -> None:
        if self.mode not in MODES:
            raise SystemExit(f"mode must be one of {MODES}")
        if not 0.0 <= self.coverage <= 1.0:
            raise SystemExit("coverage must lie in [0, 1]")
        if not 0.0 <= self.amplitude <= 1.0:
            raise SystemExit("amplitude must lie in [0, 1]")


def master(seed: int, label: str) -> bytes:
    """Deterministic 128-bit master for one CSPRNG stream in this tool."""
    text = f"dtraining/deleg6040/gate-sensitivity/{seed}/{label}"
    return hashlib.sha256(text.encode()).digest()[:16]


def value_index(*token_tensors: Any) -> Tuple[Any, Any]:
    """Compact index over every token value present in the bundle."""
    import torch

    values = torch.unique(
        torch.cat([t.reshape(-1) for t in token_tensors])).sort().values
    lut = torch.full((int(values.max()) + 1,), -1, dtype=torch.long)
    lut[values] = torch.arange(values.numel())
    return values, lut


def codebook(classes: int, dim: int, seed: int) -> Any:
    """One Haar-uniform unit direction per token value, from the v2 stream."""
    from privacy_runtime.ratchet_v2 import derive_gaussian_tensor

    raw = derive_gaussian_tensor(master(seed, "codebook"), 0, (classes, dim))
    return raw / raw.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def frame_rotations(frames: int, dim: int, seed: int, label: str) -> Any:
    """One fresh Haar rotation per released frame, as `release()` draws them."""
    import torch
    from privacy_runtime.ratchet_v2 import derive_orthogonal

    key = master(seed, f"rotation/{label}")
    return torch.stack([derive_orthogonal(key, index, dim)
                        for index in range(frames)])


def selection_mask(frames: int, seq: int, coverage: float, seed: int,
                   label: str) -> Any:
    """Nested row selection: the first round(rho*N) of a fixed permutation."""
    import torch
    from privacy_runtime.ratchet_v2 import derive_permutation

    total = frames * seq
    take = int(round(coverage * total))
    mask = torch.zeros(total, dtype=torch.bool)
    if take:
        order = derive_permutation(master(seed, f"rows/{label}"), 0, total)
        mask[order[:take]] = True
    return mask.reshape(frames, seq)


def codewords(tokens: Any, lut: Any, book: Any,
              rotation: Optional[Any]) -> Any:
    """Per-row target direction, carried through the frame rotation if given."""
    import torch

    code = book[lut[tokens]]
    if rotation is None:
        return code
    return torch.einsum("fsd,fde->fse", code, rotation)


def inject(wire: Any, code: Any, mask: Any, amplitude: float) -> Any:
    """Norm-preserving rotation of the selected rows toward their codeword."""
    import torch

    if amplitude == 0.0:
        return wire.clone()
    norms = wire.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    blend = math.sqrt(1.0 - amplitude ** 2) * (wire / norms) + amplitude * code
    blend = blend / blend.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    out = wire.clone()
    index = mask.unsqueeze(-1).expand_as(wire)
    return torch.where(index, blend * norms, out)


def decode_rate(wire: Any, tokens: Any, lut: Any, book: Any, mask: Any,
                rotation: Optional[Any]) -> Dict[str, Any]:
    """Nearest-codeword top-1 on the injected rows of the released frame.

    `rotation` un-rotates each frame first: that is the ORACLE decoder, which
    knows the per-frame secret. Passing None is the codebook-only decoder,
    which is what a compromised node could actually run.
    """
    import torch

    unit = wire / wire.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    if rotation is not None:
        unit = torch.einsum("fsd,fed->fse", unit, rotation)
    predicted = (unit.reshape(-1, book.shape[-1]) @ book.t()).argmax(-1)
    truth = lut[tokens].reshape(-1)
    flat = mask.reshape(-1)
    hits = int(((predicted == truth) & flat).sum())
    rows = int(flat.sum())
    rate = 100.0 * hits / rows if rows else 0.0
    return {"rows": rows, "correct": hits, "top1_pct": rate}


def frame_cosines(wire: Any, tokens: Any, mask: Any) -> Dict[str, float]:
    """Mean within-frame cosine between injected rows, split by token match."""
    import torch

    unit = wire / wire.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    gram = unit @ unit.transpose(-1, -2)
    pair = mask.unsqueeze(-1) & mask.unsqueeze(-2)
    pair = pair & ~torch.eye(mask.shape[1], dtype=torch.bool).unsqueeze(0)
    same = pair & (tokens.unsqueeze(-1) == tokens.unsqueeze(-2))
    diff = pair & ~same

    def mean(selection: Any) -> float:
        return (float(gram[selection].mean()) if bool(selection.any())
                else float("nan"))

    return {"same_token_pairs": int(same.sum()),
            "same_token_mean_cosine": mean(same),
            "other_token_pairs": int(diff.sum()),
            "other_token_mean_cosine": mean(diff)}


def inject_partition(wire: Any, tokens: Any, lut: Any, book: Any, spec: Spec,
                     label: str, scoreable: Any) -> Tuple[Any, Dict[str, Any]]:
    """Inject one partition and account for exactly what was added.

    `scoreable` marks rows whose token the frozen attacker is able to emit at
    all (its class set is the train partition's token values). An injected row
    outside that set can never be scored correct however cleanly it leaks, so
    the injected budget that the gate can even see is reported separately.
    """
    frames, seq, dim = wire.shape
    mask = selection_mask(frames, seq, spec.coverage, spec.seed, label)
    rows = int(mask.sum())
    account: Dict[str, Any] = {
        "partition": label, "rows_total": frames * seq, "rows_injected": rows,
        "rows_injected_scoreable": int((mask & scoreable).sum()),
        "coverage_achieved": rows / (frames * seq)}
    if not rows or spec.amplitude == 0.0:
        account["injected"] = False
        return wire.clone(), account
    rotation = (frame_rotations(frames, dim, spec.seed, label)
                if spec.mode == "invariant" else None)
    out = inject(wire, codewords(tokens, lut, book, rotation), mask,
                 spec.amplitude)
    account.update({
        "injected": True,
        "codebook_decode": decode_rate(out, tokens, lut, book, mask, None),
        "oracle_decode": decode_rate(out, tokens, lut, book, mask, rotation),
        "cosines": frame_cosines(out, tokens, mask),
        "max_row_norm_drift": float(
            (out.norm(dim=-1) - wire.norm(dim=-1)).abs().max())})
    return out, account


def write_injected(source: Path, target: Path,
                   spec: Spec) -> Dict[str, Any]:
    """Copy a bundle with the injection applied to both released partitions."""
    import torch

    bundle = torch.load(source, map_location="cpu")
    values, lut = value_index(bundle["train_tokens"], bundle["eval_tokens"])
    book = codebook(values.numel(), bundle["train_wire"].shape[-1], spec.seed)
    classes = torch.unique(bundle["train_tokens"])
    accounts = []
    for part in ("train", "eval"):
        wire, tokens = bundle[f"{part}_wire"], bundle[f"{part}_tokens"]
        new, account = inject_partition(wire.float(), tokens, lut, book, spec,
                                        part, torch.isin(tokens, classes))
        if spec.amplitude == 0.0 or account["rows_injected"] == 0:
            if not torch.equal(new, wire):
                raise SystemExit(f"{part}: zero dose changed the wire")
        bundle[f"{part}_wire"] = new
        accounts.append(account)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, target)
    return {"spec": spec.__dict__, "token_values": int(values.numel()),
            "attacker_classes": int(classes.numel()),
            "latent_dim": int(book.shape[-1]), "partitions": accounts,
            "path": str(target), "sha256": file_sha256(target)}


def file_sha256(path: Path) -> str:
    """Digest of the injected bundle actually handed to the frozen attacker."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arm_rows(artifact: Path, majority_pct: float,
             dump: Optional[Path]) -> List[Dict[str, Any]]:
    """Per-arm top-1, binomial z and (when dumped) prediction concentration."""
    import torch

    results = json.loads(artifact.read_text())["results"]
    shapes = {}
    if dump is not None and dump.exists():
        loaded = torch.load(dump, map_location="cpu")
        classes = loaded["classes"]
        for arm in loaded["arms"]:
            shapes[(arm["model"], arm["restart"])] = concentration(
                classes[arm["prediction"]])
    rows = []
    for arm in results:
        rows.append({
            "model": arm["model"], "family": FAMILY[arm["model"]],
            "restart": arm["restart"], "correct": arm["correct"],
            "total": arm["total"], "top1_pct": arm["top1_pct"],
            "binomial_z": binomial_z(arm["correct"], arm["total"],
                                     majority_pct / 100.0),
            **shapes.get((arm["model"], arm["restart"]), {})})
    return rows


def family_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Best binomial z within each arm family: which family detects first."""
    out = {}
    for family in ("coordinate", "invariant"):
        members = [r for r in rows if r["family"] == family]
        best = max(members, key=lambda r: r["correct"])
        out[family] = {"model": best["model"], "restart": best["restart"],
                       "correct": best["correct"],
                       "top1_pct": best["top1_pct"],
                       "binomial_z": best["binomial_z"]}
    return out


def score_point(spec: Spec, source: Path, work: Path,
                reuse: bool) -> Dict[str, Any]:
    """Inject one dose, re-score the frozen attacker, re-derive the gate."""
    tag = spec.tag()
    bundle = work / f"inj_{tag}.pt"
    artifact = work / f"inj_{tag}_attacker.json"
    dump = work / f"inj_{tag}_pred.pt"
    injection = write_injected(source, bundle, spec)
    if not (reuse and artifact.exists()):
        run_attacker(bundle, artifact, dump)
    gate = recalibrate(str(artifact))
    rows = arm_rows(artifact, gate["label_free_majority_pct"], dump)
    return {"spec": spec.__dict__, "tag": tag, "injection": injection,
            "gate": gate, "arms": rows, "families": family_summary(rows),
            "artifact": str(artifact)}


def build_specs(args: argparse.Namespace) -> List[Spec]:
    specs = [Spec(args.mode, coverage, amplitude, args.seed)
             for coverage in args.coverage for amplitude in args.amplitude]
    for spec in specs:
        spec.validate()
    return specs


def check_zero_dose(point: Dict[str, Any], recorded: Optional[str]) -> None:
    """The zero dose must reproduce the unmodified reading, exactly."""
    if recorded is None:
        return
    baseline = recalibrate(recorded)
    keys = ("excess_pp", "statistical_floor_pp", "best_arm_binomial_z",
            "label_free_majority_pct")
    delta = {k: (baseline[k], point["gate"][k]) for k in keys
             if abs(baseline[k] - point["gate"][k]) > 1e-9}
    if delta:
        raise SystemExit(f"zero dose did not reproduce the recorded gate: "
                         f"{delta}")
    print(f"[gate-sens] zero dose reproduces {recorded} exactly", flush=True)


def eval_account(point: Dict[str, Any]) -> Dict[str, Any]:
    """The evaluation partition's injection accounting for one dose."""
    for part in point["injection"]["partitions"]:
        if part["partition"] == "eval":
            return part
    raise SystemExit(f"{point['tag']}: no eval partition accounting")


def reference_point(points: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The zero dose, when the table contains one: the undosed reading."""
    zeros = [p for p in points if p["spec"]["coverage"] == 0.0
             or p["spec"]["amplitude"] == 0.0]
    return zeros[0] if zeros else None


def delta(point: Dict[str, Any], reference: Optional[Dict[str, Any]],
          family: str) -> str:
    """Correct rows this family gained over the undosed reading."""
    if reference is None:
        return f"{'-':>8}"
    gained = (point["families"][family]["correct"]
              - reference["families"][family]["correct"])
    return f"{gained:>+8d}"


def row(point: Dict[str, Any], reference: Optional[Dict[str, Any]]) -> str:
    spec, gate, fam = point["spec"], point["gate"], point["families"]
    part = eval_account(point)
    decode = part.get("codebook_decode", {}).get("top1_pct", 0.0)
    oracle = part.get("oracle_decode", {}).get("top1_pct", 0.0)
    return (f"{spec['mode']:>11} {spec['coverage']:>8.4f} "
            f"{spec['amplitude']:>6.3f} {part['rows_injected']:>7d} "
            f"{decode:>7.2f} {oracle:>7.2f} {gate['excess_pp']:>+9.4f} "
            f"{gate['statistical_floor_pp']:>+8.4f} "
            f"{gate['best_arm_binomial_z']:>+9.3f} "
            f"{fam['coordinate']['binomial_z']:>+9.3f} "
            f"{fam['invariant']['binomial_z']:>+9.3f} "
            f"{delta(point, reference, 'coordinate')} "
            f"{delta(point, reference, 'invariant')} "
            f"{gate['classification']:>9} {gate['legacy_verdict']:>6} "
            f"{gate['proposed_verdict']:>6}")


HEADER = (f"{'mode':>11} {'coverage':>8} {'ampl':>6} {'inj':>7} "
          f"{'dec%':>7} {'orc%':>7} {'excess':>9} {'floor':>8} "
          f"{'best z':>9} {'coord z':>9} {'inv z':>9} {'coord d':>8} "
          f"{'inv d':>8} {'class':>9} {'legacy':>6} {'floor':>6}")


def print_table(points: List[Dict[str, Any]]) -> None:
    reference = reference_point(points)
    print(HEADER)
    for point in points:
        print(row(point, reference))


def load_points(paths: List[str]) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    for path in paths:
        points.extend(json.loads(Path(path).read_text())["points"])
    return sorted(points, key=lambda p: (p["spec"]["mode"],
                                         p["spec"]["amplitude"],
                                         p["spec"]["coverage"]))


def self_test() -> int:
    """Injection algebra on a fixture: exact codeword at a=1, no-op at a=0,
    norm preserved, and nested row selection across coverage."""
    import torch

    wire = torch.randn(3, 5, 8) * 2.0
    tokens = torch.tensor([[1, 2, 1, 3, 2]] * 3)
    values, lut = value_index(tokens)
    book = codebook(values.numel(), 8, 7)
    mask = torch.ones(3, 5, dtype=torch.bool)
    code = book[lut[tokens]]
    full = inject(wire, code, mask, 1.0)
    unit = full / full.norm(dim=-1, keepdim=True)
    exact = torch.allclose(unit, code, atol=1e-6)
    zero = torch.equal(inject(wire, code, mask, 0.0), wire)
    norm = torch.allclose(full.norm(dim=-1), wire.norm(dim=-1), atol=1e-5)
    small = selection_mask(3, 5, 0.2, 7, "eval")
    large = selection_mask(3, 5, 0.6, 7, "eval")
    nested = bool((small & large).sum() == small.sum()) and int(
        small.sum()) == 3 and int(large.sum()) == 9
    ok = exact and zero and norm and nested
    print(f"  [{'PASS' if exact else 'FAIL'}] a=1 row is exactly its codeword")
    print(f"  [{'PASS' if zero else 'FAIL'}] a=0 leaves the wire untouched")
    print(f"  [{'PASS' if norm else 'FAIL'}] injection preserves row norms")
    print(f"  [{'PASS' if nested else 'FAIL'}] coverage selections are nested")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", help="retained .pt attacker bundle to dose")
    ap.add_argument("--workdir", help="scratch dir for injected bundles")
    ap.add_argument("--output", help="sweep JSON artifact")
    ap.add_argument("--mode", choices=MODES, default="coordinate")
    ap.add_argument("--coverage", type=float, nargs="+", default=[0.0],
                    help="fraction of released rows carrying the signal")
    ap.add_argument("--amplitude", type=float, nargs="+", default=[1.0],
                    help="codeword share of the row direction (a); the "
                         "injected power fraction is a^2")
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--recorded-attacker-json",
                    help="the cell's original *_attacker.json; the zero dose "
                         "is asserted against it")
    ap.add_argument("--reuse", action="store_true",
                    help="skip a frozen-attacker run whose artifact exists")
    ap.add_argument("--table", nargs="+",
                    help="print the dose-response table from sweep JSONs")
    ap.add_argument("--self-test", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        return self_test()
    if args.table:
        print_table(load_points(args.table))
        return 0
    for name in ("bundle", "workdir", "output"):
        if not getattr(args, name):
            raise SystemExit(f"--{name} is required")
    source, work = Path(args.bundle), Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    points = []
    for spec in build_specs(args):
        print(f"[gate-sens] dose {spec.tag()}", flush=True)
        point = score_point(spec, source, work, args.reuse)
        if spec.coverage == 0.0 or spec.amplitude == 0.0:
            check_zero_dose(point, args.recorded_attacker_json)
        points.append(point)
        print(row(point, reference_point(points)), flush=True)
    report = {"schema": SCHEMA, "bundle": source.name,
              "generated_utc": datetime.now(timezone.utc).isoformat(),
              "mode": args.mode, "seed": args.seed, "points": points}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print_table(points)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
