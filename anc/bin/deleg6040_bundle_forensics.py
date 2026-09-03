#!/usr/bin/env python3
"""Two checks on a retained 60/40 delegation attacker bundle.

CHECK 1 -- chaff vs real breakdown of the correct predictions.
Every released frame is 32 real evaluation rows plus 48 recycled chaff rows
(bin/deleg6040_cell.sh: --seq-len 32 --chaff-tokens 48), so 60% of every
scored row is chaff and the headline recovery number could be dominated by
chaff. The bundle does not say which released row is which: the release
permutation is drawn from a fresh CSPRNG master inside TLN and never
leaves it (bin/run_latent_native_v5_06b.py:377-419). What IS exactly
recoverable is each frame's real LABEL MULTISET, because the evaluation
blocks are a deterministic function of the corpus, the tokenizer and
seq_len (bin/run_latent_native_v5_06b.py:38-45, 226-229).

That pins, per frame and per token value v: r = how many rows carrying v are
real, c = how many are chaff. With k of those rows predicted correctly, the
real share of the correct predictions is exactly

    real_correct(v) in [max(0, k - c), min(k, r)]

summed over frames and values. The interval collapses to a point wherever a
value is carried by real rows only or by chaff rows only, and for any
predictor that emits one class over a whole frame. No row is ever guessed:
what the labels do not determine is reported as an interval.

CHECK 2 -- shuffled-label negative control.
The frozen nine-arm attacker is re-scored on the same bundle with the
evaluation labels globally permuted. The permutation preserves the label
multiset exactly, so label_free_majority_pct is unchanged by construction
and any surviving excess over it is pipeline artefact, not information.

Both checks only re-score a retained bundle. Nothing is trained and no cloud
server is contacted.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "dtraining.deleg6040.bundle_forensics.v1"


def real_eval_labels(model_dir: str, corpus: str, seq_len: int,
                     train_blocks: int, eval_blocks: int) -> List[List[int]]:
    """Regenerate the real evaluation labels exactly as the runner did."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    ids = tokenizer(Path(corpus).read_text(errors="replace"),
                    add_special_tokens=False)["input_ids"]
    width = seq_len + 1
    blocks = [ids[i:i + width]
              for i in range(0, len(ids) - width + 1, width)]
    held = blocks[:train_blocks + eval_blocks][train_blocks:]
    if len(held) != eval_blocks:
        raise SystemExit(f"corpus yields {len(held)} eval blocks, "
                         f"expected {eval_blocks}")
    return [block[1:] for block in held]


def frame_partition(observed: Sequence[Any],
                    real: Sequence[Any]) -> Tuple[Counter, Counter]:
    """Split one released frame's labels into its real and chaff multisets."""
    obs = Counter(observed)
    real_counts = Counter(real)
    missing = real_counts - obs
    if missing:
        raise SystemExit("real labels are not a sub-multiset of the released "
                         f"frame (bundle/corpus mismatch): {dict(missing)}")
    return real_counts, obs - real_counts


def arm_breakdown(labels: List[List[Any]], real_counts: List[Counter],
                  chaff_counts: List[Counter], correct: List[List[bool]],
                  classes: set) -> Dict[str, int]:
    """Exact real/chaff attribution of one arm's correct predictions."""
    keys = ("correct", "real_lo", "real_hi", "unamb_real_rows",
            "unamb_real_correct", "unamb_chaff_rows", "unamb_chaff_correct")
    acc = {key: 0 for key in keys}
    for row_labels, reals, chaffs, row_ok in zip(labels, real_counts,
                                                 chaff_counts, correct):
        hit = Counter(v for v, ok in zip(row_labels, row_ok) if ok)
        for value, seen in Counter(row_labels).items():
            k, r, c = hit[value], reals[value], chaffs[value]
            acc["real_lo"] += max(0, k - c)
            acc["real_hi"] += min(k, r)
            if value not in classes:
                continue
            if c == 0:
                acc["unamb_real_rows"] += seen
                acc["unamb_real_correct"] += k
            elif r == 0:
                acc["unamb_chaff_rows"] += seen
                acc["unamb_chaff_correct"] += k
        acc["correct"] += int(sum(row_ok))
    return acc


def row_census(real_counts: List[Counter], chaff_counts: List[Counter],
               classes: set) -> Dict[str, Any]:
    """Population sizes the breakdown is measured against."""
    real = sum(sum(c.values()) for c in real_counts)
    chaff = sum(sum(c.values()) for c in chaff_counts)
    known_real = sum(n for c in real_counts
                     for value, n in c.items() if value in classes)
    known_chaff = sum(n for c in chaff_counts
                      for value, n in c.items() if value in classes)
    return {"eval_rows": real + chaff, "real_rows": real, "chaff_rows": chaff,
            "chaff_share_of_rows": chaff / max(1, real + chaff),
            "known_rows": known_real + known_chaff,
            "known_real_rows": known_real, "known_chaff_rows": known_chaff,
            "chaff_share_of_known_rows":
                known_chaff / max(1, known_real + known_chaff)}


def arm_record(name: str, restart: int, acc: Dict[str, int], total: int,
               census: Dict[str, Any]) -> Dict[str, Any]:
    """One arm's breakdown, with rates against each scored population."""
    correct, lo, hi = acc["correct"], acc["real_lo"], acc["real_hi"]
    unamb_real, unamb_chaff = acc["unamb_real_rows"], acc["unamb_chaff_rows"]
    return {
        "model": name, "restart": restart, "correct": correct,
        "top1_pct": 100.0 * correct / total,
        "real_correct_lo": lo, "real_correct_hi": hi,
        "chaff_correct_lo": correct - hi, "chaff_correct_hi": correct - lo,
        "chaff_share_of_correct_lo": (correct - hi) / max(1, correct),
        "chaff_share_of_correct_hi": (correct - lo) / max(1, correct),
        "chaff_share_of_known_rows": census["chaff_share_of_known_rows"],
        "real_recovery_rate_lo": lo / max(1, census["known_real_rows"]),
        "real_recovery_rate_hi": hi / max(1, census["known_real_rows"]),
        "chaff_recovery_rate_lo":
            (correct - hi) / max(1, census["known_chaff_rows"]),
        "chaff_recovery_rate_hi":
            (correct - lo) / max(1, census["known_chaff_rows"]),
        "unambiguous_real_rows": unamb_real,
        "unambiguous_real_correct": acc["unamb_real_correct"],
        "unambiguous_real_rate":
            acc["unamb_real_correct"] / max(1, unamb_real),
        "unambiguous_chaff_rows": unamb_chaff,
        "unambiguous_chaff_correct": acc["unamb_chaff_correct"],
        "unambiguous_chaff_rate":
            acc["unamb_chaff_correct"] / max(1, unamb_chaff),
    }


def write_shuffled(source: Path, target: Path, seed: int) -> Dict[str, Any]:
    """Copy a bundle with its evaluation labels globally permuted."""
    import torch

    bundle = torch.load(source, map_location="cpu")
    tokens = bundle["eval_tokens"]
    original = tokens.reshape(-1).clone()
    generator = torch.Generator().manual_seed(seed)
    shuffled = original[torch.randperm(original.numel(),
                                       generator=generator)]
    if not torch.equal(original.sort().values, shuffled.sort().values):
        raise SystemExit("shuffle changed the eval label multiset")
    bundle["eval_tokens"] = shuffled.reshape(tokens.shape)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, target)
    return {"seed": seed, "rows": int(original.numel()),
            "rows_unmoved": int((original == shuffled).sum()),
            "multiset_preserved": True, "path": str(target)}


def run_attacker(bundle: Path, output: Path, dump: Optional[Path]) -> None:
    """Score the frozen nine-arm latent-probe attacker on one bundle."""
    cmd = [sys.executable, "-m", "attacker", "--attack", "latent-probe",
           "--bundle", str(bundle), "--output", str(output)]
    if dump is not None:
        cmd += ["--dump-eval-predictions", str(dump)]
    output.parent.mkdir(parents=True, exist_ok=True)
    print("[forensics] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def load_correct(dump_path: Path, bundle_path: Path) -> Dict[str, Any]:
    """Per-arm per-row correctness, exactly as the frozen attacker scores it."""
    import torch

    dump = torch.load(dump_path, map_location="cpu")
    bundle = torch.load(bundle_path, map_location="cpu")
    tokens = dump["eval_tokens"]
    if not torch.equal(tokens, bundle["eval_tokens"]):
        raise SystemExit("prediction dump does not match the bundle labels")
    classes, known = dump["classes"], dump["known_eval"]
    majority = int(torch.mode(bundle["train_tokens"].reshape(-1)).values)
    arms = []
    for arm in dump["arms"]:
        predicted = classes[arm["prediction"]]
        arms.append({"model": arm["model"], "restart": arm["restart"],
                     "correct": ((predicted == tokens) & known).tolist(),
                     "concentration": concentration(predicted)})
    return {"labels": tokens.tolist(), "classes": set(classes.tolist()),
            "majority": majority, "arms": arms}


def concentration(predicted: Any) -> Dict[str, Any]:
    """How degenerate an arm's output is: a near-constant predictor recovers
    no content whatever its top-1 reads."""
    import torch

    flat = predicted.reshape(-1)
    values, counts = torch.unique(flat, return_counts=True)
    top = int(counts.argmax())
    return {"distinct_tokens_predicted": int(values.numel()),
            "modal_token": int(values[top]),
            "modal_token_share": float(counts[top]) / flat.numel()}


def summarize(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())["summary"][0]


def check_arm_counts(rescore: Path, arms: List[Dict[str, Any]]) -> None:
    """The breakdown must reproduce the frozen attacker's own correct counts."""
    recorded = {(r["model"], r["restart"]): r["correct"]
                for r in json.loads(rescore.read_text())["results"]}
    for arm in arms:
        key = (arm["model"], arm["restart"])
        if recorded.get(key) != arm["correct"]:
            raise SystemExit(f"arm {key}: breakdown counts {arm['correct']}, "
                             f"frozen attacker recorded {recorded.get(key)}")


def restricted_view(lo: int, hi: int, control: int,
                    rows: int) -> Dict[str, Any]:
    """Raw top-1 against the majority control on one row population.

    NOT the gate statistic: the gate compares a Bonferroni-Wilson UPPER bound
    against the control over all 20480 rows, so these are strictly smaller and
    are never to be read as a gate verdict.
    """
    rows = max(1, rows)
    base = 100.0 * control / rows
    return {"rows": rows, "control_correct": control,
            "control_top1_pct": base,
            "arm_top1_pct_lo": 100.0 * lo / rows,
            "arm_top1_pct_hi": 100.0 * hi / rows,
            "top1_excess_pp_lo": 100.0 * lo / rows - base,
            "top1_excess_pp_hi": 100.0 * hi / rows - base}


def population_views(best: Dict[str, Any], control: Dict[str, int],
                     census: Dict[str, Any]) -> Dict[str, Any]:
    """The best arm read on all scoreable rows, on real only, on chaff only."""
    return {
        "all_scoreable_rows": restricted_view(
            best["correct"], best["correct"], control["correct"],
            census["known_rows"]),
        "real_rows_only": restricted_view(
            best["real_correct_lo"], best["real_correct_hi"],
            control["real"], census["known_real_rows"]),
        "chaff_rows_only": restricted_view(
            best["chaff_correct_lo"], best["chaff_correct_hi"],
            control["chaff"], census["known_chaff_rows"])}


def eval_labels(bundle_path: Path) -> List[List[int]]:
    import torch

    return torch.load(bundle_path,
                      map_location="cpu")["eval_tokens"].tolist()


def build_report(args: argparse.Namespace, scored: Dict[str, Any],
                 parts: List[Tuple[Counter, Counter]]) -> Dict[str, Any]:
    """Check 1 over every arm of the frozen attacker, plus the control split."""
    labels = scored["labels"]
    if len(labels) != len(parts):
        raise SystemExit(f"{len(labels)} released frames vs "
                         f"{len(parts)} partitioned frames")
    real_counts = [part[0] for part in parts]
    chaff_counts = [part[1] for part in parts]
    census = row_census(real_counts, chaff_counts, scored["classes"])
    total, majority = census["eval_rows"], scored["majority"]
    control = {"token": majority,
               "real": sum(c[majority] for c in real_counts),
               "chaff": sum(c[majority] for c in chaff_counts)}
    control["correct"] = control["real"] + control["chaff"]
    control["pct"] = 100.0 * control["correct"] / total
    control["chaff_share"] = control["chaff"] / max(1, control["correct"])
    arms = []
    for arm in scored["arms"]:
        record = arm_record(
            arm["model"], arm["restart"],
            arm_breakdown(labels, real_counts, chaff_counts, arm["correct"],
                          scored["classes"]), total, census)
        record.update(arm["concentration"])
        arms.append(record)
    best = max(arms, key=lambda record: record["top1_pct"])
    return {
        "rows": {"frames": len(labels), "rows_per_frame": len(labels[0]),
                 "real_per_frame": args.seq_len,
                 "chaff_per_frame": len(labels[0]) - args.seq_len, **census},
        "majority_control": control, "arms": arms, "best_arm": best,
        "best_arm_by_population": population_views(best, control, census)}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", help="retained .pt attacker bundle")
    ap.add_argument("--model", help="tokenizer/model directory")
    ap.add_argument("--corpus", help="corpus the cell was run on")
    ap.add_argument("--workdir", help="scratch dir for re-scores and dumps")
    ap.add_argument("--output", help="forensics JSON artifact")
    ap.add_argument("--recorded-attacker-json",
                    help="the cell's original *_attacker.json, for a "
                         "bit-for-bit reproduction check")
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--train-blocks", type=int, default=256)
    ap.add_argument("--eval-blocks", type=int, default=256)
    ap.add_argument("--shuffle-seed", type=int, default=20260818)
    ap.add_argument("--reuse", action="store_true",
                    help="skip a frozen-attacker run whose outputs exist")
    ap.add_argument("--self-test", action="store_true")
    return ap


def self_test() -> int:
    """Bounds on a hand-checked frame: rows a,a,a,b,c with a,a,b real, so
    'a' is 2 real + 1 chaff, 'b' real only, 'c' chaff only. Two 'a' rows and
    the 'c' row score, so real gets at least 2-1=1 and at most min(2,2)=2."""
    labels = [["a", "a", "a", "b", "c"]]
    real, chaff = frame_partition(labels[0], ["a", "a", "b"])
    acc = arm_breakdown(labels, [real], [chaff],
                        [[True, True, False, False, True]],
                        {"a", "b", "c"})
    ok = (acc["correct"] == 3 and acc["real_lo"] == 1 and acc["real_hi"] == 2
          and acc["unamb_real_rows"] == 1 and acc["unamb_chaff_rows"] == 1
          and acc["unamb_chaff_correct"] == 1)
    print(f"  [{'PASS' if ok else 'FAIL'}] real/chaff attribution bounds")
    return 0 if ok else 1


def partition_frames(args: argparse.Namespace,
                     bundle: Path) -> List[Tuple[Counter, Counter]]:
    """Regenerate the real eval labels and split every released frame by them.

    Runs before any scoring: a corpus/bundle mismatch must abort in seconds,
    not after two full nine-arm re-scores.
    """
    observed = eval_labels(bundle)
    regenerated = real_eval_labels(args.model, args.corpus, args.seq_len,
                                   args.train_blocks, args.eval_blocks)
    if len(observed) != len(regenerated):
        raise SystemExit(f"{len(observed)} released frames vs "
                         f"{len(regenerated)} regenerated eval blocks")
    parts = [frame_partition(obs, real)
             for obs, real in zip(observed, regenerated)]
    print(f"[forensics] {len(parts)} frames partitioned: "
          f"{sum(sum(p[0].values()) for p in parts)} real rows, "
          f"{sum(sum(p[1].values()) for p in parts)} chaff rows", flush=True)
    return parts


def score_both(args: argparse.Namespace, bundle: Path,
               work: Path) -> Tuple[Path, Path, Path, Dict[str, Any]]:
    """Frozen-attacker re-score of the bundle and of its label-shuffled copy."""
    stem = bundle.stem
    base_json, dump = work / f"{stem}_rescore.json", work / f"{stem}_pred.pt"
    if not (args.reuse and base_json.exists() and dump.exists()):
        run_attacker(bundle, base_json, dump)
    shuffled = work / f"{stem}_shuffled.pt"
    meta = write_shuffled(bundle, shuffled, args.shuffle_seed)
    shuffled_json = work / f"{stem}_shuffled_rescore.json"
    if not (args.reuse and shuffled_json.exists()):
        run_attacker(shuffled, shuffled_json, None)
    return base_json, dump, shuffled_json, meta


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        return self_test()
    for name in ("bundle", "model", "corpus", "workdir", "output"):
        if not getattr(args, name):
            raise SystemExit(f"--{name} is required")
    bundle, work = Path(args.bundle), Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    parts = partition_frames(args, bundle)
    base_json, dump, shuffled_json, shuffle_meta = score_both(
        args, bundle, work)

    report = build_report(args, load_correct(dump, bundle), parts)
    check_arm_counts(base_json, report["arms"])
    rescored, control = summarize(base_json), summarize(shuffled_json)
    recorded = (summarize(Path(args.recorded_attacker_json))
                if args.recorded_attacker_json else None)
    report.update({
        "schema": SCHEMA, "bundle": bundle.name,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "reproduction": {"recorded": recorded, "rescored": rescored,
                         "identical": recorded == rescored},
        "shuffled_label_control": {
            **shuffle_meta, "summary": control,
            "excess_pp_unshuffled":
                rescored["upper95_excess_over_majority_pp"],
            "excess_pp_shuffled":
                control["upper95_excess_over_majority_pp"]}})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["best_arm"], indent=2))
    print(json.dumps(report["shuffled_label_control"]["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
