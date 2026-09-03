#!/usr/bin/env python3
"""The paper's label-free control, recomputed on the evaluation rows.

main.tex:153 defines the control as the best constant predictor ON THE
EVALUATION ROWS.  attacker/attacks/latent_probe.py:215-217 scores the TRAIN
partition's mode on those rows instead.  This tool computes both from the
corpus and reports the difference, for the cells where the evaluation rows are
corpus rows -- i.e. the cells that release no chaff.  Where chaff is released
the evaluation label multiset is not a function of the corpus and this tool
refuses to guess; see bin/deleg6040_metric_stats_audit.py --help-limits.

Label regeneration is the method bin/deleg6040_bundle_forensics.py already
validated: the blocks are a
deterministic function of corpus, tokenizer and seq_len
(bin/run_latent_native_v5_06b.py:38-45, 226-229), train = blocks[:256],
eval = blocks[256:512], and a block's labels are block[1:seq_len+1].

THREE INDEPENDENT CHECKS run before any number is emitted, and each aborts:
  1. the corpus sha256 must equal paper-data/corpus_manifest_original.json;
  2. the train mode's count on the evaluation rows must equal every chaff-free
     attacker artifact's label_free_majority_pct;
  3. the share of evaluation rows whose label is inside the train class set
     must equal those artifacts' known_eval_fraction.
Checks 2 and 3 tie the regenerated labels to the committed campaign artifacts,
so a corpus that hashes correctly but tokenizes differently still fails.

THE CORPUS IS NOT IN THE REPOSITORY and must not be added to it: it is the
campaign's private training corpus.  It is reconstructible byte-exactly from a
public source, verified by the manifest sha256
(78b6bfb90cfd718f0c27d42b1fd2231b139d1dda75d7d796e6a603b2e5cd7efe):

    src  https://huggingface.co/datasets/Salesforce/wikitext/resolve/
         b08601e04326c79dfdd32d625aee71d232d685c3/
         wikitext-2-v1/train-00000-of-00001.parquet
         (sha256 dfc27e4360c639dc1fba1e403bfffd53af4a5c75d5363b5724d49bf12d07cce6)
    rows = [str(v) for v in pandas.read_parquet(src)["text"].tolist()]
    kept = [s for r in rows if (s := r.strip()) and len(s) > 200]
    open(path, "wb").write("\\n".join(kept).encode())   # NO trailing newline

Note the tokenized wikitext-2-v1 variant, the >200-character line filter and
the absent trailing newline: with the newline the file is one byte longer and
hashes differently.  bin/repro_make_corpus.py's raw-variant reconstruction
(sha256 1ac2aed3...) is a different corpus and is not usable here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "paper-data" / "corpus_manifest_original.json"
COLLECTED = REPO_ROOT / "paper-data" / "collected"
SCHEMA = "dtraining.deleg6040.paper_control.v1"


def verify_corpus(path: Path) -> dict[str, Any]:
    """Abort unless the corpus is byte-identical to the manifest's."""
    manifest = json.loads(MANIFEST.read_text())
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != manifest["sha256"]:
        raise SystemExit(f"corpus sha256 {digest} != manifest "
                         f"{manifest['sha256']}: this is a different corpus, "
                         f"and every number below is a property of the corpus")
    if len(raw) != manifest["bytes"]:
        raise SystemExit(f"corpus is {len(raw)} bytes, manifest says "
                         f"{manifest['bytes']}")
    return {"path": str(path), "sha256": digest, "bytes": len(raw),
            "manifest": str(MANIFEST.relative_to(REPO_ROOT))}


def label_partitions(model_dir: str, corpus: Path, seq_len: int,
                     train_blocks: int,
                     eval_blocks: int) -> tuple[list[int], list[int]]:
    """Train and eval label rows, exactly as the runner cuts them."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    ids = tokenizer(corpus.read_text(errors="replace"),
                    add_special_tokens=False)["input_ids"]
    width = seq_len + 1
    blocks = [ids[i:i + width]
              for i in range(0, len(ids) - width + 1, width)]
    if len(blocks) < train_blocks + eval_blocks:
        raise SystemExit(f"corpus yields {len(blocks)} blocks, need "
                         f"{train_blocks + eval_blocks}")
    train = [t for block in blocks[:train_blocks] for t in block[1:]]
    held = blocks[train_blocks:train_blocks + eval_blocks]
    return train, [t for block in held for t in block[1:]]


def chaff_free_cells(rows_per_frame: int, eval_rows: int) -> list[Path]:
    """Committed attacker artifacts whose released frames carry no chaff.

    Chaff is redrawn per run, so a released chaff row would make the control
    differ between independently run cells.  A frame shape whose cells all
    report one control count releases none; see METRIC_STATISTICS_AUDIT.md.
    """
    found = []
    for path in sorted(COLLECTED.rglob("*attacker*.json")):
        art = json.loads(path.read_text())
        if "results" not in art or not art["results"]:
            continue
        config = art["config"]
        if (config["sequence_length"] != rows_per_frame
                or art["results"][0]["total"] != eval_rows):
            continue
        found.append(path)
    if not found:
        raise SystemExit(f"no committed cell has {eval_rows} rows in frames of "
                         f"{rows_per_frame}")
    return found


def cross_check(cells: list[Path], control_correct: int, known: int,
                total: int) -> list[dict[str, Any]]:
    """Tie the regenerated labels to every committed chaff-free artifact."""
    checked = []
    for path in cells:
        summary = json.loads(path.read_text())["summary"][0]
        artifact_control = round(
            summary["label_free_majority_pct"] * total / 100.0)
        artifact_known = round(summary["known_eval_fraction"] * total)
        if artifact_control != control_correct:
            raise SystemExit(
                f"{path.name}: artifact control {artifact_control} rows, "
                f"regenerated labels give {control_correct}")
        if artifact_known != known:
            raise SystemExit(
                f"{path.name}: artifact scores {artifact_known} rows inside "
                f"the train class set, regenerated labels give {known}")
        checked.append({"file": str(path.relative_to(REPO_ROOT)),
                        "control_correct": artifact_control,
                        "known_rows": artifact_known})
    return checked


def top(counter: Counter, model_dir: str, count: int) -> list[dict[str, Any]]:
    """The most frequent labels, decoded, so a reader can sanity-check them."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    return [{"token_id": token, "count": n,
             "decoded": tokenizer.decode([token])}
            for token, n in counter.most_common(count)]


def analyse(args: argparse.Namespace) -> dict[str, Any]:
    """Both control estimators on the regenerated evaluation rows."""
    corpus = Path(args.corpus)
    provenance = verify_corpus(corpus)
    train, held = label_partitions(args.model, corpus, args.seq_len,
                                   args.train_blocks, args.eval_blocks)
    train_counts, eval_counts = Counter(train), Counter(held)
    total = len(held)
    code_token = train_counts.most_common(1)[0][0]
    paper_token, paper_correct = eval_counts.most_common(1)[0]
    code_correct = eval_counts[code_token]
    known = sum(n for token, n in eval_counts.items() if token in train_counts)
    cells = chaff_free_cells(args.seq_len, total)
    return {
        "schema": SCHEMA, "corpus": provenance, "model": args.model,
        "seq_len": args.seq_len, "train_blocks": args.train_blocks,
        "eval_blocks": args.eval_blocks, "eval_rows": total,
        "known_eval_rows": known,
        "code_control": {"token_id": code_token, "correct": code_correct,
                         "pct": 100.0 * code_correct / total,
                         "estimator": "mode(train labels), scored on eval rows",
                         "source": "attacker/attacks/latent_probe.py:215-217"},
        "paper_control": {"token_id": paper_token, "correct": paper_correct,
                          "pct": 100.0 * paper_correct / total,
                          "estimator": "argmax over eval-row label counts",
                          "source": "papers/arxiv-draft/main.tex:153"},
        "control_shift_rows": paper_correct - code_correct,
        "control_shift_pp": 100.0 * (paper_correct - code_correct) / total,
        "excess_shift_pp": -100.0 * (paper_correct - code_correct) / total,
        "estimators_agree": paper_token == code_token,
        "top_eval_labels": top(eval_counts, args.model, args.top),
        "top_train_labels": top(train_counts, args.model, args.top),
        "cells_cross_checked": cross_check(cells, code_correct, known, total),
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", help="the campaign corpus, sha256-verified")
    ap.add_argument("--model", help="tokenizer directory")
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--train-blocks", type=int, default=256)
    ap.add_argument("--eval-blocks", type=int, default=256)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--output", help="report JSON path")
    ap.add_argument("--self-test", action="store_true")
    return ap


def self_test() -> int:
    """Checks that touch neither the corpus nor a tokenizer."""
    manifest = json.loads(MANIFEST.read_text())
    cells = chaff_free_cells(32, 8192)
    controls = {json.loads(p.read_text())["summary"][0]
                ["label_free_majority_pct"] for p in cells}
    checks = [
        ("manifest carries the corpus identity",
         len(manifest["sha256"]) == 64 and manifest["bytes"] > 0),
        ("chaff-free cells found", len(cells) == 26),
        ("and they share one control", controls == {4.84619140625}),
    ]
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return 0 if all(ok for _, ok in checks) else 1


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        return self_test()
    for name in ("corpus", "model"):
        if not getattr(args, name):
            raise SystemExit(f"--{name} is required")
    report = analyse(args)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    code, paper = report["code_control"], report["paper_control"]
    print(f"corpus sha256 verified against {report['corpus']['manifest']}")
    print(f"eval rows                : {report['eval_rows']}")
    print(f"code control  (train mode): token {code['token_id']}  "
          f"{code['correct']} rows = {code['pct']:.6f}%")
    print(f"paper control (eval mode) : token {paper['token_id']}  "
          f"{paper['correct']} rows = {paper['pct']:.6f}%")
    print(f"control shift            : {report['control_shift_rows']:+d} rows "
          f"= {report['control_shift_pp']:+.6f} pp")
    print(f"EXCESS SHIFT             : {report['excess_shift_pp']:+.6f} pp "
          f"on every cell with this evaluation partition")
    print(f"cells cross-checked      : {len(report['cells_cross_checked'])}")
    for row in report["top_eval_labels"]:
        print(f"   eval  {row['token_id']:>7}  {row['count']:>6}  "
              f"{row['decoded']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
