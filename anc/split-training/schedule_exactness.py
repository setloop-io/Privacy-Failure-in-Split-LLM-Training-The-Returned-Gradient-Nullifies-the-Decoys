#!/usr/bin/env python3
"""E-S0: deterministic sync-versus-AFAB/overlap numerical-equivalence test.

The experiment holds initialization, data, optimizer, and microbatch order
fixed and varies only gradient scheduling. It includes tied and untied output
weights because tied embedding/lm_head parameters receive both head-side and
tail-side gradient streams whose floating-point accumulation order differs.
"""

import argparse
import copy
import json
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # --help remains usable without torch
    torch = nn = F = None


class ToySplit(nn.Module if nn is not None else object):
    def __init__(self, vocab, hidden, tied):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.cloud = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU())
        self.tail = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        if tied:
            self.lm_head.weight = self.embed.weight


def one_step(model, batches, schedule, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    opt.zero_grad(set_to_none=True)
    pending = []
    for ids, labels in batches:
        h = model.head(model.embed(ids))
        boundary = h.detach().requires_grad_(True)
        cloud_out = model.cloud(boundary)
        cloud_leaf = cloud_out.detach().requires_grad_(True)
        logits = model.lm_head(model.tail(cloud_leaf))
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               labels.reshape(-1)) / len(batches)
        loss.backward()
        torch.autograd.backward(cloud_out, cloud_leaf.grad)
        if schedule == "sync":
            torch.autograd.backward(h, boundary.grad)
        else:
            pending.append((h, boundary.grad.detach().clone()))
    if schedule == "afab":
        for h, grad in pending:
            torch.autograd.backward(h, grad)
    opt.step()


def compare(seed, accum, tied, vocab, hidden, seq_len, lr):
    torch.manual_seed(seed)
    base = ToySplit(vocab, hidden, tied)
    sync, afab = copy.deepcopy(base), copy.deepcopy(base)
    g = torch.Generator().manual_seed(seed + 1000)
    batches = []
    for _ in range(accum):
        block = torch.randint(0, vocab, (1, seq_len + 1), generator=g)
        batches.append((block[:, :-1], block[:, 1:]))
    one_step(sync, batches, "sync", lr)
    one_step(afab, batches, "afab", lr)
    rows = []
    for (name_a, a), (name_b, b) in zip(sync.state_dict().items(),
                                        afab.state_dict().items()):
        assert name_a == name_b
        diff = (a - b).abs().float()
        rows.append({"parameter": name_a,
                     "torch_equal": bool(torch.equal(a, b)),
                     "max_abs_diff": float(diff.max()),
                     "mean_abs_diff": float(diff.mean())})
    return {"seed": seed, "grad_accum": accum, "tied_embeddings": tied,
            "all_torch_equal": all(x["torch_equal"] for x in rows),
            "max_abs_diff": max(x["max_abs_diff"] for x in rows),
            "parameters": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--grad-accum", type=int, nargs="+", default=[1, 2, 8])
    ap.add_argument("--vocab", type=int, default=257)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--output", default="schedule_exactness.json")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if torch is None:
        ap.error("torch is required; --help works without it")
    if args.self_test:
        args.seeds, args.grad_accum = [42], [1, 2]
        args.vocab, args.hidden, args.seq_len = 31, 16, 8
    runs = [compare(seed, accum, tied, args.vocab, args.hidden,
                    args.seq_len, args.lr)
            for tied in (False, True)
            for accum in args.grad_accum for seed in args.seeds]
    result = {
        "schema": "dtraining.schedule_exactness.v1",
        "experiment": "E-S0 sync versus AFAB numerical equivalence",
        "config": {"measurement_kind": "measured", "seeds": args.seeds,
                   "grad_accum": args.grad_accum, "vocab": args.vocab,
                   "hidden": args.hidden, "seq_len": args.seq_len,
                   "lr": args.lr, "device": "cpu"},
        "evidence_status": "supporting",
        "known_limitations": [
            "Toy CPU test isolates accumulation order; a real-model remote test is still required for a deployment headline.",
            "AFAB ordering is emulated without network serialization because serialization is not expected to change arithmetic."
        ],
        "runs": runs,
        "all_torch_equal": all(r["all_torch_equal"] for r in runs),
        "all_close_1e_6": all(r["max_abs_diff"] <= 1e-6 for r in runs),
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    if args.self_test:
        assert len(runs) == 4
        print("SELF-TEST PASSED")
    else:
        print(json.dumps({"all_torch_equal": result["all_torch_equal"],
                          "all_close_1e_6": result["all_close_1e_6"]}))


if __name__ == "__main__":
    main()
