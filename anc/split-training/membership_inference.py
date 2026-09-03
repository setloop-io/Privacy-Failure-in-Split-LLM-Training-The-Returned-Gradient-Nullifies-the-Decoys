#!/usr/bin/env python3
"""E-A4 membership/property inference on captured training-boundary features.

Input is JSONL, one record per document observation:
  {"document_id": str, "condition": "split_ft"|"fedavg",
   "membership": 0|1, "property": 0|1, "features": [float, ...]}

The attacker split is document-disjoint; metrics are aggregated over three
attacker seeds and bootstrapped at document level. Features are not
manufactured here: capture must come from the matching split-FT/FedAvg run.
"""
import argparse
import hashlib
import json
import math
import os
import random
import statistics
from pathlib import Path


SCHEMA = "dtraining.ea4.membership_property.v1"


def sigmoid(z):
    if z >= 0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)


def standardize(train_x, test_x):
    n = len(train_x[0])
    means = [statistics.fmean(r[j] for r in train_x) for j in range(n)]
    stds = []
    for j in range(n):
        s = statistics.pstdev(r[j] for r in train_x)
        stds.append(s if s > 1e-12 else 1.0)
    conv = lambda rows: [[(r[j] - means[j]) / stds[j] for j in range(n)] for r in rows]
    return conv(train_x), conv(test_x)


def fit_logreg(x, y, seed, steps=1200, lr=0.05, l2=1e-3):
    rng = random.Random(seed)
    w = [rng.uniform(-1e-3, 1e-3) for _ in x[0]]
    b = 0.0
    for _ in range(steps):
        gw = [0.0] * len(w); gb = 0.0
        for row, target in zip(x, y):
            p = sigmoid(sum(a * c for a, c in zip(row, w)) + b)
            d = p - target
            for j, a in enumerate(row): gw[j] += d * a
            gb += d
        inv = 1.0 / len(x)
        for j in range(len(w)):
            w[j] -= lr * (gw[j] * inv + l2 * w[j])
        b -= lr * gb * inv
    return w, b


def scores(x, w, b):
    return [sigmoid(sum(a * c for a, c in zip(r, w)) + b) for r in x]


def roc_auc(y, s):
    pos = [v for v, t in zip(s, y) if t == 1]
    neg = [v for v, t in zip(s, y) if t == 0]
    if not pos or not neg: raise ValueError("metric split lacks both classes")
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def tpr_at_fpr(y, s, max_fpr=0.01):
    neg = sorted((v for v, t in zip(s, y) if t == 0), reverse=True)
    pos = [v for v, t in zip(s, y) if t == 1]
    allowed = math.floor(max_fpr * len(neg))
    threshold = math.inf if allowed == 0 else neg[allowed - 1]
    return sum(v > threshold for v in pos) / len(pos)


def split_docs(records, seed, frac=0.6):
    docs = sorted({str(r["document_id"]) for r in records})
    rng = random.Random(seed); rng.shuffle(docs)
    cut = max(1, min(len(docs) - 1, round(len(docs) * frac)))
    train_docs, test_docs = set(docs[:cut]), set(docs[cut:])
    return ([r for r in records if str(r["document_id"]) in train_docs],
            [r for r in records if str(r["document_id"]) in test_docs],
            train_docs, test_docs)


def validate(records):
    if len(records) < 20: raise ValueError("need at least 20 document observations")
    width = len(records[0].get("features", []))
    if width == 0: raise ValueError("empty feature vector")
    for r in records:
        if r.get("condition") not in {"split_ft", "fedavg"}: raise ValueError("bad condition")
        if r.get("membership") not in {0, 1} or r.get("property") not in {0, 1}:
            raise ValueError("labels must be binary")
        if len(r.get("features", [])) != width: raise ValueError("feature width mismatch")
        if not all(math.isfinite(float(v)) for v in r["features"]): raise ValueError("non-finite feature")
    return width


def bootstrap(y, s, seed, n=1000):
    rng = random.Random(seed); idx = list(range(len(y))); vals = []
    for _ in range(n):
        pick = [rng.choice(idx) for _ in idx]
        yy = [y[i] for i in pick]; ss = [s[i] for i in pick]
        if len(set(yy)) == 2: vals.append(roc_auc(yy, ss))
    vals.sort()
    return [vals[int(.025 * (len(vals)-1))], vals[int(.975 * (len(vals)-1))]]


def evaluate(records, label, seeds):
    runs = []
    for seed in seeds:
        tr, te, trd, ted = split_docs(records, seed)
        if trd & ted: raise AssertionError("document leakage")
        xtr, xte = standardize([r["features"] for r in tr], [r["features"] for r in te])
        ytr = [r[label] for r in tr]; yte = [r[label] for r in te]
        if len(set(ytr)) < 2 or len(set(yte)) < 2: raise ValueError(f"{label}: split lacks both classes")
        w, b = fit_logreg(xtr, ytr, seed)
        sc = scores(xte, w, b)
        runs.append({"seed": seed, "train_documents": len(trd), "test_documents": len(ted),
                     "roc_auc": roc_auc(yte, sc), "tpr_at_1pct_fpr": tpr_at_fpr(yte, sc),
                     "roc_auc_bootstrap_95ci": bootstrap(yte, sc, seed + 10000)})
    return {"runs": runs,
            "roc_auc_mean": statistics.fmean(r["roc_auc"] for r in runs),
            "roc_auc_std": statistics.pstdev(r["roc_auc"] for r in runs),
            "tpr_at_1pct_fpr_mean": statistics.fmean(r["tpr_at_1pct_fpr"] for r in runs)}


def run(paths, output, seeds):
    blobs = [(str(path), Path(path).read_bytes()) for path in paths]
    records = [json.loads(line) for _, raw in blobs
               for line in raw.decode().splitlines() if line.strip()]
    width = validate(records)
    result = {"schema": SCHEMA, "experiment": "E-A4 membership/property inference",
              "input": {"files": [{"path": path, "sha256": hashlib.sha256(raw).hexdigest()}
                                     for path, raw in blobs],
                        "records": len(records), "feature_width": width},
              "seeds": seeds, "conditions": {}}
    for condition in ("split_ft", "fedavg"):
        subset = [r for r in records if r["condition"] == condition]
        if not subset: raise ValueError(f"missing condition {condition}")
        result["conditions"][condition] = {
            "membership": evaluate(subset, "membership", seeds),
            "property": evaluate(subset, "property", seeds)}
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {output}")


def self_test():
    rng = random.Random(7); records = []
    for condition in ("split_ft", "fedavg"):
        for i in range(80):
            m, p = i % 2, (i // 2) % 2
            records.append({"document_id": f"{condition}-{i}", "condition": condition,
                            "membership": m, "property": p,
                            "features": [m + rng.gauss(0, .2), p + rng.gauss(0, .2), rng.random()]})
    validate(records)
    for label in ("membership", "property"):
        out = evaluate([r for r in records if r["condition"] == "split_ft"], label, [0,1,2])
        assert out["roc_auc_mean"] > .9
    print("SELF-TEST PASSED")


def main():
    ap = argparse.ArgumentParser(description="E-A4 feature-based membership/property attacker")
    ap.add_argument("--input", nargs="+", help="captured boundary-feature JSONL files")
    ap.add_argument("--output", default="ea4_membership_property.json")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test: return self_test()
    if not args.input: ap.error("--input is required unless --self-test")
    run(args.input, args.output, args.seeds)


if __name__ == "__main__": main()
