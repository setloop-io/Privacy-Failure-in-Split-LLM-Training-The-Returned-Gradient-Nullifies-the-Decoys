#!/usr/bin/env python3
"""Shared wire-capture / pair-loading core for all attacks.

Two sidecar schemas, selected by --mode (pinned by the deployed writers so
the framework and the harnesses cannot drift apart):

  inference (split-inference capture hook; the inference tree is not
      included in this release):
      {"session_id", "request_seq", "phase", "position", "epoch"}
      ("epoch" added by the ER ratchet arms; older captures may lack it)
  training (split-training/er_ratchet.py SIDECAR_KEYS):
      {"session_id", "mb_id", "phase", "step", "epoch"}

Every capture is a wire_NNNN.pt tensor + wire_NNNN.json sidecar pair.
Loading .pt needs torch; the schema/grouping logic is pure python so
--help/--self-test work torch-less (repo rule).
"""

import glob
import json
import os

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

INFERENCE_SIDECAR_KEYS = {"session_id", "request_seq", "phase", "position",
                          "epoch"}
INFERENCE_SIDECAR_KEYS_LEGACY = {"session_id", "request_seq", "phase",
                                 "position"}  # pre-ratchet captures
TRAINING_SIDECAR_KEYS = {"session_id", "mb_id", "phase", "step", "epoch"}

SCHEMAS = {"inference": INFERENCE_SIDECAR_KEYS,
           "training": TRAINING_SIDECAR_KEYS}


def schema_keys(mode):
    return SCHEMAS[mode]


def validate_sidecar(meta, mode, path="<sidecar>"):
    """Raise ValueError on schema drift; None epoch is allowed (ratchet off).
    Inference legacy captures without 'epoch' are accepted and normalized to
    epoch=None."""
    keys = set(meta.keys())
    want = SCHEMAS[mode]
    if mode == "inference" and keys == INFERENCE_SIDECAR_KEYS_LEGACY:
        meta = dict(meta)
        meta["epoch"] = None
        return meta
    if keys != want:
        raise ValueError(f"{path}: sidecar keys {sorted(keys)} != {mode} "
                         f"schema {sorted(want)}")
    return meta


def scan_captures(capture_dir, mode):
    """Return [(meta, pt_path), ...] sorted by the mode's alignment key.
    Pure python (no tensor loading): meta dicts are validated against the
    mode schema. Alignment order:
      training:  (step, mb_id)     — rank within step == mb order
      inference: (session_id, request_seq, position)"""
    records = []
    for jf in sorted(glob.glob(os.path.join(capture_dir, "wire_*.json"))):
        pt = jf[:-len(".json")] + ".pt"
        if not os.path.exists(pt):
            continue
        with open(jf) as f:
            meta = validate_sidecar(json.load(f), mode, jf)
        records.append((meta, pt))
    if mode == "training":
        records.sort(key=lambda r: (r[0].get("step", 0),
                                    r[0].get("mb_id", 0)))
    else:
        records.sort(key=lambda r: (r[0].get("session_id", ""),
                                    r[0].get("request_seq", 0),
                                    r[0].get("position", 0)))
    return records


def load_tensor(pt_path):
    """Load one capture tensor as float32 CPU. Torch-only."""
    if torch is None:
        raise RuntimeError("torch is required to load .pt captures")
    return torch.load(pt_path, map_location="cpu").float()


def rows_2d(t):
    """Flatten a capture to [n_rows, H]: [b, s, H] / [s, H] / [H] -> rows."""
    if t.dim() == 1:
        return t.unsqueeze(0)
    if t.dim() == 2:
        return t
    return t.reshape(-1, t.shape[-1])


def group_by_epoch(records, phase=None):
    """{epoch: [(meta, pt_path), ...]} for one wire phase (None = all).
    Preserves the scan order within each epoch."""
    out = {}
    for meta, pt in records:
        if phase is not None and meta.get("phase") != phase:
            continue
        out.setdefault(meta.get("epoch"), []).append((meta, pt))
    return out


def load_epoch_rows(records, phase=None):
    """{epoch: wire_rows [n, H]} concatenated per epoch. Torch-only."""
    out = {}
    for ep, items in group_by_epoch(records, phase).items():
        out[ep] = rows_2d(torch.cat(
            [rows_2d(load_tensor(pt)) for _, pt in items]))
    return out


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    print("capture schemas (pure python):")
    tr = {"session_id": "s", "mb_id": 3, "phase": "fwd", "step": 2,
          "epoch": 1}
    check("training sidecar validates",
          validate_sidecar(dict(tr), "training") == tr)
    inf = {"session_id": "s", "request_seq": 1, "phase": "decode",
           "position": 5, "epoch": 0}
    check("inference sidecar validates",
          validate_sidecar(dict(inf), "inference") == inf)
    legacy = {"session_id": "s", "request_seq": 1, "phase": "prefill",
              "position": 0}
    check("legacy inference sidecar normalized to epoch=None",
          validate_sidecar(dict(legacy), "inference")["epoch"] is None)
    try:
        validate_sidecar({"session_id": "s", "mb_id": 0}, "training")
        check("schema drift raises", False)
    except ValueError:
        check("schema drift raises", True)
    check("schemas round-trip through JSON unchanged",
          json.loads(json.dumps(tr)) == tr)

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        metas = [
            {"session_id": "s0", "mb_id": 1, "phase": "fwd", "step": 0,
             "epoch": 0},
            {"session_id": "s0", "mb_id": 0, "phase": "fwd", "step": 0,
             "epoch": 0},
            {"session_id": "s0", "mb_id": 0, "phase": "bwd", "step": 1,
             "epoch": 1},
        ]
        for i, m in enumerate(metas):
            with open(os.path.join(d, f"wire_{i:04d}.json"), "w") as f:
                json.dump(m, f)
            with open(os.path.join(d, f"wire_{i:04d}.pt"), "wb") as f:
                f.write(b"\x00")  # scan_captures never reads the tensor
        recs = scan_captures(d, "training")
        check("scan orders by (step, mb_id) not filename",
              [(r[0]["step"], r[0]["mb_id"]) for r in recs]
              == [(0, 0), (0, 1), (1, 0)])
        grp = group_by_epoch(recs, phase="fwd")
        check("group_by_epoch + phase filter",
              sorted(grp.keys()) == [0] and len(grp[0]) == 2)

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
