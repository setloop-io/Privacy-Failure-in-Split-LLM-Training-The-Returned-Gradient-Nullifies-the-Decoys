#!/usr/bin/env python3
"""Extract the public base model's input embedding table as a [vocab, H] .pt.

The adversary holds the public base weights, so this table is part of the
adversary view; bin/leakage_metrics.py --embeddings consumes it for the
semantic_cosine metric. Reads the safetensors shards directly (no model load).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="model directory")
    ap.add_argument("--output", required=True)
    ap.add_argument("--key", default="model.embed_tokens.weight")
    args = ap.parse_args()

    from safetensors import safe_open
    model = Path(args.model)
    index_path = model / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        shard = index["weight_map"].get(args.key)
        if shard is None:
            raise SystemExit(f"{args.key} not in the weight map")
        shards = [model / shard]
    else:
        shards = sorted(model.glob("*.safetensors"))
    for shard_path in shards:
        with safe_open(str(shard_path), framework="pt") as handle:
            if args.key in handle.keys():
                import torch
                tensor = handle.get_tensor(args.key)
                out = Path(args.output)
                out.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"embeddings": tensor.float(), "key": args.key,
                            "model": str(model)}, out)
                print(f"wrote {out} {tuple(tensor.shape)}")
                return 0
    raise SystemExit(f"{args.key} not found in {model}")


if __name__ == "__main__":
    sys.exit(main())
