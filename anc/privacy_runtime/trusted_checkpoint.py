"""Trusted-side boundary recording for the complete transcript (experiment W3.3).

The cloud side of every session is already snapshotted per optimizer step by
``LatentServer.snapshot`` in ``split-training/latent_cloud_server.py``; this
module records the trusted side: the ``TLNPrivateBoundary`` whose encoder
produces every released frame and whose decoder consumes every cloud return.

Design: ONE full ``state_dict`` snapshot at session start plus ONE sha256
usage-trace entry per training step appended to a single JSONL sidecar -- a
32-byte commitment per quantity, not a copy.  Each entry binds

  state_sha256   the full state_dict at the moment the step's frames were
                 produced (the defender optimizer trains TLN, so the state
                 drifts from the snapshot)
  encode_sha256  the protected-encode call's output (``tln.encode``)
  decode_sha256  the protected-decode call's input and output
                 (``tln.decode``)
  trace_sha256   sha256 over (step, the three hashes above)

The snapshot is self-verifying: ``state_sha256`` recomputes from the stored
tensors.  Recording consumes no RNG draws, so a run with the recorder on
reproduces a run without it bit for bit.  Nonces are never written: they
seed the DP noise streams, and a transcript carrying them would let the
cloud subtract its own noise.  Scope is the training loop: eval and
post-training probe releases never reach the cloud trainer.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import nn

SNAPSHOT_SCHEMA = "dtraining.trusted_boundary_checkpoint.v1"
USAGE_SCHEMA = "dtraining.trusted_boundary_usage.v1"
SNAPSHOT_FILENAME = "trusted_boundary_state.pt"
USAGE_FILENAME = "trusted_boundary_usage.jsonl"


def _labeled_digest(items: Iterable[tuple[str, bytes]]) -> str:
    """sha256 over length-framed (label, payload) pairs."""
    digest = hashlib.sha256()
    for label, payload in items:
        label_bytes = label.encode()
        digest.update(len(label_bytes).to_bytes(4, "big"))
        digest.update(label_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _tensor_bytes(tensor: "torch.Tensor") -> bytes:
    """Little-endian float32 value bytes, detached from device and graph."""
    import torch

    return (tensor.detach().to(device="cpu", dtype=torch.float32)
            .contiguous().numpy().astype("<f4", copy=False).tobytes())


def _tensor_items(label: str, tensor: "torch.Tensor") -> list[tuple[str, bytes]]:
    """Frame one tensor as labelled payloads binding dtype and shape."""
    return [
        (f"{label}.dtype", str(tensor.dtype).encode()),
        (f"{label}.shape", json.dumps(list(tensor.shape)).encode()),
        (f"{label}.values", _tensor_bytes(tensor)),
    ]


def state_items(state: dict) -> list[tuple[str, bytes]]:
    """Digest input for a state_dict: every key, dtype, shape and value."""
    items: list[tuple[str, bytes]] = []
    for key in sorted(state):
        items.extend(_tensor_items(f"state.{key}", state[key]))
    return items


def state_digest(module: "nn.Module") -> str:
    """sha256 over a module's full state_dict, in sorted key order."""
    return _labeled_digest(state_items(module.state_dict()))


class TrustedBoundaryRecorder:
    """Session-start snapshot plus per-step usage-trace hashes for one run.

    Two files total under ``out_dir`` (a fresh directory per run):

      trusted_boundary_state.pt     one full state_dict snapshot
      trusted_boundary_usage.jsonl  one sha256 entry per training step
    """

    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.snapshot_path = self.out_dir / SNAPSHOT_FILENAME
        self.usage_path = self.out_dir / USAGE_FILENAME
        self.entries = 0
        self._snapshotted = False

    def snapshot(self, module: "nn.Module",
                 meta: dict | None = None) -> Path:
        """Write the module's full state_dict once, before any training step."""
        import torch

        if self.snapshot_path.exists() or self.usage_path.exists():
            raise RuntimeError(
                f"trusted checkpoint directory {self.out_dir} already "
                "carries a recording; each run needs a fresh directory")
        state = module.state_dict()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": SNAPSHOT_SCHEMA,
            "module": type(module).__name__,
            "recorded_at_step": 0,
            "state_dict": {key: value.detach().to("cpu").clone()
                           for key, value in state.items()},
            "state_sha256": _labeled_digest(state_items(state)),
            "meta": dict(meta or {}),
        }, self.snapshot_path)
        self._snapshotted = True
        return self.snapshot_path

    def record_step(self, step: int, module: "nn.Module", *,
                    encode_output: "torch.Tensor",
                    decode_input: "torch.Tensor",
                    decode_output: "torch.Tensor") -> None:
        """Append one usage-trace entry for one training step.

        Call after the step's protected-decode and before the defender
        optimizer steps, so the recorded state is the one that produced
        this step's frames.
        """
        if not self._snapshotted:
            raise RuntimeError("snapshot() must run before record_step()")
        entry = {
            "schema": USAGE_SCHEMA,
            "step": int(step),
            "state_sha256": state_digest(module),
            "encode_sha256": _labeled_digest(
                _tensor_items("encode.protected_latent", encode_output)),
            "decode_sha256": _labeled_digest(
                _tensor_items("decode.input", decode_input)
                + _tensor_items("decode.output", decode_output)),
        }
        entry["trace_sha256"] = _labeled_digest(
            (name, str(entry[name]).encode()) for name in
            ("step", "state_sha256", "encode_sha256", "decode_sha256"))
        with self.usage_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.entries += 1


def self_test() -> int:
    """Exercise the recorder end to end on a real TLNPrivateBoundary."""
    import tempfile

    try:
        import torch
    except ImportError:
        print("SELF-TEST FAILED: torch is not importable. Run inside "
              "split-inference:spark; W3.3 is NOT verified without it.")
        return 1

    from privacy_runtime.latent_native import (LatentPrivacyConfig,
                                               build_latent_native_split)

    def check(condition: bool, message: str) -> None:
        if not condition:
            raise AssertionError(message)

    torch.manual_seed(0)
    config = LatentPrivacyConfig(hidden_dim=64, latent_dim=16, cloud_layers=1,
                                 cloud_heads=4, noise_multiplier=0.35,
                                 gradient_clip_norm=0.01,
                                 gradient_noise_multiplier=0.35)
    tln, _, _ = build_latent_native_split(config, vocab_size=8)
    hidden = torch.randn(1, 8, 64)
    optimizer = torch.optim.AdamW(tln.parameters(), lr=1e-2)

    with tempfile.TemporaryDirectory() as directory:
        recorder = TrustedBoundaryRecorder(directory)
        recorder.snapshot(tln, meta={"seed": 0})
        check([path.name for path in Path(directory).iterdir()]
              == [SNAPSHOT_FILENAME],
              "the snapshot must be the only file at session start")

        saved = torch.load(recorder.snapshot_path)
        check(saved["schema"] == SNAPSHOT_SCHEMA, "snapshot schema")
        check(saved["state_sha256"] == state_digest(tln),
              "snapshot digest must match the live module")
        check(saved["state_sha256"]
              == _labeled_digest(state_items(saved["state_dict"])),
              "snapshot digest must recompute from the saved file alone")
        for key, value in saved["state_dict"].items():
            check(torch.equal(value, tln.state_dict()[key].detach().cpu()),
                  f"snapshot tensor {key} differs from the module")

        for step in range(3):
            nonce = f"{step:032x}"
            latent, _ = tln.encode(hidden, nonce)
            restored, _ = tln.decode(latent, nonce, residual=hidden)
            recorder.record_step(step, tln, encode_output=latent,
                                 decode_input=latent, decode_output=restored)
            optimizer.zero_grad(set_to_none=True)
            restored.sum().backward()
            optimizer.step()

        entries = [json.loads(line) for line in
                   recorder.usage_path.read_text().splitlines()]
        check(len(entries) == 3, "three steps must write three entries")
        check([entry["step"] for entry in entries] == [0, 1, 2],
              "entries must carry the step index")
        check(entries[0]["state_sha256"] == saved["state_sha256"],
              "step 0 must be the snapshotted state")
        check(entries[1]["state_sha256"] != entries[0]["state_sha256"],
              "the optimizer step between entries must show in the state hash")
        for entry in entries:
            rebuilt = _labeled_digest(
                (name, str(entry[name]).encode()) for name in
                ("step", "state_sha256", "encode_sha256", "decode_sha256"))
            check(rebuilt == entry["trace_sha256"],
                  "trace_sha256 must re-derive from the components")

        latent, _ = tln.encode(hidden, f"{len(entries):032x}")
        check(_labeled_digest(_tensor_items("x", latent))
              != _labeled_digest(_tensor_items("x", latent + 1e-6)),
              "a changed value must change the digest")
        flat = torch.arange(8, dtype=torch.float32)
        check(_labeled_digest(_tensor_items("x", flat))
              != _labeled_digest(_tensor_items("x", flat.reshape(2, 4))),
              "a reshaped tensor must change the digest")

        try:
            TrustedBoundaryRecorder(directory).snapshot(tln)
        except RuntimeError as error:
            check("already carries" in str(error),
                  "a stale directory must be rejected")
        else:
            raise AssertionError("a stale directory must be rejected")
        try:
            TrustedBoundaryRecorder(Path(directory) / "nested").record_step(
                0, tln, encode_output=latent, decode_input=latent,
                decode_output=latent)
        except RuntimeError as error:
            check("snapshot()" in str(error),
                  "record_step before snapshot must fail")
        else:
            raise AssertionError("record_step before snapshot must fail")

    print("SELF-TEST PASSED")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(self_test() if "--self-test" in sys.argv else 2)
