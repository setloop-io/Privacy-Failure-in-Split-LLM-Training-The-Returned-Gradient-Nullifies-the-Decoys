#!/usr/bin/env python3
"""Split fine-tuning prototype — pipeline-split training over WAN.

Companion to the split-INFERENCE study. Local node owns
embedding + layers 0..SA + layers RA..end + final norm + LM head + data +
labels + loss; the "cloud" owns the middle layers SA+1..RA-1. Here
the cloud runs IN-PROCESS on the same machine behind the
`CloudWorker` class boundary — the exact seam where it moves onto the
existing binary WebSocket link later (see PROTOCOL.md).

Forward/backward across the boundary (synchronous, GPipe-style
all-forward-all-backward within a step):

    local head fwd -> h.detach().requires_grad_(True)         [boundary act]
    -> CloudWorker.forward(h) -> local tail fwd -> CE loss
    -> loss.backward()  (local tail+head grads; h.grad = dL/dh)
    -> CloudWorker.backward(h.grad)  (cloud layer grads)
    -> optimizers step (local AdamW + cloud AdamW)

Supports: full FT of cloud layers, LoRA on local layers (experiment 5),
gradient accumulation, per-microbatch timing JSON, bf16/fp16, seeds,
--smoke (2 microbatches, 1 step), --toy (tiny random model for CPU
verification on hosts without an HF checkpoint).

Usage:
    python split_trainer.py --help                      # works without torch
    python split_trainer.py --toy --smoke               # CPU smoke test
    python split_trainer.py --model ~/experiments/models/qwen3-0.6b --smoke
    python split_trainer.py --model ... --split-after 4 --resume-after 23 \
        --lora-rank 16 --micro-batch-size 1 --grad-accum 8 --steps 100
    python split_trainer.py --model ... --cloud ws://ucn:5003   # M2: remote
"""

import argparse
import hashlib
import json
import os
import random
import secrets
import time
from collections import deque
from datetime import datetime

# torch / transformers are imported lazily-guarded so `--help` works on
# hosts without them.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - torch-less host
    torch = None
    nn = None
    F = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None

try:  # only needed for --cloud (remote CloudWorker over WS, M2)
    import struct as _struct
    from websockets.sync.client import connect as _ws_connect
    from websockets.exceptions import ConnectionClosed as _WSClosed
except ImportError:  # pragma: no cover
    _struct = None
    _ws_connect = None
    _WSClosed = Exception

# [ER] per-epoch boundary rotation (E-R7). Same-dir import works torch-less.
try:
    from er_ratchet import (EpochRatchet, warn_if_weak_seed,
                            SIDECAR_KEYS as _ER_SIDECAR_KEYS)
except ImportError:  # pragma: no cover - run from another cwd
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from er_ratchet import (EpochRatchet, warn_if_weak_seed,
                                SIDECAR_KEYS as _ER_SIDECAR_KEYS)
    except ImportError:
        EpochRatchet = None
        warn_if_weak_seed = None
        _ER_SIDECAR_KEYS = None

try:
    from dp_sgd import LocalDPSGD, conservative_zcdp_epsilon
except ImportError:  # pragma: no cover
    LocalDPSGD = conservative_zcdp_epsilon = None

try:
    from privacy_runtime.activation_dp import (BidirectionalBoundaryDP,
                                               noise_for_rho,
                                               rho_for_epsilon)
except ImportError:  # pragma: no cover
    BidirectionalBoundaryDP = None
    noise_for_rho = rho_for_epsilon = None


# [ER] Wire capture (training analog of the E9 inference capture): env
# ER_CAPTURE_DIR; every boundary crossing dumps the RAW rotated wire tensor
# plus a wire_NNNN.json sidecar (schema pinned in er_ratchet.SIDECAR_KEYS).
_ER_CAPTURE_DIR = os.environ.get("ER_CAPTURE_DIR")


def _er_capture(tensor, session_id, mb_id, phase, step, epoch):
    if not _ER_CAPTURE_DIR or torch is None:
        return
    os.makedirs(_ER_CAPTURE_DIR, exist_ok=True)
    n = len([f for f in os.listdir(_ER_CAPTURE_DIR)
             if f.startswith("wire_") and f.endswith(".pt")])
    torch.save(tensor.detach().cpu(),
               os.path.join(_ER_CAPTURE_DIR, f"wire_{n:04d}.pt"))
    meta = {"session_id": session_id, "mb_id": mb_id, "phase": phase,
            "step": step, "epoch": epoch}
    assert set(meta.keys()) == _ER_SIDECAR_KEYS
    with open(os.path.join(_ER_CAPTURE_DIR, f"wire_{n:04d}.json"), "w") as mf:
        json.dump(meta, mf)


def _er_rot(x, M):
    """[ER] Rotation seam: fp32 matmul, cast back to the tensor's dtype —
    the same convention as the inference defense (h @ W out, @ W^T back)."""
    return (x.float() @ M.to(x.device)).to(x.dtype)

# Training status reporting for the spark dashboard (best-effort, atomic).
STATUS_FILE = os.environ.get(
    "TRAINING_STATUS_FILE", "/home/geo/experiments/results/training_status.json")
_status_state = {}
_status_last_write = 0.0


def _write_training_status(**fields):
    """Merge fields into the dashboard training status JSON.

    Atomic (tmp + os.replace), throttled to one write per 5s unless the new
    state is done/failed, and never fatal to the experiment."""
    global _status_last_write
    now = time.time()
    force = fields.get("state") in ("done", "failed")
    if not force and now - _status_last_write < 5:
        _status_state.update(fields)
        return
    _status_last_write = now
    _status_state.update(fields)
    _status_state["updated"] = datetime.now().astimezone().isoformat()
    try:
        os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
        tmp_path = STATUS_FILE + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(_status_state, f)
        os.replace(tmp_path, STATUS_FILE)
    except Exception:
        pass


# Tiny inline corpus (TEXT_SAMPLES convention). Extend with --corpus-file.
TEXT_SAMPLES = [
    "The process of photosynthesis converts carbon dioxide and water into glucose and oxygen using sunlight energy.",
    "The French Revolution began in 1789 with the storming of the Bastille and ended with Napoleon Bonaparte's rise to power.",
    "Quantum mechanics describes the behavior of matter and energy at the atomic and subatomic level.",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b",
    "The patient presents with elevated blood pressure of 160/95 mmHg, persistent headaches, and occasional dizziness.",
    "Hey, I was wondering if you could help me understand how neural networks work? What exactly is backpropagation?",
    "Q3 2025 revenue reached $4.2 billion, up 18% year-over-year, driven by strong cloud services growth of 32%.",
    "The mitochondria is often called the powerhouse of the cell because it generates most of the cell's ATP supply.",
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.",
    "Climate change poses significant risks to global food security through rising temperatures and extreme weather.",
]


# Toy model: a tiny GPT-like nn.Module exposing the same stage interface as
# the HF path (embed_tokens / layers / norm / lm_head), so the full training
# loop can be verified on CPU without an HF checkpoint.
if nn is not None:

    class ToyBlock(nn.Module):
        def __init__(self, hidden, nheads=4):
            super().__init__()
            self.norm1 = nn.LayerNorm(hidden)
            self.attn = nn.MultiheadAttention(hidden, nheads, batch_first=True)
            self.norm2 = nn.LayerNorm(hidden)
            self.ffn = nn.Sequential(
                nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Linear(4 * hidden, hidden))

        def forward(self, x, **_kwargs):
            h = self.norm1(x)
            t = h.shape[1]
            mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=h.device), 1)
            a, _ = self.attn(h, h, h, need_weights=False, attn_mask=mask)
            x = x + a
            x = x + self.ffn(self.norm2(x))
            return x

    class ToyCausalLM(nn.Module):
        def __init__(self, vocab=128, hidden=64, n_layers=4):
            super().__init__()
            self.vocab_size = vocab
            self.embed_tokens = nn.Embedding(vocab, hidden)
            self.layers = nn.ModuleList([ToyBlock(hidden) for _ in range(n_layers)])
            self.norm = nn.LayerNorm(hidden)
            self.lm_head = nn.Linear(hidden, vocab, bias=False)
            self.rotary_emb = None  # no position embeddings needed

        def encode(self, texts, seq_len):
            """Byte-level 'tokenizer': ord(c) % vocab, chunked to seq_len."""
            ids = [ord(c) % self.vocab_size for t in texts for c in t]
            n = (len(ids) // (seq_len + 1)) * (seq_len + 1)
            ids = ids[:n]
            return [torch.tensor(ids[i:i + seq_len + 1], dtype=torch.long)
                    for i in range(0, n, seq_len + 1)]


# LoRA: minimal self-contained implementation (no peft dependency). Applied
# to local-stage linear projections for experiment 5 (LoRA-local-only).
if nn is not None:

    class LoRALinear(nn.Module):
        """Frozen base linear + trainable low-rank residual BA."""

        def __init__(self, base, rank, alpha):
            super().__init__()
            self.base = base
            for p in self.base.parameters():
                p.requires_grad_(False)
            in_f, out_f = base.in_features, base.out_features
            dev, dt = base.weight.device, base.weight.dtype
            self.lora_A = nn.Linear(in_f, rank, bias=False).to(device=dev, dtype=dt)
            self.lora_B = nn.Linear(rank, out_f, bias=False).to(device=dev, dtype=dt)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
            nn.init.zeros_(self.lora_B.weight)
            self.scaling = alpha / rank

        def forward(self, x):
            return self.base(x) + self.lora_B(self.lora_A(x)) * self.scaling


def apply_lora(module, rank, alpha, targets=("q_proj", "v_proj")):
    """Replace direct-child linears whose name ends with a target by LoRALinear."""
    n = 0
    for parent in module.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and any(name.endswith(t) for t in targets):
                setattr(parent, name, LoRALinear(child, rank, alpha))
                n += 1
    return n


# CloudWorker: the trust-boundary seam. Today it holds the middle layers
# in-process; every method maps 1:1 onto a WS message type when it moves to a
# remote cloud_trainer_server (see PROTOCOL.md):
#   forward  -> "forward_with_graph"  (activation in, activation out, graph kept server-side)
#   backward -> "backward_grad"       (dL/d(output) in, dL/d(input) out; grads stay server-side)
#   step/zero_grad -> "optimizer_step" / folded into backward
#
# Wire-accurate autograd emulation (two detached boundary leaves):
#   h_head (local graph A) -> boundary_out = h_head.detach().requires_grad_()
#   cloud_out (cloud graph B) -> boundary_in = cloud_out.detach().requires_grad_()
#   loss.backward()                       # local tail grads + boundary_in.grad
#   cloud.backward(boundary_in.grad)      # cloud param grads; boundary_out.grad = dL/d(input)
#   autograd.backward(h_head, boundary_out.grad)   # local head/embed grads
class CloudWorker:
    def __init__(self, layers, lr, trainable=True, grad_hook=None, ratchet=None):
        self.layers = nn.ModuleList(list(layers))
        self.grad_hook = grad_hook  # compression/noise seam for experiment 3
        self.ratchet = ratchet      # [ER] EpochRatchet or None
        self._mb_next = 0           # [ER] capture mb counter
        if not trainable:
            for p in self.layers.parameters():
                p.requires_grad_(False)
        params = [p for p in self.layers.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=lr) if params else None
        self._last_input = None
        self._last_output = None
        self._last_epoch = None
        self._last_exchange_id = None

    def forward(self, hidden, layer_kwargs, step=None):
        """Run middle layers; keep the autograd graph on the cloud side.

        [ER] With a ratchet, emulate the remote fold in-process: the wire
        tensor is h @ W_t, the cloud computes canonically (fold == unwrap /
        run / rewrap, exact by covariant_fold.unfold_check), and the rotated
        return is unwrapped at the seam. The round-trips run in fp32 like the
        deployed seam, so the loss curve is comparable to the remote arm."""
        mb_id = self._mb_next
        self._mb_next += 1
        leaf = hidden
        if self.ratchet is not None:
            rows = hidden.shape[0] * hidden.shape[1]
            exchange_id = ("in-process", mb_id)
            if getattr(self.ratchet, "count_all_training_directions", False):
                directions = (("activation_request", "activation_response")
                              if step is None else
                              ("activation_request", "activation_response",
                               "gradient_request", "gradient_response"))
                ep = self.ratchet.reserve_exchange(
                    rows, exchange_id, phase="evaluation" if step is None else "train",
                    directions=directions)
                self._last_exchange_id = exchange_id
            else:
                ep = self.ratchet.advance(rows)
            W, Wt = self.ratchet.epoch_W(ep, hidden.shape[-1])
            self._last_epoch = (ep, W, Wt)
            with torch.no_grad():
                h_wire = _er_rot(hidden, W)          # what crosses the wire
                _er_capture(h_wire, "in-process", mb_id, "fwd", step, ep)
                leaf = _er_rot(h_wire, Wt).detach().requires_grad_(True)
        self._last_input = leaf
        for layer in self.layers:
            out = layer(leaf, **layer_kwargs)
            leaf = out[0] if isinstance(out, tuple) else out
        self._last_output = leaf
        if self.ratchet is not None:
            ep, W, Wt = self._last_epoch
            with torch.no_grad():
                out_wire = _er_rot(leaf, W)          # rotated return tensor
                result = _er_rot(out_wire, Wt)       # unwrapped at the seam
            if step is None and self._last_exchange_id is not None:
                self.ratchet.complete_exchange(self._last_exchange_id)
                self._last_exchange_id = None
            return result
        return leaf

    def backward(self, grad_output, step=None):
        """Backprop dL/d(cloud output) through the cloud layers; return
        dL/d(cloud input) so the local side can finish its own backward.
        Runs even when layers are frozen (input grad still needed).

        [ER] grad_output is w.r.t. the UNWRAPPED seam tensor; the wire grad
        is dL/d(rotated output) = grad @ W, and the returned grad is
        unwrapped with @ W^T (chain rule through the rotation)."""
        if self._last_output is None:
            return None
        grad_can = grad_output
        if self.ratchet is not None:
            ep, W, Wt = self._last_epoch
            with torch.no_grad():
                grad_wire = _er_rot(grad_output, W)  # dL/d(rotated output)
                _er_capture(grad_wire, "in-process", self._mb_next - 1,
                            "bwd", step, ep)
                grad_can = _er_rot(grad_wire, Wt)    # dL/d(canonical output)
        if self.grad_hook is not None:
            grad_can = self.grad_hook(grad_can)
        torch.autograd.backward(self._last_output, grad_tensors=grad_can)
        grad_input = self._last_input.grad
        self._last_input = self._last_output = None
        if self.ratchet is not None:
            ep, W, Wt = self._last_epoch
            self._last_epoch = None
            if self._last_exchange_id is not None:
                self.ratchet.complete_exchange(self._last_exchange_id)
                self._last_exchange_id = None
            with torch.no_grad():
                # dL/d(wire input) = dL/d(canonical input) @ W; unwrap @ W^T
                return _er_rot(_er_rot(grad_input, W), Wt)
        return grad_input

    def zero_grad(self):
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

    def step(self):
        if self.optimizer is not None:
            self.optimizer.step()

    def param_counts(self):
        total = sum(p.numel() for p in self.layers.parameters())
        trainable = sum(p.numel() for p in self.layers.parameters() if p.requires_grad)
        return total, trainable

    def close(self):
        pass


# RemoteCloudWorker (M2): same interface as CloudWorker, but the middle layers
# live on a remote cloud_trainer_server (PROTOCOL.md). Boundary tensors cross
# the wire in --wire-dtype (default fp16): values are CAST AT THE SEAM, so a
# bf16 local graph sees fp16-rounded boundary values and gradients. With
# --wire-dtype bf16 and a bf16 model the round-trip is exact; fp16 trades a
# ~1e-3 relative perturbation of boundary values for half the bytes and
# comparability with the fp16 inference study.
WIRE_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16,
               "fp32": torch.float32} if torch is not None else {}


def normalize_cloud_url(u):
    """Accept ws://host:5003, http://host:5002, or bare host; always return
    the ws:// URL on port 5003 (http 5002 + 1, like jacobi_server derives)."""
    if "://" in u:
        scheme, rest = u.split("://", 1)
        host = rest.split(":")[0].split("/")[0]
        if scheme in ("http", "https"):
            return f"ws://{host}:5003"
        if ":" in rest:
            return f"ws://{rest.rstrip('/')}"
        return f"ws://{host}:5003"
    return f"ws://{u.split(':')[0]}:5003"


class RemoteCloudWorker:
    def __init__(self, url, sa, ra, lr, trainable, wire_dtype="fp16",
                 grad_noise_sigma=0.0, ratchet=None,
                 obf_seed_preprovisioned=False,
                 tee_attest=False, tee_emulated=False):
        if _ws_connect is None:
            raise RuntimeError("websockets not installed; needed for --cloud")
        self.wire_dtype = WIRE_DTYPES[wire_dtype]
        self._wire_name = wire_dtype
        self.ratchet = ratchet    # [ER] EpochRatchet or None
        self._mb_epoch = {}       # [ER] mb_id -> ratchet epoch at send time
        self._mb_phase = {}
        # Fold/refold of a 7B middle stack is a synchronous GPU operation and
        # may legitimately exceed the websockets default ping timeout. The
        # application already blocks waiting for each explicit response and
        # fails loud on socket closure, so transport keepalive pings add false
        # disconnects without improving failure detection.
        self.ws = _ws_connect(normalize_cloud_url(url),
                              max_size=512 * 1024 * 1024,
                              ping_interval=None,
                              close_timeout=10)
        hello = {"op": "hello", "split_after": sa, "resume_after": ra,
                 "lr": lr, "trainable": trainable, "wire_dtype": wire_dtype}
        if grad_noise_sigma:
            hello["grad_compression"] = {"kind": "noise",
                                         "sigma": grad_noise_sigma}
        if tee_attest or tee_emulated:
            from tee_enclave import GPUConfidentialCompute
            self._tee_verifier = GPUConfidentialCompute(mode="emulated" if tee_emulated else "auto")
            hello["tee_attest"] = True
            hello["tee_emulated"] = tee_emulated
            hello["tee_nonce"] = secrets.token_hex(16)
            hello["tee_client_pub"] = self._tee_verifier.generate_ecdh_keypair().hex()
        if ratchet is not None:
            # [ER] the server derives the same W_t from the seed chain and
            # folds its middle layers per epoch (E-R7 fold mode); it never
            # receives W itself.
            if obf_seed_preprovisioned:
                hello["obf_enabled"] = True
            else:
                hello["obf_seed_base"] = ratchet.seed_base
            hello["obf_ratchet_version"] = ratchet.version
            hello["obf_transform_mode"] = ratchet.transform_mode
        self.ws.send(json.dumps(hello))
        ack = json.loads(self.ws.recv())
        if ack.get("op") == "error":
            raise RuntimeError(f"cloud rejected hello: {ack.get('error')}")
        if ratchet is not None and ack.get("obf_ratchet_version") != ratchet.version:
            raise RuntimeError("[ER] cloud ratchet-version acknowledgement "
                               f"mismatch: local={ratchet.version!r}, "
                               f"cloud={ack.get('obf_ratchet_version')!r}")
        if ratchet is not None and ack.get("obf_transform_mode") != ratchet.transform_mode:
            raise RuntimeError("[ER] cloud transform-mode acknowledgement "
                               f"mismatch: local={ratchet.transform_mode!r}, "
                               f"cloud={ack.get('obf_transform_mode')!r}")
        self.tee_channel = None
        if tee_attest:
            report = ack.get("tee_attestation_report")
            if not report:
                raise RuntimeError("[TEE] Cloud worker returned NO attestation report!")
            from tee_enclave import TEEEncryptedChannel
            session_key = self._tee_verifier.verify_remote_report(
                report, bytes.fromhex(hello["tee_nonce"]), self._tee_verifier._private_key
            )
            self.tee_channel = TEEEncryptedChannel(session_key)
            print(f"[TEE] Attestation PASSED for remote GPU '{report.get('gpu_device')}' (ECDH AES-256-GCM Channel Active)")
        if ack.get("cloud_start") != sa + 1 or ack.get("cloud_end") != ra - 1:
            self.ws.close()
            raise RuntimeError(
                f"layer-range mismatch: trainer wants {sa + 1}..{ra - 1}, "
                f"server serves {ack.get('cloud_start')}..{ack.get('cloud_end')}")
        self.session_id = ack["session_id"]
        self._counts = (ack["cloud_params"], ack["cloud_trainable_params"])
        self._mb_next = 0
        self._pending = []  # FIFO of mb_ids awaiting backward (sync path)
        self._box = {}      # (op, mb_id) -> (header, payload) response mailbox
        print(f"[cloud] session {self.session_id[:8]} on "
              f"{normalize_cloud_url(url)} — layers {ack['cloud_start']}.."
              f"{ack['cloud_end']}, {self._counts[0] / 1e6:.1f}M params "
              f"({self._counts[1] / 1e6:.1f}M trainable), wire={wire_dtype}")

    # -- framing helpers (mirror cloud_trainer_server.py) --
    def _pack_frame(self, header, payload=b""):
        if self.tee_channel is not None and payload:
            header["tee_encrypted"] = True
            payload = self.tee_channel.encrypt_payload(payload)
        h = json.dumps(header).encode()
        return _struct.pack(">I", len(h)) + h + payload

    def _unpack_frame(self, message):
        header_len = _struct.unpack(">I", message[:4])[0]
        header = json.loads(message[4:4 + header_len])
        payload = message[4 + header_len:]
        if self.tee_channel is not None:
            if payload and not header.get("tee_encrypted"):
                raise RuntimeError("TEE channel active but cloud response is plaintext")
            if payload:
                payload = self.tee_channel.decrypt_payload(payload)
        elif header.get("tee_encrypted"):
            raise RuntimeError("cloud response claims TEE encryption without a negotiated channel")
        return header, payload

    def _pack(self, t):
        t = t.detach().to(self.wire_dtype).contiguous().cpu()
        if self.wire_dtype == torch.bfloat16:
            return t.view(torch.int16).numpy().tobytes()
        return t.numpy().tobytes()

    def _unpack(self, buf, shape):
        if self.wire_dtype == torch.bfloat16:
            t = torch.frombuffer(bytearray(buf), dtype=torch.int16).view(torch.bfloat16)
        else:
            t = torch.frombuffer(bytearray(buf), dtype=self.wire_dtype)
        return t.reshape(shape).clone()

    # -- response correlation (M2b): every binary response carries the mb_id
    # of its request, so a dispatch loop can file responses into mailboxes and
    # callers can wait on ANY outstanding mb_id — this is what makes the
    # pipelined (overlap) schedule possible on a single WS connection.
    def _read_into_mailbox(self):
        try:
            msg = self.ws.recv()
        except _WSClosed as e:
            raise RuntimeError(
                f"cloud connection lost with {len(self._box)} replies "
                f"in mailbox and requests in flight (server error mid-flight? "
                f"check cloud_trainer_server log): {e}") from None
        if isinstance(msg, str):
            data = json.loads(msg)
            if data.get("op") == "error":
                raise RuntimeError(f"cloud error: {data.get('error')}")
            self._box[("text", data.get("op"))] = data
            return
        header, payload = self._unpack_frame(msg)
        if header.get("op") == "error":
            raise RuntimeError(f"cloud error: {header.get('error')}")
        self._box[(header["op"], header.get("mb_id"))] = (header, payload)

    def _wait(self, op, mb_id):
        key = (op, mb_id)
        while key not in self._box:
            self._read_into_mailbox()
        return self._box.pop(key)

    # -- async (pipelined) API: explicit mb_ids, send and wait separately --
    def send_forward(self, mb_id, hidden, layer_kwargs, step=None, block_indices=None):
        header = {"op": "forward_with_graph", "mb_id": mb_id,
                  "hidden_shape": list(hidden.shape), "dtype": self._wire_name}
        if step is not None:
            header["step"] = step
        if block_indices is not None:
            header["block_indices"] = block_indices
        if self.ratchet is not None:
            # [ER] advance the ratchet (each boundary forward adds the full
            # microbatch sequence, prefill-style) and wrap the wire tensor.
            rows = hidden.shape[0] * hidden.shape[1]
            if getattr(self.ratchet, "count_all_training_directions", False):
                directions = (("activation_request", "activation_response")
                              if step is None else
                              ("activation_request", "activation_response",
                               "gradient_request", "gradient_response"))
                ep = self.ratchet.reserve_exchange(
                    rows, mb_id, phase="evaluation" if step is None else "train",
                    directions=directions)
                self._mb_phase[mb_id] = "evaluation" if step is None else "train"
            else:
                ep = self.ratchet.advance(rows)
            self._mb_epoch[mb_id] = ep
            hidden = self.ratchet.apply(hidden, ep)
            header["obf_epoch"] = ep
        payload = self._pack(hidden)
        pe = layer_kwargs.get("position_embeddings")
        if pe is not None:
            header["has_pos_emb"] = True
            header["pe_shape"] = list(pe[0].shape)
            payload += self._pack(pe[0]) + self._pack(pe[1])
        self.ws.send(self._pack_frame(header, payload))

    def recv_forward(self, mb_id, dtype, device="cpu"):
        header, payload = self._wait("forward_result", mb_id)
        out = self._unpack(payload, header["hidden_shape"])
        if self.ratchet is not None:
            # [ER] the wire carries the ROTATED cloud output y @ W_t; unwrap
            # for the canonical local tail (y = y_rot @ W_t^T).
            out = self.ratchet.inverse(out, self._mb_epoch[mb_id])
            if self._mb_phase.get(mb_id) == "evaluation":
                self.ratchet.complete_exchange(mb_id)
                self._mb_phase.pop(mb_id, None)
                self._mb_epoch.pop(mb_id, None)
        return out.to(device=device, dtype=dtype)  # cast at the seam

    def send_backward(self, mb_id, grad_output, step=None):
        header = {"op": "backward_grad", "mb_id": mb_id,
                  "grad_shape": list(grad_output.shape), "dtype": self._wire_name}
        if step is not None:
            header["step"] = step
        if self.ratchet is not None:
            # [ER] grad_output is w.r.t. the canonical tail input; the wire
            # grad is w.r.t. the rotated cloud output: y_rot = y @ W implies
            # dL/dy_rot = dL/dy @ W.
            ep = self._mb_epoch[mb_id]
            grad_output = self.ratchet.apply(grad_output, ep)
            header["obf_epoch"] = ep
        self.ws.send(self._pack_frame(header, self._pack(grad_output)))

    def recv_backward(self, mb_id, dtype, device="cpu"):
        header, payload = self._wait("backward_result", mb_id)
        grad_input = self._unpack(payload, header["grad_shape"])
        if self.ratchet is not None:
            # [ER] the wire grad is w.r.t. the rotated boundary h @ W_t;
            # h_rot = h @ W implies dL/dh = dL/dh_rot @ W^T.
            ep = self._mb_epoch.pop(mb_id)
            grad_input = self.ratchet.inverse(grad_input, ep)
            if self._mb_phase.pop(mb_id, None) == "train":
                self.ratchet.complete_exchange(mb_id)
        return grad_input.to(device=device, dtype=dtype)

    # -- CloudWorker interface (sync path: strict request->response order) --
    def forward(self, hidden, layer_kwargs, step=None):
        mb_id = self._mb_next
        self._mb_next += 1
        self.send_forward(mb_id, hidden, layer_kwargs, step=step)
        out = self.recv_forward(mb_id, hidden.dtype, hidden.device)
        if step is not None:
            self._pending.append(mb_id)
        return out

    def backward(self, grad_output, step=None):
        mb_id = self._pending.pop(0)
        self.send_backward(mb_id, grad_output, step=step)
        return self.recv_backward(mb_id, grad_output.dtype, grad_output.device)

    def zero_grad(self):
        pass  # server zeroes inside optimizer_step

    def step(self):
        self.ws.send(json.dumps({"op": "optimizer_step"}))
        self._wait("text", "step_ack")

    def weights_snapshot(self):
        """[data-split] fetch the cloud's current middle-layer weights.

        The cloud trains these on the public stream only, so nothing private
        crosses back.  Call only when no forward/backward is in flight.
        Returns {name: tensor} in the wire dtype.
        """
        self.ws.send(json.dumps({"op": "weights_snapshot"}))
        state = {}
        while True:
            msg = self.ws.recv()
            if isinstance(msg, str):
                done = json.loads(msg)
                if done.get("op") == "weights_done":
                    if done.get("count") != len(state):
                        raise RuntimeError(
                            f"weights snapshot count mismatch: "
                            f"{done.get('count')} announced vs {len(state)} received")
                    return state
                raise RuntimeError(
                    f"unexpected control frame during weights snapshot: {done}")
            header, payload = self._unpack_frame(msg)
            if header.get("op") != "weights_part":
                raise RuntimeError(
                    f"unexpected frame during weights snapshot: {header.get('op')}")
            state[header["name"]] = self._unpack(payload, header["shape"])

    def param_counts(self):
        return self._counts

    def close(self):
        try:
            self.ws.send(json.dumps({"op": "close"}))
            self.ws.close()
        except Exception:
            pass


def unique_params(modules, require_grad=True):
    """Trainable parameters across `modules`, deduplicated by identity.

    A weight shared between two modules is ONE tensor reached by two paths.
    Flattening `m.parameters()` over a module list yields it twice, and
    torch.optim does not deduplicate within a param group: the optimizer then
    updates it twice per step, sharing one state entry, so it moves ~2x as far
    and its Adam step counter advances 2x (breaking bias correction).

    This is not hypothetical here. qwen3-0.6b and qwen35-2b set
    tie_word_embeddings=True, so embed.weight IS lm_head.weight, and every
    local-parameter list in this repo enumerates both. `bin/run_e4.sh` runs
    qwen3-0.6b. Measured on a single tied tensor, 3 steps at lr 0.1:
    single entry -> -0.3 with step count 3; duplicated -> -0.6 with step
    count 6.

    Order is preserved (first occurrence wins) so optimizer state ordering
    stays deterministic across runs.
    """
    seen, out = set(), []
    for m in modules:
        for p in m.parameters():
            if require_grad and not p.requires_grad:
                continue
            if id(p) in seen:
                continue
            seen.add(id(p))
            out.append(p)
    return out


def unique_params_self_test():
    """Torch-less checks on unique_params (tied-weight duplication).

    Uses stub modules so this runs on a host without torch. The real
    tied-checkpoint case: qwen3-0.6b reports embed.weight IS lm_head.weight.
    """
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    class P:
        def __init__(self, requires_grad=True):
            self.requires_grad = requires_grad

    class M:
        def __init__(self, *ps):
            self._ps = ps

        def parameters(self):
            return iter(self._ps)

    shared, a, b = P(), P(), P()
    embed, mid, lm_head = M(shared, a), M(b), M(shared)   # shared reached twice

    got = unique_params([embed, mid, lm_head])
    check("a weight reached by two modules is returned once",
          len(got) == 3 and sum(1 for p in got if p is shared) == 1)
    check("first-occurrence order preserved",
          got[0] is shared and got[1] is a and got[2] is b)

    frozen = P(requires_grad=False)
    check("requires_grad=False excluded by default",
          frozen not in unique_params([M(frozen, a)]))
    check("require_grad=False includes frozen params",
          len(unique_params([M(frozen, a)], require_grad=False)) == 2)

    # Identity, not equality: two DISTINCT tensors must both survive, or a
    # dedupe by value would silently drop real parameters.
    x, y = P(), P()
    check("distinct-but-equal params are NOT collapsed",
          len(unique_params([M(x), M(y)])) == 2)

    check("empty input is empty output", unique_params([]) == [])

    print("SELF-TEST PASSED" if ok else "SELF-TEST FAILED")
    return ok


def param_digest(named_modules, cloud=None):
    """sha256 per parameter tensor, keyed by module-qualified name.

    Deduplicated by identity for the same reason unique_params is: a tied
    weight is ONE tensor and must be digested once, or a schedule comparison
    would report a spurious difference in the duplicate slot.

    Digests rather than raw tensors so the result is small enough to commit as
    an artifact and to diff by eye. sha256 over the exact bytes, so this is a
    bit-exactness test -- not allclose.
    """
    import hashlib
    seen, out = set(), {}
    def add(prefix, module):
        for name, p in module.named_parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            t = p.detach().cpu().contiguous()
            h = hashlib.sha256(t.numpy().tobytes()).hexdigest()
            out[f"{prefix}.{name}" if name else prefix] = h
    for prefix, m in named_modules:
        add(prefix, m)
    if cloud is not None and hasattr(cloud, "layers"):
        add("cloud", cloud.layers)
    return out


def build_modules(args):
    """Return (embed, layers, norm, lm_head, rotary_emb, encode_fn)."""
    if args.toy:
        model = ToyCausalLM()
        model.to(args.device)
        return (model.embed_tokens, model.layers, model.norm, model.lm_head,
                None, model.encode)

    if AutoModelForCausalLM is None:
        raise RuntimeError("transformers/torch not installed; use --toy for a CPU test")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    # proven CPU-then-.to(device) loader (the accelerate device_map variant
    # died silently mid-shard-load on the Qwen3-Next hybrid; transient peak
    # ~1.6x weights)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(args.device)
    model.train()
    core = model.model  # Qwen3ForCausalLM and friends
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # [data-split] kept so train() can save the merged checkpoint when
    # --save-cloud-weights is set (the cloud's public-trained layers are
    # merged back into this object by reference).
    args._model_ref = model
    args._tokenizer_ref = tokenizer

    def encode(texts, seq_len):
        ids = []
        for t in texts:
            ids.extend(tokenizer(t, add_special_tokens=False)["input_ids"])
        n = (len(ids) // (seq_len + 1)) * (seq_len + 1)
        ids = ids[:n]
        return [torch.tensor(ids[i:i + seq_len + 1], dtype=torch.long)
                for i in range(0, n, seq_len + 1)]

    return (core.embed_tokens, core.layers, core.norm, model.lm_head,
            getattr(core, "rotary_emb", None), encode)


def make_layer_kwargs(rotary, hidden, position_ids, args):
    """Per-microbatch extras for decoder layers. sdpa/FA2 treat
    attention_mask=None as causal (is_causal path); eager needs a 4D mask."""
    kwargs = {"attention_mask": None, "position_ids": position_ids}
    impl = getattr(args, "attn_impl", "sdpa")
    if impl == "eager":
        bsz, seq = hidden.shape[0], hidden.shape[1]
        mask = torch.full((seq, seq), torch.finfo(hidden.dtype).min,
                          dtype=hidden.dtype, device=hidden.device)
        mask = torch.triu(mask, diagonal=1)[None, None, :, :].expand(bsz, 1, seq, seq)
        kwargs["attention_mask"] = mask
    if rotary is not None:
        kwargs["position_embeddings"] = rotary(hidden, position_ids)
    return kwargs


def run_layer_stack(layers, hidden, layer_kwargs):
    for layer in layers:
        out = layer(hidden, **layer_kwargs)
        hidden = out[0] if isinstance(out, tuple) else out
    return hidden


def run_sublayer_stack(layers, hidden, layer_kwargs, sublayers: tuple):
    """Run only the named residual sublayers of each decoder layer, in order.

    A Qwen3 decoder layer is two residual sublayers and nothing else:

        hidden = hidden + self_attn(input_layernorm(hidden))            "attn"
        hidden = hidden + mlp(post_attention_layernorm(hidden))         "mlp"

    so `sublayers=("attn", "mlp")` reproduces `run_layer_stack` exactly and
    either name alone runs half of every layer.  The layernorms are pre-norm
    and each sublayer owns its own, so dropping one sublayer must also drop
    that sublayer's layernorm -- keeping it would apply a norm the residual
    stream never sees in the reference model.
    """
    if not sublayers or not set(sublayers) <= {"attn", "mlp"}:
        raise ValueError("sublayers must be a non-empty subset of (attn, mlp)")
    for layer in layers:
        if "attn" in sublayers:
            out = layer.self_attn(hidden_states=layer.input_layernorm(hidden),
                                  **layer_kwargs)
            hidden = hidden + (out[0] if isinstance(out, tuple) else out)
        if "mlp" in sublayers:
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    return hidden


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.expanduser(
        "~/experiments/models/qwen3-0.6b"), help="HF model path (ignored with --toy)")
    ap.add_argument("--toy", action="store_true",
                    help="use a tiny random built-in model (CPU verification)")
    ap.add_argument("--split-after", type=int, default=4,
                    help="local keeps layers 0..SA (inclusive)")
    ap.add_argument("--resume-after", type=int, default=None,
                    help="local keeps layers RA..end (default: n_layers-2)")
    ap.add_argument("--freeze-cloud", action="store_true",
                    help="cloud middle layers frozen (experiment 5 uses this with --lora-rank)")
    ap.add_argument("--lora-rank", type=int, default=0,
                    help=">0: LoRA on local-stage q/v projections (experiment 5)")
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--micro-batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4,
                    help="microbatches per optimizer step")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device", default="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    ap.add_argument("--attn-impl", choices=["sdpa", "eager"], default="sdpa")
    ap.add_argument("--corpus-file", default=None,
                    help="extra training text, one document per line")
    ap.add_argument("--save-cloud-weights", default=None,
                    help="[data-split] after training, fetch the cloud's "
                         "middle-layer weights, merge them into the local "
                         "model, and save the assembled checkpoint + "
                         "tokenizer to this path")
    ap.add_argument("--ea4-members-only", action="store_true",
                    help="train only on the first half of corpus documents for E-A4")
    ap.add_argument("--ea4-checkpoint", default=None,
                    help="save embedding + input-side layers for E-A4 capture")
    ap.add_argument("--cloud", default=None,
                    help="remote cloud_trainer_server: ws://host:5003, "
                         "http://host:5002, or bare host (default: in-process)")
    ap.add_argument("--wire-dtype", choices=["fp16", "bf16", "fp32"],
                    default=None,
                    help="dtype of boundary tensors on the wire (remote only); default: same as --dtype (exact round-trip)")
    ap.add_argument("--grad-noise-sigma", type=float, default=0.0,
                    help="DP-style noise on boundary grad, sigma x grad RMS "
                         "(experiment 3 seam; applied cloud-side)")
    ap.add_argument("--dp-sgd", action="store_true",
                    help="E-R9 per-example DP-SGD on private local layers")
    ap.add_argument("--dp-max-grad-norm", type=float, default=1.0)
    ap.add_argument("--dp-noise-multiplier", type=float, default=1.2)
    ap.add_argument("--dp-delta", type=float, default=1e-6)
    ap.add_argument("--boundary-dp", action="store_true",
                    help="production-v3 formal replace-one Gaussian DP on "
                         "outbound activations and outbound boundary gradients")
    ap.add_argument("--boundary-dp-forward-clip", type=float, default=1.0)
    ap.add_argument("--boundary-dp-forward-noise", type=float, default=22.0)
    ap.add_argument("--boundary-dp-return-clip", type=float, default=1.0)
    ap.add_argument("--boundary-dp-return-noise", type=float, default=22.0)
    ap.add_argument("--boundary-dp-delta", type=float, default=1e-6)
    ap.add_argument("--boundary-dp-target-epsilon", type=float, default=None,
                    help="derive directional noise multipliers for the exact "
                         "planned release count and this composed epsilon")
    ap.add_argument("--boundary-dp-forward-rho-fraction", type=float,
                    default=0.6,
                    help="fraction of the target zCDP rho allocated to "
                         "forward releases (remainder protects returns)")
    ap.add_argument("--boundary-norm-audit", action="store_true",
                    help="diagnostic only: record aggregate per-token boundary "
                         "norm statistics on trusted TLN; these private-data "
                         "statistics must not calibrate a formal DP claim")
    ap.add_argument("--pipeline", choices=["sync", "overlap"], default="sync",
                    help="sync = strict request->response per microbatch "
                         "(M1/M2 behavior); overlap = async boundary crossings "
                         "so local work of other microbatches hides the RTT "
                         "(experiment 4; requires --cloud)")
    ap.add_argument("--max-inflight", type=int, default=2,
                    help="max cloud forwards outstanding in overlap mode; "
                         "bounds cloud pending-graph memory. 1 == sync schedule")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--obf-seed-base", type=int, default=None,
                    help="[ER] 128-bit master S for the portable v2 ratchet; "
                         "required to enable per-epoch "
                         "boundary rotation (E-R7 fold mode)")
    ap.add_argument("--obf-seed-file", default=None,
                    help="read private seed from a mode-0600 file (preferred)")
    ap.add_argument("--obf-seed-preprovisioned", action="store_true",
                    help="remote cloud already has the same private seed; "
                         "do not transmit it in the WebSocket hello")
    ap.add_argument("--obf-ratchet-tokens", type=int, default=0,
                    help="[ER] boundary tokens per epoch for ratcheted "
                         "rotation (0=off); each training forward adds "
                         "seq_len x micro_batch_size (prefill-style)")
    ap.add_argument("--obf-budget-events", type=int, default=0,
                    help="[ER] evidence budget B for budget-triggered rotation "
                         "(0=off); served boundary tokens +1 per token. "
                         "When --obf-seed-base is set without N or B, defaults to B=128 (half the E-R1a max_safe_epoch=256, depths 1/4/8, 12B+27B)")
    ap.add_argument("--obf-count-all-directions", action="store_true",
                    help="production-v3: reserve activation request/response "
                         "and gradient request/response rows before each "
                         "training exchange; evaluation reserves both "
                         "activation directions")
    ap.add_argument("--obf-ratchet-version", choices=["v1", "v2"],
                    default="v2",
                    help="portable v2 is mandatory for production; v1 exists "
                         "only to reproduce historical evidence")
    ap.add_argument("--obf-transform-mode",
                    choices=["dense", "dense_sandwich",
                             "structured_hadamard"], default="dense",
                    help="dense covariant weight fold or experimental fast "
                         "signed-permuted Hadamard wire sandwich; the fast "
                         "mode requires attacker revalidation")
    ap.add_argument("--allow-unsafe-oversized-ratchet-forward",
                    action="store_true",
                    help="UNSAFE historical replay only: permit one forward "
                         "to exceed the declared per-key pair cadence")
    ap.add_argument("--act-noise-sigma", type=float, default=0.0,
                    help="relative-RMS forward activation DP noise scale (sigma_act)")
    ap.add_argument("--tee-attest", action="store_true",
                    help="require hardware GPU Confidential Compute (TEE) attestation report from cloud worker")
    ap.add_argument("--tee-emulated", action="store_true",
                    help="allow emulated GPU Confidential Compute attestation for lab testing")
    ap.add_argument("--smoke", action="store_true",
                    help="2 microbatches, 1 step, seq-len 32 (CPU-runnable)")
    ap.add_argument("--output", default="split_training_results.json")
    ap.add_argument("--param-digest", action="store_true",
                    help="include a per-parameter sha256 in the result JSON. "
                         "Bit-exactness evidence for issue #45; compare two "
                         "runs with bin/check_bitexact.sh, which handles the "
                         "loopback cloud server and the restart between arms.")
    ap.add_argument("--self-test", action="store_true",
                    help="verify unique_params dedupes shared weights "
                         "(issue #45); torch-less, no model needed")
    args = ap.parse_args()

    if args.obf_seed_base is not None and args.obf_seed_file:
        ap.error("use only one of --obf-seed-base and --obf-seed-file")
    if args.obf_ratchet_tokens and args.obf_budget_events:
        ap.error("choose only one of --obf-ratchet-tokens and --obf-budget-events")
    if args.obf_seed_file:
        if os.stat(args.obf_seed_file).st_mode & 0o077:
            ap.error("--obf-seed-file must be mode 0600")
        with open(args.obf_seed_file) as f:
            args.obf_seed_base = int(f.read().strip())
    if args.obf_seed_preprovisioned and not args.obf_seed_file:
        ap.error("--obf-seed-preprovisioned requires --obf-seed-file")
    if args.obf_seed_preprovisioned and not args.cloud:
        ap.error("--obf-seed-preprovisioned requires --cloud")
    if args.dp_sgd:
        if LocalDPSGD is None:
            ap.error("dp_sgd.py not importable")
        if not args.freeze_cloud:
            ap.error("--dp-sgd requires --freeze-cloud")
        if args.micro_batch_size != 1:
            ap.error("--dp-sgd requires --micro-batch-size 1")
        if args.pipeline != "sync":
            ap.error("--dp-sgd requires --pipeline sync")
    if args.boundary_dp:
        if BidirectionalBoundaryDP is None:
            ap.error("privacy_runtime.activation_dp is not importable")
        if args.pipeline != "sync":
            ap.error("--boundary-dp requires --pipeline sync")
        for name in ("boundary_dp_forward_clip", "boundary_dp_forward_noise",
                     "boundary_dp_return_clip", "boundary_dp_return_noise"):
            if getattr(args, name) <= 0:
                ap.error(f"--{name.replace('_', '-')} must be positive")
        if not 0 < args.boundary_dp_delta < 1:
            ap.error("--boundary-dp-delta must be in (0,1)")
        if (args.boundary_dp_target_epsilon is not None
                and args.boundary_dp_target_epsilon <= 0):
            ap.error("--boundary-dp-target-epsilon must be positive")
        if not 0 < args.boundary_dp_forward_rho_fraction < 1:
            ap.error("--boundary-dp-forward-rho-fraction must be in (0,1)")

    if args.smoke:
        args.grad_accum = 2
        args.steps = 1
        args.seq_len = min(args.seq_len, 32)
        if args.toy is False:
            args.seq_len = min(args.seq_len, 64)

    if args.self_test:
        raise SystemExit(0 if unique_params_self_test() else 1)

    if torch is None:
        ap.error("torch is not installed; install it or run --help only")

    # [ER] Evidence-based default cadence: E-R1a measured max_safe_epoch=256
    # (50% recovery bracket (256, 1024]) on Mistral-NeMo-12B and Qwen3.6-27B,
    # depths 1/4/8 — rotation_lifetime_{12b,27b}_bf16_20260811T175827.json.
    # Budget-triggered rotation at B=128 applies a 2x safety margin below the
    # measured safe limit (12B pooled cross-epoch accumulation drifts higher
    # than 27B) without a fixed token cadence.
    if (args.obf_seed_base is not None
            and args.obf_ratchet_tokens == 0
            and args.obf_budget_events == 0):
        args.obf_budget_events = 128
        print("[ER] no cadence specified; defaulting to budget-triggered "
              "rotation B=128 (E-R1a max_safe_epoch=256, 2x margin)")
    if args.obf_seed_base is not None:
        cadence = args.obf_budget_events or args.obf_ratchet_tokens
        rows_per_forward = args.micro_batch_size * args.seq_len
        observable_rows = rows_per_forward * (4 if args.obf_count_all_directions else 1)
        if (cadence and observable_rows > cadence
                and not args.allow_unsafe_oversized_ratchet_forward):
            ap.error("production ratchet budget violated before launch: "
                     f"observable_rows={observable_rows} exceeds "
                     f"cadence={cadence}; reduce the forward or explicitly "
                     "select the unsafe historical-replay flag")
    if args.obf_seed_base is not None and EpochRatchet is None:
        ap.error("er_ratchet.py not importable; needed for --obf-seed-base")
    if args.obf_seed_base is not None and warn_if_weak_seed is not None:
        warn_if_weak_seed(args.obf_seed_base, where="split_trainer")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    _write_training_status(state="running", task="split_training",
                           model=args.model if not args.toy else "toy",
                           split_after=args.split_after,
                           resume_after=args.resume_after,
                           lora_rank=args.lora_rank,
                           started=datetime.now().astimezone().isoformat())

    try:
        result = train(args)
    except Exception as e:
        _write_training_status(state="failed", error=str(e))
        raise

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    _write_training_status(state="done", result_file=args.output,
                           final_loss=result["steps"][-1]["loss"])
    print(json.dumps({k: result[k] for k in ("config", "steps")}, indent=2))
    print(f"\nWrote {args.output}")


def train(args):
    embed, layers, norm, lm_head, rotary, encode = build_modules(args)
    n_layers = len(layers)
    if n_layers < 3:
        raise ValueError(f"need >= 3 layers to split, got {n_layers}")
    # Clamp so head/cloud/tail are all non-empty: 0 <= sa, sa+2 <= ra <= n-1.
    sa = max(0, min(args.split_after, n_layers - 3))
    ra = args.resume_after if args.resume_after is not None else n_layers - 2
    ra = min(max(ra, sa + 2), n_layers - 1)
    if (sa, ra) != (args.split_after, args.resume_after if args.resume_after is not None else n_layers - 2):
        print(f"[split] clamped to split-after={sa}, resume-after={ra} "
              f"for {n_layers} layers")

    head_layers = nn.ModuleList(list(layers[: sa + 1]))       # local
    cloud_layers = list(layers[sa + 1: ra])                   # -> CloudWorker
    tail_layers = nn.ModuleList(list(layers[ra:]))            # local

    if args.lora_rank > 0:
        n_head = apply_lora(nn.ModuleList([*head_layers, *tail_layers]),
                            args.lora_rank, args.lora_alpha)
        print(f"[lora] wrapped {n_head} local projections "
              f"(rank={args.lora_rank}, alpha={args.lora_alpha})")

    local_trainables = ([m for m in [embed, *head_layers, *tail_layers, norm, lm_head]])
    local_params = unique_params(local_trainables)
    local_opt = torch.optim.AdamW(local_params, lr=args.lr) if local_params else None
    dp = (LocalDPSGD(local_params, args.dp_max_grad_norm,
                     args.dp_noise_multiplier, args.grad_accum, args.device)
          if args.dp_sgd else None)
    boundary_dp = None

    grad_hook = None
    if args.grad_noise_sigma > 0 and not args.cloud:
        sigma = args.grad_noise_sigma
        def grad_hook(g):  # same defense as the server applies, in-process
            return g + torch.randn_like(g) * (sigma * g.pow(2).mean().sqrt())

    # [ER] per-epoch boundary rotation (E-R7). The local side holds ONLY
    # W_t (seed-chain derived); the cloud folds its middle-layer weights per
    # epoch, so wire tensors stay rotated in both directions.
    ratchet = None
    if args.obf_seed_base is not None:
        ratchet = EpochRatchet(args.obf_seed_base,
                               ratchet_tokens=args.obf_ratchet_tokens,
                               budget_events=args.obf_budget_events,
                               strict_budget=not args.allow_unsafe_oversized_ratchet_forward,
                               version=args.obf_ratchet_version,
                               transform_mode=args.obf_transform_mode)
        ratchet.count_all_training_directions = args.obf_count_all_directions
        print(f"[ER] ratchet enabled: N={args.obf_ratchet_tokens} "
              f"B={args.obf_budget_events} version={args.obf_ratchet_version} "
              f"transform={args.obf_transform_mode} "
              "(private seed redacted)")

    if args.cloud:
        wire_dtype = args.wire_dtype or args.dtype  # default: match model dtype (exact round-trip)
        cloud = RemoteCloudWorker(args.cloud, sa, ra, lr=args.lr,
                                  trainable=not args.freeze_cloud,
                                  wire_dtype=wire_dtype,
                                  grad_noise_sigma=args.grad_noise_sigma,
                                  ratchet=ratchet,
                                  obf_seed_preprovisioned=args.obf_seed_preprovisioned,
                                  tee_attest=args.tee_attest,
                                  tee_emulated=args.tee_emulated)
    else:
        cloud = CloudWorker(cloud_layers, lr=args.lr,
                            trainable=not args.freeze_cloud, grad_hook=grad_hook,
                            ratchet=ratchet)

    texts = list(TEXT_SAMPLES)
    if args.corpus_file:
        with open(args.corpus_file) as f:
            texts.extend(l.strip() for l in f if l.strip())
    if args.ea4_members_only:
        texts = texts[:max(1, len(texts) // 2)]
    n_eval_docs = max(1, len(texts) // 10)
    train_texts = texts[:-n_eval_docs]
    eval_texts = texts[-n_eval_docs:]
    
    train_blocks = encode(train_texts, args.seq_len)
    eval_blocks = encode(eval_texts, args.seq_len)

    if args.boundary_dp:
        forward_noise = args.boundary_dp_forward_noise
        return_noise = args.boundary_dp_return_noise
        if args.boundary_dp_target_epsilon is not None:
            total_rho = rho_for_epsilon(args.boundary_dp_target_epsilon,
                                        args.boundary_dp_delta)
            f_rho = total_rho * args.boundary_dp_forward_rho_fraction
            r_rho = total_rho - f_rho
            rows_per_microbatch = args.micro_batch_size * args.seq_len
            forward_releases = ((args.steps * args.grad_accum + len(eval_blocks))
                                * rows_per_microbatch)
            return_releases = (args.steps * args.grad_accum
                               * rows_per_microbatch)
            forward_noise = noise_for_rho(forward_releases, f_rho)
            return_noise = noise_for_rho(return_releases, r_rho)
            print("[boundary-dp] planned composed epsilon="
                  f"{args.boundary_dp_target_epsilon:g}: "
                  f"forward_sigma={forward_noise:.4f}, "
                  f"return_sigma={return_noise:.4f}, "
                  f"releases={forward_releases}+{return_releases}")
        boundary_dp = BidirectionalBoundaryDP(
            args.boundary_dp_forward_clip, forward_noise,
            args.boundary_dp_return_clip, return_noise,
            args.boundary_dp_delta, adjacency="replace_one")
    
    if len(train_blocks) < args.micro_batch_size:
        raise ValueError(f"corpus too small: {len(train_blocks)} train blocks < micro-batch "
                         f"{args.micro_batch_size}; add --corpus-file or shrink --seq-len")
    
    print(f"[data] {len(texts)} docs -> {len(train_blocks)+len(eval_blocks)} blocks of seq_len+1={args.seq_len + 1} "
          f"(train={len(train_blocks)}, eval={len(eval_blocks)})")

    timing = []
    step_log = []
    rng = random.Random(args.seed)
    act_bytes = grad_bytes = 0

    def _norm_meta(tensor):
        norms = tensor.detach().float().reshape(-1, tensor.shape[-1]).norm(2, dim=1)
        return {"min": float(norms.min()), "median": float(norms.median()),
                "p95": float(torch.quantile(norms, 0.95)),
                "max": float(norms.max()), "rows": int(norms.numel()),
                "privacy_status": "private_diagnostic_not_formal_dp_calibration"}

    if args.pipeline == "overlap" and not args.cloud:
        raise ValueError("--pipeline overlap requires --cloud "
                         "(in-process cloud has no network latency to hide)")

    def _mb_head(step, mb):
        """Local head forward + boundary leaf for one microbatch."""
        rec = {"step": step, "microbatch": mb}
        t_mb = time.perf_counter()
        b_indices = [rng.randrange(len(train_blocks)) for _ in range(args.micro_batch_size)]
        batch = torch.stack([train_blocks[i] for i in b_indices]).to(args.device)
        rec["block_indices"] = b_indices
        input_ids, labels = batch[:, :-1], batch[:, 1:]
        position_ids = torch.arange(input_ids.shape[1], device=args.device).unsqueeze(0)
        t = time.perf_counter()
        hidden = embed(input_ids)
        lk = make_layer_kwargs(rotary, hidden, position_ids, args)
        head_out = run_layer_stack(head_layers, hidden, lk)
        if args.boundary_norm_audit:
            rec["boundary_forward_norms"] = _norm_meta(head_out)
        if getattr(args, "act_noise_sigma", 0.0) > 0.0:
            sigma_frac = args.act_noise_sigma
            rms = head_out.detach().float().pow(2).mean(dim=-1, keepdim=True).sqrt()
            noise = torch.randn_like(head_out.float()) * (sigma_frac * rms)
            head_out = (head_out.float() + noise).to(head_out.dtype)
        if boundary_dp is not None:
            head_out, dp_meta = boundary_dp.protect_forward(head_out)
            rec["boundary_dp_forward"] = dp_meta
        rec["t_local_fwd"] = time.perf_counter() - t
        boundary_out = head_out.detach().requires_grad_(True)
        return rec, {"rec": rec, "t_mb": t_mb, "labels": labels, "lk": lk,
                     "head_out": head_out, "boundary_out": boundary_out}

    for step in range(args.steps):
        local_opt and local_opt.zero_grad(set_to_none=True)
        cloud.zero_grad()
        step_loss = 0.0
        t0 = time.perf_counter()
        blocked = 0.0   # time actually waiting on the cloud
        rtt_sum = 0.0   # sum of cloud round-trip times (serial estimate)

        if args.pipeline == "overlap":
            # GPipe-AFAB with async boundary crossings (DESIGN.md §4): issue
            # cloud fwd(mb i+1) before consuming fwd(mb i); local tail/head
            # work of earlier microbatches hides the RTT. At most
            # --max-inflight forwards outstanding -> bounds cloud pending-graph
            # memory. Server processes frames in arrival order; replies are
            # correlated by mb_id (RemoteCloudWorker mailboxes).
            inflight_fwd = deque()
            inflight_bwd = deque()
            states = {}

            def _finish_tail(j):
                nonlocal step_loss, blocked, rtt_sum, grad_bytes
                st = states[j]
                rec = st["rec"]
                t = time.perf_counter()
                cloud_out = cloud.recv_forward(j, st["head_out"].dtype, args.device)
                now = time.perf_counter()
                rec["t_cloud_fwd"] = now - st["t_send_fwd"]
                rec["t_wait_cloud_fwd"] = now - t
                blocked += rec["t_wait_cloud_fwd"]
                rtt_sum += rec["t_cloud_fwd"]
                boundary_in = cloud_out.detach().requires_grad_(True)
                t = time.perf_counter()
                hidden = run_layer_stack(tail_layers, boundary_in, st["lk"])
                logits = lm_head(norm(hidden))
                loss = F.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]),
                    st["labels"].reshape(-1)) / args.grad_accum
                rec["loss"] = loss.item() * args.grad_accum
                rec["t_tail_fwd"] = time.perf_counter() - t
                t = time.perf_counter()
                loss.backward()
                rec["t_local_bwd"] = time.perf_counter() - t
                if args.boundary_norm_audit:
                    rec["boundary_return_norms"] = _norm_meta(boundary_in.grad)
                grad_bytes += boundary_in.grad.numel() * boundary_in.grad.element_size()
                cloud.send_backward(j, boundary_in.grad, step=st["rec"]["step"])
                st["t_send_bwd"] = time.perf_counter()
                st["boundary_bytes"] = (st["boundary_out"].numel()
                                        * st["boundary_out"].element_size())
                st["grad_bytes"] = (boundary_in.grad.numel()
                                    * boundary_in.grad.element_size())
                inflight_bwd.append(j)
                step_loss += rec["loss"]

            def _finish_head(j):
                nonlocal blocked, rtt_sum
                st = states[j]
                rec = st["rec"]
                t = time.perf_counter()
                grad_input = cloud.recv_backward(j, st["head_out"].dtype, args.device)
                now = time.perf_counter()
                rec["t_cloud_bwd"] = now - st["t_send_bwd"]
                rec["t_wait_cloud_bwd"] = now - t
                blocked += rec["t_wait_cloud_bwd"]
                rtt_sum += rec["t_cloud_bwd"]
                t = time.perf_counter()
                torch.autograd.backward(st["head_out"], grad_tensors=grad_input)
                rec["t_head_bwd"] = time.perf_counter() - t
                rec["bytes_crossed"] = st["boundary_bytes"] + st["grad_bytes"]
                rec["t_total"] = time.perf_counter() - st["t_mb"]
                timing.append(rec)

            for mb in range(args.grad_accum):
                rec, st = _mb_head(step, mb)
                states[mb] = st
                act_bytes += (st["boundary_out"].numel()
                              * st["boundary_out"].element_size())
                cloud.send_forward(mb, st["boundary_out"], st["lk"], step=step, block_indices=st["rec"].get("block_indices"))
                st["t_send_fwd"] = time.perf_counter()
                inflight_fwd.append(mb)
                if len(inflight_fwd) >= args.max_inflight:
                    _finish_tail(inflight_fwd.popleft())
            while inflight_fwd:
                _finish_tail(inflight_fwd.popleft())
            while inflight_bwd:
                _finish_head(inflight_bwd.popleft())

        else:  # synchronous (M1/M2) path
            for mb in range(args.grad_accum):
                if dp is not None:
                    local_opt.zero_grad(set_to_none=True)
                rec, st = _mb_head(step, mb)
                head_out = st["head_out"]
                boundary_out = st["boundary_out"]
                lk = st["lk"]
                labels = st["labels"]
                act_bytes += boundary_out.numel() * boundary_out.element_size()

                # boundary crossing: local -> cloud (forward)
                t = time.perf_counter()
                cloud_out = cloud.forward(boundary_out, lk, step=step)
                rec["t_cloud_fwd"] = time.perf_counter() - t
                rec["t_wait_cloud_fwd"] = rec["t_cloud_fwd"]

                # boundary crossing: cloud -> local (forward return)
                boundary_in = cloud_out.detach().requires_grad_(True)

                # local tail forward + loss
                t = time.perf_counter()
                hidden = run_layer_stack(tail_layers, boundary_in, lk)
                logits = lm_head(norm(hidden))
                loss = F.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))
                rec["loss"] = loss.item()
                if dp is None:
                    loss = loss / args.grad_accum
                rec["t_tail_fwd"] = time.perf_counter() - t

                # local backward to the boundary (tail only)
                t = time.perf_counter()
                loss.backward()
                rec["t_local_bwd"] = time.perf_counter() - t

                # boundary crossing: cloud <- local (dL/dactivation)
                grad_to_cloud = boundary_in.grad
                if args.boundary_norm_audit:
                    rec["boundary_return_norms"] = _norm_meta(grad_to_cloud)
                if boundary_dp is not None:
                    grad_to_cloud, dp_meta = boundary_dp.protect_return(grad_to_cloud)
                    rec["boundary_dp_return"] = dp_meta
                grad_bytes += grad_to_cloud.numel() * grad_to_cloud.element_size()
                t = time.perf_counter()
                grad_input = cloud.backward(grad_to_cloud, step=step)
                rec["t_cloud_bwd"] = time.perf_counter() - t
                rec["t_wait_cloud_bwd"] = rec["t_cloud_bwd"]

                # finish local backward through the head
                torch.autograd.backward(head_out, grad_tensors=grad_input)
                if dp is not None:
                    grad_norm, clip_scale = dp.clip_and_accumulate()
                    rec["dp_grad_norm_before_clip"] = grad_norm
                    rec["dp_clip_scale"] = clip_scale
                rec["bytes_crossed"] = (boundary_out.numel() * boundary_out.element_size()
                                        + grad_to_cloud.numel() * grad_to_cloud.element_size())
                rec["t_total"] = time.perf_counter() - st["t_mb"]
                step_loss += rec["loss"]
                timing.append(rec)

        if dp is not None:
            dp.materialize_noisy_average()
        local_opt and local_opt.step()
        cloud.step()
        avg = step_loss / args.grad_accum
        step_log.append({"step": step, "loss": avg,
                         "t_step": time.perf_counter() - t0,
                         "t_blocked_cloud": blocked,
                         "overlap_savings_s": rtt_sum - blocked})
        print(f"[step {step}] loss={avg:.4f} "
              f"({(time.perf_counter() - t0):.2f}s, blocked={blocked:.2f}s)")
        _write_training_status(state="running", step=step, loss=avg)

    n_local = sum(p.numel() for p in unique_params(local_trainables,
                                                   require_grad=False))
    n_cloud, n_cloud_train = cloud.param_counts()
    if args.ea4_checkpoint:
        os.makedirs(os.path.dirname(os.path.abspath(args.ea4_checkpoint)), exist_ok=True)
        torch.save({"schema": "dtraining.ea4.boundary_checkpoint.v1",
                    "condition": "split_ft", "split_after": sa,
                    "embed": embed.state_dict(), "head": head_layers.state_dict(),
                    "seed": args.seed}, args.ea4_checkpoint)
        print(f"[E-A4] saved input-boundary checkpoint {args.ea4_checkpoint}")
    # [Held-out Evaluation] Evaluate final loss over ALL held-out eval_blocks.
    # Training never sampled from these blocks (hard partition above).
    eval_loss = None
    eval_losses = []
    if len(eval_blocks) > 0:
        with torch.no_grad():
            for ei in range(len(eval_blocks)):
                eval_batch = torch.stack(eval_blocks[ei:ei+1]).to(args.device)
                e_ids, e_labels = eval_batch[:, :-1], eval_batch[:, 1:]
                e_pos = torch.arange(e_ids.shape[1], device=args.device).unsqueeze(0)
                e_hidden = embed(e_ids)
                e_lk = make_layer_kwargs(rotary, e_hidden, e_pos, args)
                e_head_out = run_layer_stack(head_layers, e_hidden, e_lk)
                # Formal boundary DP is part of the deployed mechanism, so
                # held-out utility includes its noise and its releases compose.
                if boundary_dp is not None:
                    e_head_out, _ = boundary_dp.protect_forward(e_head_out)
                # Legacy empirical act_noise remains disabled on eval.
                e_boundary = e_head_out.detach()
                e_cloud_out = cloud.forward(e_boundary, e_lk)
                e_tail_hidden = run_layer_stack(tail_layers, e_cloud_out, e_lk)
                e_logits = lm_head(norm(e_tail_hidden))
                el = F.cross_entropy(e_logits.float().reshape(-1, e_logits.shape[-1]),
                                     e_labels.reshape(-1)).item()
                eval_losses.append(el)
            eval_loss = sum(eval_losses) / len(eval_losses)
            eval_loss_std = torch.tensor(eval_losses).std(unbiased=False).item()
            print(f"[eval] held-out eval_loss={eval_loss:.4f} "
                  f"(mean of {len(eval_losses)} blocks, population_std={eval_loss_std:.4f})")

    digest = None
    if getattr(args, "param_digest", False):
        digest = param_digest(
            [("embed", embed), ("head", head_layers), ("tail", tail_layers),
             ("norm", norm), ("lm_head", lm_head)], cloud)
    if getattr(args, "save_cloud_weights", None):
        # [data-split] merge the cloud's (public-stream-trained) middle
        # layers back into the local model and save the assembled
        # checkpoint.  The weights cross UCN -> TLN only because they
        # were trained exclusively on public data.
        if not hasattr(cloud, "weights_snapshot"):
            raise ValueError("--save-cloud-weights requires a remote --cloud")
        state = cloud.weights_snapshot()
        target = nn.ModuleList(list(layers[sa + 1: ra]))
        target.load_state_dict({k: v.to(target[0].input_layernorm.weight.dtype)
                                for k, v in state.items()})
        args._model_ref.save_pretrained(args.save_cloud_weights)
        args._tokenizer_ref.save_pretrained(args.save_cloud_weights)
        print(f"[data-split] merged {len(state)} cloud tensors; assembled "
              f"checkpoint saved to {args.save_cloud_weights}")
    cloud.close()
    return {
        "param_digest": digest,
        "eval_loss": eval_loss,
        "eval_loss_std": eval_loss_std if eval_losses else None,
        "eval_blocks": len(eval_losses),
        "final_train_loss": step_log[-1]["loss"] if step_log else None,
        "config": {
            "model": "toy" if args.toy else args.model,
            "n_layers": n_layers, "split_after": sa, "resume_after": ra,
            "local_params": n_local, "cloud_params": n_cloud,
            "cloud_trainable_params": n_cloud_train,
            "cloud": args.cloud or "in-process",
            "wire_dtype": args.wire_dtype if args.cloud else None,
            "pipeline": args.pipeline,
            "max_inflight": args.max_inflight if args.pipeline == "overlap" else None,
            "lora_rank": args.lora_rank, "lr": args.lr,
            "seq_len": args.seq_len, "micro_batch_size": args.micro_batch_size,
            "grad_accum": args.grad_accum, "steps": args.steps,
            "dtype": args.dtype, "device": args.device, "seed": args.seed,
            "corpus_file": getattr(args, "corpus_file", None),
            "act_noise_sigma": getattr(args, "act_noise_sigma", 0.0),
            "tee": {
                "requested": getattr(args, "tee_attest", False),
                "mode": ("emulated" if getattr(args, "tee_emulated", False)
                         else "hardware" if getattr(args, "tee_attest", False)
                         else "none"),
                "note": ("Emulated attestation does NOT provide hardware isolation; "
                         "UCN host root can still inspect VRAM and process memory."
                         if getattr(args, "tee_emulated", False) else None),
            },
            "act_noise": (None if getattr(args, "act_noise_sigma", 0.0) == 0.0 else {
                "mechanism": "relative_rms_gaussian_noise",
                "sigma_act": args.act_noise_sigma,
                "scope": "live_forward_boundary_activations",
                "formal_dp": False,
                "note": "Empirical noise injection for privacy amplification. "
                        "NOT a formal (epsilon,delta)-DP mechanism: the per-query "
                        "sensitivity bound for arbitrary-length sequences has not "
                        "been derived. Do not cite this as a DP guarantee.",
            }),
            "dp_sgd": (None if dp is None else {
                "scope": "private_local_weight_updates_only",
                "cloud_frozen": True,
                "per_example_clipping": True,
                "max_grad_norm": args.dp_max_grad_norm,
                "noise_multiplier": args.dp_noise_multiplier,
                "accountant": conservative_zcdp_epsilon(
                    args.steps, args.dp_noise_multiplier, args.dp_delta),
                "boundary_activations_covered_by_weight_dp": False,
                "note": "Weight DP-SGD bounds model update leakage; live activation noise is empirical and TEE is emulated unless hardware evidence is supplied."
            }),
            "boundary_dp": (None if boundary_dp is None else {
                "scope": "outbound_forward_activations_and_outbound_boundary_gradients",
                "adjacency_unit": "one_boundary_token_row",
                "cloud_response_before_local_receipt_covered": False,
                "formal_dp": True,
                "accountant": boundary_dp.report(),
            }),
            # [ER] rotation provenance (null when the ratchet is off)
            "obf": (None if ratchet is None else {
                "mode": ("fold" if ratchet.transform_mode == "dense"
                         else "wire_sandwich"),
                "transform_mode": ratchet.transform_mode,
                "version": ratchet.version,
                "seed_base": "redacted",
                "seed_delivery": ("preprovisioned_on_cloud"
                                  if args.obf_seed_preprovisioned
                                  else "sent_in_hello" if args.cloud
                                  else "in_process"),
                "ratchet_tokens": args.obf_ratchet_tokens,
                "budget_events": args.obf_budget_events,
                "count_all_training_directions": args.obf_count_all_directions,
                "strict_budget": ratchet.strict_budget,
                "oversized_forwards": ratchet.oversized_forwards,
                "served_tokens": ratchet.served,
                "served_observable_rows": ratchet.served,
                "final_epoch": ratchet.epoch,
                "directional_accounting": ratchet.snapshot(),
                "privacy_scope": "capture_resistance_negative_control",
                "compromised_cloud_privacy": False,
            }),
        },
        "steps": step_log,
        "microbatch_timing": timing,
        "bytes_crossed_total": {"activations": act_bytes, "grads": grad_bytes},
        "note": ("remote cloud: t_cloud_* includes network round-trip"
                 if args.cloud else
                 "cloud runs in-process; timing excludes network. See PROTOCOL.md."),
    }
if __name__ == "__main__":
    main()
