#!/usr/bin/env python3
"""Split fine-tuning - Cloud Trainer Server (M2).

Runs the MIDDLE layers [cloud_start..cloud_end] (inclusive) of a causal LM as
a *training* stage: keeps per-session autograd graphs and optimizer state,
receives boundary activations, returns outputs, then receives dL/d(output)
and returns dL/d(input). Wire protocol per split-training/PROTOCOL.md:

  - WS (binary): [4B big-endian header_len][JSON header][raw tensor bytes]
      ops: forward_with_graph -> forward_result
           backward_grad      -> backward_result
  - WS (text JSON control): hello -> hello_ack, optimizer_step -> step_ack,
    close
  - HTTP/Flask: /health on port 5002 (ws on 5003), the same convention the
    inference server used.

Weight provenance (v1): both sides load the SAME checkpoint independently
(--model path, or --toy with the same --seed). The server never receives
weights over the wire.

Grad compression seam (experiment 3): hello may carry
grad_compression={"kind":"noise","sigma":s} — Gaussian noise with std
s * per-tensor RMS added to the boundary gradient before cloud backward.
Quantization/top-k land in M3.

[ER] E-R7 fold mode: hello may carry obf_seed_base=S; frames then carry
"obf_epoch" and the server folds its middle layers per epoch
(covariant_fold.fold_layer on the seed-chain W_t), never unwrapping wire
tensors. Env ER_CAPTURE_DIR dumps raw wire tensors + JSON sidecars
(schema: er_ratchet.SIDECAR_KEYS) for the wire evaluator
(attacker/captures.py training mode).

Usage:
  python cloud_trainer_server.py --help        # works without torch
  python cloud_trainer_server.py --toy --device cpu --cloud-start 2 --cloud-end 2
  python cloud_trainer_server.py --model <hf-model> --cloud-start 5 --cloud-end 25
"""

import argparse
import asyncio
import copy
import hashlib
import json
import os
import secrets
import struct
import threading
import time
import traceback
import uuid

# Guarded heavy imports so `--help` works on torch-less hosts.
try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None
try:
    import websockets
except ImportError:  # pragma: no cover
    websockets = None
try:
    from flask import Flask, jsonify
except ImportError:  # pragma: no cover
    Flask = None
    jsonify = None
try:
    from transformers import AutoModelForCausalLM
except ImportError:  # pragma: no cover
    AutoModelForCausalLM = None

# Toy model + layer-stack helper reused from the trainer (same repo dir).
ToyCausalLM = None
run_layer_stack = None
if torch is not None:
    try:
        from split_trainer import ToyCausalLM, run_layer_stack
    except ImportError:  # pragma: no cover - run from repo root
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from split_trainer import ToyCausalLM, run_layer_stack

# [ER] E-R7 fold mode: per-epoch W from the shared seed chain + the exact
# covariant fold. Same guarded-import convention as above.
derive_epoch_W = None
fold_layer = None
unfold_grads_into = None
StructuredHadamard = None
if torch is not None:
    try:
        from er_ratchet import derive_epoch_W
        from covariant_fold import fold_layer, unfold_grads_into
        from privacy_runtime.structured_transform import StructuredHadamard
    except ImportError:  # pragma: no cover
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        try:
            from er_ratchet import derive_epoch_W
            from covariant_fold import fold_layer, unfold_grads_into
            from privacy_runtime.structured_transform import StructuredHadamard
        except ImportError:
            pass

WS_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16,
             "fp32": torch.float32} if torch is not None else {}

app = Flask(__name__) if Flask is not None else None
CLOUD_LAYERS = None          # nn.ModuleList of the middle layers
MODEL_DTYPE = None
DEVICE = "cpu"
CLOUD_START = CLOUD_END = -1
N_LAYERS = 0

# {session_id: {optimizer, pending: {mb_id: (input_leaf, output)},
#               mb_since_step, last_access, wire_dtype, grad_noise_sigma}}
sessions = {}
PREPROVISIONED_OBF_SEED = None
SESSION_TIMEOUT = 600  # seconds; PROTOCOL.md §3
LATENCY_MS = 0  # artificial per-call delay (experiment-4 loopback testing)
WEIGHT_VERSION = 0  # [ER] bumped at every optimizer_step; folded caches
# built from older canonical weights are stale and must be refolded


# Tensor (de)serialization without numpy dependency issues: bf16 has no numpy
# dtype, so go through a uint16 view.
def pack_tensor(t, wire_dtype):
    t = t.detach().to(wire_dtype).contiguous().cpu()
    if wire_dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().tobytes()
    return t.numpy().tobytes()


def unpack_tensor(buf, shape, wire_dtype, device):
    if wire_dtype == torch.bfloat16:
        t = torch.frombuffer(bytearray(buf), dtype=torch.int16).view(torch.bfloat16)
    else:
        t = torch.frombuffer(bytearray(buf), dtype=wire_dtype)
    return t.reshape(shape).clone().to(device)


def pack_frame(header, payload=b""):
    h = json.dumps(header).encode()
    return struct.pack(">I", len(h)) + h + payload


def unpack_frame(message):
    header_len = struct.unpack(">I", message[:4])[0]
    header = json.loads(message[4:4 + header_len])
    return header, message[4 + header_len:]


def _numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def load_model(args):
    """Load the checkpoint and slice out the middle layers."""
    global CLOUD_LAYERS, MODEL_DTYPE, DEVICE, CLOUD_START, CLOUD_END, N_LAYERS
    DEVICE = args.device
    if args.toy:
        torch.manual_seed(args.seed)  # MUST match the trainer's seed (v1)
        model = ToyCausalLM()
        layers = model.layers
        MODEL_DTYPE = torch.float32
    else:
        if AutoModelForCausalLM is None:
            raise RuntimeError("transformers not installed; use --toy")
        MODEL_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16,
                       "fp32": torch.float32}[args.dtype]
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=MODEL_DTYPE)
        layers = model.model.layers
    N_LAYERS = len(layers)
    CLOUD_START = args.cloud_start
    CLOUD_END = args.cloud_end if args.cloud_end is not None else N_LAYERS - 2
    if not (0 <= CLOUD_START <= CLOUD_END <= N_LAYERS - 1):
        raise ValueError(f"bad layer range {CLOUD_START}..{CLOUD_END} for {N_LAYERS} layers")
    CLOUD_LAYERS = nn.ModuleList(list(layers[CLOUD_START: CLOUD_END + 1]))
    CLOUD_LAYERS.to(DEVICE).train()  # gradients ENABLED (unlike inference)
    if not args.toy:
        del model  # keep only the middle slice referenced
    n = sum(p.numel() for p in CLOUD_LAYERS.parameters())
    print(f"[model] {N_LAYERS} layers total, serving {CLOUD_START}-{CLOUD_END} "
          f"({n / 1e6:.1f}M params, {MODEL_DTYPE}, device={DEVICE})")


def cleanup_old_sessions():
    now = time.time()
    expired = [sid for sid, s in sessions.items()
               if now - s["last_access"] > SESSION_TIMEOUT]
    for sid in expired:
        leaked = len(sessions[sid]["pending"])
        del sessions[sid]
        print(f"[session] expired {sid[:8]} (leaked pending graphs: {leaked})")


def make_optimizer(lr, trainable):
    params = [p for p in CLOUD_LAYERS.parameters() if p.requires_grad]
    if not trainable:
        for p in CLOUD_LAYERS.parameters():
            p.requires_grad_(False)
        params = []
    return torch.optim.AdamW(params, lr=lr) if params else None


# [ER] E-R7 fold mode. The client transmits "obf_epoch" in every frame
# header; the server derives the SAME W_t from the hello's obf_seed_base
# (seed chain sha256("S:t"), no key shipping) and folds its middle-layer
# weights with covariant_fold.fold_layer (fp64, exact). The server NEVER
# unwraps tensors: wire tensors stay rotated in both directions and the
# folded layers map h @ W_t -> y @ W_t by construction
# (covariant_fold.unfold_check). Backward populates folded-basis grads,
# which unfold_grads_into maps back onto the canonical weights, so the
# optimizer update matches the non-rotated baseline exactly.
#
# Cache discipline: ONLY the current epoch's folded copy is held (one extra
# copy of the middle stack); it is evicted on epoch change. A backward that
# references a non-current epoch fails loud — sync pipelines never straddle
# a rotation boundary (fwd/bwd pairs complete before the next forward).
def fold_for_epoch(session, epoch):
    """Return (folded_layers, W) for `epoch`, refolding on epoch change OR
    when the canonical weights moved (optimizer step) since the cache was
    built."""
    if fold_layer is None:
        raise RuntimeError("[ER] covariant_fold not importable on this host")
    cache = session.get("obf_cache")
    if (cache and cache["epoch"] == epoch
            and cache["wversion"] == WEIGHT_VERSION):
        return cache["layers"], cache["W"]
    hidden_dim = CLOUD_LAYERS[0].input_layernorm.weight.numel()
    t0 = time.perf_counter()
    W = derive_epoch_W(session["obf_seed_base"], epoch, hidden_dim,
                       version=session["obf_ratchet_version"])
    folded = copy.deepcopy(CLOUD_LAYERS)  # from CANONICAL weights (post-step)
    for layer in folded:
        fold_layer(layer, W)
    dt = time.perf_counter() - t0
    reason = "epoch change" if not cache or cache["epoch"] != epoch \
        else "post-step refold"
    session["obf_cache"] = {"epoch": epoch, "layers": folded, "W": W,
                            "wversion": WEIGHT_VERSION}
    print(f"[ER] folded middle layers for epoch {epoch} in {dt:.2f}s "
          f"(seed_base=redacted, {reason}; previous copy "
          "evicted)")
    return folded, W


def session_epoch(session, header):
    """[ER] Validate and return the frame's obf_epoch (None when the ratchet
    is off). Fails loud on the no-silent-downgrade cases."""
    obf_epoch = header.get("obf_epoch")
    if obf_epoch is None:
        if session.get("obf_seed_base") is not None:
            raise RuntimeError(
                "[ER] frame without obf_epoch in a rotated session — "
                "refusing silent downgrade")
        return None
    if session.get("obf_seed_base") is None:
        raise RuntimeError(
            "[ER] obf_epoch arrived but the session hello carried no "
            "obf_seed_base — refusing to process an unkeyed rotation")
    if not isinstance(obf_epoch, int) or isinstance(obf_epoch, bool) or obf_epoch < 0:
        raise RuntimeError("[ER] obf_epoch must be a non-negative integer")
    return obf_epoch


def _er_capture_bytes(session_id, header, phase, raw_bytes, epoch):
    cap_dir = os.environ.get("ER_CAPTURE_DIR")
    if not cap_dir:
        return
    os.makedirs(cap_dir, exist_ok=True)
    n = len([f for f in os.listdir(cap_dir) if f.startswith("wire_") and f.endswith(".bin")])
    with open(os.path.join(cap_dir, f"wire_{n:04d}.bin"), "wb") as f:
        f.write(raw_bytes)
    meta = {"session_id": session_id, "mb_id": header.get("mb_id"),
            "phase": phase, "step": header.get("step"), "epoch": epoch, "format": "bytes"}
    with open(os.path.join(cap_dir, f"wire_{n:04d}.json"), "w") as mf:
        json.dump(meta, mf)


def _er_capture(session_id, header, phase, tensor, epoch, prefix="cloud"):
    """[ER] Wire capture (training analog of the E9 capture in
    cloud_server_kv.py): env ER_CAPTURE_DIR; dump the RAW (rotated) wire
    tensor + a wire_NNNN.json sidecar. Schema pinned in
    er_ratchet.SIDECAR_KEYS = {session_id, mb_id, phase, step, epoch}."""
    cap_dir = os.environ.get("ER_CAPTURE_DIR")
    if not cap_dir:
        return
    os.makedirs(cap_dir, exist_ok=True)
    n = len([f for f in os.listdir(cap_dir)
             if f.startswith(f"{prefix}_") and f.endswith(".pt")])
    torch.save(tensor.detach().cpu(), os.path.join(cap_dir, f"{prefix}_{n:04d}.pt"))
    meta = {"session_id": session_id, "mb_id": header.get("mb_id"),
            "phase": phase, "step": header.get("step"), "epoch": epoch}
    if "block_indices" in header:
        meta["block_indices"] = header["block_indices"]
    with open(os.path.join(cap_dir, f"{prefix}_{n:04d}.json"), "w") as mf:
        json.dump(meta, mf)


def handle_forward(session, header, tensor_data):
    wire_dtype = WS_DTYPES[header.get("dtype", session["wire_dtype"])]
    hs_shape = header["hidden_shape"]
    hs_bytes = _numel(hs_shape) * (2 if wire_dtype in (torch.float16, torch.bfloat16) else 4)
    hidden = unpack_tensor(tensor_data[:hs_bytes], hs_shape, wire_dtype, DEVICE)

    # [ER] fold mode: pick (or refold) the middle stack for this epoch. The
    # wire tensor stays rotated — it is NEVER unwrapped server-side.
    epoch = session_epoch(session, header)
    _er_capture(session.get("session_id"), header, "fwd", hidden, epoch)
    layers = CLOUD_LAYERS
    sandwich = (epoch is not None and
                session.get("obf_transform_mode") != "dense")
    if epoch is not None and not sandwich:
        layers, _W = fold_for_epoch(session, epoch)

    kwargs = {"attention_mask": None,
              "position_ids": torch.arange(hs_shape[1], device=DEVICE).unsqueeze(0)}
    off = hs_bytes
    if header.get("has_pos_emb"):
        pe_shape = header["pe_shape"]
        pe_bytes = _numel(pe_shape) * (2 if wire_dtype in (torch.float16, torch.bfloat16) else 4)
        cos = unpack_tensor(tensor_data[off:off + pe_bytes], pe_shape, wire_dtype, DEVICE)
        sin = unpack_tensor(tensor_data[off + pe_bytes:off + 2 * pe_bytes], pe_shape,
                            wire_dtype, DEVICE)
        kwargs["position_embeddings"] = (cos.to(MODEL_DTYPE), sin.to(MODEL_DTYPE))

    t0 = time.perf_counter()
    is_training = header.get("step") is not None
    leaf = hidden.to(MODEL_DTYPE).detach().requires_grad_(is_training)

    def compute():
        if sandwich:
            if session["obf_transform_mode"] == "structured_hadamard":
                transform = StructuredHadamard(
                    session["obf_seed_base"], epoch, leaf.shape[-1],
                    device=leaf.device, dtype=leaf.dtype)
                canonical = transform.inverse(leaf)
            else:
                cache = session.get("obf_transform_cache")
                if cache is None or cache["epoch"] != epoch:
                    W = derive_epoch_W(session["obf_seed_base"], epoch,
                                       leaf.shape[-1],
                                       version=session["obf_ratchet_version"])
                    W = W.to(device=leaf.device, dtype=leaf.dtype)
                    cache = {"epoch": epoch, "W": W,
                             "Wt": W.T.contiguous()}
                    session["obf_transform_cache"] = cache
                canonical = leaf @ cache["Wt"]
            canonical_out = run_layer_stack(layers, canonical, kwargs)
            return (transform.apply(canonical_out)
                    if session["obf_transform_mode"] == "structured_hadamard"
                    else canonical_out @ cache["W"])
        return run_layer_stack(layers, leaf, kwargs)

    if is_training:
        out = compute()
    else:
        with torch.no_grad():
            out = compute()
    mb_id = header["mb_id"]
    if is_training:
        session["pending"][mb_id] = (leaf, out, epoch)
        session["mb_since_step"] += 1

    payload = pack_tensor(out, wire_dtype)
    resp = pack_frame({"op": "forward_result", "mb_id": mb_id,
                       "hidden_shape": list(out.shape),
                       "dtype": header.get("dtype", session["wire_dtype"]),
                       "cloud_time_ms": round((time.perf_counter() - t0) * 1000, 2)},
                      payload)
    return resp


def handle_backward(session, header, tensor_data):
    wire_dtype = WS_DTYPES[header.get("dtype", session["wire_dtype"])]
    g_shape = header["grad_shape"]
    grad = unpack_tensor(tensor_data, g_shape, wire_dtype, DEVICE).to(MODEL_DTYPE)
    mb_id = header["mb_id"]

    # [ER] the wire grad is w.r.t. the ROTATED cloud output; the folded
    # graph consumes it directly (no unwrap). The backward must reference a
    # pending forward of the SAME epoch — mixed epochs fail loud.
    epoch = session_epoch(session, header)
    _er_capture(session.get("session_id"), header, "bwd", grad, epoch)
    leaf, out, fwd_epoch = session["pending"].pop(mb_id)
    if fwd_epoch != epoch:
        raise RuntimeError(
            f"[ER] backward epoch {epoch} does not match the pending "
            f"forward's epoch {fwd_epoch} (mb_id={mb_id}) — mixed epochs "
            "within a session are refused")
    if epoch is not None and session.get("obf_transform_mode") == "dense":
        cache = session.get("obf_cache")
        if cache is None or cache["epoch"] != epoch:
            raise RuntimeError(
                f"[ER] backward for epoch {epoch} after the folded weights "
                "were evicted (overlap pipeline straddling a rotation "
                "boundary is not supported; use --pipeline sync)")

    sigma = session.get("grad_noise_sigma", 0.0)
    if sigma:  # experiment-3 seam: DP-style noise calibrated to grad RMS
        rms = grad.pow(2).mean().sqrt()
        grad = grad + torch.randn_like(grad) * (sigma * rms)

    t0 = time.perf_counter()
    torch.autograd.backward(out, grad_tensors=grad)
    grad_input = leaf.grad
    if (epoch is not None and session["optimizer"] is not None
            and session.get("obf_transform_mode") == "dense"):
        # [ER] map folded-basis grads back onto the canonical weights so the
        # optimizer update matches the non-rotated baseline exactly.
        unfold_grads_into(CLOUD_LAYERS, session["obf_cache"]["layers"],
                          session["obf_cache"]["W"])

    payload = pack_tensor(grad_input, wire_dtype)
    resp = pack_frame({"op": "backward_result", "mb_id": mb_id,
                       "grad_shape": list(grad_input.shape),
                       "dtype": header.get("dtype", session["wire_dtype"]),
                       "cloud_time_ms": round((time.perf_counter() - t0) * 1000, 2)},
                      payload)
    return resp


async def ws_handler(websocket):
    """One connection = one training session.

    1. Client sends text JSON hello; server replies hello_ack (or error+close).
    2. Binary frames: forward_with_graph / backward_grad.
    3. Text frames: optimizer_step, close.
    v1: a disconnect aborts the session — no graph recovery (PROTOCOL.md §6).
    """
    cleanup_old_sessions()
    session_id = str(uuid.uuid4())
    session = None
    try:
        hello_raw = await websocket.recv()
        hello = json.loads(hello_raw)
        if hello.get("op") != "hello":
            await websocket.send(json.dumps({"op": "error",
                                             "error": "first message must be hello"}))
            return
        trainable = hello.get("trainable", True)
        requested_obf = (hello.get("obf_enabled")
                         or hello.get("obf_seed_base") is not None)
        ratchet_version = hello.get("obf_ratchet_version")
        transform_mode = hello.get("obf_transform_mode", "dense")
        if hello.get("obf_enabled") and hello.get("obf_seed_base") is not None:
            await websocket.send(json.dumps({
                "op": "error",
                "error": "ambiguous obfuscation setup: choose preprovisioned or inline seed",
            }))
            return
        if hello.get("obf_enabled") and PREPROVISIONED_OBF_SEED is None:
            await websocket.send(json.dumps({
                "op": "error",
                "error": "preprovisioned obfuscation requested but no server seed is configured",
            }))
            return
        if requested_obf and ratchet_version not in ("v1", "v2"):
            await websocket.send(json.dumps({
                "op": "error",
                "error": "rotated sessions must declare ratchet version v1 or v2",
            }))
            return
        if requested_obf and transform_mode not in (
                "dense", "dense_sandwich", "structured_hadamard"):
            await websocket.send(json.dumps({
                "op": "error", "error": "unsupported obfuscation transform mode",
            }))
            return
        if not requested_obf and ratchet_version is not None:
            await websocket.send(json.dumps({
                "op": "error",
                "error": "ratchet version supplied for an unrotated session",
            }))
            return
        session = {
            "session_id": session_id,
            "optimizer": make_optimizer(hello.get("lr", 1e-4), trainable),
            "pending": {},
            "mb_since_step": 0,
            "last_access": time.time(),
            "wire_dtype": hello.get("wire_dtype", "fp16"),
            "grad_noise_sigma": (hello.get("grad_compression") or {}).get("sigma", 0.0),
            # [ER] fold mode: seed base from hello; folded-weights cache
            # (single current epoch) created lazily on the first rotated
            # forward.
            "obf_seed_base": (PREPROVISIONED_OBF_SEED
                              if hello.get("obf_enabled")
                              else hello.get("obf_seed_base")),
            "obf_ratchet_version": ratchet_version,
            "obf_transform_mode": transform_mode,
            "obf_cache": None,
            "obf_transform_cache": None,
        }
        # [TEE] Confidential Compute hardware attestation probe & encryption channel
        tee_report = None
        tee_channel = None
        if hello.get("tee_attest") or hello.get("tee_enclave"):
            try:
                from tee_enclave import GPUConfidentialCompute, TEEEncryptedChannel
                probe = GPUConfidentialCompute(mode="emulated" if hello.get("tee_emulated") else "auto")
                nonce_hex = hello.get("tee_nonce", secrets.token_hex(16))
                client_pub_hex = hello.get("tee_client_pub")
                if not client_pub_hex:
                    raise ValueError("Client did not provide ECDH public key")
                tee_report, session_key = probe.get_attestation_report(
                    bytes.fromhex(nonce_hex), bytes.fromhex(client_pub_hex)
                )
                if tee_report.get("attestation_status") == "PASSED":
                    tee_channel = TEEEncryptedChannel(session_key)
            except Exception as exc:
                tee_report = {"attestation_status": "FAILED_ERROR", "error": str(exc)}

        session["tee_channel"] = tee_channel
        sessions[session_id] = session
        n_params = sum(p.numel() for p in CLOUD_LAYERS.parameters())
        n_train = sum(p.numel() for p in CLOUD_LAYERS.parameters() if p.requires_grad)
        await websocket.send(json.dumps({
            "op": "hello_ack", "session_id": session_id,
            "cloud_start": CLOUD_START, "cloud_end": CLOUD_END,
            "n_cloud_layers": CLOUD_END - CLOUD_START + 1,
            "cloud_params": n_params, "cloud_trainable_params": n_train,
            "obf_ratchet_version": session["obf_ratchet_version"],
            "obf_transform_mode": session["obf_transform_mode"],
            "tee_attestation_report": tee_report,
        }))
        print(f"[WS] session {session_id[:8]} hello "
              f"(lr={hello.get('lr')}, trainable={trainable}, "
              f"wire={session['wire_dtype']}, noise_sigma={session['grad_noise_sigma']}, "
              f"obf_seed_base={'present-redacted' if session['obf_seed_base'] is not None else None}, "
              f"tee_attest={'passed' if tee_report and tee_report.get('attestation_status')=='PASSED' else False})")

        async for message in websocket:
            session["last_access"] = time.time()
            if isinstance(message, str):
                data = json.loads(message)
                op = data.get("op")
                if op == "optimizer_step":
                    global WEIGHT_VERSION
                    if session["optimizer"] is not None:
                        session["optimizer"].step()
                        session["optimizer"].zero_grad(set_to_none=True)
                        # [ER] canonical weights moved: any cached folded
                        # copy is now stale (refolded lazily on next use)
                        WEIGHT_VERSION += 1
                    consumed = session["mb_since_step"]
                    session["mb_since_step"] = 0
                    await websocket.send(json.dumps(
                        {"op": "step_ack", "mb_consumed": consumed}))
                elif op == "weights_snapshot":
                    # [data-split] send the current middle-layer weights back
                    # to the trusted node. These weights are trained on the
                    # PUBLIC stream only, so nothing private crosses here.
                    # Call only when no forward/backward is in flight.
                    n = 0
                    wire_dtype = WS_DTYPES[session["wire_dtype"]]
                    for name, tensor in CLOUD_LAYERS.state_dict().items():
                        payload = pack_tensor(tensor, wire_dtype)
                        await websocket.send(pack_frame(
                            {"op": "weights_part", "name": name,
                             "shape": list(tensor.shape),
                             "dtype": session["wire_dtype"]}, payload))
                        n += 1
                    await websocket.send(json.dumps(
                        {"op": "weights_done", "count": n}))
                elif op == "close":
                    break
                continue

            header, tensor_data = unpack_frame(message)
            op = header.get("op")
            
            # [ER] Capture raw wire bytes before TEE decryption (Passive Observer)
            if op in ("forward_with_graph", "backward_grad"):
                _er_capture_bytes(session.get("session_id"), header, "fwd" if op == "forward_with_graph" else "bwd",
                                  tensor_data, session_epoch(session, header))

            # [TEE] Reject unencrypted payloads when a TEE channel is active.
            if session.get("tee_channel") is not None:
                if not header.get("tee_encrypted"):
                    await websocket.send(pack_frame(
                        {"op": "error",
                         "error": "TEE channel active but frame is not encrypted. "
                                  "Protocol downgrade rejected."}))
                    continue
                tensor_data = session["tee_channel"].decrypt_payload(tensor_data)
            elif header.get("tee_encrypted"):
                await websocket.send(pack_frame(
                    {"op": "error",
                     "error": "Frame claims tee_encrypted but no TEE channel was negotiated."}))
                continue
            
            if op in ("forward_with_graph", "backward_grad"):
                fn = handle_forward if op == "forward_with_graph" else handle_backward
                if LATENCY_MS:
                    async def run(fn=fn, h=header, td=tensor_data):
                        await asyncio.sleep(LATENCY_MS / 1000.0)
                        resp = await asyncio.to_thread(fn, session, h, td)
                        if session.get("tee_channel") is not None:
                            rh, rp = unpack_frame(resp)
                            rh["tee_encrypted"] = True
                            rp = session["tee_channel"].encrypt_payload(rp)
                            resp = pack_frame(rh, rp)
                        await websocket.send(resp)
                    asyncio.create_task(run())
                else:
                    resp = fn(session, header, tensor_data)
                    if session.get("tee_channel") is not None:
                        rh, rp = unpack_frame(resp)
                        rh["tee_encrypted"] = True
                        rp = session["tee_channel"].encrypt_payload(rp)
                        resp = pack_frame(rh, rp)
                    await websocket.send(resp)
            else:
                await websocket.send(pack_frame(
                    {"op": "error", "error": f"unknown op {op!r}"}))

    except websockets.exceptions.ConnectionClosed:
        print(f"[WS] connection dropped for session {session_id[:8]} "
              f"(aborting run — v1 has no graph recovery)")
    except Exception as e:
        print(f"[WS] error in session {session_id[:8]}: {e}")
        traceback.print_exc()
    finally:
        if session_id in sessions:
            leaked = len(sessions[session_id]["pending"])
            del sessions[session_id]
        else:
            leaked = 0
        print(f"[WS] session {session_id[:8]} ended "
              f"(leaked pending graphs: {leaked})")


async def start_ws_server(port):
    async with websockets.serve(
        ws_handler, "0.0.0.0", port,
        max_size=512 * 1024 * 1024,  # training frames are larger than inference
        # handle_forward is intentionally synchronous and an epoch refold can
        # occupy this event loop for minutes. Application-level request/reply
        # semantics detect failures; protocol pings would kill healthy folds.
        ping_interval=None,
    ):
        print(f"WebSocket trainer server on port {port}")
        await asyncio.Future()


if app is not None:

    @app.route("/health", methods=["GET"])
    def health():
        pending = sum(len(s["pending"]) for s in sessions.values())
        return jsonify({
            "status": "ok",
            "mode": "training",
            "layers": f"{CLOUD_START}-{CLOUD_END}",
            "n_layers": N_LAYERS,
            "active_sessions": len(sessions),
            "pending_graphs": pending,
            "device": DEVICE,
            "dtype": str(MODEL_DTYPE),
        })


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/qwen3-0.6b")
    ap.add_argument("--toy", action="store_true",
                    help="tiny built-in model for CPU loopback tests")
    ap.add_argument("--cloud-start", type=int, required=False, default=None,
                    help="first cloud layer (inclusive)")
    ap.add_argument("--cloud-end", type=int, default=None,
                    help="last cloud layer (inclusive, default n_layers-2)")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device",
                    default="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42,
                    help="toy-model seed; MUST match the trainer's --seed")
    ap.add_argument("--http-port", type=int, default=5002)
    ap.add_argument("--ws-port", type=int, default=5003)
    ap.add_argument("--latency-ms", type=float,
                    default=float(os.environ.get("CLOUD_LATENCY_MS", 0)),
                    help="artificial per-call delay (experiment-4 loopback "
                         "testing; env CLOUD_LATENCY_MS)")
    ap.add_argument("--obf-seed-file", default=None,
                    help="mode-0600 pre-provisioned E-R9 seed; keeps it off WS")
    args = ap.parse_args()

    global PREPROVISIONED_OBF_SEED
    if args.obf_seed_file:
        if os.stat(args.obf_seed_file).st_mode & 0o077:
            ap.error("--obf-seed-file must be mode 0600")
        with open(args.obf_seed_file) as f:
            PREPROVISIONED_OBF_SEED = int(f.read().strip())

    if torch is None or websockets is None or Flask is None:
        ap.error("torch/websockets/flask not installed; --help works without them")

    if args.cloud_start is None:
        ap.error("--cloud-start is required (must equal trainer's split-after + 1)")

    global LATENCY_MS
    LATENCY_MS = args.latency_ms
    if LATENCY_MS:
        print(f"[test] artificial latency: {LATENCY_MS}ms per forward/backward call")
    load_model(args)

    def run_ws():
        asyncio.run(start_ws_server(port=args.ws_port))

    threading.Thread(target=run_ws, daemon=True).start()
    print(f"HTTP health server on port {args.http_port}...")
    app.run(host="0.0.0.0", port=args.http_port, threaded=True)


if __name__ == "__main__":
    main()
