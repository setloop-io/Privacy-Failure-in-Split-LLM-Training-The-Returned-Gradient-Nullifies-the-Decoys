#!/usr/bin/env python3
"""Fail-closed D-only training server for latent-native v5."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import struct
import sys
import time
import traceback
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Guarded heavy imports so `--help` works on torch-less hosts.
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None
try:
    import websockets
except ImportError:  # pragma: no cover
    websockets = None
try:
    from privacy_runtime.latent_native import (assert_ucn_latent_only,
                                               build_ucn_latent_middle)
except ImportError:  # pragma: no cover
    assert_ucn_latent_only = None
    build_ucn_latent_middle = None
from privacy_runtime.remote_transcript import RemoteTranscript


def pack_frame(header, payload=b""):
    encoded = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack(">I", len(encoded)) + encoded + payload


def unpack_frame(message):
    size = struct.unpack(">I", message[:4])[0]
    return json.loads(message[4:4 + size]), message[4 + size:]


def tensor_from(payload, shape, device):
    expected = 4
    for dimension in shape:
        expected *= dimension
    if len(payload) != expected:
        raise ValueError("payload length does not match declared shape")
    return torch.frombuffer(bytearray(payload), dtype=torch.float32).reshape(
        shape).clone().to(device)


def tensor_bytes(value):
    return value.detach().float().contiguous().cpu().numpy().tobytes()


def validate_latent_shape(shape, latent_dim):
    """Validate headers before payload allocation or tensor construction."""
    if (not isinstance(shape, list) or len(shape) != 3
            or any(not isinstance(value, int) or isinstance(value, bool)
                   or value <= 0 for value in shape)
            or shape[-1] != latent_dim):
        raise ValueError("non-D latent frame rejected before allocation")
    return tuple(shape)


def _positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


class LatentServer:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.sessions = {}
        self.max_seen_width = 0

    def new_session_model(self, seed):
        """Build an independently seeded cloud model for one connection."""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = build_ucn_latent_middle(
                self.args.latent_dim, self.args.cloud_layers,
                self.args.cloud_heads, self.args.cloud_kind,
                getattr(self.args, "cloud_experts", 1),
                getattr(self.args, "cloud_hidden", 0))
        model = model.to(self.device).train()
        assert_ucn_latent_only(
            model, self.args.latent_dim, self.args.forbidden_hidden_dim,
            allowed_internal_width=getattr(self.args, "cloud_hidden", 0)
            or None)
        return model

    def capture(self, session_id, mb_id, phase, tensor, step):
        """Wire capture sidecar, schema-compatible with split_trainer.py.

        `step` is the optimizer step counted from the client's optimizer_step
        control op (the same quantity split_trainer passes to _er_capture).
        `epoch` is null: this server has no key ratchet, so writing mb_id
        there would fabricate a rotation window that does not exist.
        """
        if not self.args.capture_dir:
            return
        root = Path(self.args.capture_dir)
        root.mkdir(parents=True, exist_ok=True)
        index = len(list(root.glob("wire_*.pt")))
        stem = root / f"wire_{index:06d}"
        torch.save(tensor.detach().float().cpu(), stem.with_suffix(".pt"))
        stem.with_suffix(".json").write_text(json.dumps({
            "session_id": session_id, "mb_id": mb_id, "phase": phase,
            "step": step, "epoch": None,
        }, sort_keys=True) + "\n")

    def transcript(self, session_id):
        root = getattr(self.args, "transcript_dir", None)
        if not root:
            return None
        return RemoteTranscript(
            root, session_id,
            state_interval=getattr(self.args, "transcript_state_interval", 1))

    @staticmethod
    def snapshot(transcript, session, session_id, reason):
        if transcript is None:
            return
        transcript.record_state({
            "schema": "dtraining.remote_transcript.state.v1",
            "session_id": session_id,
            "optimizer_step": int(session["step"]),
            "model_state_dict": session["model"].state_dict(),
            "optimizer_state_dict": session["optimizer"].state_dict(),
        }, step=int(session["step"]), reason=reason)

    async def handler(self, websocket):
        session_id = str(uuid.uuid4())
        session = None
        transcript = None
        clean_close = False
        try:
            hello_message = await websocket.recv()
            if not isinstance(hello_message, str):
                raise ValueError("latent-native-v5 text hello required")
            hello = json.loads(hello_message)
            transcript = self.transcript(session_id)
            if transcript is not None:
                transcript.record_text("received", hello_message, event="hello",
                                       parsed=hello)
            if (hello.get("op") != "hello"
                    or hello.get("protocol") != "latent-native-v5"):
                raise ValueError("latent-native-v5 hello required")
            if hello.get("latent_dim") != self.args.latent_dim:
                raise ValueError("requested latent width does not match server")
            if hello.get("wire_dtype") != "fp32":
                raise ValueError("only explicit fp32 latent wire is supported")
            if hello.get("cloud_kind", "transformer") != self.args.cloud_kind:
                raise ValueError("requested cloud kind does not match server")
            if int(hello.get("cloud_experts", 1)) != self.args.cloud_experts:
                raise ValueError("requested expert count does not match server")
            if int(hello.get("cloud_layers", self.args.cloud_layers)
                   ) != self.args.cloud_layers:
                raise ValueError("requested layer count does not match server")
            if int(hello.get("cloud_hidden", 0)
                   ) != getattr(self.args, "cloud_hidden", 0):
                raise ValueError("requested hidden width does not match server")
            cloud_seed = hello.get("cloud_seed")
            if (not isinstance(cloud_seed, int)
                    or isinstance(cloud_seed, bool) or cloud_seed < 0):
                raise ValueError("non-negative integer cloud_seed required")
            model = self.new_session_model(cloud_seed)
            session = {
                "pending": {},
                "model": model,
                "optimizer": torch.optim.AdamW(
                    model.parameters(), lr=float(hello.get("lr", 3e-4))),
                "active_delta": float(hello.get("active_delta", 0.0)),
                "tampered": 0,
                "step": 0,
            }
            self.sessions[session_id] = session
            shapes = {name: list(value.shape)
                      for name, value in model.state_dict().items()}
            digest = hashlib.sha256(json.dumps(
                shapes, sort_keys=True).encode()).hexdigest()
            hello_ack = {
                "op": "hello_ack", "session_id": session_id,
                "latent_dim": self.args.latent_dim,
                "cloud_kind": self.args.cloud_kind,
                "cloud_experts": self.args.cloud_experts,
                "cloud_layers": self.args.cloud_layers,
                "cloud_hidden": getattr(self.args, "cloud_hidden", 0),
                "cloud_seed": cloud_seed,
                "latent_only_audit": True,
                "max_parameter_dimension": max(
                    max(value.shape or (1,))
                    for value in model.state_dict().values()),
                "state_shape_digest": digest,
            }
            hello_ack_message = json.dumps(hello_ack)
            await websocket.send(hello_ack_message)
            if transcript is not None:
                transcript.record_text("sent", hello_ack_message,
                                       event="hello_ack", parsed=hello_ack)
                self.snapshot(transcript, session, session_id, "initial")
            async for message in websocket:
                if isinstance(message, str):
                    control = json.loads(message)
                    if transcript is not None:
                        transcript.record_text("received", message,
                                               event=control.get("op", "unknown"),
                                               step=session["step"],
                                               parsed=control)
                    if control.get("op") == "optimizer_step":
                        step_before = session["step"]
                        session["optimizer"].step()
                        session["optimizer"].zero_grad(set_to_none=True)
                        # frames captured from here on belong to the next step
                        session["step"] += 1
                        step_ack = {
                            "op": "step_ack", "pending": len(session["pending"]),
                            "tampered": session["tampered"]}
                        step_ack_message = json.dumps(step_ack)
                        await websocket.send(step_ack_message)
                        if transcript is not None:
                            # step_ack echoes the pre-increment step: the
                            # transcript verifier pairs request/response
                            # events by (step, mb_id) identity.
                            transcript.record_text("sent", step_ack_message,
                                                   event="step_ack",
                                                   step=step_before,
                                                   parsed=step_ack)
                            self.snapshot(transcript, session, session_id,
                                          "after_optimizer_step")
                    elif control.get("op") == "close":
                        clean_close = True
                        break
                    else:
                        raise ValueError("unknown latent control operation")
                    continue
                header, payload = unpack_frame(message)
                shape = header.get("shape")
                validate_latent_shape(shape, self.args.latent_dim)
                self.max_seen_width = max(self.max_seen_width, shape[-1])
                value = tensor_from(payload, shape, self.device)
                mb_id = int(header["mb_id"])
                if transcript is not None:
                    transcript.record_wire("received", message,
                                           event=header.get("op", "unknown"),
                                           header=header, step=session["step"],
                                           mb_id=mb_id)
                if header.get("op") == "forward":
                    self.capture(session_id, mb_id, "fwd", value, session["step"])
                    training = bool(header.get("training"))
                    leaf = value.detach().requires_grad_(training)
                    started = time.perf_counter()
                    output = session["model"](leaf)
                    tampered = False
                    if session["active_delta"]:
                        direction = torch.sign(output.detach()).clamp(min=-1, max=1)
                        output = output + session["active_delta"] * direction
                        session["tampered"] += 1
                        tampered = True
                    if self.args.malicious_delta:
                        # True active adversary: perturbs returns and does NOT
                        # report it (unlike active_delta, which is declared in
                        # the reply metadata). For the Byzantine-verify test.
                        direction = torch.sign(output.detach()).clamp(min=-1, max=1)
                        output = output + self.args.malicious_delta * direction
                    if training:
                        session["pending"][mb_id] = (leaf, output)
                    self.capture(session_id, mb_id, "return", output, session["step"])
                    response = pack_frame({
                        "op": "forward_result", "mb_id": mb_id,
                        "shape": list(output.shape), "tampered": tampered,
                        "cloud_ms": 1000 * (time.perf_counter() - started),
                    }, tensor_bytes(output))
                    await websocket.send(response)
                    if transcript is not None:
                        response_header, _ = unpack_frame(response)
                        transcript.record_wire("sent", response,
                                               event="forward_result",
                                               header=response_header,
                                               step=session["step"], mb_id=mb_id)
                elif header.get("op") == "backward":
                    if mb_id not in session["pending"]:
                        raise ValueError("backward has no pending forward")
                    leaf, output = session["pending"].pop(mb_id)
                    self.capture(session_id, mb_id, "bwd", value, session["step"])
                    torch.autograd.backward(output, value)
                    response = pack_frame({
                        "op": "backward_result", "mb_id": mb_id,
                        "shape": list(leaf.grad.shape),
                    }, tensor_bytes(leaf.grad))
                    await websocket.send(response)
                    if transcript is not None:
                        response_header, _ = unpack_frame(response)
                        transcript.record_wire("sent", response,
                                               event="backward_result",
                                               header=response_header,
                                               step=session["step"], mb_id=mb_id)
                else:
                    raise ValueError("unknown latent frame operation")
        except Exception as exc:
            traceback.print_exc()
            if transcript is not None:
                transcript.record_error(str(exc))
            try:
                await websocket.send(pack_frame({"op": "error",
                                                 "error": str(exc)}))
            except Exception:
                pass
        finally:
            self.sessions.pop(session_id, None)
            if transcript is not None:
                transcript.finalize("complete" if clean_close else "incomplete")
            print(json.dumps({"session": session_id, "event": "closed",
                              "max_seen_width": self.max_seen_width}))


async def serve(args):
    server = LatentServer(args)
    tls_context = None
    if args.tls_cert or args.tls_key:
        import ssl
        if not (args.tls_cert and args.tls_key):
            raise ValueError("--tls-cert and --tls-key are required together")
        tls_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls_context.minimum_version = ssl.TLSVersion.TLSv1_3
        tls_context.load_cert_chain(args.tls_cert, args.tls_key)
    async with websockets.serve(server.handler, "0.0.0.0", args.port,
                                max_size=64 * 1024 * 1024, ssl=tls_context,
                                ping_interval=None):
        print(json.dumps({"status": "ready", "port": args.port,
                          "tls": tls_context is not None,
                          "latent_dim": args.latent_dim,
                          "forbidden_hidden_dim": args.forbidden_hidden_dim}))
        await asyncio.Future()


def build_parser():
    """Standalone seam so torch-less hosts can exercise the CLI surface."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--forbidden-hidden-dim", type=int, default=1024)
    parser.add_argument("--cloud-layers", type=int, default=2)
    parser.add_argument("--cloud-heads", type=int, default=4)
    parser.add_argument("--cloud-kind",
                        choices=("transformer", "equivariant", "monomial",
                                 "monomial_moe", "monomial_moe_radial",
                                 "invariant_mlp", "invariant_mlp_deep"),
                        default="transformer")
    parser.add_argument("--cloud-experts", type=int, default=1,
                        help="expert count for the monomial_moe cloud")
    parser.add_argument("--cloud-hidden", type=int, default=0,
                        help="internal latent-space width for the "
                             "invariant_mlp_deep cloud (0 = unset)")
    parser.add_argument("--device",
                        default="cuda" if torch is not None
                        and torch.cuda.is_available() else "cpu")
    parser.add_argument("--port", type=int, default=5013)
    parser.add_argument("--tls-cert", help="PEM certificate for wss serving")
    parser.add_argument("--tls-key", help="PEM private key for wss serving")
    parser.add_argument("--capture-dir")
    parser.add_argument("--transcript-dir",
                        help="remote-node directory for a complete-view, "
                             "SHA-256-manifested session transcript")
    parser.add_argument("--transcript-state-interval", type=_positive_int,
                        default=1,
                        help="persist a remote state checkpoint only every N "
                             "optimizer steps (default 1 = every step). A 40k-step "
                             "session otherwise writes ~40k state checkpoints on "
                             "top of the message files; the transcript manifest "
                             "declares N so the verifier still passes")
    parser.add_argument("--malicious-delta", type=float, default=0.0,
                        help="true active adversary: perturb returns without "
                             "reporting it (Byzantine-verify test only)")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if (torch is None or websockets is None
            or build_ucn_latent_middle is None):
        parser.error("torch/websockets/privacy_runtime not installed; "
                     "--help works without them")
    if args.device == "cuda":
        # Force CUDA context init before serving: a cold event loop can stall
        # the first connection's TLS handshake.
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
