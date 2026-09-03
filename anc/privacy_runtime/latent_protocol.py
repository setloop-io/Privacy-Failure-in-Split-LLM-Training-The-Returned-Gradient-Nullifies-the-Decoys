"""Synchronous client for the fail-closed latent-only cloud protocol."""

from __future__ import annotations

import json
import struct


def _pack_frame(header: dict, payload: bytes = b"") -> bytes:
    encoded = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack(">I", len(encoded)) + encoded + payload


def _unpack_frame(message: bytes) -> tuple[dict, bytes]:
    size = struct.unpack(">I", message[:4])[0]
    return json.loads(message[4:4 + size]), message[4 + size:]


class RemoteLatentCloud:
    def __init__(self, url: str, latent_dim: int, lr: float,
                 active_delta: float = 0.0, cloud_kind: str = "transformer",
                 cloud_seed: int | None = None, tls_ca: str | None = None,
                 cloud_experts: int = 1, cloud_layers: int = 2,
                 cloud_hidden: int = 0):
        import hashlib
        import secrets as _secrets
        import ssl

        import torch
        from websockets.sync.client import connect

        # The trusted side's RNG seed must never cross the wire: an
        # explicitly provided seed is domain-separated one-way (reproducible
        # yet uninformative about the trusted stream); no seed -> fresh
        # random per connection (strongest per-session isolation).
        if cloud_seed is None:
            cloud_seed = _secrets.randbits(63)
        else:
            cloud_seed = int.from_bytes(
                hashlib.sha256(
                    b"dtraining/latent-cloud-seed/v1\x00"
                    + int(cloud_seed).to_bytes(8, "big")).digest()[:8],
                "big")

        self.torch = torch
        self.latent_dim = int(latent_dim)
        tls_context = None
        if url.startswith("wss://"):
            # Pinned CA: trust only the provided CA, never the system store.
            if not tls_ca:
                raise ValueError("wss:// cloud requires a pinned --tls-ca")
            tls_context = ssl.create_default_context(
                ssl.Purpose.SERVER_AUTH, cafile=tls_ca)
            tls_context.check_hostname = True
            tls_context.minimum_version = ssl.TLSVersion.TLSv1_3
        elif tls_ca:
            raise ValueError("--tls-ca given but cloud URL is not wss://")
        # Retry broadly: under heavy startup load (model + corpus load),
        # the first handshake intermittently dies mid-upgrade
        # (InvalidMessage/EOFError).
        last_error = None
        for attempt in range(6):
            try:
                self.ws = connect(url, max_size=64 * 1024 * 1024,
                                  ssl=tls_context, ping_interval=None,
                                  close_timeout=10, open_timeout=15)
                break
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                last_error = exc
                import time
                time.sleep(min(2 * (attempt + 1), 8))
        else:
            raise RuntimeError(
                f"latent cloud unreachable after retries: {last_error}")
        self.ws.send(json.dumps({
            "op": "hello", "protocol": "latent-native-v5",
            "latent_dim": self.latent_dim, "wire_dtype": "fp32",
            "lr": float(lr), "active_delta": float(active_delta),
            "cloud_kind": cloud_kind, "cloud_seed": int(cloud_seed),
            "cloud_experts": int(cloud_experts),
            "cloud_layers": int(cloud_layers),
            "cloud_hidden": int(cloud_hidden),
        }))
        ack = json.loads(self.ws.recv())
        if ack.get("op") != "hello_ack":
            raise RuntimeError(f"latent cloud rejected hello: {ack}")
        if ack.get("latent_dim") != self.latent_dim:
            raise RuntimeError("latent width acknowledgement mismatch")
        if not ack.get("latent_only_audit"):
            raise RuntimeError("cloud did not pass its latent-only audit")
        if ack.get("cloud_kind") != cloud_kind:
            raise RuntimeError("cloud kind acknowledgement mismatch")
        if int(ack.get("cloud_experts", 1)) != int(cloud_experts):
            raise RuntimeError("cloud expert count acknowledgement mismatch")
        if ack.get("cloud_layers") is not None and int(
                ack["cloud_layers"]) != int(cloud_layers):
            raise RuntimeError("cloud layer count acknowledgement mismatch")
        if int(ack.get("cloud_hidden", 0)) != int(cloud_hidden):
            raise RuntimeError("cloud hidden width acknowledgement mismatch")
        if ack.get("cloud_seed") != int(cloud_seed):
            raise RuntimeError("cloud seed acknowledgement mismatch")
        self.session_id = ack["session_id"]
        self.audit = ack
        self._next_mb = 0

    @staticmethod
    def _bytes(tensor) -> bytes:
        return tensor.detach().float().contiguous().cpu().numpy().tobytes()

    def _tensor(self, payload: bytes, shape, device):
        value = self.torch.frombuffer(bytearray(payload),
                                      dtype=self.torch.float32)
        return value.reshape(shape).clone().to(device)

    def forward(self, latent, training: bool = True):
        if latent.ndim != 3 or latent.shape[-1] != self.latent_dim:
            raise ValueError("remote cloud input must be [batch,tokens,D]")
        mb_id = self._next_mb
        self._next_mb += 1
        header = {"op": "forward", "mb_id": mb_id,
                  "shape": list(latent.shape), "training": bool(training)}
        self.ws.send(_pack_frame(header, self._bytes(latent)))
        reply, payload = _unpack_frame(self.ws.recv())
        if reply.get("op") == "error":
            raise RuntimeError(reply.get("error"))
        if reply.get("op") != "forward_result" or reply.get("mb_id") != mb_id:
            raise RuntimeError("invalid latent-cloud forward response")
        output = self._tensor(payload, reply["shape"], latent.device)
        if training:
            output.requires_grad_(True)
        return output, mb_id, reply

    def forward_many(self, latents, window: int = 64):
        """Coalesce independent evaluation blocks into bounded requests.

        TLN applies an independent coordinate rotation, row permutation and
        token gauge before this method is called. Coalescing only stacks those
        already-protected blocks on the batch axis; no secret is shared and no
        canonical ordering is restored. Bounded requests avoid one RTT and one
        tiny cloud-kernel launch per block without unbounded socket buffering.
        """
        if window <= 0:
            raise ValueError("forward_many window must be positive")
        results = []
        for start in range(0, len(latents), window):
            chunk = latents[start:start + window]
            if not chunk:
                continue
            shape = chunk[0].shape
            if any(value.ndim != 3
                   or value.shape[-1] != self.latent_dim
                   or value.shape[1:] != shape[1:]
                   or value.device != chunk[0].device for value in chunk):
                raise ValueError(
                    "coalesced cloud inputs must share [tokens,D] and device")
            sizes = [value.shape[0] for value in chunk]
            coalesced = self.torch.cat(chunk, dim=0)
            output, mb_id, reply = self.forward(coalesced, training=False)
            pieces = output.split(sizes, dim=0)
            results.extend((piece, mb_id, dict(reply)) for piece in pieces)
        return results

    def backward(self, mb_id: int, grad_output):
        header = {"op": "backward", "mb_id": int(mb_id),
                  "shape": list(grad_output.shape)}
        self.ws.send(_pack_frame(header, self._bytes(grad_output)))
        reply, payload = _unpack_frame(self.ws.recv())
        if reply.get("op") == "error":
            raise RuntimeError(reply.get("error"))
        if reply.get("op") != "backward_result" or reply.get("mb_id") != mb_id:
            raise RuntimeError("invalid latent-cloud backward response")
        return self._tensor(payload, reply["shape"], grad_output.device)

    def step(self):
        self.ws.send(json.dumps({"op": "optimizer_step"}))
        reply = json.loads(self.ws.recv())
        if reply.get("op") != "step_ack":
            raise RuntimeError(f"cloud optimizer step failed: {reply}")
        return reply

    def close(self):
        try:
            self.ws.send(json.dumps({"op": "close"}))
        finally:
            self.ws.close()
