"""CSPRNG-driven request randomization and replay-resistant sampling."""

from __future__ import annotations

import hashlib
import json
import secrets


class ReplayResistantSampler:
    """Sample private block identifiers without exposing deterministic order."""

    def __init__(self, block_count: int):
        if block_count <= 0:
            raise ValueError("block_count must be positive")
        self.block_count = block_count
        self._seen_nonces = set()

    def sample(self, count: int) -> tuple[list[int], str]:
        if not 0 < count <= self.block_count:
            raise ValueError("invalid sample count")
        indices = list(range(self.block_count))
        for i in range(self.block_count - 1, 0, -1):
            j = secrets.randbelow(i + 1)
            indices[i], indices[j] = indices[j], indices[i]
        nonce = secrets.token_hex(16)
        return indices[:count], nonce

    def public_commitment(self, indices: list[int], nonce: str) -> str:
        payload = json.dumps({"indices": indices, "nonce": nonce},
                             sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    def accept_nonce(self, nonce: str) -> None:
        if not isinstance(nonce, str) or len(nonce) < 32:
            raise ValueError("nonce must contain at least 128 bits")
        if nonce in self._seen_nonces:
            raise RuntimeError("request nonce replay refused")
        self._seen_nonces.add(nonce)

    def refuse_replay(self, nonce: str) -> None:
        """Compatibility alias: accept a fresh nonce or refuse its replay."""
        self.accept_nonce(nonce)


def randomize_template(slots: list[str], padding_tokens: list[str],
                       minimum_padding: int = 1,
                       maximum_padding: int = 8) -> tuple[str, dict]:
    """Permute semantic slots and insert CSPRNG-selected dummy padding."""
    if not slots or not padding_tokens:
        raise ValueError("slots and padding_tokens are required")
    if not 0 <= minimum_padding <= maximum_padding:
        raise ValueError("invalid padding range")
    shuffled = list(slots)
    for i in range(len(shuffled) - 1, 0, -1):
        j = secrets.randbelow(i + 1)
        shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
    pad_count = minimum_padding + secrets.randbelow(
        maximum_padding - minimum_padding + 1)
    pads = [padding_tokens[secrets.randbelow(len(padding_tokens))]
            for _ in range(pad_count)]
    combined = shuffled + pads
    for i in range(len(combined) - 1, 0, -1):
        j = secrets.randbelow(i + 1)
        combined[i], combined[j] = combined[j], combined[i]
    return " ".join(combined), {
        "slot_count": len(slots), "padding_count": pad_count,
        "manifest_commitment": hashlib.sha256(
            "\x1f".join(combined).encode()).hexdigest(),
    }
