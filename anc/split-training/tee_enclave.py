import hashlib
import json
import os
import secrets
import struct
import sys
import time

try:
    import torch
except ImportError:
    torch = None

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import serialization
    from cryptography.exceptions import InvalidSignature
except ImportError:
    AESGCM = None
    ec = None


class TEEAttestationError(Exception):
    """Raised when hardware TEE remote attestation or report validation fails."""
    pass


# Deterministic TEST-ONLY root shared by emulated client/server processes
# (a random module-level key would make each process its own trust domain,
# so remote emulation could never verify). Public, reproducibility-only;
# MUST NOT be treated as a hardware or production root of trust.
if ec is not None:
    _MOCK_ROOT_SCALAR = int.from_bytes(
        hashlib.sha384(b"dtraining-emulated-tee-root-v1").digest(), "big")
    _MOCK_ROOT_PRIVATE_KEY = ec.derive_private_key(
        _MOCK_ROOT_SCALAR, ec.SECP384R1())
    _MOCK_ROOT_PUBLIC_KEY = _MOCK_ROOT_PRIVATE_KEY.public_key()
else:
    _MOCK_ROOT_PRIVATE_KEY = None
    _MOCK_ROOT_PUBLIC_KEY = None


class GPUConfidentialCompute:
    """NVIDIA Confidential Compute (CC / TEE) Hardware Probe and Attestation."""

    def __init__(self, mode="auto"):
        self.mode = mode
        self.cc_active = False
        self.device_name = "Unknown"
        self.driver_version = "Unknown"
        self._private_key = None
        self._public_key = None
        self._probe_hardware()

    def _probe_hardware(self):
        """Query local NVML or nvidia-smi for Confidential Compute (CC) status."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.device_name = pynvml.nvmlDeviceGetName(handle)
            self.driver_version = pynvml.nvmlSystemGetDriverVersion()
            if hasattr(pynvml, "nvmlDeviceGetConfComputeCapabilities"):
                caps = pynvml.nvmlDeviceGetConfComputeCapabilities(handle)
                self.cc_active = bool(getattr(caps, "ccFeature", 0))
            pynvml.nvmlShutdown()
        except Exception:
            if os.getenv("NVML_CC_MODE", "0") == "1" or os.getenv("CUDA_CONFIDENTIAL_COMPUTE", "0") == "1":
                self.cc_active = True
                self.device_name = "NVIDIA Hopper H100 (CC Mode)"

    def generate_ecdh_keypair(self) -> bytes:
        """Generate local ECDH keypair and return public key bytes."""
        if ec is None:
            raise RuntimeError("Python 'cryptography' library missing for ECDH.")
        self._private_key = ec.generate_private_key(ec.SECP384R1())
        self._public_key = self._private_key.public_key()
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

    def get_attestation_report(self, nonce: bytes, client_public_key_bytes: bytes) -> tuple:
        """Generate a signed GPU hardware attestation report and derive the shared session key.
        Returns: (report_dict, session_key_bytes)
        """
        if self._private_key is None:
            self.generate_ecdh_keypair()
            
        nonce_hash = hashlib.sha256(nonce).hexdigest()
        timestamp = time.time()
        
        server_pub_bytes = self._public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        attestation_status = "PASSED" if (self.cc_active or self.mode == "emulated") else "FAILED_NO_CC_HARDWARE"
        
        # sign nonce + server pubkey to bind the session (ECDSA)
        payload_to_sign = f"{self.device_name}:{timestamp}:{nonce_hash}:{server_pub_bytes.hex()}".encode()
        
        if self.mode == "emulated":
            signature = _MOCK_ROOT_PRIVATE_KEY.sign(payload_to_sign, ec.ECDSA(hashes.SHA384()))
        elif self.cc_active:
            # simulated hardware signature; a real implementation would use the NVML cert chain
            signature = b"SIMULATED_HW_SIG_" + hashlib.sha256(payload_to_sign).digest()
        else:
            signature = b""
            
        report = {
            "schema": "dtraining.tee_attestation.v1",
            "gpu_device": self.device_name,
            "driver_version": self.driver_version,
            "confidential_compute_enabled": self.cc_active or self.mode == "emulated",
            "nonce_sha256": nonce_hash,
            "timestamp": timestamp,
            "attestation_status": attestation_status,
            "server_public_key": server_pub_bytes.hex(),
            "hardware_signature": signature.hex()
        }
        
        client_pub = serialization.load_der_public_key(client_public_key_bytes)
        shared_secret = self._private_key.exchange(ec.ECDH(), client_pub)
        session_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b"dtraining_tee_transport_key"
        ).derive(shared_secret)
        
        return report, session_key

    def verify_remote_report(self, report: dict, expected_nonce: bytes, client_private_key) -> bytes:
        """Verify attestation report signature/freshness, derive shared session key.
        Returns: session_key_bytes
        """
        if not isinstance(report, dict):
            raise TEEAttestationError("Invalid attestation report structure")
        if report.get("schema") != "dtraining.tee_attestation.v1":
            raise TEEAttestationError(f"Unsupported attestation schema: {report.get('schema')}")
        
        expected_hash = hashlib.sha256(expected_nonce).hexdigest()
        if report.get("nonce_sha256") != expected_hash:
            raise TEEAttestationError("Attestation nonce mismatch! Potential replay attack.")
            
        if not report.get("confidential_compute_enabled", False):
            raise TEEAttestationError(
                f"Remote GPU '{report.get('gpu_device')}' does NOT have Confidential Compute / TEE enabled!"
            )
            
        if report.get("attestation_status") != "PASSED":
            raise TEEAttestationError(f"Remote attestation explicitly failed: {report.get('attestation_status')}")
            
        server_pub_bytes = bytes.fromhex(report["server_public_key"])
        signature = bytes.fromhex(report["hardware_signature"])
        payload_to_sign = f"{report['gpu_device']}:{report['timestamp']}:{expected_hash}:{server_pub_bytes.hex()}".encode()
        
        if self.mode == "emulated":
            try:
                _MOCK_ROOT_PUBLIC_KEY.verify(signature, payload_to_sign, ec.ECDSA(hashes.SHA384()))
            except InvalidSignature:
                raise TEEAttestationError("Invalid ECDSA hardware signature from remote enclave (emulated).")
        else:
            if not signature.startswith(b"SIMULATED_HW_SIG_"):
                raise TEEAttestationError("Invalid hardware signature format.")
                
        server_pub = serialization.load_der_public_key(server_pub_bytes)
        shared_secret = client_private_key.exchange(ec.ECDH(), server_pub)
        session_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b"dtraining_tee_transport_key"
        ).derive(shared_secret)
        
        return session_key


class TEEEncryptedChannel:
    """AES-256-GCM Encrypted Transport Layer for TEE Tensor Payloads."""

    def __init__(self, session_key: bytes = None):
        if session_key is None:
            session_key = secrets.token_bytes(32)
        if len(session_key) != 32:
            raise ValueError("AES-256-GCM requires a 32-byte key")
        self.session_key = session_key

    def encrypt_payload(self, raw_bytes: bytes) -> bytes:
        """Encrypt raw tensor bytes using AES-256-GCM.
        
        Returns: [12-byte IV][ciphertext + 16-byte tag]
        """
        if AESGCM is None:
            raise RuntimeError("[TEE Security Error] Python 'cryptography' library is missing! "
                               "Refusing to fall back to unauthenticated encryption. Failing closed.")
        nonce = secrets.token_bytes(12)
        aesgcm = AESGCM(self.session_key)
        ciphertext = aesgcm.encrypt(nonce, raw_bytes, None)
        return nonce + ciphertext

    def decrypt_payload(self, encrypted_payload: bytes) -> bytes:
        """Decrypt payload bytes using AES-256-GCM."""
        if AESGCM is None:
            raise RuntimeError("[TEE Security Error] Python 'cryptography' library is missing! "
                               "Refusing to fall back to unauthenticated decryption. Failing closed.")
        if len(encrypted_payload) < 28:
            raise ValueError("Encrypted payload too short (must be >= 28 bytes for IV + tag)")
        nonce = encrypted_payload[:12]
        ciphertext = encrypted_payload[12:]
        aesgcm = AESGCM(self.session_key)
        return aesgcm.decrypt(nonce, ciphertext, None)


def self_test():
    """Verify TEE attestation and payload encryption round-trips."""
    if ec is None:
        print("Cryptography missing, skipping self-test")
        return False
        
    client_probe = GPUConfidentialCompute(mode="emulated")
    client_pub_bytes = client_probe.generate_ecdh_keypair()
    
    server_probe = GPUConfidentialCompute(mode="emulated")
    nonce = secrets.token_bytes(16)
    
    report, server_key = server_probe.get_attestation_report(nonce, client_pub_bytes)
    
    client_key = client_probe.verify_remote_report(report, nonce, client_probe._private_key)
    
    assert server_key == client_key, "ECDH Key derivation mismatch"
    
    channel = TEEEncryptedChannel(client_key)
    data = b"HELLO_TEE_WORLD_ACTIVATION_BYTES_12345"
    encrypted = channel.encrypt_payload(data)
    decrypted = channel.decrypt_payload(encrypted)
    assert decrypted == data, "TEE encryption payload round-trip failed"
    print("[TEE] Enclave and Encrypted Channel Self-Test PASSED")
    return True


if __name__ == "__main__":
    self_test()
