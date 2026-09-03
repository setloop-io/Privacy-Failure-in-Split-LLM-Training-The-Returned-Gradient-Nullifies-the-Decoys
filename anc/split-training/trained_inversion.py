#!/usr/bin/env python3
"""Trained MLP inversion attack on split-TRAINING boundaries (E4).

Strong-attacker sequel to gradient_inversion.py. That DLG-style attack is a
WEAK attacker (<=15% token recovery at depth 1, ~0% at depth >= 4 on
Qwen3-0.6B); meanwhile the split-INFERENCE study showed a TRAINED MLP
decoder recovers ~63% of tokens from boundary activations at depth 2. This
script ports that attacker to the training boundary and asks the question
the training-privacy story needs answered:

    Does fine-tuning drift protect? I.e., does an attacker trained on
    (h*, token) pairs from the PUBLIC BASE model keep working as the
    local fine-tune moves the boundary distribution away from base?

Threat model: semi-honest cloud during fine-tuning observes, per
microbatch, the boundary activation h* AND the boundary gradient
g* = dL/dh at the split point (after d local layers). Goal: recover the
input tokens of the victim's microbatch.

Attack A (activation-based, the proven one): train the InversionDecoder
MLP (defined below) on
(h*_i, token_i) pairs collected from PUBLIC text (--corpus-file) passed
through the PUBLIC base model's first d layers. Apply it — unchanged — to
victim boundary activations captured at train steps {0, 10, 100} of a real
split fine-tune (full-FT and LoRA-local configs).

Attack B (gradient-enriched): same decoder but input is concat(h*, g*)
per position (g* collected on the base model the same way, via a
next-token CE backward at the boundary). Tests whether the gradient the
cloud already sees adds recovery power over activations alone.

Both boundaries. The split crosses the untrusted network twice: the local head
SENDS the activation after layer sa (carries the PROMPT) and the cloud RETURNS
the activation after layer ra-1 (carries the RESPONSE). The split-inference
study's write-up (not included in this release) calls
these equally shallow and asks for symmetric local depth; --boundary selects
which one is attacked and --tail-depths makes the output-side depth a variable
instead of the hardcoded 2 layers. Defaults (input, tail 2) reproduce the
original configuration exactly.

Honest-scope notes (results are an UPPER-threating bound for the cloud):
  - attack training pairs come from the base checkpoint (exact at step 0);
  - victim docs are the LAST --victim-docs docs of the corpus and never
    enter the attack training/val pools (document-disjoint);
  - input side: per-position prediction h*[i] -> token[i], same convention as
    defense_experiment.py. Output side: h*[i] -> token[i+1] (the token that
    position predicts) — see collect_output_pairs for the derivation;
  - every privacy figure is emitted with utility_loss (held-out CE) at the same
    checkpoint, because a recovery drop caused by model damage is not a defense.

Usage:
    python trained_inversion.py --help        # works without torch
    python trained_inversion.py --toy --quick # CPU machinery check
    python trained_inversion.py --model <hf-model> --corpus-file <docs.txt> --output ti.json
    python trained_inversion.py --boundary output --tail-depths 2 4 8 --model <hf-model>
"""

import argparse
import copy
import gc
import json
import os
import sys
import time

# Guarded heavy imports: `--help` must work on torch-less hosts.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None
    DataLoader = None
    TensorDataset = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from split_trainer import (TEXT_SAMPLES, CloudWorker, _write_training_status,
                               apply_lora, build_modules, make_layer_kwargs,
                               run_layer_stack, unique_params)
except ImportError:  # pragma: no cover - torch-less: split_trainer still imports
    TEXT_SAMPLES = []
    CloudWorker = None
    _write_training_status = lambda **k: None
    apply_lora = build_modules = make_layer_kwargs = run_layer_stack = None
    unique_params = None


# Inversion decoder MLP (input dim widened for Attack B's concat(h*, g*)).
if nn is not None:

    class InversionDecoder(nn.Module):
        """Small MLP that predicts token IDs from split-point features."""

        def __init__(self, in_dim, vocab_size, intermediate_dim=2048):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, intermediate_dim),
                nn.GELU(),
                nn.LayerNorm(intermediate_dim),
                nn.Linear(intermediate_dim, intermediate_dim),
                nn.GELU(),
                nn.LayerNorm(intermediate_dim),
                nn.Linear(intermediate_dim, vocab_size),
            )

        def forward(self, x):
            return self.net(x)


def seed_all(seed):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Numerical precision, recorded rather than inferred.
#
# `--dtype fp32` sets the MODEL dtype and never touched the fp32 matmul mode.
# The NGC container ships float32_matmul_precision="high", so every "fp32"
# matmul in this family actually executed as TF32. Traceable magnitude, from
# the committed pair e8_obfuscation_06b_fp32_{tf32,highest}.json: over the same
# six cells, "highest" tightens max_roundtrip_deviation_relative by 366x-1487x
# (widest cell, depth 4 prompt 0: 3.3296e-04 -> 2.2390e-07). These helpers make
# the mode selectable AND make every artifact state what actually ran, read
# back out of torch -- never echoed from a flag.

MATMUL_PRECISION_CHOICES = ("highest", "high", "medium")

# cuBLAS needs this at handle-creation time to make its reduction order
# reproducible; torch.use_deterministic_algorithms() raises without it.
CUBLAS_DETERMINISTIC_WORKSPACE = ":4096:8"


def add_numerics_args(ap):
    """The two numerics flags, identical across the E8 scripts.

    Both default to "leave torch alone" so an invocation that omits them
    keeps the container's defaults untouched.
    """
    ap.add_argument("--matmul-precision", choices=MATMUL_PRECISION_CHOICES,
                    default=None,
                    help="torch fp32 matmul mode. Default: inherit the "
                         "container's (NGC ships 'high' = TF32). 'highest' is "
                         "true fp32 and measured 366x-1487x tighter at the "
                         "boundary round-trip across the six committed cells.")
    ap.add_argument("--deterministic", action="store_true",
                    help="torch.use_deterministic_algorithms(True) + the cuBLAS "
                         "deterministic workspace, to pin reduction order. "
                         "'highest' alone does NOT make cuBLAS reproducible.")


def apply_numerics(args):
    """Apply the numerics flags, then report what torch says actually took hold.

    Returns the dict that goes into `config`. Read-back, not echo: if a request
    silently fails to take effect, the artifact records the truth.
    """
    if getattr(args, "matmul_precision", None):
        torch.set_float32_matmul_precision(args.matmul_precision)
        # matmul mode does not cover cuDNN; no convolutions here, so this
        # cannot move a number, but "true fp32" should mean it everywhere.
        torch.backends.cudnn.allow_tf32 = args.matmul_precision != "highest"
    if getattr(args, "deterministic", False):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",
                              CUBLAS_DETERMINISTIC_WORKSPACE)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    return numerics_config()


def numerics_config():
    """The effective numerics, read back out of torch and the environment."""
    return {
        "matmul_precision_effective": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_version": torch.__version__,
    }


def tensor_sha256(t):
    """sha256 of a tensor's raw bytes, for cross-run identity checks."""
    import hashlib
    a = t.detach().cpu().contiguous()
    return hashlib.sha256(memoryview(a.numpy()).tobytes()).hexdigest()


# Provenance recorded in every result JSON: which code, which corpus bytes,
# which docs were the victims, on which host/image. MODEL_MANIFEST_ALGO is
# bumped whenever the manifest definition changes, so a v1 hash is never
# compared against a v2 one. v1 walked the whole directory including
# .cache/huggingface/download/*, whose per-download .lock/.metadata sizes made
# the hash fingerprint the DOWNLOAD, not the model: two byte-identical copies
# of qwen3-0.6b on two nodes produced different manifests, and 23 of that
# model's 33 files were cache bookkeeping. v1 hashes are therefore not
# evidence of anything and must not be compared across machines.
MODEL_MANIFEST_ALGO = "v2-sizes-no-cache"

# Directories excluded from the manifest: tool bookkeeping, not model content.
MANIFEST_EXCLUDED_DIRS = (".cache", ".git", "__pycache__")


def manifest_entries(model_path):
    """Sorted (relpath, size) for the files that DEFINE the model.

    Pure and torch-less so --self-test can exercise it. Mirrors the exclusion
    in bin/make_manifest.sh.
    """
    entries = []
    for root, dirs, files in os.walk(model_path):
        dirs[:] = [d for d in dirs if d not in MANIFEST_EXCLUDED_DIRS]
        for fn in files:
            p = os.path.join(root, fn)
            entries.append((os.path.relpath(p, model_path), os.path.getsize(p)))
    return sorted(entries)


def model_manifest(model_path):
    """Substitution/reorder check over model files only.

    Size-based, not content-based: full content hashing of a 54GB model is not
    practical per run. `model_manifest_n_files` is reported alongside so an
    unexpected extra or missing file is visible rather than silently folded
    into the digest.
    """
    import hashlib
    entries = manifest_entries(model_path)
    h = hashlib.sha256()
    for rel, size in entries:
        h.update(f"{rel}\0{size}\n".encode())
    return {"model_manifest_sha256": h.hexdigest(),
            "model_manifest_algo": MODEL_MANIFEST_ALGO,
            "model_manifest_n_files": len(entries)}


def manifest_self_test():
    """Torch-less checks on the manifest definition."""
    import shutil
    import tempfile
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    root = tempfile.mkdtemp()
    try:
        def build(base):
            os.makedirs(os.path.join(base, ".cache", "huggingface", "download"),
                        exist_ok=True)
            for rel, body in (("config.json", b"{}"),
                              ("model.safetensors", b"\x00" * 1024),
                              ("tokenizer.json", b"tok")):
                with open(os.path.join(base, rel), "wb") as f:
                    f.write(body)

        a, b = os.path.join(root, "a"), os.path.join(root, "b")
        build(a)
        build(b)
        # Same model, different download bookkeeping — the v1 failure mode.
        for base, blob in ((a, b"x" * 7), (b, b"y" * 4096)):
            with open(os.path.join(base, ".cache", "huggingface", "download",
                                   "config.json.metadata"), "wb") as f:
                f.write(blob)

        ma, mb = model_manifest(a), model_manifest(b)
        check("identical models with differing .cache agree",
              ma["model_manifest_sha256"] == mb["model_manifest_sha256"])
        check("only model files counted (3, not the cache entries)",
              ma["model_manifest_n_files"] == 3)
        check("algo version recorded",
              ma["model_manifest_algo"] == MODEL_MANIFEST_ALGO)
        check("entries sorted by relpath",
              [r for r, _ in manifest_entries(a)] ==
              sorted(r for r, _ in manifest_entries(a)))

        # A real substitution must still be caught: same name, different size.
        with open(os.path.join(b, "model.safetensors"), "wb") as f:
            f.write(b"\x00" * 2048)
        check("size change IS detected",
              model_manifest(b)["model_manifest_sha256"] !=
              ma["model_manifest_sha256"])

        # An extra model file must be caught, and must move the file count.
        with open(os.path.join(a, "extra.safetensors"), "wb") as f:
            f.write(b"z")
        m2 = model_manifest(a)
        check("added model file IS detected",
              m2["model_manifest_sha256"] != ma["model_manifest_sha256"] and
              m2["model_manifest_n_files"] == 4)

        # Negative: cache churn alone must NOT move the digest.
        with open(os.path.join(a, ".cache", "huggingface", "download",
                               "another.lock"), "wb") as f:
            f.write(b"lock")
        check("cache churn alone does NOT move the digest",
              model_manifest(a)["model_manifest_sha256"] ==
              m2["model_manifest_sha256"])
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print("SELF-TEST PASSED" if ok else "SELF-TEST FAILED")
    return ok


def make_provenance(corpus_file, corpus_source, n_docs, victim_doc_indices,
                    model_path=None, docs=None):
    import hashlib
    import socket
    import subprocess
    victim_doc_indices = list(victim_doc_indices)
    prov = {"corpus_source": corpus_source,
            "corpus_n_docs": n_docs,
            "victim_doc_indices": victim_doc_indices,
            "container_image": os.environ.get("CONTAINER_IMAGE", "unknown"),
            "hostname": socket.gethostname()}
    # Hash the selected logical documents as well as the source file.  The
    # indices document the ordering contract; these hashes make a reordered
    # or edited corpus detectable even when the caller cannot retain the file.
    if docs is not None:
        prov["victim_doc_sha256"] = {
            str(i): hashlib.sha256(docs[i].encode("utf-8")).hexdigest()
            for i in victim_doc_indices if 0 <= i < len(docs)}
    else:
        prov["victim_doc_sha256"] = {}
    try:
        prov["dtraining_commit"] = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C",
             os.path.dirname(os.path.abspath(__file__)),
             "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:
        prov["dtraining_commit"] = "unknown"
    if model_path and os.path.isdir(model_path):
        try:
            prov.update(model_manifest(model_path))
            prov["model_path"] = os.path.basename(os.path.normpath(model_path))
        except Exception:
            prov["model_manifest_sha256"] = "unknown"
            prov["model_manifest_algo"] = MODEL_MANIFEST_ALGO
    if corpus_file:
        try:
            h = hashlib.sha256()
            with open(corpus_file, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            prov["corpus_sha256"] = h.hexdigest()
        except Exception:
            prov["corpus_sha256"] = "unknown"
    else:
        prov["corpus_sha256"] = None
    return prov


def mean_std(vals):
    m = sum(vals) / len(vals)
    v = sum((x - m) ** 2 for x in vals) / max(1, len(vals) - 1)
    return round(m, 4), round(v ** 0.5, 4)


DEFAULT_TAIL_DEPTH = 2  # layers the local node keeps at the END (output side)


def split_indices(depth, n_layers, tail_depth=DEFAULT_TAIL_DEPTH):
    """Boundary arithmetic only — no torch, so it is importable/testable on a
    torch-less host. Returns (sa, ra).

    sa = last layer kept locally on the INPUT side (head = 0..sa).
    ra = first layer kept locally on the OUTPUT side (tail = ra..n_layers-1),
         so the local node keeps `tail_depth` layers at the end.

    Clamps, unchanged from the original hardcoded form:
      - sa <= n_layers-3, leaving room for >=1 cloud layer and >=1 tail layer;
      - ra >= sa+2, so the cloud stack is never empty;
      - ra <= n_layers-1, so the tail is never empty.
    tail_depth=2 reproduces the previous `ra = min(max(n_layers-2, sa+2),
    n_layers-1)` exactly, which is why it is the default.
    """
    assert tail_depth >= 1, f"tail_depth must be >= 1, got {tail_depth}"
    sa = max(0, min(depth, n_layers - 3))
    ra = min(max(n_layers - tail_depth, sa + 2), n_layers - 1)
    return sa, ra


def split_at(layers, depth, n_layers, tail_depth=DEFAULT_TAIL_DEPTH):
    """Local head = 0..depth inclusive; cloud = depth+1..ra-1; tail = ra..
    Same clamping as gradient_inversion.split_at (toy clamps depths).
    `tail_depth` defaults to the original hardcoded 2-layer tail."""
    sa, ra = split_indices(depth, n_layers, tail_depth)
    return (nn.ModuleList(list(layers[: sa + 1])),
            nn.ModuleList(list(layers[sa + 1: ra])),
            nn.ModuleList(list(layers[ra:])), sa, ra)


# Boundary selection. The split crosses the untrusted network TWICE:
#
#   input side  -- activation the local head SENDS after layer sa. Carries the
#                  user's PROMPT.
#   output side -- activation the cloud RETURNS after layer ra-1, the tensor
#                  the local tail consumes. Carries the model's RESPONSE.
#
# Both are equally shallow in the deployed config (2 layers from the embedding,
# 2 from the LM head). Everything below is parameterised
# by which of the two is under attack.
#
# LABEL CONVENTION (the load-bearing choice; see collect_output_pairs).
def _boundary_leaf(head, cloud, hidden, lk, boundary):
    """Materialise the chosen network-crossing tensor as an autograd leaf, so
    torch.autograd.grad can report the gradient observed at that same seam."""
    if boundary == "input":
        return run_layer_stack(head, hidden, lk).detach().requires_grad_(True)
    # Output side: everything up to the seam is the head PLUS the whole cloud
    # stack (~28 layers on a 32-layer model). That graph is detached away
    # immediately, so build it under no_grad rather than allocate it. Values are
    # unchanged — grad mode does not affect dropout, which keys off .training.
    with torch.no_grad():
        h_out = run_layer_stack(cloud, run_layer_stack(head, hidden, lk), lk)
    return h_out.detach().requires_grad_(True)


def _downstream_loss(feat, cloud, tail, norm, lm_head, lk, labels, boundary):
    """Next-token CE recomputed from `feat` downstream, so the gradient wrt
    `feat` is exactly dL/d(boundary tensor)."""
    out = run_layer_stack(cloud, feat, lk) if boundary == "input" else feat
    out = run_layer_stack(tail, out, lk)
    logits = lm_head(norm(out))
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                           labels.reshape(-1))


def _collect_pairs(embed, head, cloud, tail, norm, lm_head, rotary, encode,
                   docs, args, with_grad, boundary):
    """Pair collection on the PUBLIC BASE model: run each doc's blocks through
    embed + head (+ cloud, output side), collect (feature[i], token[i]) pairs;
    optionally also g*[i] via a next-token CE backward at that boundary (full
    model, no optimizer step)."""
    assert boundary in ("input", "output"), boundary
    h_all, g_all, tok_all = [], [], []
    n_pairs = 0
    for doc in docs:
        for block in encode([doc], args.seq_len):
            ids = block[:-1].unsqueeze(0).to(args.device)
            labels = block[1:].unsqueeze(0).to(args.device)
            position_ids = torch.arange(ids.shape[1], device=args.device).unsqueeze(0)
            hidden = embed(ids)
            lk = make_layer_kwargs(rotary, hidden, position_ids, args)
            feat = _boundary_leaf(head, cloud, hidden, lk, boundary)
            if with_grad:
                loss = _downstream_loss(feat, cloud, tail, norm, lm_head, lk,
                                        labels, boundary)
                g_all.append(torch.autograd.grad(loss, feat)[0].detach()[0]
                             .float().cpu())
            h_all.append(feat.detach()[0].float().cpu())
            tok_all.append((ids if boundary == "input" else labels)[0].cpu())
            n_pairs += ids.shape[1]
            if n_pairs >= args.max_pairs:
                break
        if n_pairs >= args.max_pairs:
            break
    g_out = torch.cat(g_all) if with_grad else None
    return torch.cat(h_all), g_out, torch.cat(tok_all)


def collect_base_pairs(embed, head, cloud, tail, norm, lm_head, rotary,
                       encode, docs, args, with_grad):
    """INPUT-side pairs: (activation the cloud receives, token INPUT at that
    position). Signature is frozen -- e7_private_ft, e8_obfuscation,
    e8_robustness and seq_inversion all call it positionally,
    and seq_inversion passes cloud=tail=None (legal: with_grad=False never
    touches them)."""
    return _collect_pairs(embed, head, cloud, tail, norm, lm_head, rotary,
                          encode, docs, args, with_grad, "input")


def collect_output_pairs(embed, head, cloud, tail, norm, lm_head, rotary,
                         encode, docs, args, with_grad):
    """OUTPUT-side pairs: (activation the cloud RETURNS after layer ra-1, the
    NEXT token -- block[i+1] -- that position predicts).

    Why the next token and not the input token at that position:

    1. Function of the tensor. h_out[i] sits `tail_depth` layers + final norm +
       lm_head away from BEING the next-token distribution. As tail_depth -> 0
       the map h_out[i] -> block[i+1] converges to the model's own LM head, so
       recovery is bounded only by the model's own accuracy. That limit is what
       makes the number mean "how much of the answer can the cloud read off the
       return tensor". Labelling with block[i] has no such limit: it would
       measure residual-stream persistence of a token that entered the network
       upstream of the cloud, which is an input-side quantity already covered by
       collect_base_pairs.
    2. Threat model. On the return path the sensitive object is the RESPONSE.
       During autoregressive decoding the local tail turns h_out[i] into the
       emitted token block[i+1]; the cloud holding h_out[i] therefore sees the
       model's answer one tail-stack early. block[i+1] IS the leaked plaintext.
    3. Symmetry. Each side is labelled with the plaintext nearest to it in the
       computation: input side with the token the tensor is a function OF,
       output side with the token the tensor is a function FOR.

    Honest caveats for whoever reports these numbers:
      - collection is teacher-forced on corpus text, so block[i+1] is the
        GROUND-TRUTH continuation, not necessarily the model's argmax. Top-1 is
        therefore bounded by the model's own agreement with the corpus plus any
        extra leakage, NOT by 100%. At real generation time the emitted token is
        the argmax by construction, so this convention transfers and, if
        anything, understates the leak on self-generated text.
      - consequently the reference point for output-side recovery is the model's
        own next-token accuracy, not only uniform 1/vocab chance. Quote both.
      - a decoder scored against block[i] instead would also beat chance (late
        residual streams retain current-token identity). That is a different,
        weaker claim (the PREVIOUS, already-emitted response token) and is not
        the readout the paper's symmetric-vulnerability passage is about."""
    return _collect_pairs(embed, head, cloud, tail, norm, lm_head, rotary,
                          encode, docs, args, with_grad, "output")


# Real split fine-tune with boundary capture at train-step checkpoints.
# Same boundary-leaf protocol as split_trainer.train /
# gradient_inversion.run_split_training, in-process CloudWorker.
def finetune_and_capture(embed, head, cloud_layers, tail, norm, lm_head,
                         rotary, encode, train_docs, victim_ids, checkpoints,
                         args, config, eval_ids=None, boundary="input"):
    """Run a split fine-tune; after steps in `checkpoints` (0 = at init,
    before any update), capture per-position (h*, g*) for each victim block at
    the selected `boundary`. Both sides are genuinely observed by the cloud: it
    receives head_leaf and returns dL/d(head_leaf), and it produces tail_leaf
    and receives dL/d(tail_leaf) — both named in the step loop below.
    Returns {step: (h, g)} with h,g [total_positions, hidden] on CPU."""
    checkpoints = sorted(set(checkpoints))
    cloud = CloudWorker(cloud_layers, lr=args.lr,
                        trainable=(config == "full"))
    if config == "lora-local":
        n = apply_lora(nn.ModuleList([*head, *tail]), args.lora_rank,
                       args.lora_alpha)
        print(f"    [lora] wrapped {n} local projections "
              f"(rank={args.lora_rank}, alpha={args.lora_alpha})")
        # apply_lora freezes only the base weights of the q/v projections it
        # wraps; embed and lm_head stay trainable, which the name "lora-local"
        # does not imply — and at 27B (vocab 248320, untied embeddings) their
        # grads + AdamW moments are ~18GB, which OOM'd the depth-8 cell (11
        # local layers vs 4 at depth 1) on the 94GB H100. Freeze them so the
        # config trains local-layer weights + LoRA adapters only.
        for p in list(embed.parameters()) + list(lm_head.parameters()):
            p.requires_grad_(False)
    local_mods = [embed, *head, *tail, norm, lm_head]
    local_params = unique_params(local_mods)
    opt = torch.optim.AdamW(local_params, lr=args.lr) if local_params else None

    blocks = []
    for doc in train_docs:
        blocks.extend(encode([doc], args.seq_len))
    if not blocks:
        raise ValueError("no training blocks; enlarge corpus or shrink --seq-len")

    def capture():
        hs, gs = [], []
        for ids in victim_ids:
            ids = ids.to(args.device)
            input_ids, labels = ids[:-1].unsqueeze(0), ids[1:].unsqueeze(0)
            position_ids = torch.arange(input_ids.shape[1],
                                        device=args.device).unsqueeze(0)
            hidden = embed(input_ids)
            lk = make_layer_kwargs(rotary, hidden, position_ids, args)
            feat = _boundary_leaf(head, cloud_layers, hidden, lk, boundary)
            loss = _downstream_loss(feat, cloud_layers, tail, norm, lm_head, lk,
                                    labels, boundary)
            g = torch.autograd.grad(loss, feat)[0].detach()
            hs.append(feat.detach()[0].float().cpu())
            gs.append(g[0].float().cpu())
        return torch.cat(hs), torch.cat(gs)

    captured = {}
    utility = {}

    def heldout_loss():
        """Utility co-metric: CE loss on docs never seen in fine-tuning.
        (Must NOT be the FT docs themselves — that would measure memorization,
        as seen in the first corrected run where utility hit 0.035.)"""
        if not eval_ids:
            raise ValueError("empty held-out eval set — refusing to fall back to FT docs")
        eval_set = eval_ids
        losses = []
        with torch.no_grad():
            for ids in eval_set:
                ids = ids.to(args.device)
                input_ids, labels = ids[:-1].unsqueeze(0), ids[1:].unsqueeze(0)
                position_ids = torch.arange(input_ids.shape[1],
                                            device=args.device).unsqueeze(0)
                hidden = embed(input_ids)
                lk = make_layer_kwargs(rotary, hidden, position_ids, args)
                out = run_layer_stack(head, hidden, lk)
                out = run_layer_stack(cloud_layers, out, lk)
                out = run_layer_stack(tail, out, lk)
                logits = lm_head(norm(out))
                losses.append(F.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]),
                    labels.reshape(-1)).item())
        return sum(losses) / len(losses)

    if 0 in checkpoints:
        captured[0] = capture()
        utility[0] = heldout_loss()
        print(f"    [train] step 0 utility_loss={utility[0]:.4f}")
    max_step = checkpoints[-1]
    for step in range(1, max_step + 1):
        if opt is not None:
            opt.zero_grad(set_to_none=True)
        cloud.zero_grad()
        ids = blocks[(step - 1) % len(blocks)].unsqueeze(0).to(args.device)
        input_ids, labels = ids[:, :-1], ids[:, 1:]
        position_ids = torch.arange(input_ids.shape[1], device=args.device).unsqueeze(0)
        hidden = embed(input_ids)
        lk = make_layer_kwargs(rotary, hidden, position_ids, args)
        # NB: named *_leaf, not `boundary` — `boundary` is this function's
        # side-selector parameter and capture() closes over it.
        head_out = run_layer_stack(head, hidden, lk)
        head_leaf = head_out.detach().requires_grad_(True)   # input-side seam
        cloud_out = cloud.forward(head_leaf, lk)
        tail_leaf = cloud_out.detach().requires_grad_(True)  # output-side seam
        out = run_layer_stack(tail, tail_leaf, lk)
        logits = lm_head(norm(out))
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                               labels.reshape(-1))
        loss.backward()
        grad_input = cloud.backward(tail_leaf.grad)
        torch.autograd.backward(head_out, grad_tensors=grad_input)
        if opt is not None:
            opt.step()
        cloud.step()
        if step in checkpoints:
            captured[step] = capture()
            utility[step] = heldout_loss()
            print(f"    [train] step {step}/{max_step} loss={loss.item():.4f} "
                  f"utility={utility[step]:.4f} (captured boundary)")
        elif step % 10 == 0 or step == max_step:
            print(f"    [train] step {step}/{max_step} loss={loss.item():.4f}")
    if args.device.startswith("cuda"):
        print(f"    [mem] finetune peak: "
              f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    return captured, utility


# Decoder training / evaluation (mirrors defense_experiment.py).
def evaluate_decoder(decoder, feats, token_ids, device, batch_size=1024):
    decoder.eval()
    top1_hits, top5_hits, n = 0.0, 0.0, 0
    with torch.no_grad():
        for i in range(0, feats.shape[0], batch_size):  # batched: full-vocab logits don't fit at 131K/248K vocab
            xb = feats[i:i + batch_size].float().to(device)
            tb = token_ids[i:i + batch_size].to(device)
            logits = decoder(xb)
            top1_hits += (logits.argmax(dim=-1) == tb).sum().item()
            top5_hits += (logits.topk(min(5, logits.shape[-1]), dim=-1).indices
                          == tb.unsqueeze(1)).any(dim=1).sum().item()
            n += tb.shape[0]
    return round(100 * top1_hits / n, 2), round(100 * top5_hits / n, 2)


def train_decoder(train_x, train_y, val_x, val_y, in_dim, vocab_size, args, tag):
    device = args.device
    ds = TensorDataset(train_x.float().to(device), train_y.to(device))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    decoder = InversionDecoder(in_dim, vocab_size).to(device)
    opt = torch.optim.AdamW(decoder.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()
    best_val, best_state = -1.0, None
    for epoch in range(args.epochs):
        decoder.train()
        tot = 0.0
        for bx, by in dl:
            loss = crit(decoder(bx), by)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()
        vtop1, _ = evaluate_decoder(decoder, val_x, val_y, device)
        _write_training_status(phase="train", state="running", run_id=tag,
                               epoch=epoch + 1, epochs=args.epochs,
                               loss=round(tot / max(len(dl), 1), 4),
                               top1=vtop1, metric_name="val_top1",
                               metric_value=vtop1)
        if vtop1 > best_val:
            best_val = vtop1
            best_state = {k: v.detach().clone()
                          for k, v in decoder.state_dict().items()}
    if best_state is not None:
        decoder.load_state_dict(best_state)
    decoder.best_val_top1 = best_val
    return decoder


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.expanduser(
        "~/experiments/models/qwen3-0.6b"), help="HF model path (ignored with --toy)")
    ap.add_argument("--toy", action="store_true",
                    help="tiny random built-in model (CPU machinery check only; "
                         "depths are clamped to the toy's 4 layers)")
    ap.add_argument("--corpus-file", default=None,
                    help="public attack-training text, one document per line; "
                         "the LAST --victim-docs documents are held out as the "
                         "victim fine-tuning data and never enter attack training")
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 4, 8],
                    help="INPUT-side split depths (local layers 0..d)")
    ap.add_argument("--tail-depths", type=int, nargs="+",
                    default=[DEFAULT_TAIL_DEPTH],
                    help="OUTPUT-side depths: layers the local node keeps at "
                         "the END (tail = ra..n-1). Default 2 = the historical "
                         "hardcoded tail, so existing runs are unchanged. "
                         "paper.txt:354-358 recommends matching --depths "
                         "(e.g. --depths 3 --tail-depths 4 on 32 layers "
                         "= local layers 0-3 and 28-31)")
    ap.add_argument("--boundary", choices=["input", "output"], default="input",
                    help="which network-crossing tensor to attack. 'input': "
                         "activation sent after layer sa, labelled with the "
                         "token at that position (the user's PROMPT). "
                         "'output': activation returned after layer ra-1, "
                         "labelled with the NEXT token that position predicts "
                         "(the model's RESPONSE). See collect_output_pairs "
                         "for why the label differs")
    ap.add_argument("--configs", nargs="+", default=["full", "lora-local"],
                    choices=["full", "lora-local"])
    ap.add_argument("--train-steps-list", type=int, nargs="+", default=[0, 10, 100],
                    help="fine-tune steps at which the boundary is attacked")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--victim-docs", type=int, default=8,
                    help="documents held out from the END of the corpus as the "
                         "victim's private fine-tuning data")
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="fraction of attack docs held out (document-disjoint) "
                         "for best-epoch selection")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--max-pairs", type=int, default=20000,
                    help="cap on (feature, token) pairs per attack split")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-5,
                    help="fine-tune LR (default 1e-5: utility-safe; 1e-4 was shown to damage models — see REPORT review #4)")
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device",
                    default="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    ap.add_argument("--attn-impl", choices=["sdpa", "eager"], default="sdpa")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true",
                    help="depth 1, first --tail-depths entry only, both "
                         "configs, steps {0,2}, 1 seed, 2 victim docs, 5 "
                         "epochs, seq 16 (<=5 min CPU on --toy)")
    ap.add_argument("--output", default="trained_inversion_results.json")
    ap.add_argument("--self-test", action="store_true",
                    help="verify the model-manifest definition (issue #36); "
                         "torch-less, no model needed")
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(0 if manifest_self_test() else 1)

    if torch is None or build_modules is None:
        ap.error("torch/transformers not installed; install them or run --help only")

    if args.quick:
        args.depths = [1]
        args.tail_depths = args.tail_depths[:1]  # one grid cell, as promised
        args.train_steps_list = [0, 2]
        args.seeds = [0]
        args.victim_docs = 2
        args.epochs = 5
        args.seq_len = 16
        args.max_pairs = 2000

    seed_all(args.seed)
    _write_training_status(state="running", task="trained_inversion",
                           depths=args.depths, tail_depths=args.tail_depths,
                           boundary=args.boundary, configs=args.configs,
                           train_steps=args.train_steps_list,
                           started=time.strftime("%Y-%m-%dT%H:%M:%S%z"))

    embed, layers, norm, lm_head, rotary, encode = build_modules(args)
    n_layers = len(layers)
    vocab_size = (lm_head.weight.shape[0] if not args.toy
                  else embed.weight.shape[0])
    hidden_dim = embed.weight.shape[1]
    print(f"[model] {'toy' if args.toy else args.model}: {n_layers} layers, "
          f"hidden={hidden_dim}, vocab={vocab_size}, device={args.device}")

    # victim docs from the END of the corpus
    # --corpus-file REPLACES TEXT_SAMPLES (no mixing, so results are
    # attributable to one source).
    if args.corpus_file:
        corpus_source = "corpus_file"
        with open(args.corpus_file) as f:
            docs = [l.strip() for l in f if len(l.strip()) > 500]  # real docs only — wikitext short lines are formatting artifacts
    else:
        corpus_source = "TEXT_SAMPLES"
        docs = list(TEXT_SAMPLES)
    if len(docs) < args.victim_docs + 4:
        raise ValueError(f"corpus too small: {len(docs)} docs; need victim-docs "
                         f"({args.victim_docs}) + at least 4 attack docs")
    victim_docs = docs[-args.victim_docs:]
    attack_docs = docs[:-args.victim_docs]
    provenance = make_provenance(
        args.corpus_file, corpus_source, len(docs),
        range(len(docs) - args.victim_docs, len(docs)),
        model_path=getattr(args, 'model', None), docs=docs)
    # document-disjoint val split from the attack pool
    n_val = max(1, int(round(args.val_frac * len(attack_docs))))
    val_docs, train_docs_pool = attack_docs[:n_val], attack_docs[n_val:]
    print(f"[data] {len(docs)} docs: {len(train_docs_pool)} attack-train, "
          f"{n_val} attack-val, {len(victim_docs)} victim (held out from end)")

    # victim blocks: one block per victim doc (truncate, not chunk-mix).
    # Label must match the boundary (see collect_output_pairs): input side is
    # labelled with the token AT each position, output side with the NEXT one.
    # A block is seq_len+1 long, so both slices are seq_len wide and align
    # position-for-position with the captured activations.
    victim_ids = []
    victim_tokens = []
    for doc in victim_docs:
        b = encode([doc], args.seq_len)
        if b:
            victim_ids.append(b[0])
            victim_tokens.append(b[0][:-1] if args.boundary == "input"
                                 else b[0][1:])
    if not victim_ids:
        raise ValueError("no victim doc long enough to yield a block")
    victim_tok = torch.cat(victim_tokens)
    print(f"[data] {len(victim_ids)} victim blocks, {victim_tok.shape[0]} "
          f"attacked positions (seq_len={args.seq_len}), "
          f"boundary={args.boundary}")
    # resolution floor: the smallest non-zero top-1 this eval can express
    floor_pct = round(100.0 / victim_tok.shape[0], 4)

    # held-out utility set: attack-pool docs, never fine-tuned on (the FT pool is
    # the victim docs) — this measures generalization, not memorization
    eval_ids = []
    for doc in val_docs:
        if len(eval_ids) >= 3:
            break
        b = encode([doc], args.seq_len)
        if b:
            eval_ids.append(b[0])
    if not eval_ids:
        raise ValueError("no held-out eval blocks: val docs too short for seq_len")

    results = []
    summary = []

    # out is built up-front over live list references and dumped after EVERY
    # grid cell — the 27B cells run for hours and a crash must not lose them
    out = {
        "config": {"model": "toy" if args.toy else args.model,
                   "n_layers": n_layers, "depths": args.depths,
                   "tail_depths": args.tail_depths, "boundary": args.boundary,
                   "configs": args.configs,
                   "train_steps_list": args.train_steps_list,
                   "seeds": args.seeds, "victim_docs": args.victim_docs,
                   "val_frac": args.val_frac, "seq_len": args.seq_len,
                   "max_pairs": args.max_pairs, "epochs": args.epochs,
                   "lr": args.lr, "lora_rank": args.lora_rank,
                   "dtype": args.dtype, "device": args.device,
                   "quick": args.quick},
        "threat_model": "semi-honest cloud observes the boundary activation h* "
                        "and its gradient g*=dL/dh per microbatch; attacker "
                        "trains an MLP decoder on PUBLIC text through the PUBLIC "
                        "base model and applies it unchanged to the victim's "
                        "training-time boundary tensors (document-disjoint; "
                        "victim docs held out from the END of the corpus). "
                        f"boundary={args.boundary}: "
                        + ("h* is the activation SENT after layer sa, labelled "
                           "with the token at that position (the user's PROMPT)"
                           if args.boundary == "input" else
                           "h* is the activation the cloud RETURNS after layer "
                           "ra-1, labelled with the NEXT token that position "
                           "predicts (the model's RESPONSE); the cloud both "
                           "produces this tensor and receives dL/dh* for it"),
        "interpretation": "top-1 vs train_step: flat/high => fine-tuning drift "
                          "does NOT protect (base-model attacker transfers); "
                          "decaying => drift defends. B vs A at the same step: "
                          "gradients add recovery power iff B > A. Every "
                          "privacy figure carries utility_loss (held-out CE) at "
                          "the same checkpoint: recovery that falls while "
                          "utility_loss rises is model damage, not a defense. "
                          "Treat any top-1 below resolution_floor_top1_pct as "
                          "'below measurement resolution', not as 0."
                          + ("" if args.boundary == "input" else
                             " OUTPUT side: the reference point is the model's "
                             "OWN next-token accuracy, not only uniform "
                             "1/vocab chance -- the decoder is reading a "
                             "representation the LM head is about to decode."),
        "random_baseline_top1_pct": round(100.0 / vocab_size, 4),
        "provenance": provenance,
        "resolution_floor_top1_pct": floor_pct,
        "summary": summary,
        "results": results,
    }
    def dump_out():
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)

    collect_pairs = (collect_base_pairs if args.boundary == "input"
                     else collect_output_pairs)
    # flat (input-depth, output-depth) grid: one extra dimension, and with the
    # default --tail-depths [2] it is exactly the old single-depth sweep
    grid = [(d, t) for d in args.depths for t in args.tail_depths]

    for depth, tail_depth in grid:
        if embed is None:  # freed after the previous cell's pair collection
            embed, layers, norm, lm_head, rotary, encode = build_modules(args)
        head, cloud_l, tail, sa, ra = split_at(layers, depth, n_layers, tail_depth)
        eff_tail = n_layers - ra
        cell = f"sa={sa} ra={ra} (head={sa + 1}L, cloud={ra - sa - 1}L, tail={eff_tail}L)"
        if sa != depth or eff_tail != tail_depth:
            print(f"[split] depth {depth}/tail {tail_depth} clamped to {cell} "
                  f"({n_layers} layers)")

        # attack training pairs from the PUBLIC BASE model
        t0 = time.time()
        print(f"[collect] {cell} boundary={args.boundary}: base-model pairs "
              f"(with gradients)...")
        tr_h, tr_g, tr_tok = collect_pairs(
            embed, head, cloud_l, tail, norm, lm_head, rotary, encode,
            train_docs_pool, args, with_grad=True)
        va_h, va_g, va_tok = collect_pairs(
            embed, head, cloud_l, tail, norm, lm_head, rotary, encode,
            val_docs, args, with_grad=True)
        print(f"[collect] train={tr_h.shape[0]} val={va_h.shape[0]} pairs "
              f"({time.time() - t0:.1f}s)")

        # free the base model before per-config finetune copies are built —
        # at 27B, resident base (~54GB) + one config copy (~54GB) exceeds a
        # 94GB H100; nothing below needs the base weights (decoder training
        # and eval use the collected pair tensors; configs rebuild their own).
        # head/cloud_l/tail are views into `layers` — they must go too or the
        # delete above frees nothing.
        if not args.toy:
            del embed, layers, norm, lm_head, head, cloud_l, tail
            embed = layers = norm = lm_head = None
            gc.collect()
            torch.cuda.empty_cache()

        feats = {
            "A_act": (tr_h, va_h),
            "B_act+grad": (torch.cat([tr_h, tr_g], dim=-1),
                           torch.cat([va_h, va_g], dim=-1)),
        }

        # train one decoder per (attack, seed) on base-model pairs
        decoders = {}
        for attack_name, (xtr, xva) in feats.items():
            for seed in args.seeds:
                seed_all(args.seed + seed)
                tag = (f"trained_inv_{args.boundary}_d{sa}_t{eff_tail}"
                       f"_{attack_name}_seed{seed}")
                dec = train_decoder(xtr, tr_tok, xva, va_tok,
                                    xtr.shape[1], vocab_size, args, tag)
                vtop1, vtop5 = evaluate_decoder(dec, xva, va_tok, args.device)
                # park on CPU: at 27B each decoder head is hidden×vocab
                # (~1GB fp32) and 6 of them plus the 54GB finetune copy OOM
                # a 94GB H100; they are only needed again at attack_eval
                decoders[(attack_name, seed)] = dec.to("cpu")
                torch.cuda.empty_cache()
                print(f"[attack] depth={sa} tail={eff_tail} {attack_name} "
                      f"seed={seed}: val top-1={vtop1:.2f}% top-5={vtop5:.2f}%")
                results.append({"phase": "attack_train", "depth": sa,
                                "tail_depth": eff_tail, "ra": ra,
                                "boundary": args.boundary,
                                "attack": attack_name, "seed": seed,
                                "val_top1": vtop1, "val_top5": vtop5})

        # real split fine-tune per config; attack at each checkpoint
        for config in args.configs:
            # fresh stage copies per (depth, config): training mutates weights
            if args.toy:
                w_embed = copy.deepcopy(embed)
                w_layers = copy.deepcopy(layers)
                w_norm = copy.deepcopy(norm)
                w_lm_head = copy.deepcopy(lm_head)
            else:
                a = argparse.Namespace(**vars(args))
                w_embed, w_layers, w_norm, w_lm_head, _, _ = build_modules(a)
            w_head, w_cloud, w_tail, sa2, _ = split_at(w_layers, depth,
                                                       n_layers, tail_depth)
            print(f"[finetune] {cell} config={config}: "
                  f"{max(args.train_steps_list)} steps, captures at "
                  f"{args.train_steps_list}")
            _write_training_status(state="running", phase="finetune",
                                   depth=sa, tail_depth=eff_tail,
                                   boundary=args.boundary, config=config)
            captured, utility = finetune_and_capture(
                w_embed, w_head, w_cloud, w_tail, w_norm, w_lm_head, rotary,
                encode, victim_docs, victim_ids, args.train_steps_list,
                args, config, eval_ids=eval_ids, boundary=args.boundary)

            for step in sorted(captured):
                h_star, g_star = captured[step]
                step_utility = utility.get(step)
                victim_feats = {
                    "A_act": h_star,
                    "B_act+grad": torch.cat([h_star, g_star], dim=-1),
                }
                for attack_name, vx in victim_feats.items():
                    t1s, t5s = [], []
                    for seed in args.seeds:
                        dec = decoders[(attack_name, seed)].to(args.device)
                        top1, top5 = evaluate_decoder(dec, vx, victim_tok,
                                                      args.device)
                        decoders[(attack_name, seed)] = dec.to("cpu")
                        t1s.append(top1)
                        t5s.append(top5)
                        results.append({"phase": "attack_eval", "depth": sa,
                                        "tail_depth": eff_tail, "ra": ra,
                                        "boundary": args.boundary,
                                        "config": config, "train_step": step,
                                        "attack": attack_name, "seed": seed,
                                        "top1": top1, "top5": top5,
                                        "utility_loss": step_utility})
                    m1, s1 = mean_std(t1s)
                    m5, s5 = mean_std(t5s)
                    # privacy never ships without its utility pair (an
                    # apparent defense was once model damage)
                    summary.append({"depth": sa, "tail_depth": eff_tail,
                                    "ra": ra, "boundary": args.boundary,
                                    "config": config,
                                    "train_step": step, "attack": attack_name,
                                    "top1_mean": m1, "top1_std": s1,
                                    "top5_mean": m5, "top5_std": s5,
                                    "utility_loss": step_utility,
                                    "resolution_floor_top1_pct": floor_pct,
                                    "n_seeds": len(t1s)})
                    print(f"[eval] {args.boundary} depth={sa} tail={eff_tail} "
                          f"config={config} step={step} "
                          f"{attack_name}: top-1={m1:.2f}+-{s1:.2f}% "
                          f"utility={step_utility:.3f} "
                          f"top-5={m5:.2f}+-{s5:.2f}%")

            # free per-cell model copies + captured tensors — the CUDA caching
            # allocator retains them otherwise (unified-memory OOM freeze).
            # w_head/w_cloud/w_tail are views into w_layers — del them too.
            del w_embed, w_layers, w_norm, w_lm_head, w_head, w_cloud, w_tail, captured
            gc.collect()
            torch.cuda.empty_cache()

        # free per-depth decoders and pair tensors
        del decoders, feats, tr_h, tr_g, tr_tok, va_h, va_g, va_tok
        gc.collect()
        torch.cuda.empty_cache()
        dump_out()  # crash-safe: every completed depth is on disk

    dump_out()
    _write_training_status(state="done", result_file=args.output)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
