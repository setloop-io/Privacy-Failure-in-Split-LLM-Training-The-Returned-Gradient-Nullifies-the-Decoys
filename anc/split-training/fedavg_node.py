#!/usr/bin/env python3
"""FedAvg between the two Spark nodes (experiment 4b) — data-parallel
"training lookahead": amortize a WAN synchronization over K local steps,
the data-parallel analogue of the pipeline overlap schedule (DESIGN.md §7).

Both nodes hold the FULL model and fine-tune (AdamW, causal-LM next-token
loss) on PRIVATE, disjoint shards of the corpus. The corpus is shuffled by
--seed before a deterministic index-parity split (server = even documents,
client = odd), so repetition seeds change data order while preserving disjoint
peer shards. Every --k-local steps the
nodes rendezvous and exchange WEIGHT DELTAS against the last sync point:

    delta_node = flat_fp32(params) - flat_fp32(ref)         # ref = last avg
    avg        = (delta_server + delta_client) / 2
    params    <- ref + avg          (both nodes, identical update)

Wire protocol: same binary framing as cloud_server_kv.py /
cloud_trainer_server.py — [4B big-endian header_len][JSON header][raw bytes]
— on ws port 5004 (FedAvg gets its own port; inference on 5001 and split
training on 5003 are untouched). One persistent connection per run:

  client -> text  {"op":"hello", n_params, sync_dtype}     -> hello_ack
  client -> binary {"op":"delta_chunk", round, seq, last}  + 64MB chunks
  server -> binary {"op":"avg_chunk",   round, seq, last}  + 64MB chunks

Delta is ONE logical flat fp32 vector (parameters_to_vector order), streamed
in 64MB chunks — per-tensor frames would pay header+scheduling overhead
hundreds of times per sync for zero benefit at these sizes. --sync-dtype
bf16 halves the bytes (see note below). Single client per server (2-node
setup); AdamW state is NOT reset after averaging (standard FedAvg practice
— documented as a source of drift, not reset bias).

Correctness (see --selftest): at --k-local 1, identical seeds and identical
init, both nodes end every sync with bit-identical parameters, and the
averaged delta equals the mean of the two local deltas. Step-0 loss of each
node equals a single-node reference on the same block with the same seed —
note the data-order caveat: each node's "step 0" is the first block of ITS
shard (server: doc 0, client: doc 1), so the two nodes' step-0 losses differ
from each other by construction; the invariant is vs the matching
single-node reference, not vs each other.

Usage:
  python fedavg_node.py --help                    # works without torch
  python fedavg_node.py --selftest --toy          # CPU: both roles, loopback
  python fedavg_node.py --role server --model ... --corpus-file ...
  python fedavg_node.py --role client --cloud ws://10.10.10.2:5004 ...
"""

import argparse
import json
import os
import random
import struct
import sys
import threading
import time

# Guarded heavy imports: `--help` works on torch-less hosts.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None
try:
    import asyncio
    import websockets
    from websockets.sync.client import connect as _ws_connect
    from websockets.exceptions import ConnectionClosed as _WSClosed
except ImportError:  # pragma: no cover
    asyncio = None
    websockets = None
    _ws_connect = None
    _WSClosed = Exception

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from split_trainer import (TEXT_SAMPLES, _write_training_status,
                               build_modules, unique_params)
except ImportError:  # pragma: no cover
    TEXT_SAMPLES = []
    _write_training_status = lambda **k: None
    build_modules = unique_params = None

CHUNK_BYTES = 64 * 1024 * 1024  # 64MB per binary frame
SYNC_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16} if torch else {}
WS_PORT = 5004


# Framing helpers (identical wire format to cloud_trainer_server.py).
def pack_frame(header, payload=b""):
    h = json.dumps(header).encode()
    return struct.pack(">I", len(h)) + h + payload


def unpack_frame(message):
    header_len = struct.unpack(">I", message[:4])[0]
    return json.loads(message[4:4 + header_len]), message[4 + header_len:]


def vector_to_bytes(vec, sync_dtype):
    v = vec.detach().to(sync_dtype).contiguous().cpu()
    if sync_dtype == torch.bfloat16:
        return v.view(torch.int16).numpy().tobytes()
    return v.numpy().tobytes()


def bytes_to_vector(buf, n_params, sync_dtype, device):
    if sync_dtype == torch.bfloat16:
        v = torch.frombuffer(bytearray(buf), dtype=torch.int16).view(torch.bfloat16)
    else:
        v = torch.frombuffer(bytearray(buf), dtype=sync_dtype)
    assert v.numel() == n_params, f"delta size mismatch: {v.numel()} vs {n_params}"
    return v.clone().float().to(device)


def iter_chunks(buf):
    for off in range(0, len(buf), CHUNK_BYTES):
        yield off // CHUNK_BYTES, buf[off:off + CHUNK_BYTES], \
            off + CHUNK_BYTES >= len(buf)


def load_model_seeded(args):
    """Identical init on both nodes: seed immediately before construction
    (toy draws from global RNG; HF from_pretrained is deterministic)."""
    torch.manual_seed(args.seed)
    if args.toy and (args.toy_hidden != 64 or args.toy_layers != 4):
        # sized toy for CPU leak tests (bypasses split_trainer.build_modules)
        from split_trainer import ToyCausalLM
        model = ToyCausalLM(vocab=128, hidden=args.toy_hidden,
                            n_layers=args.toy_layers)
        model.to(args.device)
        embed, layers, norm, lm_head = (model.embed_tokens, model.layers,
                                        model.norm, model.lm_head)
        rotary, encode = None, model.encode
    else:
        embed, layers, norm, lm_head, rotary, encode = build_modules(args)
    params = unique_params([embed, layers, norm, lm_head], require_grad=False)
    return (embed, layers, norm, lm_head, rotary, encode, params)


def load_docs(args):
    docs = list(TEXT_SAMPLES)
    if args.corpus_file:
        with open(args.corpus_file) as f:
            docs = [l.strip() for l in f if l.strip()]
    if args.ea4_members_only:
        docs = docs[:max(2, len(docs) // 2)]
    # Apply the repetition seed after selecting the E-A4 member pool. Both
    # peers therefore derive the same permutation and retain disjoint parity
    # shards, while different training seeds exercise different data order.
    random.Random(args.seed).shuffle(docs)
    return docs


def shard_docs(docs, role):
    """Deterministic disjoint split by doc index parity."""
    return docs[0::2] if role == "server" else docs[1::2]


# Sync coordinator: rendezvous between the server's training thread and the
# WS handler thread. One outstanding round at a time (synchronous FedAvg).
class SyncCoordinator:
    def __init__(self):
        self._cond = threading.Condition()
        self._deltas = {}    # round -> {who: vector}
        self._avgs = {}      # round -> [avg vector, n_consumed]
        self._delivered = {} # round -> threading.Event (set after last chunk sent)
        self._aborted = None # set to a reason string on peer disconnect

    def abort(self, reason):
        """Fail all current and future rendezvous immediately (peer gone)."""
        with self._cond:
            self._aborted = reason
            self._cond.notify_all()

    def mark_delivered(self, round_no):
        with self._cond:
            self._delivered.setdefault(round_no, threading.Event()).set()
            self._cond.notify_all()

    def wait_delivered(self, round_no, timeout=600):
        """Block until the WS thread has finished SENDING this round's avg —
        keeps the server process alive long enough on its final round."""
        with self._cond:
            ev = self._delivered.setdefault(round_no, threading.Event())
        ok = ev.wait(timeout)
        self._delivered.pop(round_no, None)  # don't accumulate events
        return ok

    def submit(self, delta, who, round_no, timeout=1800):
        with self._cond:
            if self._aborted:
                raise RuntimeError(f"FedAvg aborted: {self._aborted}")
            self._deltas.setdefault(round_no, {})[who] = delta
            if len(self._deltas[round_no]) == 2:
                d = self._deltas.pop(round_no)
                self._avgs[round_no] = [(d["server"] + d["client"]) / 2, 0]
                self._cond.notify_all()
            ok = self._cond.wait_for(
                lambda: round_no in self._avgs or self._aborted is not None,
                timeout=timeout)
            if self._aborted and round_no not in self._avgs:
                # free our delta; the peer is never coming
                self._deltas.pop(round_no, None)
                raise RuntimeError(f"FedAvg aborted in round {round_no}: "
                                   f"{self._aborted}")
            if not ok:
                self._deltas.pop(round_no, None)
                raise TimeoutError(f"FedAvg rendezvous timed out in round {round_no} "
                                   f"({timeout}s; peer never arrived)")
            entry = self._avgs[round_no]
            avg = entry[0]
            entry[1] += 1
            if entry[1] == 2:  # both sides consumed -> free the big vector
                del self._avgs[round_no]
            return avg


# WebSocket sync server (server role only; runs in a daemon thread).
def make_ws_handler(coordinator, n_params, sync_dtype, device, log,
                    sync_timeout=1800):
    async def ws_handler(websocket):
        hello = json.loads(await websocket.recv())
        if hello.get("op") != "hello":
            await websocket.send(json.dumps({"op": "error", "error": "need hello"}))
            return
        if hello.get("n_params") != n_params:
            await websocket.send(json.dumps(
                {"op": "error",
                 "error": f"n_params mismatch: {hello.get('n_params')} vs {n_params}"}))
            return
        await websocket.send(json.dumps({"op": "hello_ack", "n_params": n_params}))
        log(f"[sync] client hello (n_params={n_params}, dtype={sync_dtype})")

        try:
            while True:
                # Collect one full delta (chunked binary frames)
                buf = bytearray()
                round_no = None
                while True:
                    message = await websocket.recv()
                    header, payload = unpack_frame(message)
                    if header.get("op") != "delta_chunk":
                        await websocket.send(pack_frame(
                            {"op": "error", "error": f"unexpected {header.get('op')}"}))
                        break
                    round_no = header["round"]
                    buf.extend(payload)
                    if header.get("last"):
                        break
                client_delta = bytes_to_vector(bytes(buf), n_params,
                                               SYNC_DTYPES[sync_dtype], device)
                buf_len = len(buf)
                del buf  # free the multi-GB staging buffer before the wait
                log(f"[sync] round {round_no}: received client delta "
                    f"({buf_len / 1e6:.0f}MB), waiting for local delta...")
                # Run the blocking rendezvous OFF the event loop: a bare
                # Condition.wait here would freeze the loop (no ping/pong,
                # no other coroutine) for the whole K-step training phase.
                avg = await asyncio.to_thread(coordinator.submit, client_delta,
                                              "client", round_no, sync_timeout)
                client_delta = None  # drop our reference; coordinator owns avg
                out = vector_to_bytes(avg, SYNC_DTYPES[sync_dtype])
                for seq, chunk, last in iter_chunks(out):
                    await websocket.send(pack_frame(
                        {"op": "avg_chunk", "round": round_no,
                         "seq": seq, "last": last}, chunk))
                del out
                coordinator.mark_delivered(round_no)
                log(f"[sync] round {round_no}: sent averaged delta")
        except _WSClosed:
            log("[sync] client disconnected")
            coordinator.abort("client disconnected")
        except Exception as e:
            log(f"[sync] error: {e}")
            coordinator.abort(f"sync server error: {e}")
    return ws_handler


def start_sync_server(coordinator, n_params, sync_dtype, device, port, log,
                      sync_timeout=1800):
    async def serve():
        handler = make_ws_handler(coordinator, n_params, sync_dtype, device,
                                  log, sync_timeout)
        async with websockets.serve(handler, "0.0.0.0", port,
                                    max_size=CHUNK_BYTES + 1024 * 1024,
                                    ping_interval=None, ping_timeout=None):
            log(f"[sync] FedAvg sync server on ws://0.0.0.0:{port}")
            await asyncio.Future()
    threading.Thread(target=lambda: asyncio.run(serve()), daemon=True).start()


# Training step (plain full-FT AdamW, causal-LM loss)
def train_step(embed, layers, norm, lm_head, ids, opt, device, rotary=None):
    ids = ids.unsqueeze(0).to(device)
    input_ids, labels = ids[:, :-1], ids[:, 1:]
    hidden = embed(input_ids)
    kwargs = {}
    if rotary is not None:
        # transformers 5.x decoder layers require position_embeddings
        position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        kwargs = {"position_ids": position_ids,
                  "position_embeddings": rotary(hidden, position_ids)}
    for layer in layers:
        out = layer(hidden, **kwargs)
        hidden = out[0] if isinstance(out, tuple) else out
    logits = lm_head(norm(hidden))
    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                           labels.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return loss.item()


def flat(params):
    return nn.utils.parameters_to_vector(params).detach().float()


def _vms_gb():
    """Virtual memory in GB — unlike RSS this is NOT hidden by the macOS
    memory compressor, so retained-but-idle tensors still show up."""
    try:
        import psutil
        return psutil.Process().memory_info().vms / 1e9
    except ImportError:
        return 0.0


def _rss_gb():
    """Current process RSS in GB (psutil preferred; ru_maxrss fallback —
    max-RSS is monotonic but still shows per-round growth)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except ImportError:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def assign_params(params, vec):
    """Copy a flat vector into params IN PLACE. (nn.utils.vector_to_parameters
    assigns param.data as a VIEW into vec — that aliases params to our ref
    tensor and let training mutate ref between rounds; found via selftest.)"""
    with torch.no_grad():
        pointer = 0
        for p in params:
            n = p.numel()
            p.copy_(vec[pointer:pointer + n].view_as(p).to(p.dtype))
            pointer += n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", choices=["server", "client"], default=None)
    ap.add_argument("--cloud", default=None,
                    help="ws://host:5004 (or http://host:5005 / bare host) — client only")
    ap.add_argument("--model", default=os.path.expanduser(
        "~/experiments/models/qwen3-0.6b"))
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--corpus-file", default=None)
    ap.add_argument("--ea4-members-only", action="store_true",
                    help="train only on first half of corpus before node sharding")
    ap.add_argument("--ea4-checkpoint", default=None,
                    help="save embedding + first five layers after averaging")
    ap.add_argument("--k-local", type=int, default=10,
                    help="local steps between FedAvg syncs")
    ap.add_argument("--total-steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--sync-dtype", choices=["fp32", "bf16"], default="fp32",
                    help="wire dtype for weight deltas (fp32 exact; bf16 = "
                         "half the sync bytes, small averaging error)")
    ap.add_argument("--attn-impl", choices=["sdpa", "eager"], default="sdpa")
    ap.add_argument("--device",
                    default="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ws-port", type=int, default=WS_PORT)
    ap.add_argument("--sync-timeout", type=float, default=1800,
                    help="seconds to wait for a sync rendezvous / avg response "
                         "before failing loudly (default 1800)")
    ap.add_argument("--toy-hidden", type=int, default=64)
    ap.add_argument("--toy-layers", type=int, default=4,
                    help="toy model size knobs (leak testing with realistic "
                         "parameter counts on CPU)")
    ap.add_argument("--debug-trace", action="store_true",
                    help="retain per-round delta/avg vectors for selftest "
                         "verification (memory-heavy; never use in real runs)")
    ap.add_argument("--smoke", action="store_true",
                    help="2 steps, K=1 (one sync per step)")
    ap.add_argument("--selftest", action="store_true",
                    help="both roles in one process over loopback WS "
                         "(CPU verification with --toy)")
    ap.add_argument("--output", default="fedavg_results.json")
    args = ap.parse_args()

    if torch is None:
        ap.error("torch is not installed; install it or run --help only")
    if args.smoke:
        args.total_steps = 2
        args.k_local = 1

    if args.selftest:
        selftest(args)
        return
    if args.role is None:
        ap.error("--role server|client required (or --selftest)")

    _write_training_status(state="running", task="fedavg", role=args.role,
                           k_local=args.k_local, total_steps=args.total_steps)
    if args.role == "server":
        # n_params known only after model load; load once here and reuse below
        coordinator = SyncCoordinator()
        embed, layers, norm, lm_head, rotary, encode, params = load_model_seeded(args)
        n_params = sum(p.numel() for p in params)
        start_sync_server(coordinator, n_params, args.sync_dtype, args.device,
                          args.ws_port, print, sync_timeout=args.sync_timeout)
        # run with the already-loaded model
        partial = {}
        try:
            result = _run_server_with_prebuilt(args, coordinator,
                                               (embed, layers, norm, lm_head,
                                                rotary, encode, params),
                                               _partial=partial)
        except Exception as e:
            result = {"steps": partial.get("steps", []),
                      "syncs": partial.get("syncs", []), "error": str(e)}
            print(f"[server] run failed: {e} — writing partial results anyway")
            _write_training_status(state="failed", error=str(e))
    else:
        if not args.cloud:
            ap.error("--role client requires --cloud ws://host:5004")
        url = normalize_cloud_url(args.cloud)
        embed, layers, norm, lm_head, rotary, encode, params = load_model_seeded(args)
        n_params = sum(p.numel() for p in params)
        ws = _ws_connect(url, max_size=CHUNK_BYTES + 1024 * 1024,
                         ping_interval=None, ping_timeout=None)
        ws.send(json.dumps({"op": "hello", "n_params": n_params,
                            "sync_dtype": args.sync_dtype}))
        ack = json.loads(ws.recv())
        if ack.get("op") == "error":
            raise RuntimeError(f"sync server rejected hello: {ack.get('error')}")
        print(f"[client] connected to {url} (n_params={n_params})")
        partial = {}
        try:
            result = _run_client_with_prebuilt(args, ws,
                                               (embed, layers, norm, lm_head,
                                                rotary, encode, params),
                                               _partial=partial)
        except Exception as e:
            result = {"steps": partial.get("steps", []),
                      "syncs": partial.get("syncs", []), "error": str(e)}
            print(f"[client] run failed: {e} — writing partial results anyway")
            _write_training_status(state="failed", error=str(e))

    out = {"config": {"role": args.role, "model": "toy" if args.toy else args.model,
                      "k_local": args.k_local, "total_steps": args.total_steps,
                      "lr": args.lr, "seq_len": args.seq_len, "dtype": args.dtype,
                      "sync_dtype": args.sync_dtype, "seed": args.seed,
                      "device": args.device},
           "steps": result["steps"], "syncs": result["syncs"]}
    if "error" in result:
        out["error"] = result["error"]
    if args.debug_trace:
        kept = result.get("trace", [])
        nbytes = sum(t["delta"].numel() * 4 + t["avg"].numel() * 4
                     + t["ref_after"].numel() * 4 for t in kept)
        print(f"[debug] trace retained: {len(kept)} rounds, {nbytes / 1e9:.2f}GB")
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    _write_training_status(state="done", result_file=args.output)
    print(f"Wrote {args.output}")


def normalize_cloud_url(u):
    if "://" in u:
        scheme, rest = u.split("://", 1)
        host = rest.split(":")[0].split("/")[0]
        if scheme in ("http", "https"):
            port = int(rest.split(":")[1].split("/")[0]) if ":" in rest else 5005
            return f"ws://{host}:{port - 1}"
        return f"ws://{rest.rstrip('/')}" if ":" in rest else f"ws://{host}:{WS_PORT}"
    return f"ws://{u.split(':')[0]}:{WS_PORT}"


def _run_server_with_prebuilt(args, coordinator, mods, _partial=None):
    return _run_with_prebuilt(args, mods, coordinator=coordinator,
                              _partial=_partial)


def _run_client_with_prebuilt(args, ws, mods, _partial=None):
    return _run_with_prebuilt(args, mods, client_ws=ws, _partial=_partial)


def _run_with_prebuilt(args, mods, coordinator=None, client_ws=None,
                       _partial=None):
    """run_node body with an already-loaded model (avoids double loads)."""
    embed, layers, norm, lm_head, rotary, encode, params = mods
    opt = torch.optim.AdamW(params, lr=args.lr)
    docs = shard_docs(load_docs(args), args.role)
    blocks = encode(docs, args.seq_len)
    if not blocks:
        raise ValueError("shard produced no training blocks; check corpus")
    print(f"[{args.role}] {len(docs)} docs (shard) -> {len(blocks)} blocks")

    n_params = sum(p.numel() for p in params)
    sync_dtype = SYNC_DTYPES[args.sync_dtype]
    ref = flat(params)
    step_log, sync_log, trace = [], [], []
    if _partial is not None:  # live views so a crash still yields partial JSON
        _partial["steps"] = step_log
        _partial["syncs"] = sync_log
    k = args.k_local
    sync_timeout = args.sync_timeout

    for step in range(args.total_steps):
        loss = train_step(embed, layers, norm, lm_head,
                          blocks[step % len(blocks)], opt, args.device, rotary)
        step_log.append({"step": step, "loss": loss})
        print(f"[{args.role} step {step}] loss={loss:.4f}")
        _write_training_status(state="running", role=args.role,
                               run_id=f"fedavg_{args.role}_k{args.k_local}",
                               script="fedavg_node.py",
                               model=os.path.basename(args.model.rstrip("/")),
                               step=step, epoch=step, epochs=args.total_steps,
                               loss=loss)

        if (step + 1) % k != 0 and (step + 1) != args.total_steps:
            continue
        round_no = (step + 1) // k - 1
        t_sync = time.perf_counter()
        delta = flat(params) - ref
        prev_ref = ref
        if args.role == "server":
            avg = coordinator.submit(delta, "server", round_no,
                                     timeout=sync_timeout)
            nbytes = n_params * (2 if args.sync_dtype == "bf16" else 4)
        else:
            payload = vector_to_bytes(delta, sync_dtype)
            nbytes = len(payload)
            for seq, chunk, last in iter_chunks(payload):
                client_ws.send(pack_frame({"op": "delta_chunk", "round": round_no,
                                           "seq": seq, "last": last}, chunk))
            del payload  # drop the multi-GB staging buffer before waiting
            buf = bytearray()
            while True:
                try:
                    header, chunk = unpack_frame(
                        client_ws.recv(timeout=sync_timeout))
                except TimeoutError:
                    raise RuntimeError(
                        f"timed out after {sync_timeout}s waiting for the "
                        f"averaged delta of round {round_no} (server stalled "
                        f"or link down) — aborting instead of hanging") from None
                if header.get("op") == "error":
                    raise RuntimeError(f"sync server error: {header.get('error')}")
                buf.extend(chunk)
                if header.get("last"):
                    break
            avg = bytes_to_vector(bytes(buf), n_params, sync_dtype, args.device)
            del buf
        ref = ref + avg
        if os.environ.get("FEDAVG_DEBUG"):
            print(f"[{args.role}] round {round_no} inline inc_err="
                  f"{((ref - prev_ref) - avg).abs().max().item():.3e}")
        assign_params(params, ref)
        t_sync = time.perf_counter() - t_sync
        if args.debug_trace:
            # selftest correctness checks only — retaining these vectors every
            # round IS the leak in a long run; never enable outside selftest
            trace.append({"round": round_no, "delta": delta.clone(),
                          "avg": avg.clone(), "ref_after": ref.clone().cpu()})
        del delta, avg  # per-round vectors die here; nothing may outlive them
        sync_log.append({"round": round_no, "after_step": step,
                         "t_sync": t_sync, "bytes": nbytes,
                         "rss_gb": round(_rss_gb(), 2)})
        # watchdog: warn loudly if a round suddenly takes >> longer
        times = [s["t_sync"] for s in sync_log[:-1]]
        if times and t_sync > max(60.0, 5 * sorted(times)[len(times) // 2]):
            print(f"[{args.role}] WATCHDOG: round {round_no} took {t_sync:.1f}s "
                  f"(median previous {sorted(times)[len(times) // 2]:.1f}s)")
        print(f"[{args.role}] sync round {round_no} done in {t_sync:.2f}s "
              f"(rss={sync_log[-1]['rss_gb']:.2f}GB vms={_vms_gb():.2f}GB)")

    if args.role == "server" and sync_log:
        # don't let main() exit and reap the daemon WS thread before the
        # final round's avg has actually been sent to the client
        last = sync_log[-1]["round"]
        if coordinator.wait_delivered(last, timeout=600):
            print(f"[server] final sync round {last} delivered to client")
        else:
            print(f"[server] WARNING: final sync round {last} not confirmed "
                  f"delivered (client gone?)")

    if args.ea4_checkpoint:
        os.makedirs(os.path.dirname(os.path.abspath(args.ea4_checkpoint)), exist_ok=True)
        torch.save({"schema": "dtraining.ea4.boundary_checkpoint.v1",
                    "condition": "fedavg", "split_after": min(4, len(layers)-1),
                    "embed": embed.state_dict(),
                    "head": torch.nn.ModuleList(list(layers[:5])).state_dict(),
                    "seed": args.seed}, args.ea4_checkpoint)
        print(f"[E-A4] saved input-boundary checkpoint {args.ea4_checkpoint}")
    return {"steps": step_log, "syncs": sync_log, "ref": ref, "params": params,
            "trace": trace}


def selftest(args):
    """Both roles in one process over loopback WS (CPU + --toy friendly).

    Checks:
      1. post-sync parameters are bit-identical across the two nodes;
      2. the averaged delta equals the exact mean of the two local deltas;
      3. each node's step-0 loss equals a single-node reference on the same
         block with the same seed (data order: first block of own shard).
    """
    if websockets is None or _ws_connect is None:
        raise RuntimeError("websockets required for --selftest")
    args.role = "server"
    args.debug_trace = True  # selftest checks need the per-round vectors
    coordinator = SyncCoordinator()

    embed, layers, norm, lm_head, rotary, encode, params = load_model_seeded(args)
    n_params = sum(p.numel() for p in params)
    start_sync_server(coordinator, n_params, args.sync_dtype, args.device,
                      args.ws_port, print)
    time.sleep(1.0)  # let the WS thread bind

    # single-node reference losses (same seed, same first block per shard)
    refs = {}
    for role in ("server", "client"):
        e2, l2, n2, h2, _, enc2, p2 = load_model_seeded(args)
        docs = shard_docs(load_docs(args), role)
        b = enc2(docs, args.seq_len)[0]
        ids = b.unsqueeze(0)
        with torch.no_grad():
            hidden = e2(ids[:, :-1])
            for layer in l2:
                out = layer(hidden)
                hidden = out[0] if isinstance(out, tuple) else out
            logits = h2(n2(hidden))
            refs[role] = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                ids[:, 1:].reshape(-1)).item()

    holder = {}

    def run_server():
        args.role = "server"
        holder["server"] = _run_with_prebuilt(
            args, (embed, layers, norm, lm_head, rotary, encode, params),
            coordinator=coordinator)

    t = threading.Thread(target=run_server)
    t.start()

    args_client = argparse.Namespace(**vars(args))
    args_client.role = "client"
    em, la, no, hd, ro, enc, pa = load_model_seeded(args_client)
    ws = None
    for _ in range(20):  # wait for the sync server thread to bind
        try:
            ws = _ws_connect(f"ws://127.0.0.1:{args.ws_port}",
                             max_size=CHUNK_BYTES + 1024 * 1024,
                             ping_interval=None, ping_timeout=None,
                             open_timeout=5)
            break
        except (OSError, TimeoutError):
            time.sleep(0.5)
    if ws is None:
        raise RuntimeError("selftest: sync server never came up")
    ws.send(json.dumps({"op": "hello", "n_params": n_params,
                        "sync_dtype": args.sync_dtype}))
    ack = json.loads(ws.recv())
    assert ack.get("op") == "hello_ack", ack
    holder["client"] = _run_with_prebuilt(args_client, (em, la, no, hd, ro, enc, pa),
                                          client_ws=ws)
    t.join(timeout=300)
    ws.close()

    sv, cl = holder["server"], holder["client"]
    ok = True

    # 1. identical post-sync parameters (fp32 wire: bit-exact; bf16 wire:
    #    client's avg is bf16-rounded, so allow its rounding error)
    tol1 = 0.0 if args.sync_dtype == "fp32" else 1e-5
    diffs = [(a - b).abs().max().item() for a, b in
             zip([p.detach() for p in sv["params"]],
                 [p.detach() for p in cl["params"]])]
    same = all(d <= tol1 for d in diffs)
    print(f"[selftest] post-sync params identical (tol={tol1}): {same} "
          f"(max abs diff {max(diffs):.3e}, {( sum(1 for d in diffs if d>tol1) )} "
          f"of {len(diffs)} tensors differ)")
    rd = (sv["ref"] - cl["ref"]).abs().max().item()
    ps = (flat(sv["params"]).cpu() - sv["ref"].cpu()).abs().max().item()
    pc = (flat(cl["params"]).cpu() - cl["ref"].cpu()).abs().max().item()
    print(f"[selftest] ref diff sv-vs-cl: {rd:.3e} | |params-ref| "
          f"server: {ps:.3e} client: {pc:.3e}")
    # reconstruct ref from the recorded traces: ref0 + sum(round avgs)
    _, _, _, _, _, _, p0 = load_model_seeded(args)
    ref0 = flat(p0)
    for name, res in (("server", sv), ("client", cl)):
        recon = ref0.clone()
        prev = None
        for tr in res["trace"]:
            recon = recon + tr["avg"].cpu()
            gap = (recon - tr["ref_after"]).abs().max().item()
            inc_err = None
            if prev is not None:
                inc_err = ((tr["ref_after"] - prev["ref_after"]) - tr["avg"]).abs().max().item()
            print(f"[selftest] {name} round {tr['round']}: "
                  f"|recon - ref_after| = {gap:.3e} inc_err={inc_err}")
            prev = tr
    ok &= same

    # 2. averaged delta == exact mean of the two LOCAL deltas, per round
    #    (fp32 wire: bit-exact; bf16 wire: half-ulp tolerance)
    tol = 0.0 if args.sync_dtype == "fp32" else 1e-2
    for tr_s, tr_c in zip(sv["trace"], cl["trace"]):
        r = tr_s["round"]
        mean = (tr_s["delta"] + tr_c["delta"]) / 2
        m_s = torch.allclose(tr_s["avg"], mean, atol=tol)
        m_c = torch.allclose(tr_c["avg"], mean, atol=tol)
        agree = torch.allclose(tr_s["avg"], tr_c["avg"], atol=tol)
        dmax = (tr_s["avg"] - tr_c["avg"]).abs().max().item()
        print(f"[selftest] round {r}: avg == mean(deltas) server:{m_s} "
              f"client:{m_c} | nodes agree:{agree} (avg max diff {dmax:.3e})")
        ok &= m_s and m_c and agree

    # 3. step-0 losses vs single-node references
    for role, res in (("server", sv), ("client", cl)):
        got = res["steps"][0]["loss"]
        want = refs[role]
        match = abs(got - want) < 1e-6
        print(f"[selftest] {role} step-0 loss {got:.6f} == reference "
              f"{want:.6f}: {match}")
        ok &= match

    print(f"[selftest] {'PASS' if ok else 'FAIL'}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
