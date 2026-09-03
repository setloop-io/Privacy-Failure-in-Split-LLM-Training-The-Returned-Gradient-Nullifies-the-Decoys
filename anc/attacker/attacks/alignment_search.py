#!/usr/bin/env python3
"""alignment-search (E-R8) — approximate-alignment attacker vs the
jittered scaffold.

E-R3's jitter arm (rotation_lifetime.py, jittered_scaffold) is only
validated against a PASSIVE-alignment adversary — one that assumes wire
row i corresponds to assumed prefix position i. The jittered arm's 0%
could reflect that the attacker doesn't try to realign. This attacker
does: it alternates (ICP-style) between

  1. de-rotating the captured wire rows with the current W_hat, and
  2. edit-distance / DTW realignment (attacker.dtw, PURE-PYTHON core,
     torch-free testable) of the de-rotated sequence against the candidate
     prefix positions of the known scaffold, harvesting matched pairs,
  3. re-solving W_hat = polar(lstsq) on the matched pairs

until the match set stabilizes. If the jittered arm falls to this, the
E-R3 "0% recovery" claim needs the realignment-aware qualifier.

Modes: training and inference (any surface that serves known-prefix
sessions under a jittered scaffold). Labeled: partial — the scaffold prefix
is known, the jitter is not.

Usage:
    python -m attacker --mode training --attack alignment-search --help
    python -m attacker --mode training --attack alignment-search --toy \
        --quick --output /tmp/er8.json
"""

import argparse
from types import SimpleNamespace

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .. import artifacts
from ..dtw import dtw_distance, edit_alignment
from ..solve_primitives import h_wire, solve_w
from .common import (add_common_args, journal_error, nn_mean_decode,
                     recovery_with_what_nn, require_torch)

EXPERIMENT_ID = "er8_alignment_search"
MODES = ("training", "inference")
REQUIRES_LABELS = "partial (known scaffold prefix, unknown jitter)"
DESCRIPTION = ("E-R8: edit-distance/DTW realignment of jittered-scaffold "
               "captures before the W solve — tests whether the jitter "
               "arm's 0% survives an attacker that tries to realign")


def _make_runtime_args(cfg, device, max_pairs=20000, epochs=30,
                       batch_size=256):
    """Build split-training arguments without nested-class name lookup."""
    return SimpleNamespace(
        model=cfg["model"], dtype=cfg["dtype"], device=device,
        toy=False, seq_len=cfg["seq_len"], attn_impl="sdpa",
        max_pairs=max_pairs, epochs=epochs, batch_size=batch_size,
    )


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--sessions", type=int, default=8,
                    help="known-prefix sessions to accumulate over")
    ap.add_argument("--prefix-len", type=int, default=32,
                    help="tokens per session (S)")
    ap.add_argument("--aligner", choices=["edit", "dtw"], default="edit",
                    help="edit = insert/delete alignment (matches the E-R3 "
                         "jitter mechanism); dtw = banded DTW")
    ap.add_argument("--rounds", type=int, default=6,
                    help="ICP realign/solve rounds")
    ap.add_argument("--match-quantile", type=float, default=0.5,
                    help="keep matches with row distance <= this quantile "
                         "of all match distances (re-solve pair hygiene)")
    ap.add_argument("--capture-dir", default=None,
                    help="path to real wire captures (if omitted, runs synthetic toy mode)")
    ap.add_argument("--run-json", default=None,
                    help="the trainer's run JSON for canonical replay")
    ap.add_argument("--corpus-file", default=None,
                    help="corpus path to regenerate the trainer's block stream")
    ap.add_argument("--solve-rows", type=int, default=16384,
                    help="maximum rows to solve on per epoch (real mode)")
    ap.add_argument("--victim-rows", type=int, default=1024,
                    help="victim rows to evaluate decoding on (real mode)")
    ap.add_argument("--device", default=None,
                    help="torch device (defaults to cuda if available, else cpu)")
    ap.add_argument("--decoder-max-pairs", type=int, default=20000,
                    help="public pairs used to train the attack decoder")
    ap.add_argument("--decoder-epochs", type=int, default=30)
    ap.add_argument("--decoder-batch-size", type=int, default=256)
    return ap


def jitter_rows(h_rows, pad_scale, seed):
    """Activation-level equivalent of rotation_lifetime.jitter_prefix:
    insert k in {1..8} random rows at random positions, then permute 4
    template slots. The wire carries THIS order; the attacker's assumed
    order is the un-jittered one."""
    g = torch.Generator().manual_seed(seed)
    rows = [r for r in h_rows]
    k = int(torch.randint(1, 9, (1,), generator=g).item())
    positions = torch.randperm(len(rows) + k, generator=g)[:k].tolist()
    for pos in sorted(positions):
        rows.insert(min(pos, len(rows)),
                    torch.randn(h_rows.shape[1], generator=g) * pad_scale)
    n = len(rows)
    q = max(1, n // 4)
    slots = [rows[i * q:(i + 1) * q] for i in range(3)] + [rows[3 * q:]]
    order = torch.randperm(4, generator=g).tolist()
    return torch.stack([t for i in order for t in slots[i]])


def realign_pairs(h_assumed, h_wire_derot, aligner, match_quantile):
    """Match wire rows (de-rotated) to assumed prefix rows via the chosen
    aligner. Returns index pairs (assumed_i, wire_j) filtered to the close
    half of match distances."""
    a = h_assumed.tolist()
    b = h_wire_derot.tolist()
    if aligner == "dtw":
        _, path = dtw_distance(a, b, window=max(len(a), len(b)) // 2)
        pairs = path
    else:
        pairs = edit_alignment(a, b)
    if not pairs:
        return []
    ds = torch.tensor([((h_assumed[i] - h_wire_derot[j]) ** 2).sum().item()
                       for i, j in pairs])
    thr = torch.quantile(ds, match_quantile).item()
    return [p for p, d in zip(pairs, ds.tolist()) if d <= thr]


def attack_sessions(h_assumed_all, h_wire_all, w0, victim, args, seed):
    """One (seed) accumulation run over the sessions.
    h_assumed_all/h_wire_all: [sessions*S(+jitter rows), H].
    Returns (top1, diagnostics)."""
    w_hat = torch.eye(args.hidden, dtype=torch.float64)
    for _ in range(args.rounds):
        derot = (h_wire_all.double() @ w_hat.T).float()
        pairs = realign_pairs(h_assumed_all, derot, args.aligner,
                              args.match_quantile)
        if len(pairs) < args.hidden // 2:
            break
        idx_a = torch.tensor([p[0] for p in pairs])
        idx_w = torch.tensor([p[1] for p in pairs])
        w_new, tag = solve_w(h_assumed_all[idx_a], h_wire_all[idx_w])
        if w_new is None:
            return None, {"error": tag, "n_pairs": len(pairs)}
        w_hat = w_new
    derot = (h_wire_all.double() @ w_hat.T).float()
    pairs = realign_pairs(h_assumed_all, derot, args.aligner,
                          args.match_quantile)
    top1 = recovery_with_what_nn(victim["h_wire"], w_hat,
                                 victim["ref_h"], victim["ref_tok"],
                                 victim["victim_tok"])
    return top1, {"n_pairs": len(pairs), "rounds": args.rounds}


def run_toy(args):
    from ..synthetic import make_toy_world
    if args.quick:
        args.sessions, args.prefix_len = 4, 24
        args.seeds, args.solve_seeds = [0, 1], [0]
    out = artifacts.make_artifact(
        "dtraining.attacker.er8.v1",
        {"attack": EXPERIMENT_ID, "mode": args.mode, "toy": True,
         "hidden": args.hidden, "sessions": args.sessions,
         "prefix_len": args.prefix_len, "aligner": args.aligner,
         "rounds": args.rounds, "match_quantile": args.match_quantile,
         "seeds": args.seeds, "seed": args.seed},
        "honest-but-curious cloud under a JITTERED known-prefix scaffold "
        "(E-R3 arm b). The attacker knows the scaffold prefix but not the "
        "per-session pad insertion / slot permutation, and performs "
        "edit-distance/DTW realignment between captured boundary "
        "activations and candidate prefix positions before the W solve.",
        interpretation="If jittered_scaffold recovery rises well above the "
                       "label-free band under this attacker, the E-R3 0% "
                       "claim is an artifact of the passive-alignment "
                       "adversary and needs the realignment-aware "
                       "qualifier.")
    for arm in ("fixed_scaffold", "jittered_scaffold"):
        per_seed = []
        for seed in args.seeds:
            world = make_toy_world(hidden=args.hidden, n_public=4096,
                                   n_victim=512, n_epochs=1,
                                   master_seed=args.seed + 1000 * seed,
                                   seed=args.seed + seed)
            w0 = world["Ws"][0]
            s_tok = args.prefix_len
            assumed, wire = [], []
            for si in range(args.sessions):
                h_sess = world["public_h"][si * s_tok:(si + 1) * s_tok]
                if arm == "jittered_scaffold":
                    h_true = jitter_rows(h_sess, pad_scale=1.0,
                                         seed=args.seed + 31337 + si)
                else:
                    h_true = h_sess
                assumed.append(h_sess)
                wire.append(h_wire(h_true, w0))
            h_assumed = torch.cat(assumed)
            h_w = torch.cat(wire)
            victim = {"h_wire": h_wire(world["victim_h"], w0),
                      "ref_h": world["public_h"],
                      "ref_tok": world["public_tok"],
                      "victim_tok": world["victim_tok"]}
            # passive-alignment baseline for the same arm (what E-R3 tested)
            n = min(h_assumed.shape[0], h_w.shape[0])
            w_pass, tag = solve_w(h_assumed[:n], h_w[:n])
            passive = (recovery_with_what_nn(victim["h_wire"], w_pass,
                                             victim["ref_h"],
                                             victim["ref_tok"],
                                             victim["victim_tok"])
                       if w_pass is not None else None)
            try:
                top1, diag = attack_sessions(h_assumed, h_w, w0, victim,
                                             args, seed)
            except RuntimeError as e:
                journal_error(args.output, EXPERIMENT_ID,
                              {"arm": arm, "seed": seed}, e)
                continue
            band = nn_mean_decode(
                (victim["h_wire"].double()
                 @ world["Ws"][0].double().T).float() * 0
                + world["victim_h"] * 0
                + torch.randn_like(world["victim_h"]),
                victim["ref_h"], victim["ref_tok"], victim["victim_tok"])
            rec = {"experiment": EXPERIMENT_ID, "arm": arm, "seed": seed,
                   "aligner": args.aligner,
                   "alignment_search_top1": top1,
                   "passive_alignment_top1": passive,
                   "labelfree_band_top1": band, **diag}
            artifacts.append_jsonl(args.output, rec)
            out["results"].append(rec)
            per_seed.append(top1)
            print(f"[er8] arm={arm} seed={seed}: alignment-search="
                  f"{top1}% passive={passive}% band={band}%")
        if per_seed:
            out["summary"].append({
                "experiment": EXPERIMENT_ID, "arm": arm,
                "alignment_search_mean": sum(per_seed) / len(per_seed),
                "n_seeds": len(per_seed)})
    artifacts.write_artifact(args.output, out)
    return 0


def run_real(args):
    import glob, json, os, random, sys
    out = artifacts.make_artifact(
        "dtraining.attacker.er8.v1",
        {"attack": EXPERIMENT_ID, "mode": args.mode, "toy": False,
         "aligner": args.aligner, "rounds": args.rounds,
         "match_quantile": args.match_quantile},
        "E-R8 approximate-alignment attacker against real E-R9 captures.",
        interpretation="If recovery exceeds label-free baseline, the jittered defense fails."
    )
    
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "split-training")))
    from split_trainer import build_modules, make_layer_kwargs, run_layer_stack, TEXT_SAMPLES
    from trained_inversion import collect_base_pairs, train_decoder, seed_all
    from rotation_lifetime import recovery_with_what
    
    cfg = json.load(open(args.run_json))["config"]
    device = args.device or ("cuda" if (hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu")
    args.device = device
    seed = cfg["seed"]
    seq_len = cfg["seq_len"]
    mbs = cfg["micro_batch_size"]
    grad_accum = cfg["grad_accum"]
    steps = cfg["steps"]
    sa, ra = cfg["split_after"], cfg["resume_after"]
    
    seed_all(seed)
    
    # Plain namespace, not a nested class body: ``seq_len = seq_len`` would
    # resolve the right-hand side in the class namespace and raise NameError.
    A = _make_runtime_args(
        cfg, args.device, args.decoder_max_pairs, args.decoder_epochs,
        args.decoder_batch_size)
        
    embed, layers, norm, lm_head, rotary, encode = build_modules(A)
    head = torch.nn.ModuleList(list(layers[: sa + 1]))
    middle = torch.nn.ModuleList(list(layers[sa + 1: ra]))
    tail = torch.nn.ModuleList(list(layers[ra:]))
    vocab_size = lm_head.weight.shape[0]
    
    texts = list(TEXT_SAMPLES)
    if args.corpus_file:
        with open(args.corpus_file) as f:
            texts.extend(l.strip() for l in f if l.strip())
    n_eval_docs = max(1, len(texts) // 10)
    train_texts = texts[:-n_eval_docs]
    eval_texts = texts[-n_eval_docs:]
    train_blocks = encode(train_texts, seq_len)
    
    pub_docs = [t for t in texts if len(t) > 300]
    if not pub_docs or not any(encode([d], seq_len) for d in pub_docs[:20]):
        joined = "\n".join(texts)
        pub_docs = [joined[i:i + 4000] for i in range(0, len(joined), 4000)
                    if len(joined[i:i + 4000]) > 100]
    n_tr = max(1, int(0.8 * len(pub_docs)))
    tr_docs, va_docs = pub_docs[:n_tr], pub_docs[n_tr:] or pub_docs[:1]
    
    tr_h, _, tr_tok = collect_base_pairs(embed, head, middle, tail, norm,
                                         lm_head, rotary, encode, tr_docs, A,
                                         with_grad=False)
    try:
        va_h, _, va_tok = collect_base_pairs(embed, head, middle, tail, norm,
                                             lm_head, rotary, encode, va_docs, A,
                                             with_grad=False)
    except ValueError:
        va_h, va_tok = tr_h, tr_tok
        
    dec = train_decoder(tr_h, tr_tok, va_h, va_tok, tr_h.shape[1], vocab_size, A, "er_train_wire")
    
    # Canonical replay on the assumed prefix
    replay = {}
    rng = random.Random(seed)
    for step in range(steps):
        for mb in range(grad_accum):
            batch = torch.stack([train_blocks[rng.randrange(len(train_blocks))]
                                 for _ in range(mbs)]).to(args.device)
            input_ids = batch[:, :-1]
            pos = torch.arange(input_ids.shape[1], device=args.device).unsqueeze(0)
            hidden = embed(input_ids)
            lk = make_layer_kwargs(rotary, hidden, pos, A)
            head_out = run_layer_stack(head, hidden, lk)
            replay[(step, mb)] = {
                "h_assumed": head_out.detach().float().cpu(),
                "tok": input_ids.cpu(),
            }
            
    records = []
    for jf in sorted(glob.glob(os.path.join(args.capture_dir, "cloud_*.json"))):
        pt = jf[:-len(".json")] + ".pt"
        if os.path.exists(pt):
            records.append((json.load(open(jf)), pt))
            
    out_map = {}
    fwd_records = [(m, p) for m, p in records if m.get("phase") == "fwd"]
    # Evaluation captures have no training step and cannot be aligned to the
    # canonical training replay. Keep them out of the E-R8 solve explicitly.
    sel = [(m, p) for m, p in fwd_records
           if isinstance(m.get("step"), int) and isinstance(m.get("mb_id"), int)]
    excluded = len(fwd_records) - len(sel)
    if excluded:
        print(f"[er8] excluded {excluded} non-training forward captures")
    sel.sort(key=lambda r: (r[0]["step"], r[0]["mb_id"]))
    rank = {}
    for meta, pt in sel:
        step = meta["step"]
        mb = rank.get(step, 0)
        rank[step] = mb + 1
        wire = torch.load(pt).float().cpu()
        ep = meta.get("epoch")
        w_rows = wire.reshape(-1, wire.shape[-1])
        
        if "block_indices" in meta:
            b_idxs = meta["block_indices"]
            batch = torch.stack([train_blocks[idx % len(train_blocks)] for idx in b_idxs]).to(args.device)
            input_ids = batch[:, :-1]
            pos = torch.arange(input_ids.shape[1], device=args.device).unsqueeze(0)
            hidden = embed(input_ids)
            lk = make_layer_kwargs(rotary, hidden, pos, A)
            head_out = run_layer_stack(head, hidden, lk)
            c_rows = head_out.detach().float().cpu().reshape(-1, head_out.shape[-1])
            t_rows = input_ids.cpu().reshape(-1)
        else:
            k = (step, mb)
            if k not in replay: continue
            c_rows = replay[k]["h_assumed"].reshape(-1, replay[k]["h_assumed"].shape[-1])
            t_rows = replay[k]["tok"].reshape(-1)
            
        w_, c_, t_ = out_map.get(ep, ([], [], []))
        w_.append(w_rows); c_.append(c_rows); t_.append(t_rows)
        out_map[ep] = (w_, c_, t_)
        
    fwd = {ep: (torch.cat(w_), torch.cat(c_), torch.cat(t_)) for ep, (w_, c_, t_) in out_map.items()}
    
    gen0 = torch.Generator().manual_seed(0)
    per_epoch = []
    for ep in sorted(fwd):
        w, c, t = fwd[ep]
        n = w.shape[0]
        if n < 16: continue
        
        n_solve = min(args.solve_rows, n // 2)
        # Preserve sequence ordering by taking contiguous chunks, DO NOT shuffle
        si = slice(0, n_solve)
        vi = slice(n_solve, n_solve + args.victim_rows)
        
        # PCA/covariance initialization for ICP (aligns the principal
        # components of both spaces)
        c_cent = c[si] - c[si].mean(dim=0)
        w_cent = w[si] - w[si].mean(dim=0)
        cov_c = (c_cent.T @ c_cent) / c_cent.shape[0]
        cov_w = (w_cent.T @ w_cent) / w_cent.shape[0]
        _, U_c = torch.linalg.eigh(cov_c)
        _, U_w = torch.linalg.eigh(cov_w)
        # W = U_c @ U_w.T under H_wire = H @ W row-vector convention
        w_hat = (U_c @ U_w.T).double()
        solve_called = False
        successful_solve = False
        solver_tag = "pca_unsolved"
        for _ in range(args.rounds):
            derot = (w[si].double() @ w_hat.T).float()
            pairs = realign_pairs(c[si], derot, args.aligner, args.match_quantile)
            if len(pairs) < c.shape[1] // 2: break
            idx_a = torch.tensor([p[0] for p in pairs])
            idx_w = torch.tensor([p[1] for p in pairs])
            solve_called = True
            w_new, tag = solve_w(c[si][idx_a], w[si][idx_w])
            if w_new is None: break
            successful_solve = True
            solver_tag = tag or "polar_lstsq_icp"
            w_hat = w_new
            
        derot = (w[si].double() @ w_hat.T).float()
        pairs = realign_pairs(c[si], derot, args.aligner, args.match_quantile)
        
        top1 = recovery_with_what(dec, w[vi], w_hat, t[vi], args.device)
        w_pass, _ = solve_w(c[si], w[si])
        passive = recovery_with_what(dec, w[vi], w_pass, t[vi], args.device) if w_pass is not None else 0.0
        
        required_rank = c.shape[1] // 2
        evidence_sufficient = len(pairs) >= required_rank
        solve_succeeded = successful_solve and evidence_sufficient
        if not evidence_sufficient:
            outcome = "insufficient_epoch_evidence"
        elif solve_succeeded:
            outcome = "solved"
        else:
            outcome = "solver_failed"
        
        rec = {"experiment": EXPERIMENT_ID, "epoch": ep, "aligner": args.aligner,
               "alignment_search_top1": top1, "passive_alignment_top1": passive,
               "n_pairs": len(pairs),
               "n_solve_pairs": len(pairs),
               "required_rank": required_rank,
               "evidence_sufficient": evidence_sufficient,
               "solve_attempted": solve_called,
               "solve_succeeded": solve_succeeded,
               "solver_tag": solver_tag,
               "outcome": outcome}
        artifacts.append_jsonl(args.output, rec)
        out["results"].append(rec)
        per_epoch.append(top1)
        print(f"[er8] epoch={ep}: alignment-search={top1:.2f}% passive={passive:.2f}% (pairs={len(pairs)}, solve_succeeded={solve_succeeded})")
        
    if per_epoch:
        out["summary"].append({
            "experiment": EXPERIMENT_ID,
            "alignment_search_mean": sum(per_epoch) / len(per_epoch),
            "n_epochs": len(per_epoch)
        })
    artifacts.write_artifact(args.output, out)
    return 0


def run(args):
    require_torch(EXPERIMENT_ID)
    if args.capture_dir:
        return run_real(args)
    return run_toy(args)
