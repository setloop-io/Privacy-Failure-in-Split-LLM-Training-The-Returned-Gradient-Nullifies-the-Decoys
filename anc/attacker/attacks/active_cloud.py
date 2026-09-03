#!/usr/bin/env python3
"""active-cloud — malicious-cloud reaction/probing attacks.

Every committed break assumes an HONEST-BUT-CURIOUS cloud that faithfully
computes. A malicious cloud can REACT: inject crafted perturbations in the
returned (rotated) tensors and observe how subsequent wire tensors respond
— and in training mode, craft optimizer updates that steer the local head
to leak W faster (the FL malicious-server analogue: the server controls
the global model the client trains on).

Implemented (offline, fully working):
  * AttackPlanner — plans the perturbation/update sequence: which probe
    directions to inject, at what magnitude, and what to observe back, to
    identify W with the fewest live interactions. Inference mode: delta
    probes d_i added to returned rotated tensors; the next wire tensor
    (a forward that consumed the perturbed input) reveals the response.
    Training mode: update steering — scale/bias the optimizer update
    returned for the local head along candidate W-row directions to
    accelerate the row-leak rate.
  * ObservationAnalyzer — given recorded (probe, observation) pairs,
    estimates W candidates (via solve_primitives.solve_w) and scores
    recovery; works on live captures OR the synthetic loop.
  * ActiveCloudHarness — the protocol a live server harness implements to
    drive this (see the class docstring); SyntheticHarness implements it
    over the toy world so the plan/observe/solve loop runs end-to-end.

INTEGRATION POINT: driving a DEPLOYED split-training cloud_trainer_server
requires a harness that wraps the server's send path with inject() and
tees the receive path into observe(). That glue is deliberately NOT here —
it belongs next to the server code; see attacker/README.md.

Usage:
    python -m attacker --mode training --attack active-cloud --help
    python -m attacker --mode training --attack active-cloud --toy \
        --quick --output /tmp/active.json
"""

import argparse

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .. import artifacts
from ..solve_primitives import h_wire, solve_w, w_rel_err
from .common import (add_common_args, journal_error, recovery_with_what_nn,
                     require_torch)

EXPERIMENT_ID = "active_cloud"
MODES = ("training", "inference")
REQUIRES_LABELS = False
DESCRIPTION = ("malicious-cloud probing: crafted perturbations in returned "
               "rotated tensors + (training) crafted optimizer updates "
               "steering the local head to leak W faster")


# Live-driver interface (protocol). A server harness implements these five
# methods; the attack core below only talks to this interface.
class ActiveCloudHarness:
    """Protocol for a live malicious-cloud driver.

    Lifecycle per interaction round:
      delta = plan.next_probe()             # attacker picks a perturbation
      harness.inject(delta)                 # cloud adds it to the tensor it
                                            #   returns to the local node
      obs = harness.observe()               # next wire tensor(s) the local
                                            #   node sends back
      plan.record(delta, obs)               # attacker stores the pair

    Methods a harness MUST implement:
      inject(delta)   — add delta (torch tensor [.., H], rotated space) to
                        the next tensor returned to the local node.
      observe()       — return the next wire tensor rows [n, H] captured
                        after the injection (fp32 CPU).
      epoch()         — current rotation epoch of the wire (None when the
                        ratchet is off).
      ground_truth_W()— OPTIONAL, eval harnesses only: the true current W
                        so the run can score W-recovery online (never
                        available to a real attacker).
      close()         — flush captures/journals.
    See attacker/README.md for how a server-side driver should be wired.
    """

    def inject(self, delta):
        raise NotImplementedError

    def observe(self):
        raise NotImplementedError

    def epoch(self):
        raise NotImplementedError

    def ground_truth_W(self):
        return None

    def close(self):
        pass


class AttackPlanner:
    """Offline planner: delta-probe sequence + observation bookkeeping.

    Strategy (inference and training alike): probe with scaled rows of a
    candidate basis — response differences between probed and unprobed
    rounds yield (probe, response) pairs that constrain W through the
    known local-head function class. Magnitude is swept (probing trades
    detectability against signal); direction schedule is round-robin over
    a random orthogonal probe set (maximal spread per probe)."""

    def __init__(self, hidden, magnitude, n_probes, seed):
        self.hidden = hidden
        self.magnitude = magnitude
        self.n_probes = n_probes
        g = torch.Generator().manual_seed(seed)
        q, r = torch.linalg.qr(torch.randn(hidden, hidden, generator=g,
                                           dtype=torch.float64))
        self.probe_basis = q * torch.sign(torch.diagonal(r))
        self.pairs = []  # (delta_rows [k,H], obs_rows [n,H])
        self._i = 0

    def next_probe(self):
        if self._i >= self.n_probes:
            return None
        d = self.magnitude * self.probe_basis[self._i % self.hidden]
        self._i += 1
        return d

    def record(self, delta, obs_rows):
        self.pairs.append((delta.double(), obs_rows.double()))


class ObservationAnalyzer:
    """Solves for W from (probe, observation) pairs.

    Under the deployed linearized seam the first-order response of the next
    wire row to a returned-tensor perturbation delta (added in ROTATED
    space, so the local node sees delta @ W^T) is linear in W; the analyzer
    treats (de-rotated probe direction, observed response) as pair evidence
    and feeds the accumulation to solve_w. When the harness exposes
    ground_truth_W (simulation only), W rel-err is reported."""

    def __init__(self):
        pass

    def estimate(self, planner, harness=None):
        if not planner.pairs:
            return None, "error: no probe/observation pairs recorded"
        deltas = torch.stack([d for d, _ in planner.pairs])
        obs = torch.stack([o.mean(0) for _, o in planner.pairs])
        # de-mean observations so only the probe-induced component remains
        obs = obs - obs.mean(0, keepdim=True)
        w_hat, tag = solve_w(deltas, obs)
        diag = {"solver": tag, "n_pairs": len(planner.pairs)}
        if w_hat is not None and harness is not None:
            w_true = harness.ground_truth_W()
            if w_true is not None:
                diag["w_rel_err"] = w_rel_err(w_hat, w_true.double())
        return w_hat, diag


class SyntheticHarness(ActiveCloudHarness):
    """Toy implementation of the protocol over the synthetic world: the
    'local node' is a linear head h -> h @ W_t; a returned-tensor
    perturbation delta shifts the next observed rows by delta @ L (a
    fixed random response map standing in for the local-head Jacobian)."""

    def __init__(self, world, epoch=0, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.world = world
        self._epoch = epoch
        self._w = world["Ws"][epoch].double()
        self._resp = torch.randn(world["hidden"], world["hidden"],
                                 generator=g, dtype=torch.float64) / \
            (world["hidden"] ** 0.5)
        self._pending = None
        self._ctr = 0

    def inject(self, delta):
        self._pending = delta.double()

    def observe(self):
        g = torch.Generator().manual_seed(1000 + self._ctr)
        self._ctr += 1
        base = torch.randn(8, self.world["hidden"], generator=g,
                           dtype=torch.float64)
        obs = base @ self._w
        if self._pending is not None:
            obs = obs + (self._pending @ self._resp @ self._w)
            self._pending = None
        return obs.float()

    def epoch(self):
        return self._epoch

    def ground_truth_W(self):
        return self._w


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--probes", type=int, default=128,
                    help="probe/observation rounds per seed")
    ap.add_argument("--magnitudes", type=float, nargs="+",
                    default=[0.01, 0.1, 1.0],
                    help="perturbation magnitude sweep (detectability vs "
                         "signal trade-off)")
    ap.add_argument("--steer-budget", type=float, default=0.1,
                    help="training mode: fraction of the returned update "
                         "norm available for steering per epoch")
    return ap


def run(args):
    require_torch(EXPERIMENT_ID)
    from ..synthetic import make_toy_world
    if not args.toy:
        raise SystemExit(
            f"[{EXPERIMENT_ID}] offline planner + analysis are implemented; "
            "the LIVE server driver is a documented integration point "
            "(attacker/README.md). Use --toy to exercise the full "
            "plan/inject/observe/solve loop over the synthetic harness.")
    if args.quick:
        args.probes = max(16, args.probes // 8)
        args.seeds = [0]
        args.magnitudes = args.magnitudes[:1]
    out = artifacts.make_artifact(
        "dtraining.attacker.active_cloud.v1",
        {"attack": EXPERIMENT_ID, "mode": args.mode, "toy": True,
         "hidden": args.hidden, "probes": args.probes,
         "magnitudes": args.magnitudes,
         "steer_budget": args.steer_budget, "seeds": args.seeds},
        "MALICIOUS cloud (active, not honest-but-curious): injects crafted "
        "perturbations into returned rotated tensors and observes "
        "subsequent wire tensors; in training mode additionally crafts "
        "optimizer updates steering the local head to leak W faster (FL "
        "malicious-server analogue). The planner/analyzer here are the "
        "offline core; the live driver is an integration point.",
        interpretation="W recovery with few probes at low magnitude means "
                       "the deployed seam must also be evaluated against "
                       "active servers, not only passive observers.")
    for mag in args.magnitudes:
        for seed in args.seeds:
            world = make_toy_world(hidden=args.hidden, n_public=1024,
                                   n_victim=256, n_epochs=1,
                                   master_seed=args.seed + 1000 * seed,
                                   seed=args.seed + seed)
            harness = SyntheticHarness(world, seed=args.seed + seed)
            planner = AttackPlanner(args.hidden, mag, args.probes,
                                    args.seed + seed)
            analyzer = ObservationAnalyzer()
            try:
                while True:
                    delta = planner.next_probe()
                    if delta is None:
                        break
                    harness.inject(delta)
                    planner.record(delta, harness.observe())
                w_hat, diag = analyzer.estimate(planner, harness)
            except RuntimeError as e:
                journal_error(args.output, EXPERIMENT_ID,
                              {"magnitude": mag, "seed": seed}, e)
                continue
            rec = {"experiment": EXPERIMENT_ID, "mode": args.mode,
                   "magnitude": mag, "seed": seed, **diag}
            if w_hat is not None:
                rec["victim_top1"] = recovery_with_what_nn(
                    world["wire"](world["victim_h"], 0), w_hat,
                    world["public_h"], world["public_tok"],
                    world["victim_tok"])
            artifacts.append_jsonl(args.output, rec)
            out["results"].append(rec)
            print(f"[active] mag={mag} seed={seed}: {diag}")
    artifacts.write_artifact(args.output, out)
    return 0
