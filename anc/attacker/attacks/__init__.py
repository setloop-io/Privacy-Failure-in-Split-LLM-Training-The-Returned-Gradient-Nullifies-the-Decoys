"""Attack registry.

Each attack module exposes:
  EXPERIMENT_ID    str, used in artifacts/journals (collection/ledger conv.)
  MODES            tuple of "training" / "inference" it applies to
  REQUIRES_LABELS  bool — labeled (oracle/known-plaintext) vs label-free
  DESCRIPTION      one-liner for --list-attacks
  build_parser()   argparse.ArgumentParser for the attack's own flags
                   (MUST work torch-less: import-time torch is forbidden)
  run(args)        executes; returns exit code
  self_test()      optional; pure-python fixtures (torch parts guarded)
"""

from importlib import import_module

ATTACKS = {
    # framework counterparts of the standalone split-training scripts
    "accumulation": "attacker.attacks.accumulation",        # E-R1a
    "known-prefix": "attacker.attacks.known_prefix",        # E-R3
    "stale-key": "attacker.attacks.stale_key",              # E-R4
    "sharded": "attacker.attacks.sharded",                  # E-R5
    "wire-eval": "attacker.attacks.wire_eval",              # e9 / er_train
    "gradient-inversion": "attacker.attacks.gradient_inversion",  # DLG++
    "membership": "attacker.attacks.membership",            # E-A4
    "output-inversion": "attacker.attacks.output_inversion",
    "latent-probe": "attacker.attacks.latent_probe",
    "latent-sensitivity": "attacker.attacks.latent_sensitivity",
    "latent-matching": "attacker.attacks.latent_matching",
    # capabilities with no standalone-script counterpart
    "alignment-search": "attacker.attacks.alignment_search",   # E-R8
    "active-cloud": "attacker.attacks.active_cloud",
    "subspace-joint": "attacker.attacks.subspace_joint",
    "ica-bss": "attacker.attacks.ica_bss",
    "leak-accumulation": "attacker.attacks.leak_accumulation",
    "max-effort": "attacker.attacks.max_effort",
    "full-history": "attacker.attacks.full_history",  # W4.1/W4.5 full-view consumer
    "deep-inversion": "attacker.attacks.deep_inversion_probe",  # positive-control probe
}


def load(name):
    return import_module(ATTACKS[name])


def for_mode(mode):
    return [n for n in ATTACKS if mode in load(n).MODES]
