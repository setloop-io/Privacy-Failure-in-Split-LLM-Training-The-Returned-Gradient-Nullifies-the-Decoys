"""Unified attacker framework for the split-inference / split-training
privacy study: framework counterparts of the standalone split-training/
attack scripts plus further attacker capabilities (see attacks/ATTACKS).

Usage:
    python -m attacker --list-attacks
    python -m attacker --self-test
    python -m attacker --mode training --attack accumulation --help
    python -m attacker --mode inference --attack alignment-search --toy ...
"""
