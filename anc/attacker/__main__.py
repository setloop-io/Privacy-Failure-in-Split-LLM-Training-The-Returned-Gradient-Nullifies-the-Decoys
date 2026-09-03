#!/usr/bin/env python3
"""python -m attacker — unified attacker framework dispatcher.

    python -m attacker --list-attacks [--mode training]
    python -m attacker --self-test
    python -m attacker --mode MODE --attack NAME [attack flags...]
    python -m attacker --mode MODE --attack NAME --help   (torch-less)

Mode selects the wire-capture schema (attacker.captures.SCHEMAS) and the
available attack set: training adds gradient / membership / active-update
attacks, inference adds serving / output-side attacks.

Every attack module is imported lazily and MUST keep --help torch-less
(repo rule, CI-enforced): torch may only be touched inside run().
"""

import argparse
import sys

from .attacks import ATTACKS, for_mode, load

MODES = ("training", "inference")


def build_parser():
    ap = argparse.ArgumentParser(
        prog="python -m attacker",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=MODES, default=None,
                    help="surface the framework operates on: selects the "
                         "capture schema and the available attack set")
    ap.add_argument("--attack", choices=sorted(ATTACKS), default=None,
                    help="attack to run (remaining args go to its parser)")
    ap.add_argument("--list-attacks", action="store_true",
                    help="list attacks (optionally filtered by --mode)")
    ap.add_argument("--self-test", action="store_true",
                    help="run every module's pure-python fixture checks "
                         "(torch sections run when torch is present)")
    return ap


def self_test():
    ok = True
    modules = ["attacker.solve_primitives", "attacker.dtw", "attacker.ica",
               "attacker.captures", "attacker.artifacts"]
    from importlib import import_module
    for modname in modules + [ATTACKS[n] for n in sorted(ATTACKS)]:
        mod = import_module(modname)
        fn = getattr(mod, "self_test", None)
        if fn is None:
            continue
        print(f"\n== {modname} ==")
        try:
            ok = (fn() == 0) and ok
        except Exception as e:  # a failing fixture must not hide the rest
            print(f"  [FAIL] {modname}.self_test raised: {e}")
            ok = False
    print("\nFRAMEWORK SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def list_attacks(mode):
    for name in sorted(ATTACKS):
        mod = load(name)
        if mode and mode not in mod.MODES:
            continue
        labels = ("labeled" if mod.REQUIRES_LABELS is True else
                  "label-free" if mod.REQUIRES_LABELS is False
                  else str(mod.REQUIRES_LABELS))
        print(f"  {name:20s} [{mod.EXPERIMENT_ID}] "
              f"modes={','.join(mod.MODES)} labels={labels}")
        print(f"      {mod.DESCRIPTION}")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = build_parser()
    args, rest = ap.parse_known_args(argv)

    if args.self_test:
        return self_test()
    if args.list_attacks or (args.attack is None):
        if args.attack is None and not args.list_attacks:
            ap.print_help()
            print()
        return list_attacks(args.mode)

    mod = load(args.attack)
    if args.mode and args.mode not in mod.MODES:
        print(f"error: attack '{args.attack}' does not apply to mode "
              f"'{args.mode}' (modes: {', '.join(mod.MODES)})",
              file=sys.stderr)
        return 2
    sub = mod.build_parser()
    sub.prog = f"python -m attacker --mode {args.mode or '<mode>'} " \
               f"--attack {args.attack}"
    sub_args = sub.parse_args(rest)
    sub_args.mode = args.mode or (mod.MODES[0] if len(mod.MODES) == 1
                                  else None)
    if sub_args.mode is None:
        print("error: --mode is required for this attack "
              f"(modes: {', '.join(mod.MODES)})", file=sys.stderr)
        return 2
    return mod.run(sub_args)


if __name__ == "__main__":
    raise SystemExit(main())
