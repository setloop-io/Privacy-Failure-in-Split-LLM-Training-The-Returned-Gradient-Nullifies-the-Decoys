"""Sealed evaluator-side manifest (experiment W3.2).

The remote transcript is deliberately label-free -- `remote_transcript.py`
never accepts trusted-side data, and the `hello` frame carries no run, arm,
seed, or corpus identity -- so it cannot describe itself.  This manifest is
the trusted-side half: sealed away from attack implementations, it maps a
session id to its experimental identity and supplies the `arm`, `seed`, and
`cluster_id` labels the summarizer requires.

The separation is enforced by construction, not by convention:

  * `AttackerView` is what an attack implementation may hold.  It exposes
    the join key and the cluster id -- enough to emit well-formed records --
    and nothing else; no accessor returns a label, an arm name, or a seed.
  * `EvaluatorManifest.score()` is the only place labels and predictions
    meet, and it runs on the trusted side after the attack has returned.

`assert_sealed()` checks a serialised attacker view for trusted-side keys
and raises if any leaked in, so the property is tested rather than asserted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = "dtraining.evaluator_manifest.v1"

#: Keys that must never appear in anything handed to an attack implementation.
TRUSTED_ONLY = frozenset({
    "labels", "eval_tokens", "tokens", "token_ids", "arm", "seed", "corpus_split",
    "canonical", "canonical_latents", "gauge", "gauge_seed", "encoder", "decoder",
    "split_hash", "majority_class", "known_eval",
})


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class AttackerView:
    """Everything an attack implementation is allowed to know about a session."""

    join_key: str
    session_id: str
    cluster_ids: tuple[str, ...]

    def as_dict(self) -> dict:
        return {"join_key": self.join_key, "session_id": self.session_id,
                "cluster_ids": list(self.cluster_ids)}


@dataclass
class SessionRecord:
    """Trusted-side identity of one captured session."""

    session_id: str
    arm: str
    seed: int
    channel: int
    corpus_split: str
    split_sha256: str
    document_ids: list[str]
    labels: list[int] = field(default_factory=list)

    @property
    def join_key(self) -> str:
        """Opaque handle. Derived from the session id and a run salt, so it carries
        no arm, seed, or split information an attacker could read off it."""
        return _digest(f"{self.session_id}|{self.split_sha256}")

    def attacker_view(self) -> AttackerView:
        return AttackerView(join_key=self.join_key, session_id=self.session_id,
                            cluster_ids=tuple(self.document_ids))


class EvaluatorManifest:
    """The sealed side. Never hand this object, or anything derived from its private
    fields, to an attack implementation."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._sessions: dict[str, SessionRecord] = {}

    def add(self, record: SessionRecord) -> None:
        if record.session_id in self._sessions:
            raise ValueError(f"duplicate session: {record.session_id}")
        self._sessions[record.session_id] = record

    def attacker_views(self) -> list[AttackerView]:
        return [record.attacker_view() for record in self._sessions.values()]

    def _by_join_key(self, join_key: str) -> SessionRecord:
        for record in self._sessions.values():
            if record.join_key == join_key:
                return record
        raise KeyError(f"unknown join key: {join_key}")

    def score(self, join_key: str, attack: str, metric: str,
              per_cluster_values: dict[str, float]) -> list[dict]:
        """Turn an attack's per-cluster output into summarizer-ready JSONL records.

        This is where arm and seed are attached -- on the trusted side, after the
        attack has returned, from the manifest rather than from anything the attack
        was given.
        """
        record = self._by_join_key(join_key)
        known = set(record.document_ids)
        unknown = sorted(set(per_cluster_values) - known)
        if unknown:
            raise ValueError(f"attack returned unknown cluster ids: {unknown[:3]}")
        return [{"arm": record.arm, "attack": attack, "metric": metric,
                 "seed": record.seed, "cluster_id": cluster, "value": float(value)}
                for cluster, value in sorted(per_cluster_values.items())]

    def to_json(self) -> str:
        return json.dumps({
            "schema": SCHEMA, "run_id": self.run_id,
            "sessions": [{
                "session_id": r.session_id, "arm": r.arm, "seed": r.seed,
                "channel": r.channel, "corpus_split": r.corpus_split,
                "split_sha256": r.split_sha256, "join_key": r.join_key,
                "clusters": len(r.document_ids),
            } for r in self._sessions.values()],
        }, indent=2, sort_keys=True)

    def write(self, path: Path) -> None:
        path.write_text(self.to_json() + "\n")


def assert_sealed(payload) -> None:
    """Raise if anything handed to an attack implementation carries trusted-side data."""
    def walk(node, trail: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in TRUSTED_ONLY:
                    raise ValueError(f"trusted-side key '{key}' leaked at {trail}")
                walk(value, f"{trail}.{key}")
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                walk(value, f"{trail}[{index}]")

    walk(payload.as_dict() if isinstance(payload, AttackerView) else payload, "$")


def self_test() -> int:
    manifest = EvaluatorManifest(run_id="fixture-run")
    manifest.add(SessionRecord(
        session_id="session_a", arm="defended", seed=42, channel=0,
        corpus_split="held-out", split_sha256="78b6bfb9",
        document_ids=["doc0", "doc1"], labels=[1, 2, 3]))
    manifest.add(SessionRecord(
        session_id="session_b", arm="naked_full_width", seed=43, channel=0,
        corpus_split="held-out", split_sha256="78b6bfb9",
        document_ids=["doc0", "doc1"], labels=[1, 2, 3]))

    views = manifest.attacker_views()
    checks = {}

    try:
        for view in views:
            assert_sealed(view)
        checks["attacker_view_carries_no_trusted_keys"] = True
    except ValueError:
        checks["attacker_view_carries_no_trusted_keys"] = False

    # Structural, not substring: a hex digest will contain "42" by chance, so
    # searching the serialised view for the seed is a false-positive generator.
    checks["view_exposes_only_three_fields"] = all(
        set(v.as_dict()) == {"join_key", "session_id", "cluster_ids"} for v in views)

    # The join key must carry no arm or seed information: hold session identity fixed,
    # vary arm and seed, and the key must not move.
    a = SessionRecord(session_id="s", arm="defended", seed=42, channel=0,
                      corpus_split="held-out", split_sha256="78b6bfb9",
                      document_ids=["doc0"])
    b = SessionRecord(session_id="s", arm="naked_full_width", seed=99, channel=3,
                      corpus_split="held-out", split_sha256="78b6bfb9",
                      document_ids=["doc0"])
    checks["join_key_independent_of_arm_and_seed"] = a.join_key == b.join_key
    checks["join_keys_are_distinct_across_sessions"] = views[0].join_key != views[1].join_key
    checks["cluster_ids_available_to_attacker"] = views[0].cluster_ids == ("doc0", "doc1")

    records = manifest.score(views[0].join_key, "gradient_only", "token_top1",
                             {"doc0": 0.11, "doc1": 0.09})
    required = {"arm", "attack", "metric", "seed", "cluster_id", "value"}
    checks["scored_records_match_summarizer_schema"] = all(
        required == set(r) for r in records)
    checks["arm_attached_on_trusted_side"] = records[0]["arm"] == "defended"
    checks["seed_attached_on_trusted_side"] = records[0]["seed"] == 42

    try:
        manifest.score(views[0].join_key, "a", "b", {"doc0": 1.0, "not-a-doc": 1.0})
        checks["unknown_cluster_id_rejected"] = False
    except ValueError:
        checks["unknown_cluster_id_rejected"] = True

    try:
        assert_sealed({"frames": [{"eval_tokens": [1, 2]}]})
        checks["sealing_check_detects_a_leak"] = False
    except ValueError:
        checks["sealing_check_detects_a_leak"] = True

    for name, passed in checks.items():
        print(f"  {'ok  ' if passed else 'FAIL'} {name}")
    ok = all(checks.values())
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
