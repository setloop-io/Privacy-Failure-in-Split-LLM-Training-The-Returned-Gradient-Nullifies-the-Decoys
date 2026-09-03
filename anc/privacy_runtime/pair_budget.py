"""Shared fail-closed pair-budget accounting for training and inference."""

from __future__ import annotations


class PairBudget:
    """Choose one transform epoch per boundary forward.

    Attention requires one basis for a whole context, so this accountant does
    not pretend to rotate inside a forward. In strict mode a forward larger
    than the cadence is rejected before any state mutation.
    """

    def __init__(self, ratchet_tokens: int = 0, budget_events: int = 0,
                 strict: bool = True):
        if (not isinstance(ratchet_tokens, int)
                or not isinstance(budget_events, int)
                or isinstance(ratchet_tokens, bool)
                or isinstance(budget_events, bool)
                or ratchet_tokens < 0 or budget_events < 0):
            raise ValueError("ratchet_tokens and budget_events must be non-negative integers")
        if ratchet_tokens and budget_events:
            raise ValueError("select fixed-token or evidence-budget mode, not both")
        self.ratchet_tokens = ratchet_tokens
        self.budget_events = budget_events
        self.strict = bool(strict)
        self.served = 0
        self.evidence = 0
        self.epoch = 0
        self.oversized_forwards = 0
        self.by_direction = {"forward": 0, "return": 0,
                             "activation_request": 0,
                             "activation_response": 0,
                             "gradient_request": 0,
                             "gradient_response": 0,
                             "retry": 0, "evaluation": 0}
        self._open_exchanges = {}
        self._completed_exchanges = set()

    @property
    def cadence(self) -> int:
        return self.budget_events or self.ratchet_tokens

    def add_labelable_evidence(self, count: int) -> None:
        """Add side-channel evidence before the next forward chooses an epoch."""
        self._positive_int(count, "count")
        if self.budget_events:
            self.evidence += count

    def advance(self, rows: int) -> int:
        """Account one forward and return the epoch that forward must use."""
        self._positive_int(rows, "rows")
        self._check_forward_fits(rows)
        self.served += rows
        if self.budget_events:
            self.evidence += rows
            advances, self.evidence = divmod(self.evidence,
                                              self.budget_events)
            self.epoch += advances
            return self.epoch
        if self.ratchet_tokens:
            return self.served // self.ratchet_tokens
        return 0

    def reserve_exchange(self, rows: int, exchange_id,
                         include_return: bool = True,
                         phase: str = "train",
                         directions: tuple[str, ...] | None = None) -> int:
        """Reserve every observable row before a request is transmitted.

        A forward and its return must use one epoch, so production accounting
        cannot wait until the response arrives to discover that the budget was
        exceeded.  Reserving the complete exchange up front also prevents
        retries and evaluation traffic from becoming uncounted side channels.
        Duplicate exchange identifiers fail closed.
        """
        self._positive_int(rows, "rows")
        if exchange_id is None:
            raise ValueError("exchange_id is required")
        if exchange_id in self._open_exchanges or exchange_id in self._completed_exchanges:
            self.by_direction["retry"] += rows
            raise RuntimeError("duplicate/retried privacy exchange refused")
        if phase not in ("train", "inference", "evaluation"):
            raise ValueError("phase must be train, inference, or evaluation")
        if directions is None:
            directions = (("forward", "return") if include_return
                          else ("forward",))
        if not directions or len(set(directions)) != len(directions):
            raise ValueError("directions must be a non-empty unique tuple")
        unknown = [d for d in directions if d not in self.by_direction]
        if unknown:
            raise ValueError(f"unknown privacy directions: {unknown}")
        total = rows * len(directions)
        epoch = self.advance(total)
        for direction in directions:
            self.by_direction[direction] += rows
        if phase == "evaluation":
            self.by_direction["evaluation"] += total
        self._open_exchanges[exchange_id] = {
            "epoch": epoch, "rows": rows, "directions": directions,
            "phase": phase,
        }
        return epoch

    def complete_exchange(self, exchange_id) -> None:
        """Mark a reserved request/response exchange complete exactly once."""
        if exchange_id not in self._open_exchanges:
            raise RuntimeError("unreserved or already-completed exchange")
        del self._open_exchanges[exchange_id]
        self._completed_exchanges.add(exchange_id)

    def snapshot(self) -> dict:
        return {
            "cadence": self.cadence,
            "served_observable_rows": self.served,
            "evidence": self.evidence,
            "epoch": self.epoch,
            "oversized_exchanges": self.oversized_forwards,
            "by_direction": dict(self.by_direction),
            "open_exchanges": len(self._open_exchanges),
            "completed_exchanges": len(self._completed_exchanges),
        }

    @staticmethod
    def _positive_int(value: int, name: str) -> None:
        if (not isinstance(value, int) or isinstance(value, bool)
                or value <= 0):
            raise ValueError(f"{name} must be a positive integer")

    def _check_forward_fits(self, rows: int) -> None:
        if not self.cadence or rows <= self.cadence:
            return
        self.oversized_forwards += 1
        message = (f"one atomic exchange carried {rows} observable rows "
                   f"against cadence {self.cadence}; an exchange cannot "
                   "rotate its privacy state mid-context")
        if self.strict:
            raise RuntimeError("[ER] PRODUCTION REFUSAL (#89): " + message)
        if self.oversized_forwards == 1:
            print("[ER] UNSAFE LEGACY WARNING (#89): " + message)
