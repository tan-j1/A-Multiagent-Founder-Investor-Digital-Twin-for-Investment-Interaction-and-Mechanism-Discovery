"""Public contracts for a text-only founder and investor-twin pitch.

These structures describe observable speech and pitch-level proposal events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


Scalar = str | int | float


@dataclass(frozen=True)
class FounderClaim:
    """A founder-supplied claim anchored to a literal span of the public turn."""

    field: str
    value: Scalar
    quote: str
    corrects_prior: bool = False


@dataclass(frozen=True)
class ClaimRecord:
    claim_id: str
    field: str
    value: Scalar
    quote: str
    source_turn: int
    replaces: str | None = None
    corrects_prior: bool = False


@dataclass(frozen=True)
class TwinReply:
    """One investor's proposed public text and structured action.

    `offer_evidence` is required for an offer or revision and must be a span
    from this investor's own generated text that states the proposed terms.
    """

    text: str
    action: str
    cash_usd: int | None = None
    equity_pct: float | None = None
    offer_evidence: str | None = None
    joint_scope: tuple[str, ...] = ()
    joint_evidence: str | None = None
    evidence_ids: tuple[str, ...] = ()
    peer_span_ids: tuple[str, ...] = ()
    parse_status: str = "parsed"
    request_id: str = ""
    seed: int = -1
    model_identity: dict[str, str | None] = field(default_factory=dict)


@dataclass(frozen=True)
class PublicTurn:
    turn: int
    actor_id: str
    role: str
    action: str
    text: str
    visible_prior_turns: int
    prefix_hash: str
    event_hash: str
    offer_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Offer:
    offer_id: str
    investor_id: str
    cash_usd: int
    equity_pct: float
    source_turn: int
    evidence_span: str
    status: str = "offered"
    replaces: str | None = None
    joint_scope: tuple[str, ...] = ()
    joint_evidence: str | None = None


@dataclass(frozen=True)
class PanelConfig:
    pitch_id: str
    investor_order: tuple[str, ...]
    scenario_id: str = "base"
    max_investor_turns: int = 12
    protocol_version: str = "twin-panel-v2"
    policy_manifest: dict[str, dict[str, str | None]] = field(default_factory=dict)
    max_retries: int = 2


class TwinBackend(Protocol):
    """Interface contract for a twin's response to a public pitch prefix.

    Historical memory may be injected behind the backend. A backend declares
    ``stateless = True`` to opt in to reproducible forks.
    """

    def policy_identity(self, investor_id: str) -> dict[str, str | None]:
        ...

    def respond(self, investor_id: str, public_view: dict[str, Any]) -> TwinReply:
        ...
