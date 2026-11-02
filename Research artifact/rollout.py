"""Version-pinned complete-pitch orchestration for a founder and investor twins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from panel import TwinPanel
from schema import FounderClaim, PublicTurn, TwinBackend


@dataclass(frozen=True)
class FounderMove:
    action: str
    text: str = ""
    claims: tuple[FounderClaim, ...] = ()
    offer_id: str | None = None
    cash_usd: int | None = None
    equity_pct: float | None = None
    evidence_span: str | None = None


class FounderPolicy(Protocol):
    """Choose one observable founder move from a committed public prefix."""

    def respond(self, public_view: dict[str, Any]) -> FounderMove | None:
        ...


def apply_founder_move(panel: TwinPanel, move: FounderMove) -> PublicTurn:
    if not isinstance(move, FounderMove):
        raise ValueError("founder policy must return a FounderMove")
    if move.action == "disclose":
        return panel.founder_say(move.text, move.claims)
    if move.action == "counter":
        if (move.offer_id is None or move.cash_usd is None
                or move.equity_pct is None or move.evidence_span is None):
            raise ValueError("founder counter lacks a referenced offer or terms")
        return panel.founder_counter(move.offer_id, move.cash_usd,
                                     move.equity_pct, move.text,
                                     move.evidence_span)
    if move.action in {"accept", "reject"}:
        if move.offer_id is None:
            raise ValueError("founder decision needs an offer ID")
        return panel.founder_decide(move.offer_id, move.action,
                                   move.text or None)
    if move.action == "close":
        return panel.close(move.text or "I end the pitch.")
    if move.action == "unsupported":
        return panel.mark_unsupported()
    raise ValueError("founder policy returned an unknown move")


def run_rollout(
    pitch_id: str,
    investors: Mapping[str, TwinBackend],
    founder_policy: FounderPolicy,
    *,
    scenario_id: str = "base",
    max_investor_turns: int = 12,
    max_retries: int = 2,
) -> TwinPanel:
    """Run Algorithm 1 through a deal, no-deal, censored, or unsupported end."""
    if not callable(getattr(founder_policy, "respond", None)):
        raise ValueError("founder policy must implement respond(public_view)")
    panel = TwinPanel(pitch_id, investors, scenario_id=scenario_id,
                      max_investor_turns=max_investor_turns,
                      max_retries=max_retries)
    try:
        first = founder_policy.respond(panel.founder_view())
    except Exception as exc:
        panel.terminate("malformed_model_output", rejected_actor="founder",
                        error_type=type(exc).__name__)
        return panel
    if not isinstance(first, FounderMove) or first.action != "disclose":
        panel.terminate("malformed_model_output", rejected_actor="founder",
                        error_type="InvalidInitialFounderMove")
        return panel
    try:
        apply_founder_move(panel, first)
    except Exception as exc:
        panel.terminate("malformed_model_output", rejected_actor="founder",
                        error_type=type(exc).__name__)
        return panel
    while panel.end_reason is None:
        panel.step(allow_final_founder=True)
        if panel.end_reason is not None:
            break
        try:
            move = founder_policy.respond(panel.founder_view())
            if move is not None:
                apply_founder_move(panel, move)
        except Exception as exc:
            if panel.end_reason is None:
                panel.terminate("malformed_model_output", rejected_actor="founder",
                                error_type=type(exc).__name__)
        if (panel.end_reason is None
                and panel.investor_turns >= panel.config.max_investor_turns):
            panel.censor()
    return panel
