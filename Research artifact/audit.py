"""Independent trace replay and panel-level consistency checks.

The replay consumes only the public event stream, not investor backends. It
checks the hash chain and reconstructs claims, offer ownership, and endings.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import re
from typing import Any, Iterable

from panel import (TwinPanel, _check_claim_text,
                   _check_founder_disclosure_text, _check_nonoffer_action,
                   _check_terms, event_hash, genesis_hash)
from schema import ClaimRecord, Offer, PanelConfig, PublicTurn, TwinReply


def _fail(message: str) -> None:
    raise ValueError(f"trace replay: {message}")


def replay_trace(config: PanelConfig,
                 turns: Iterable[PublicTurn]) -> dict[str, Any]:
    """Rebuild observable state from a recorded public trace."""
    if (set(config.policy_manifest) != set(config.investor_order)
            or any(not isinstance(identity, dict) or not identity
                   for identity in config.policy_manifest.values())):
        _fail("investor policies are not pinned in the replay manifest")
    events = tuple(turns)
    offers: dict[str, Offer] = {}
    claims: list[ClaimRecord] = []
    current = {name: None for name in config.investor_order}
    exited: set[str] = set()
    end_reason = None
    investor_turns = 0
    prefix = genesis_hash(config)
    founder_seen = False

    for index, event in enumerate(events, 1):
        if end_reason is not None:
            _fail("event appears after a terminal turn")
        fields = {"turn": event.turn, "actor_id": event.actor_id,
                  "role": event.role, "action": event.action, "text": event.text,
                  "visible_prior_turns": event.visible_prior_turns,
                  "offer_id": event.offer_id, "payload": event.payload,
                  "metadata": event.metadata}
        if event.turn != index or event.visible_prior_turns != index - 1:
            _fail("turn order or visible-prefix length differs")
        if event.prefix_hash != prefix or event.event_hash != event_hash(prefix, fields):
            _fail("public text or structured event differs from its hash chain")
        prefix = event.event_hash

        if event.role == "founder":
            if event.actor_id != "founder":
                _fail("founder turn has another actor")
            founder_seen = True
            if event.action == "disclose":
                if event.offer_id is not None or set(event.payload) != {"claims"}:
                    _fail("founder disclosure has unexpected structured fields")
                try:
                    _check_founder_disclosure_text(event.text)
                except ValueError as exc:
                    _fail(f"founder disclosure conflicts with a decision: {exc}")
                fields_this_turn: set[str] = set()
                for raw in event.payload.get("claims", []):
                    claim = ClaimRecord(**raw)
                    try:
                        _check_claim_text(claim.field, claim.value,
                                          claim.quote, event.text)
                    except (TypeError, ValueError) as exc:
                        _fail(f"claim lacks own-text evidence: {exc}")
                    if (claim.source_turn != index or claim.field in fields_this_turn
                            or type(claim.corrects_prior) is not bool):
                        _fail("claim provenance or field repetition is invalid")
                    fields_this_turn.add(claim.field)
                    prior = next((item for item in reversed(claims)
                                  if item.field == claim.field), None)
                    if (claim.claim_id != f"c{len(claims) + 1}"
                            or claim.replaces != (prior.claim_id if prior else None)):
                        _fail("claim identity or correction link is inconsistent")
                    if (claim.corrects_prior and prior is None
                            or prior is not None and prior.value != claim.value
                            and not claim.corrects_prior):
                        _fail("claim correction flag is inconsistent")
                    claims.append(claim)
            elif event.action == "counter":
                selected = offers.get(event.offer_id)
                data = event.payload
                if (selected is None or selected.status != "offered"
                        or data.get("offer_id") != event.offer_id
                        or event.offer_id not in event.text
                        or set(data) != {"offer_id", "cash_usd",
                                         "equity_pct", "evidence_span"}):
                    _fail("counterproposal does not reference an active offer")
                try:
                    _check_terms(TwinReply(event.text, "counter",
                                           data.get("cash_usd"),
                                           data.get("equity_pct"),
                                           data.get("evidence_span")))
                except (TypeError, ValueError) as exc:
                    _fail(f"counterproposal terms lack own-text evidence: {exc}")
                if ((data["cash_usd"], float(data["equity_pct"]))
                        == (selected.cash_usd, selected.equity_pct)):
                    _fail("counterproposal repeats the active offer")
            elif event.action in {"accept", "reject"}:
                selected = offers.get(event.offer_id)
                if selected is None or selected.status != "offered":
                    _fail("founder decision does not reference an active offer")
                if event.text != f"I {event.action} offer {event.offer_id}.":
                    _fail("founder decision text contradicts structured action")
                if event.payload != {"offer_id": event.offer_id,
                                     "decision": event.action}:
                    _fail("founder decision payload differs from public action")
                offers[event.offer_id] = replace(
                    selected, status="accepted" if event.action == "accept" else "rejected")
                current[selected.investor_id] = None
                if event.action == "accept":
                    for offer_id, offer in tuple(offers.items()):
                        if offer.status == "offered":
                            offers[offer_id] = replace(offer, status="expired")
                            current[offer.investor_id] = None
                    end_reason = "accepted"
            elif event.action == "close":
                if (event.text != "I end the pitch."
                        or event.offer_id is not None or event.payload):
                    _fail("founder closing text contradicts structured action")
                for offer_id, offer in tuple(offers.items()):
                    if offer.status == "offered":
                        offers[offer_id] = replace(offer, status="expired")
                        current[offer.investor_id] = None
                end_reason = "founder_closed"
            else:
                _fail("unsupported founder action")

        elif event.role == "investor":
            name = event.actor_id
            if not founder_seen or name not in current or name in exited:
                _fail("investor turn is out of order or has unknown owner")
            meta = event.metadata
            if (set(meta) != {"request_id", "seed", "model_identity", "evidence_ids",
                              "peer_span_ids", "parse_status", "rejected_attempts"}
                    or meta["model_identity"] != config.policy_manifest.get(name, {})
                    or meta["parse_status"] != "parsed"
                    or not isinstance(meta["rejected_attempts"], list)
                    or len(meta["rejected_attempts"]) > config.max_retries
                    or any(not isinstance(item, str) or not item
                           for item in meta["rejected_attempts"])
                    or not isinstance(meta["evidence_ids"], list)
                    or not isinstance(meta["peer_span_ids"], list)
                    or any(not isinstance(item, str) or not item
                           for item in (*meta["evidence_ids"], *meta["peer_span_ids"]))
                    or not isinstance(meta["request_id"], str)
                    or re.fullmatch(r"[0-9a-f]{64}", meta["request_id"]) is None
                    or type(meta["seed"]) is not int or meta["seed"] < 0):
                _fail("investor response metadata disagrees with the pinned request")
            prior_events = events[:index - 1]
            for span_id in meta["peer_span_ids"]:
                parts = span_id.split(":")
                if (len(parts) != 4 or parts[0] != "peer" or parts[2] != "0"
                        or not parts[3].isdigit() or not any(
                            prior.role == "investor" and prior.actor_id != name
                            and prior.event_hash == parts[1]
                            and len(prior.text) == int(parts[3])
                            for prior in prior_events)):
                    _fail("peer span does not identify a prior other-investor turn")
            for evidence_id in meta["evidence_ids"]:
                kind, separator, marker = evidence_id.partition(":")
                if (not separator or not marker
                        or kind not in {"claim", "turn", "case"}
                        or kind == "claim" and not any(
                            claim.claim_id == marker for claim in claims)
                        or kind == "turn" and not any(
                            prior.event_hash == marker for prior in prior_events)):
                    _fail("evidence identifier is not attributable to a prior source")
            investor_turns += 1
            if investor_turns > config.max_investor_turns:
                _fail("investor turn budget exceeded")
            own = current[name]
            if event.action in {"offer", "revise"}:
                data = event.payload
                if (event.action == "offer" and own is not None
                        or event.action == "revise" and own is None
                        or data.get("replaces") != own
                        or event.offer_id != f"o{len(offers) + 1}"
                        or set(data) != {"cash_usd", "equity_pct",
                                         "evidence_span", "replaces",
                                         "joint_scope", "joint_evidence"}):
                    _fail("offer state transition or ID is invalid")
                joint_scope = data["joint_scope"]
                joint_evidence = data["joint_evidence"]
                if (not isinstance(joint_scope, list)
                        or len(set(joint_scope)) != len(joint_scope)
                        or name in joint_scope
                        or any(other not in current for other in joint_scope)
                        or (not joint_scope and joint_evidence is not None)
                        or (joint_scope and
                            (not isinstance(joint_evidence, str)
                             or event.text.count(joint_evidence) != 1
                             or re.search(r"\b(?:joint|together|with|joining|partner|co-invest|collaborate)\b",
                                          joint_evidence, re.IGNORECASE) is None
                             or any(other not in joint_evidence
                                    for other in joint_scope)))):
                    _fail("joint scope lacks own-speaker text evidence")
                try:
                    _check_terms(TwinReply(event.text, event.action,
                                           data.get("cash_usd"),
                                           data.get("equity_pct"),
                                           data.get("evidence_span")))
                except (TypeError, ValueError) as exc:
                    _fail(f"offer terms lack own-text evidence: {exc}")
                if own is not None and (data["cash_usd"], float(data["equity_pct"])) == (
                        offers[own].cash_usd, offers[own].equity_pct):
                    _fail("revision repeats the active offer")
                if own is not None:
                    offers[own] = replace(offers[own], status="superseded")
                offers[event.offer_id] = Offer(
                    event.offer_id, name, data["cash_usd"], data["equity_pct"],
                    index, data["evidence_span"], replaces=own,
                    joint_scope=tuple(joint_scope), joint_evidence=joint_evidence)
                current[name] = event.offer_id
            elif event.action in {"withdraw", "exit"}:
                try:
                    _check_nonoffer_action(event.action, event.text)
                except ValueError as exc:
                    _fail(f"investor action lacks own-text evidence: {exc}")
                if (event.offer_id is not None
                        or event.payload != {"withdrawn_offer_id": own}):
                    _fail("withdrawal references another speaker's offer")
                if event.action == "withdraw" and own is None:
                    _fail("withdrawal without an active own offer")
                if own is not None:
                    offers[own] = replace(offers[own], status="withdrawn")
                    current[name] = None
                if event.action == "exit":
                    exited.add(name)
            elif event.action in {"ask", "comment"}:
                if event.offer_id is not None or event.payload:
                    _fail("non-offer turn contains structured offer fields")
                try:
                    _check_nonoffer_action(event.action, event.text)
                except ValueError as exc:
                    _fail(f"non-offer turn has contradictory text: {exc}")
            else:
                _fail("unsupported investor action")
            if len(exited) == len(current):
                end_reason = "all_exited"

        elif event.role == "controller":
            reason = event.payload.get("reason")
            expected = {
                "turn_budget_exhausted": ("censor", "Session stopped at the prespecified turn budget."),
                "malformed_model_output": ("censor", "Session stopped after an invalid model response."),
                "ambiguous_acceptance": ("censor", "Session stopped after an ambiguous founder decision."),
                "unsupported_arrangement": ("unsupported", "Session stopped at an unsupported arrangement."),
            }.get(reason)
            if (event.actor_id != "controller" or expected is None
                    or (event.action, event.text) != expected
                    or event.offer_id is not None
                    or event.payload != {"reason": reason}
                    or set(event.metadata) != {"rejected_actor", "error_type"}
                    or any(value is not None and not isinstance(value, str)
                           for value in event.metadata.values())
                    or (reason == "turn_budget_exhausted"
                        and investor_turns != config.max_investor_turns)):
                _fail("invalid controller termination event")
            end_reason = reason
        else:
            _fail("unknown public role")

    accepted = [offer for offer in offers.values() if offer.status == "accepted"]
    if len(accepted) > 1:
        _fail("more than one offer accepted")
    chosen = accepted[0] if accepted else None
    status = ("deal" if chosen
              else "unsupported" if end_reason == "unsupported_arrangement"
              else "censored" if end_reason in {"turn_budget_exhausted",
                                               "malformed_model_output",
                                               "ambiguous_acceptance"}
              else "no_deal" if end_reason else "in_progress")
    return {"status": status, "end_reason": end_reason,
            "turns": len(events), "investor_turns": investor_turns,
            "head_hash": prefix,
            "policy_manifest": config.policy_manifest,
            "offers": [asdict(offer) for offer in offers.values()],
            "claim_history": [asdict(claim) for claim in claims],
            "investor_final_states": {
                name: {"exited": name in exited, "active_offer_id": current[name]}
                for name in config.investor_order},
            "accepted_offer_id": chosen.offer_id if chosen else None,
            "accepted_investor": chosen.investor_id if chosen else None,
            "accepted_cash_usd": chosen.cash_usd if chosen else None,
            "accepted_equity_pct": chosen.equity_pct if chosen else None}


def verify_panel(panel: TwinPanel) -> dict[str, Any]:
    """Reconcile a live panel's outcome against independent trace replay."""
    replayed, outcome = replay_trace(panel.config, panel.turns), panel.outcome()
    for key, value in replayed.items():
        if outcome[key] != value:
            _fail(f"outcome field {key} differs from replay")
    return {"status": "PASS", "events": replayed["turns"],
            "head_hash": replayed["head_hash"],
            "outcome": replayed["status"]}


def compare_branches(left: TwinPanel, right: TwinPanel,
                     shared_turns: int) -> dict[str, Any]:
    """Confirm two scenarios diverged only after the declared common prefix."""
    if (left.pitch_id != right.pitch_id or shared_turns < 0
            or shared_turns > min(len(left.turns), len(right.turns))):
        _fail("branches do not share a valid pitch prefix")
    if (left.turns[:shared_turns] != right.turns[:shared_turns]
            or left.config.investor_order != right.config.investor_order
            or left.config.max_investor_turns != right.config.max_investor_turns
            or left.config.protocol_version != right.config.protocol_version
            or left.config.policy_manifest != right.config.policy_manifest):
        _fail("branches differ before the founder intervention")
    prefix_hash = (left.turns[shared_turns - 1].event_hash if shared_turns
                   else genesis_hash(left.config))
    return {"shared_turns": shared_turns, "shared_prefix_hash": prefix_hash,
            "left_status": left.outcome()["status"],
            "right_status": right.outcome()["status"],
            "left_head_hash": left.head_hash,
            "right_head_hash": right.head_hash}
