"""Auditable text-only founder and multi-investor panel orchestration.

An investor backend receives the entire public text prefix and returns one
structured public turn.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json
import math
import re
from typing import Any, Iterable, Mapping

from schema import (ClaimRecord, FounderClaim, Offer, PanelConfig, PublicTurn,
                    TwinBackend, TwinReply)


INVESTOR_ACTIONS = frozenset(("ask", "comment", "offer", "revise", "withdraw", "exit"))
PROPOSAL_PATTERN = re.compile(
    r"\b(?:I|we)(?:\s+(?:will|would|can|am going to|are going to|am|are)"
    r"|['’](?:ll|d|m going to|re going to|m|re))?\s+"
    r"(?:offer(?:ing)?|giv(?:e|ing)|invest(?:ing)?|do|propos(?:e|ing)|"
    r"put(?:ting)? up|revis(?:e|ing) my offer to)\b", re.IGNORECASE)
TERMS_PATTERN = re.compile(
    r"(?:\$\s*(?P<dollars>\d[\d,]*(?:\.\d+)?)\s*(?P<dollar_scale>k|m|thousand|million)?"
    r"|(?P<usd>\d[\d,]*(?:\.\d+)?)\s*(?P<usd_scale>k|m|thousand|million)?\s*(?:USD|dollars))"
    r"\s*(?:for|in exchange for)\s*(?:a\s+)?"
    r"(?P<equity>\d+(?:\.\d+)?)\s*(?:%|percent)(?!\w)", re.IGNORECASE)
MONEY_PATTERN = re.compile(
    r"\$\s*\d|\d[\d,]*(?:\.\d+)?\s*(?:k|m|thousand|million)?\s*(?:USD|dollars)\b"
    r"|\broyalt(?:y|ies)\b|%|\bpercent\b", re.IGNORECASE)
EXIT_PATTERN = re.compile(
    r"\bI(?:['’]m| am) out\b|\bI exit\b|\bI(?:['’]m| am) dropping out\b", re.IGNORECASE)
WITHDRAW_PATTERN = re.compile(
    r"\bI withdraw my offer\b|\bI(?:['’]m| am) (?:withdrawing|pulling|taking back) my offer\b"
    r"|\bmy offer is (?:off the table|withdrawn)\b", re.IGNORECASE)
FOUNDER_DECISION_PATTERN = re.compile(
    r"\bI (?:accept|reject) offer o\d+\b|\bI end the pitch\b", re.IGNORECASE)
_SENTENCE = re.compile(r"\S.*?(?:[.!?](?=\s|$)|$)", re.DOTALL)
_SCALE = {"": 1, "k": 1_000, "thousand": 1_000, "m": 1_000_000, "million": 1_000_000}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def genesis_hash(config: PanelConfig) -> str:
    # Scenario IDs are excluded so counterfactual branches share a prefix hash.
    material = {"pitch_id": config.pitch_id, "investor_order": config.investor_order,
                "max_investor_turns": config.max_investor_turns,
                "protocol_version": config.protocol_version,
                "policy_manifest": config.policy_manifest,
                "max_retries": config.max_retries}
    return sha256(canonical_json(material).encode("utf-8")).hexdigest()


def event_hash(prefix_hash: str, fields: dict[str, Any]) -> str:
    return sha256((prefix_hash + "\n" + canonical_json(fields)).encode("utf-8")).hexdigest()


def _contains_number(text: str, value: int | float) -> bool:
    compact = re.sub(r"(?<=\d),(?=\d)", "", text)
    number = f"{value:g}" if isinstance(value, float) else str(value)
    return re.search(rf"(?<![\d.]){re.escape(number)}(?![\d.])", compact) is not None


def _check_claim_text(field: str, value: str | int | float,
                      quote: str, text: str) -> None:
    if (not isinstance(field, str) or not field
            or type(value) not in (str, int, float)
            or (type(value) is float and not math.isfinite(value))
            or not isinstance(quote, str) or not quote
            or not isinstance(text, str) or quote not in text):
        raise ValueError("claim needs typed value and own-turn quote")
    if type(value) is str:
        if value.casefold() not in quote.casefold():
            raise ValueError("claimed value absent from its quote")
        return
    if not _contains_number(quote, value):
        raise ValueError("claimed value absent from its quote")
    number = f"{value:g}" if type(value) is float else str(value)
    normalized = re.sub(r"(?<=\d),(?=\d)", "", quote)
    if (field.endswith("_usd")
            and re.search(rf"(?<![\d.]){re.escape(number)}(?![\d.])\s*USD\b",
                          normalized, re.IGNORECASE) is None):
        raise ValueError("USD claim lacks a currency-labelled value")
    if (field.endswith("_pct")
            and re.search(rf"(?<![\d.]){re.escape(number)}(?![\d.])\s*"
                          r"(?:%|percent\b)", normalized, re.IGNORECASE) is None):
        raise ValueError("percentage claim lacks a percent-labelled value")


def _quoted_ranges(text: str) -> list[tuple[int, int]]:
    ranges = []
    for opening, closing in (('"', '"'), ("“", "”"), ("«", "»")):
        start = 0
        while (left := text.find(opening, start)) >= 0:
            right = text.find(closing, left + 1)
            if right < 0:
                ranges.append((left, len(text)))
                break
            ranges.append((left, right + 1))
            start = right + 1
    return ranges


def own_sentences(text: str) -> list[tuple[str, str]]:
    """Each sentence as (literal text, same text with quoted words blanked)."""
    characters = list(text)
    for left, right in _quoted_ranges(text):
        characters[left:right] = " " * (right - left)
    masked = "".join(characters)
    result = []
    for match in _SENTENCE.finditer(masked):
        own = match.group().strip()
        if own:
            result.append((text[match.start():match.end()].strip(), own))
    return result


def parse_terms(text: str) -> list[tuple[int, float]]:
    """Read every explicit cash-for-equity statement in one span."""
    terms = []
    for match in TERMS_PATTERN.finditer(text):
        amount = match["dollars"] or match["usd"]
        scale = (match["dollar_scale"] or match["usd_scale"] or "").casefold()
        cash = float(amount.replace(",", "")) * _SCALE[scale]
        equity = float(match["equity"])
        if (not math.isfinite(cash) or cash != int(cash) or cash <= 0
                or not 0 < equity <= 100):
            raise ValueError("stated terms are outside the cash-for-equity schema")
        terms.append((int(cash), equity))
    return terms


def proposal_sentences(text: str) -> list[tuple[str, str]]:
    """Sentences whose unquoted words propose money or equity in the first person."""
    return [(literal, own) for literal, own in own_sentences(text)
            if PROPOSAL_PATTERN.search(own) and MONEY_PATTERN.search(own)]


def _has_own(pattern: re.Pattern[str], text: str) -> bool:
    return any(pattern.search(own) for _, own in own_sentences(text))


def _check_founder_disclosure_text(text: str) -> None:
    if FOUNDER_DECISION_PATTERN.search(text):
        raise ValueError("founder decision must use a decision turn")


def _check_nonoffer_action(action: str, text: str) -> None:
    if FOUNDER_DECISION_PATTERN.search(text):
        raise ValueError("investor text contains a founder decision")
    if proposal_sentences(text):
        raise ValueError("non-offer text proposes a transaction")
    exits, withdraws = _has_own(EXIT_PATTERN, text), _has_own(WITHDRAW_PATTERN, text)
    if action == "exit" and not exits:
        raise ValueError("exit needs the investor's own exit statement")
    if action == "withdraw" and (not withdraws or exits):
        raise ValueError("withdrawal needs the investor's own withdrawal statement")
    if action in {"ask", "comment"} and (exits or withdraws):
        raise ValueError("ask/comment text contains a transactional statement")


def _check_terms(reply: TwinReply) -> None:
    """The structured terms must equal those stated in the speaker's own words."""
    if (not isinstance(reply.text, str)
            or type(reply.cash_usd) is not int or reply.cash_usd <= 0
            or type(reply.equity_pct) not in (int, float)
            or not 0 < reply.equity_pct <= 100
            or not isinstance(reply.offer_evidence, str)
            or not reply.offer_evidence.strip()
            or reply.offer_evidence != reply.offer_evidence.strip()
            or reply.text.count(reply.offer_evidence) != 1):
        raise ValueError("offer needs supported terms and one own-text evidence span")
    if reply.action not in {"offer", "revise", "counter"}:
        raise ValueError("unknown cash-for-equity action")
    if FOUNDER_DECISION_PATTERN.search(reply.text) or (
            reply.action != "counter" and (_has_own(EXIT_PATTERN, reply.text)
                                           or _has_own(WITHDRAW_PATTERN, reply.text))):
        raise ValueError("offer text contains a conflicting declaration")
    start = reply.text.find(reply.offer_evidence)
    end = start + len(reply.offer_evidence)
    if any(start < right and left < end for left, right in _quoted_ranges(reply.text)):
        raise ValueError("offer evidence overlaps quoted text")
    stated = (reply.cash_usd, float(reply.equity_pct))
    span = reply.offer_evidence
    if not PROPOSAL_PATTERN.search(span) or parse_terms(span) != [stated]:
        raise ValueError("evidence span does not state these terms in the first person")
    for _, own in proposal_sentences(reply.text):
        if any(terms != stated for terms in parse_terms(own)):
            raise ValueError("public text states conflicting terms")


class TwinPanel:
    """Route a founder and several investor backends through one pitch.

    An investor sees every earlier public turn. One generated reply is
    committed before the next investor runs. Only an explicit founder choice
    can accept an offer and turn the session outcome into a deal.
    """

    def __init__(self, pitch_id: str, investors: Mapping[str, TwinBackend], *,
                  scenario_id: str = "base", max_investor_turns: int = 12,
                  max_retries: int = 2):
        if (not pitch_id or len(investors) < 2 or any(not key for key in investors)
                or max_investor_turns < 1
                or type(max_retries) is not int or max_retries < 0):
            raise ValueError("a pitch, at least two investors, a positive budget, "
                             "and a nonnegative retry budget are required")
        self.pitch_id = pitch_id
        self.investors = dict(investors)
        identities = {}
        for name, backend in self.investors.items():
            identity_method = getattr(backend, "policy_identity", None)
            if not callable(identity_method):
                raise ValueError("investor backend must declare its policy identity")
            identity = identity_method(name)
            if (not isinstance(identity, dict) or not identity
                    or any(not isinstance(key, str)
                           or value is not None and not isinstance(value, str)
                           for key, value in identity.items())):
                raise ValueError("investor policy identity must be a string mapping")
            identities[name] = deepcopy(identity)
        self.config = PanelConfig(pitch_id, tuple(investors), scenario_id,
                                  max_investor_turns, policy_manifest=identities,
                                  max_retries=max_retries)
        self.turns: list[PublicTurn] = []
        self.facts: dict[str, str | int | float] = {}
        self.claim_history: list[ClaimRecord] = []
        self.offers: dict[str, Offer] = {}
        self.current_offer: dict[str, str | None] = {key: None for key in investors}
        self.exited: set[str] = set()
        self.end_reason: str | None = None
        self.investor_turns = 0
        self.next_index = 0
        self.fork_origin: tuple[int, str] | None = None
        self._snapshots = {0: self._snapshot()}

    def _snapshot(self) -> dict[str, Any]:
        return deepcopy({"turns": self.turns, "facts": self.facts,
                         "claim_history": self.claim_history, "offers": self.offers,
                         "current_offer": self.current_offer, "exited": self.exited,
                         "end_reason": self.end_reason,
                         "investor_turns": self.investor_turns,
                         "next_index": self.next_index})

    def _restore(self, snapshot: dict[str, Any]) -> None:
        for name, value in deepcopy(snapshot).items():
            setattr(self, name, value)

    @property
    def head_hash(self) -> str:
        return self.turns[-1].event_hash if self.turns else genesis_hash(self.config)

    def _open(self) -> None:
        if self.end_reason is not None:
            raise ValueError("pitch has ended")

    def _record(self, actor_id: str, role: str, action: str, text: str,
                offer_id: str | None = None,
                payload: dict[str, Any] | None = None,
                metadata: dict[str, Any] | None = None) -> PublicTurn:
        fields = {"turn": len(self.turns) + 1, "actor_id": actor_id,
                  "role": role, "action": action, "text": text,
                  "visible_prior_turns": len(self.turns), "offer_id": offer_id,
                  "payload": payload or {}, "metadata": metadata or {}}
        prefix = self.head_hash
        event = PublicTurn(prefix_hash=prefix,
                           event_hash=event_hash(prefix, fields), **fields)
        self.turns.append(event)
        self._snapshots[len(self.turns)] = self._snapshot()
        return deepcopy(event)

    def founder_say(self, text: str,
                    claims: Iterable[FounderClaim] = ()) -> PublicTurn:
        """Record founder-supplied claims with literal public evidence spans.

        A literal match establishes provenance. Changed values require an
        explicit correction and preserve the earlier claim.
        """
        self._open()
        if not isinstance(text, str) or not text.strip():
            raise ValueError("founder text cannot be empty")
        _check_founder_disclosure_text(text)
        items = tuple(claims)
        if any(not isinstance(item, FounderClaim)
               or type(item.corrects_prior) is not bool for item in items):
            raise ValueError("claims must be FounderClaim records")
        if len({item.field for item in items}) != len(items):
            raise ValueError("duplicate claim field in one founder turn")
        facts, history, recorded = deepcopy(self.facts), deepcopy(self.claim_history), []
        for item in items:
            _check_claim_text(item.field, item.value, item.quote, text)
            prior = next((claim for claim in reversed(history)
                          if claim.field == item.field), None)
            if item.corrects_prior and prior is None:
                raise ValueError("correction has no earlier claim")
            if prior and prior.value != item.value and not item.corrects_prior:
                raise ValueError("changed claim needs explicit correction")
            record = ClaimRecord(f"c{len(history) + 1}", item.field,
                                 item.value, item.quote, len(self.turns) + 1,
                                 prior.claim_id if prior else None,
                                 item.corrects_prior)
            history.append(record)
            facts[item.field] = item.value
            recorded.append(asdict(record))
        self.facts, self.claim_history = facts, history
        return self._record("founder", "founder", "disclose", text,
                            payload={"claims": recorded})

    def founder_counter(self, offer_id: str, cash_usd: int,
                        equity_pct: float, text: str,
                        evidence_span: str) -> PublicTurn:
        """Make a public founder counterproposal; investor acceptance is separate."""
        self._open()
        offer = self.offers.get(offer_id)
        if offer is None or offer.status != "offered":
            raise ValueError("counterproposal needs an active offer")
        _check_terms(TwinReply(text, "counter", cash_usd,
                               equity_pct, evidence_span))
        if (cash_usd, float(equity_pct)) == (offer.cash_usd, offer.equity_pct):
            raise ValueError("counterproposal must change terms")
        if offer_id not in text:
            raise ValueError("counterproposal must identify the referenced offer")
        return self._record("founder", "founder", "counter", text,
                            offer_id, {"offer_id": offer_id,
                                       "cash_usd": cash_usd,
                                       "equity_pct": float(equity_pct),
                                       "evidence_span": evidence_span})

    def public_view(self, investor_id: str, *, attempt: int = 0) -> dict[str, Any]:
        if investor_id not in self.investors:
            raise ValueError("unknown investor")
        return deepcopy({
            "attempt": attempt,
            "pitch_id": self.pitch_id,
            "session_id": f"{self.pitch_id}:{self.config.scenario_id}",
            "turn_id": len(self.turns) + 1,
            "investor_id": investor_id,
            "investor_order": list(self.config.investor_order),
            "policy_identity": self.config.policy_manifest[investor_id],
            "prefix_hash": self.head_hash,
            "visible_prior_turns": len(self.turns),
            "remaining_investor_turns": self.config.max_investor_turns - self.investor_turns,
            "founder_claims": self.facts,
            "claim_history": [asdict(claim) for claim in self.claim_history],
            "turns": [asdict(turn) for turn in self.turns],
            "offers": [asdict(offer) for offer in self.offers.values()],
            "own_offer_id": self.current_offer[investor_id],
            "exited_investors": sorted(self.exited),
        })

    def founder_view(self) -> dict[str, Any]:
        """Return the same committed public history for the founder policy."""
        return deepcopy({
            "pitch_id": self.pitch_id,
            "session_id": f"{self.pitch_id}:{self.config.scenario_id}",
            "turn_id": len(self.turns) + 1,
            "prefix_hash": self.head_hash,
            "turns": [asdict(turn) for turn in self.turns],
            "founder_claims": self.facts,
            "claim_history": [asdict(claim) for claim in self.claim_history],
            "offers": [asdict(offer) for offer in self.offers.values()],
            "remaining_investor_turns": self.config.max_investor_turns - self.investor_turns,
            "policy_manifest": self.config.policy_manifest,
        })

    def _scheduled_investor(self, requested: str | None) -> tuple[str, int]:
        roster = self.config.investor_order
        if requested is not None:
            if requested not in self.investors or requested in self.exited:
                raise ValueError("unknown or exited investor")
            return requested, (roster.index(requested) + 1) % len(roster)
        for offset in range(len(roster)):
            index = (self.next_index + offset) % len(roster)
            if roster[index] not in self.exited:
                return roster[index], (index + 1) % len(roster)
        raise ValueError("no active investor")

    def step(self, investor_id: str | None = None, *,
             allow_final_founder: bool = False) -> PublicTurn:
        """Validate one generated turn, then publish it to the shared ledger.

        A rejected reply leaves the state unchanged and is retried up to
        ``max_retries`` times; an unsupported arrangement ends the session at
        once, and exhausted retries censor it.
        """
        self._open()
        if not any(turn.role == "founder" for turn in self.turns):
            raise ValueError("the founder must begin the pitch")
        if self.investor_turns >= self.config.max_investor_turns:
            return self.censor()
        investor_id, next_index = self._scheduled_investor(investor_id)
        before = self._snapshot()
        rejected: list[str] = []
        for attempt in range(self.config.max_retries + 1):
            view = self.public_view(investor_id, attempt=attempt)
            try:
                reply = self.investors[investor_id].respond(investor_id, view)
                event = self._commit_reply(investor_id, next_index, reply, rejected)
                break
            except Exception as exc:
                self._restore(before)
                for turn in tuple(self._snapshots):
                    if turn > len(self.turns):
                        del self._snapshots[turn]
                if getattr(exc, "reason", None) == "unsupported_arrangement":
                    return self.terminate("unsupported_arrangement",
                                          rejected_actor=investor_id,
                                          error_type=type(exc).__name__)
                rejected.append(type(exc).__name__)
        else:
            return self.terminate("malformed_model_output", rejected_actor=investor_id,
                                  error_type=rejected[-1])
        if (self.end_reason is None and not allow_final_founder
                and self.investor_turns >= self.config.max_investor_turns):
            self.censor()
        return event

    def _commit_reply(self, investor_id: str, next_index: int,
                      reply: TwinReply, rejected: list[str]) -> PublicTurn:
        if (not isinstance(reply, TwinReply) or not isinstance(reply.text, str)
                or not reply.text.strip()):
            raise ValueError("investor backend returned no public text")
        if reply.action not in INVESTOR_ACTIONS:
            raise ValueError("unsupported investor action")
        if (not isinstance(reply.request_id, str)
                or re.fullmatch(r"[0-9a-f]{64}", reply.request_id) is None
                or type(reply.seed) is not int or reply.seed < 0
                or reply.model_identity != self.config.policy_manifest[investor_id]
                or reply.parse_status != "parsed"
                or not isinstance(reply.evidence_ids, tuple)
                or not isinstance(reply.peer_span_ids, tuple)
                or any(not isinstance(item, str) or not item
                       for item in (*reply.evidence_ids, *reply.peer_span_ids))):
            raise ValueError("response provenance differs from its pinned policy")
        if (not isinstance(reply.joint_scope, tuple)
                or len(set(reply.joint_scope)) != len(reply.joint_scope)
                or investor_id in reply.joint_scope
                or any(name not in self.investors for name in reply.joint_scope)
                or (not reply.joint_scope and reply.joint_evidence is not None)
                or (reply.joint_scope and
                    (not isinstance(reply.joint_evidence, str)
                     or reply.text.count(reply.joint_evidence) != 1
                     or re.search(r"\b(?:joint|together|with|joining|partner|co-invest|collaborate)\b",
                                  reply.joint_evidence, re.IGNORECASE) is None
                     or any(name not in reply.joint_evidence
                            for name in reply.joint_scope)))):
            raise ValueError("joint scope lacks own-speaker textual evidence")
        offers, current, exited = (deepcopy(self.offers),
                                   deepcopy(self.current_offer), set(self.exited))
        offer_id = None
        payload: dict[str, Any] = {}
        if reply.action in {"offer", "revise"}:
            _check_terms(reply)
            previous = current[investor_id]
            if (reply.action == "offer" and previous is not None
                    or reply.action == "revise" and previous is None):
                raise ValueError("offer/revision inconsistent with own prior offer")
            if previous is not None:
                old = offers[previous]
                if (old.cash_usd, old.equity_pct) == (reply.cash_usd, float(reply.equity_pct)):
                    raise ValueError("revision must change terms")
            offer_id = f"o{len(offers) + 1}"
            if previous is not None:
                offers[previous] = replace(offers[previous], status="superseded")
            offers[offer_id] = Offer(offer_id, investor_id, reply.cash_usd,
                                     float(reply.equity_pct), len(self.turns) + 1,
                                     reply.offer_evidence, replaces=previous,
                                     joint_scope=reply.joint_scope,
                                     joint_evidence=reply.joint_evidence)
            current[investor_id] = offer_id
            payload = {"cash_usd": reply.cash_usd,
                       "equity_pct": float(reply.equity_pct),
                       "evidence_span": reply.offer_evidence,
                       "replaces": previous,
                       "joint_scope": list(reply.joint_scope),
                       "joint_evidence": reply.joint_evidence}
        else:
            if any(value is not None for value in
                   (reply.cash_usd, reply.equity_pct, reply.offer_evidence,
                    reply.joint_evidence)) or reply.joint_scope:
                raise ValueError("non-offer action cannot carry terms or offer evidence")
            _check_nonoffer_action(reply.action, reply.text)
            if reply.action in {"withdraw", "exit"}:
                previous = current[investor_id]
                if reply.action == "withdraw" and previous is None:
                    raise ValueError("no own offer to withdraw")
                payload = {"withdrawn_offer_id": previous}
                if previous is not None:
                    offers[previous] = replace(offers[previous], status="withdrawn")
                    current[investor_id] = None
                if reply.action == "exit":
                    exited.add(investor_id)
        self.offers, self.current_offer, self.exited = offers, current, exited
        self.investor_turns += 1
        self.next_index = next_index
        if len(self.exited) == len(self.investors):
            self.end_reason = "all_exited"
        metadata = {"request_id": reply.request_id,
                    "seed": reply.seed,
                    "model_identity": deepcopy(reply.model_identity),
                    "evidence_ids": list(reply.evidence_ids),
                    "peer_span_ids": list(reply.peer_span_ids),
                    "parse_status": reply.parse_status,
                    "rejected_attempts": list(rejected)}
        event = self._record(investor_id, "investor", reply.action,
                             reply.text, offer_id, payload, metadata)
        return event

    def round(self) -> tuple[PublicTurn, ...]:
        """Give each investor active at the round start one scheduled turn."""
        self._open()
        events = []
        scheduled = [name for name in self.config.investor_order
                     if name not in self.exited]
        for investor_id in scheduled:
            if self.end_reason is not None:
                break
            event = self.step(investor_id)
            events.append(event)
            if event.role == "controller":
                break
        return tuple(events)

    def founder_decide(self, offer_id: str, decision: str,
                       text: str | None = None) -> PublicTurn:
        """Record the founder's explicit choice of one still-active offer."""
        self._open()
        if decision not in {"accept", "reject"}:
            raise ValueError("founder decision must be accept or reject")
        offer = self.offers.get(offer_id)
        if offer is None or offer.status != "offered":
            if decision == "accept":
                return self.terminate("ambiguous_acceptance",
                                      rejected_actor="founder",
                                      error_type="UnresolvedOfferReference")
            raise ValueError("offer is not active")
        canonical = f"I {decision} offer {offer_id}."
        text = text or canonical
        if text != canonical:
            return self.terminate("ambiguous_acceptance",
                                  rejected_actor="founder",
                                  error_type="AmbiguousFounderDecision")
        self.offers[offer_id] = replace(offer, status="accepted" if decision == "accept"
                                        else "rejected")
        self.current_offer[offer.investor_id] = None
        if decision == "accept":
            for key, other in tuple(self.offers.items()):
                if other.status == "offered":
                    self.offers[key] = replace(other, status="expired")
                    self.current_offer[other.investor_id] = None
            self.end_reason = "accepted"
        return self._record("founder", "founder", decision, text, offer_id,
                            {"offer_id": offer_id, "decision": decision})

    def close(self, text: str = "I end the pitch.") -> PublicTurn:
        """End the pitch without a deal if no offer was accepted."""
        self._open()
        if text != "I end the pitch.":
            raise ValueError("closing turn must use the canonical public declaration")
        for key, offer in tuple(self.offers.items()):
            if offer.status == "offered":
                self.offers[key] = replace(offer, status="expired")
        self.current_offer = {key: None for key in self.investors}
        self.end_reason = "founder_closed"
        return self._record("founder", "founder", "close", text)

    def terminate(self, reason: str, *, rejected_actor: str | None = None,
                  error_type: str | None = None) -> PublicTurn:
        """Record a typed censored or unsupported terminal observation."""
        self._open()
        if reason == "turn_budget_exhausted" and self.investor_turns < self.config.max_investor_turns:
            raise ValueError("turn budget is not exhausted")
        if reason not in {"turn_budget_exhausted", "malformed_model_output",
                          "ambiguous_acceptance", "unsupported_arrangement"}:
            raise ValueError("unknown terminal reason")
        action = "unsupported" if reason == "unsupported_arrangement" else "censor"
        text = {
            "turn_budget_exhausted": "Session stopped at the prespecified turn budget.",
            "malformed_model_output": "Session stopped after an invalid model response.",
            "ambiguous_acceptance": "Session stopped after an ambiguous founder decision.",
            "unsupported_arrangement": "Session stopped at an unsupported arrangement.",
        }[reason]
        metadata = {"rejected_actor": rejected_actor, "error_type": error_type}
        self.end_reason = reason
        return self._record("controller", "controller", action, text,
                            payload={"reason": reason}, metadata=metadata)

    def censor(self) -> PublicTurn:
        return self.terminate("turn_budget_exhausted")

    def mark_unsupported(self) -> PublicTurn:
        return self.terminate("unsupported_arrangement")

    def fork(self, scenario_id: str, *, at_turn: int | None = None) -> TwinPanel:
        """Branch from one committed prefix, including its claims and offers."""
        if not scenario_id or scenario_id == self.config.scenario_id:
            raise ValueError("branch needs a distinct scenario ID")
        if not all(getattr(backend, "stateless", False)
                   for backend in self.investors.values()):
            raise ValueError("fork requires declared stateless backends")
        at_turn = len(self.turns) if at_turn is None else at_turn
        if at_turn not in self._snapshots:
            raise ValueError("unknown committed turn")
        branch = TwinPanel(self.pitch_id, self.investors,
                           scenario_id=scenario_id,
                           max_investor_turns=self.config.max_investor_turns,
                           max_retries=self.config.max_retries)
        if branch.config.policy_manifest != self.config.policy_manifest:
            raise ValueError("branch policy versions differ from the pinned prefix")
        branch._restore(self._snapshots[at_turn])
        branch._snapshots = {turn: deepcopy(snapshot)
                             for turn, snapshot in self._snapshots.items()
                             if turn <= at_turn}
        branch.fork_origin = (at_turn, branch.head_hash)
        return branch

    def outcome(self) -> dict[str, Any]:
        """Summarize observable offers and the session's final deal status."""
        accepted = [offer for offer in self.offers.values()
                    if offer.status == "accepted"]
        if len(accepted) > 1:
            raise AssertionError("multiple accepted offers")
        selected = accepted[0] if accepted else None
        status = ("deal" if selected
                  else "unsupported" if self.end_reason == "unsupported_arrangement"
                  else "censored" if self.end_reason in {
                      "turn_budget_exhausted", "malformed_model_output",
                      "ambiguous_acceptance"}
                  else "no_deal" if self.end_reason else "in_progress")
        return {"pitch_id": self.pitch_id,
                "scenario_id": self.config.scenario_id,
                "status": status,
                "end_reason": self.end_reason,
                "turns": len(self.turns),
                "investor_turns": self.investor_turns,
                "head_hash": self.head_hash,
                "policy_manifest": deepcopy(self.config.policy_manifest),
                "fork_origin": self.fork_origin,
                "offers": [asdict(offer) for offer in self.offers.values()],
                "claim_history": [asdict(claim) for claim in self.claim_history],
                "investor_final_states": {
                    name: {"exited": name in self.exited,
                           "active_offer_id": self.current_offer[name]}
                    for name in self.config.investor_order},
                "accepted_offer_id": selected.offer_id if selected else None,
                "accepted_investor": selected.investor_id if selected else None,
                "accepted_cash_usd": selected.cash_usd if selected else None,
                "accepted_equity_pct": selected.equity_pct if selected else None}
