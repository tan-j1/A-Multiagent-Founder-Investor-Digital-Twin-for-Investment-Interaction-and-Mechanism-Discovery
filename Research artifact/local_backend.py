"""Versioned boundary between a public pitch state and a local text generator.

The generator and optional historical retriever are injected. The backend
returns one checked proposal; ``TwinPanel.step`` remains responsible
for the final state transition and for rejecting illegal public declarations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Protocol, Sequence

from knowledge import (KnowledgeSnapshot, retrieve as retrieve_knowledge,
                       visible_prefix)
from panel import (INVESTOR_ACTIONS, PROPOSAL_PATTERN, _check_nonoffer_action,
                   _check_terms, canonical_json, event_hash)
import prompt_builder
from schema import TwinReply


SCHEMA_VERSION = "local-twin-reply-v2"
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_JOINT_WORD = re.compile(r"\b(?:joint(?:ly)?|together|with|joining|partner(?:ing)?|"
                         r"co[- ]invest(?:ing)?|collaborat(?:e|ing))\b", re.I)
_PROPOSAL_TEXT = PROPOSAL_PATTERN
_UNSUPPORTED_TEXT = {
    "royalty": re.compile(r"\broyalt(?:y|ies)\b", re.I),
    "contingent": re.compile(r"\b(?:contingent|subject to|provided that|if you)\b", re.I),
    "in_kind": re.compile(r"\bin[- ]kind\b", re.I),
    "other": PROPOSAL_PATTERN,
}
_OUTPUT_KEYS = frozenset((
    "request_id", "session_id", "turn_id", "investor_id", "prefix_hash",
    "visible_prior_turns", "own_offer_id", "action", "text",
    "arrangement_type", "cash_usd", "equity_pct", "offer_evidence",
    "joint_scope", "joint_evidence", "evidence_ids", "peer_span_ids",
    "parse_status", "unsupported_reason",
))


class BackendContractError(ValueError):
    """The input, retrieval result, or generated response violates a contract."""

    reason = "invalid_reply"


class UnsupportedArrangementError(BackendContractError):
    """A legible proposed transaction exceeds the cash-equity offer schema."""

    reason = "unsupported_arrangement"


@dataclass(frozen=True)
class HistoricalCase:
    """Training-side analogy; never a fact about the current venture.

    ``public_prefix`` ends before ``target_turn``. ``observed_reply`` is the
    historical target's public turn, not a label or a current-session target.
    ``relevance_score`` records the snapshot ranking score. The retriever must
    supply a pinned ``index_version`` and train partition.
    """

    case_id: str
    investor_id: str
    venture_id: str
    index_version: str
    partition: str
    source_turn: int
    target_turn: int
    public_prefix: str
    observed_reply: str
    relevance_score: float = 0.0


HistoricalRetriever = Callable[
    [str, Mapping[str, Any]], Sequence[HistoricalCase]
]


@dataclass(frozen=True)
class GenerationRequest:
    """One deterministic, inspectable local-model call."""

    messages: tuple[dict[str, str], dict[str, str]]
    request_id: str
    session_id: str
    turn_id: int
    investor_id: str
    prefix_hash: str
    model_identity: Mapping[str, str | None]
    model_version: str
    schema_version: str
    seed: int


class LocalTextGenerator(Protocol):
    """Local inference with hash-verified backbone, tokenizer, and adapter.

    A generator may declare ``stateless = True`` only if repeated calls with
    the same request and seed produce the same result. This declaration lets
    the panel safely fork a shared prefix for counterfactual rollouts.
    """

    stateless: bool

    def artifact_identity(self, investor_id: str) -> Mapping[str, str]:
        ...

    def generate(self, request: GenerationRequest) -> str:
        ...


def _fail(message: str) -> None:
    raise BackendContractError(message)


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{field} must be nonempty text")
    return value


def _hash(value: Any, field: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_PROMPT_IMPORT_SHA256 = _file_sha256(Path(prompt_builder.__file__).resolve(strict=True))


def _numeric(value: Any, field: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        _fail(f"{field} must be finite numeric data")
    return float(value)


def _names_mentioned(text: str, names: Sequence[str]) -> bool:
    return all(re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text) is not None
               for name in names)


def _checked_view(investor_id: str, view: Mapping[str, Any]) -> dict[str, Any]:
    """Whitelist and verify the pre-target public snapshot from the panel."""
    if not isinstance(view, Mapping):
        _fail("public_view must be a mapping")
    _nonempty(investor_id, "investor_id")
    if view.get("investor_id") != investor_id:
        _fail("public_view belongs to a different investor")
    pitch_id = _nonempty(view.get("pitch_id"), "pitch_id")
    session_id = _nonempty(view.get("session_id"), "session_id")
    if not session_id.startswith(f"{pitch_id}:"):
        _fail("session_id does not belong to this pitch")
    prefix_hash = _hash(view.get("prefix_hash"), "prefix_hash")
    turns = view.get("turns")
    n = view.get("visible_prior_turns")
    if not isinstance(turns, list) or type(n) is not int or n != len(turns):
        _fail("visible_prior_turns must equal the supplied prefix length")
    turn_id = view.get("turn_id")
    if type(turn_id) is not int or turn_id != n + 1:
        _fail("turn_id must follow the committed prefix")
    investor_order = view.get("investor_order")
    if (not isinstance(investor_order, (list, tuple))
            or len(investor_order) < 2
            or any(not isinstance(name, str) or not name for name in investor_order)
            or len(set(investor_order)) != len(investor_order)
            or investor_id not in investor_order):
        _fail("investor_order does not identify a multi-investor panel")
    declared_policy = view.get("policy_identity")
    if not isinstance(declared_policy, Mapping):
        _fail("public_view lacks its pinned investor policy")
    if not turns or not any(isinstance(t, Mapping) and t.get("role") == "founder"
                            for t in turns):
        _fail("an investor reply requires an earlier founder turn")
    remaining = view.get("remaining_investor_turns")
    if type(remaining) is not int or remaining < 1:
        _fail("no investor-turn budget remains")

    public_turns: list[dict[str, Any]] = []
    prior_hash: str | None = None
    for index, turn in enumerate(turns):
        if not isinstance(turn, Mapping):
            _fail(f"turn {index + 1} is not a record")
        number = turn.get("turn")
        visible = turn.get("visible_prior_turns")
        actor = turn.get("actor_id")
        role = turn.get("role")
        action = turn.get("action")
        utterance = turn.get("text")
        offer_id = turn.get("offer_id")
        payload = turn.get("payload")
        metadata = turn.get("metadata", {})
        if (type(number) is not int or number != index + 1
                or type(visible) is not int or visible != index
                or not isinstance(actor, str) or not actor
                or role not in ("founder", "investor", "controller")
                or not isinstance(action, str) or not action
                or not isinstance(utterance, str) or not utterance.strip()
                or (offer_id is not None and
                    (not isinstance(offer_id, str) or not offer_id))
                or not isinstance(payload, dict)
                or not isinstance(metadata, dict)):
            _fail(f"turn {index + 1} has an invalid public event shape")
        link = _hash(turn.get("prefix_hash"), f"turn {number} prefix_hash")
        digest = _hash(turn.get("event_hash"), f"turn {number} event_hash")
        if prior_hash is not None and link != prior_hash:
            _fail(f"turn {number} does not follow the previous public event")
        fields = {"turn": number, "actor_id": actor, "role": role,
                  "action": action, "text": utterance,
                  "visible_prior_turns": visible, "offer_id": offer_id,
                  "payload": payload, "metadata": metadata}
        try:
            if event_hash(link, fields) != digest:
                _fail(f"turn {number} fails its event digest")
        except (TypeError, ValueError) as exc:
            raise BackendContractError(f"turn {number} is not JSON-safe") from exc
        public_turns.append({"turn": number, "actor_id": actor,
                             "role": role, "action": action,
                             "text": utterance, "offer_id": offer_id,
                             "event_hash": digest})
        prior_hash = digest
    if prior_hash != prefix_hash:
        _fail("public_view prefix_hash is not the last committed event")

    facts = view.get("founder_claims")
    if not isinstance(facts, Mapping):
        _fail("founder_claims must be a mapping")
    clean_facts = {}
    for field, value in facts.items():
        _nonempty(field, "founder claim field")
        if type(value) not in (str, int, float) or (
                type(value) is float and not math.isfinite(value)):
            _fail(f"founder claim {field} has an unsupported value")
        clean_facts[field] = value

    history = view.get("claim_history")
    if not isinstance(history, list):
        _fail("claim_history must be a list")
    clean_history = []
    for claim in history:
        if not isinstance(claim, Mapping):
            _fail("claim_history contains a nonrecord")
        source = claim.get("source_turn")
        quote = claim.get("quote")
        if (type(source) is not int or not 1 <= source <= n
                or not isinstance(quote, str) or not quote
                or public_turns[source - 1]["role"] != "founder"
                or quote not in public_turns[source - 1]["text"]):
            _fail("founder claim is not supported by its source turn")
        clean_history.append({key: claim.get(key) for key in (
            "claim_id", "field", "value", "quote", "source_turn",
            "replaces", "corrects_prior")})

    raw_offers = view.get("offers")
    if not isinstance(raw_offers, list):
        _fail("offers must be a list")
    offers: dict[str, dict[str, Any]] = {}
    for offer in raw_offers:
        if not isinstance(offer, Mapping):
            _fail("offers contains a nonrecord")
        offer_id = _nonempty(offer.get("offer_id"), "offer_id")
        owner = _nonempty(offer.get("investor_id"), "offer owner")
        source = offer.get("source_turn")
        cash = offer.get("cash_usd")
        equity = offer.get("equity_pct")
        evidence = _nonempty(offer.get("evidence_span"), "offer evidence")
        status = offer.get("status")
        replaces = offer.get("replaces")
        joint_scope = offer.get("joint_scope", ())
        joint_evidence = offer.get("joint_evidence")
        if (offer_id in offers or type(source) is not int
                or not 1 <= source <= n or type(cash) is not int or cash <= 0
                or not 0 < _numeric(equity, "offer equity_pct") <= 100
                or status not in ("offered", "superseded", "withdrawn",
                                  "rejected", "accepted", "expired")
                or (replaces is not None and
                    (not isinstance(replaces, str) or not replaces))
                or not isinstance(joint_scope, (tuple, list))
                or len(set(joint_scope)) != len(joint_scope)
                or any(not isinstance(name, str) or name not in investor_order
                       or name == owner for name in joint_scope)
                or (joint_scope and (not isinstance(joint_evidence, str)
                                     or not joint_evidence))
                or (not joint_scope and joint_evidence is not None)):
            _fail(f"offer {offer_id} has an invalid record")
        source_turn = public_turns[source - 1]
        if (source_turn["role"] != "investor"
                or source_turn["actor_id"] != owner
                or source_turn["offer_id"] != offer_id
                or source_turn["action"] != ("revise" if replaces else "offer")
                or evidence not in source_turn["text"]
                or (joint_evidence is not None and
                    (joint_evidence not in source_turn["text"]
                     or not _names_mentioned(joint_evidence, joint_scope)))):
            _fail(f"offer {offer_id} lacks its owner's public evidence")
        offers[offer_id] = {"offer_id": offer_id, "investor_id": owner,
                            "cash_usd": cash, "equity_pct": equity,
                            "source_turn": source, "evidence_span": evidence,
                            "status": status, "replaces": replaces,
                            "joint_scope": list(joint_scope),
                            "joint_evidence": joint_evidence}
    for offer in offers.values():
        prior = offers.get(offer["replaces"])
        if offer["replaces"] is not None and (
                prior is None or prior["investor_id"] != offer["investor_id"]
                or prior["source_turn"] >= offer["source_turn"]):
            _fail("offer revision does not link to an earlier own offer")

    own_offer_id = view.get("own_offer_id")
    if own_offer_id is not None and (
            not isinstance(own_offer_id, str)
            or own_offer_id not in offers
            or offers[own_offer_id]["investor_id"] != investor_id
            or offers[own_offer_id]["status"] != "offered"):
        _fail("own_offer_id does not identify this investor's active offer")
    active_own = [item["offer_id"] for item in offers.values()
                  if item["investor_id"] == investor_id
                  and item["status"] == "offered"]
    if active_own != ([own_offer_id] if own_offer_id else []):
        _fail("active offer ownership disagrees with own_offer_id")

    exited = view.get("exited_investors")
    if (not isinstance(exited, list) or any(
            not isinstance(item, str) or not item for item in exited)
            or len(set(exited)) != len(exited) or investor_id in exited):
        _fail("exited_investors is inconsistent with the target investor")
    attempt = view.get("attempt")
    if type(attempt) is not int or attempt < 0:
        _fail("attempt must be a nonnegative retry index")
    return {"attempt": attempt, "pitch_id": pitch_id, "session_id": session_id,
            "turn_id": turn_id, "investor_id": investor_id,
            "prefix_hash": prefix_hash, "visible_prior_turns": n,
            "remaining_investor_turns": remaining,
            "founder_claims": clean_facts, "claim_history": clean_history,
            "turns": public_turns, "offers": list(offers.values()),
            "own_offer_id": own_offer_id, "exited_investors": exited,
            "investor_order": list(investor_order),
            "policy_identity": dict(declared_policy)}


def _checked_cases(cases: Sequence[HistoricalCase], *, investor_id: str,
                   pitch_id: str, index_version: str,
                   max_cases: int) -> list[dict[str, Any]]:
    if not isinstance(cases, (list, tuple)) or len(cases) > max_cases:
        _fail("retriever must return a bounded sequence of historical cases")
    seen_ids: set[str] = set()
    seen_ventures: set[str] = set()
    result = []
    for case in cases:
        if not isinstance(case, HistoricalCase):
            _fail("retriever returned a non-HistoricalCase record")
        if (not isinstance(case.case_id, str) or not case.case_id
                or case.case_id in seen_ids
                or case.investor_id != investor_id
                or not isinstance(case.venture_id, str)
                or not case.venture_id or case.venture_id == pitch_id
                or case.venture_id in seen_ventures
                or case.index_version != index_version
                or case.partition != "train"
                or type(case.source_turn) is not int
                or type(case.target_turn) is not int
                or not 0 < case.source_turn < case.target_turn
                or not isinstance(case.public_prefix, str)
                or not case.public_prefix.strip()
                or not isinstance(case.observed_reply, str)
                or not case.observed_reply.strip()
                or len(case.public_prefix) > 12_000
                or len(case.observed_reply) > 4_000
                or type(case.relevance_score) not in (int, float)
                or not math.isfinite(case.relevance_score)):
            _fail("historical case violates investor, venture, snapshot, or time boundary")
        try:
            prefix = json.loads(case.public_prefix,
                                object_pairs_hook=_no_duplicate_keys,
                                parse_constant=lambda token: _fail(
                                    f"nonfinite historical prefix constant {token}"))
            safe_prefix = visible_prefix(prefix)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            raise BackendContractError("historical public_prefix is malformed") from exc
        if (prefix != safe_prefix
                or safe_prefix["case_id"] != case.venture_id
                or safe_prefix["target_investor"] != case.investor_id
                or safe_prefix["source_turn"] != case.source_turn
                or safe_prefix["target_turn"] != case.target_turn):
            _fail("historical prefix contains hidden fields or disagrees with case metadata")
        seen_ids.add(case.case_id)
        seen_ventures.add(case.venture_id)
        result.append(asdict(case))
    return result


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            _fail(f"model JSON repeats field {key}")
        result[key] = value
    return result


def _reference_catalogs(view: Mapping[str, Any],
                        cases: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]],
                                                                       list[dict[str, Any]]]:
    """Expose only prior public evidence and eligible training examples."""
    evidence: list[dict[str, Any]] = []
    spans: list[dict[str, Any]] = []
    for claim in view["claim_history"]:
        claim_id = _nonempty(claim.get("claim_id"), "claim_id")
        evidence.append({"evidence_id": f"claim:{claim_id}",
                         "kind": "founder_claim", "turn_id": claim["source_turn"],
                         "quote": claim["quote"]})
    for turn in view["turns"]:
        evidence.append({"evidence_id": f"turn:{turn['event_hash']}",
                         "kind": "public_turn", "turn_id": turn["turn"],
                         "quote": turn["text"]})
        if turn["role"] == "investor" and turn["actor_id"] != view["investor_id"]:
            text = turn["text"]
            spans.append({"peer_span_id": f"peer:{turn['event_hash']}:0:{len(text)}",
                          "turn_id": turn["turn"], "investor_id": turn["actor_id"],
                          "start": 0, "end": len(text), "quote": text})
    for case in cases:
        evidence.append({"evidence_id": f"case:{case['case_id']}",
                         "kind": "training_case", "quote": case["observed_reply"]})
    return evidence, spans


def _checked_ids(value: Any, catalog: Sequence[Mapping[str, Any]],
                 key: str) -> tuple[str, ...]:
    if (not isinstance(value, list) or any(not isinstance(item, str) or not item
                                           for item in value)
            or len(value) != len(set(value))):
        _fail(f"{key} must be a list of unique IDs")
    permitted = {entry[key[:-1] if key.endswith("s") else key]
                 for entry in catalog}
    if not set(value) <= permitted:
        _fail(f"{key} cites a source outside the public prefix or training snapshot")
    return tuple(value)


def _parse_reply(raw: Any, request: Mapping[str, Any]) -> TwinReply:
    if not isinstance(raw, str) or not raw.strip():
        _fail("generator returned no JSON text")
    try:
        obj = json.loads(raw, object_pairs_hook=_no_duplicate_keys,
                         parse_constant=lambda token: _fail(
                             f"nonfinite JSON constant {token}"))
    except json.JSONDecodeError as exc:
        raise BackendContractError("generator returned malformed JSON") from exc
    if not isinstance(obj, dict) or set(obj) != _OUTPUT_KEYS:
        _fail("generator response has missing or unsupported fields")
    for key in ("request_id", "session_id", "turn_id", "investor_id", "prefix_hash",
                "visible_prior_turns", "own_offer_id"):
        if type(obj[key]) is not type(request[key]) or obj[key] != request[key]:
            _fail(f"generated {key} does not match the committed request")
    action = obj["action"]
    text = obj["text"]
    if action not in INVESTOR_ACTIONS or not isinstance(text, str) or not text.strip():
        _fail("generated action or public text is invalid")
    evidence_ids = _checked_ids(obj["evidence_ids"], request["evidence_catalog"],
                                "evidence_ids")
    peer_span_ids = _checked_ids(obj["peer_span_ids"], request["peer_span_catalog"],
                                 "peer_span_ids")
    joint_scope = obj["joint_scope"]
    if (not isinstance(joint_scope, list)
            or any(not isinstance(name, str) or not name for name in joint_scope)
            or len(joint_scope) != len(set(joint_scope))
            or any(name == request["investor_id"]
                   or name not in request["investor_order"] for name in joint_scope)):
        _fail("joint_scope must list other investors in this panel exactly once")
    joint_evidence = obj["joint_evidence"]
    if joint_scope:
        if (action not in ("offer", "revise")
                or not isinstance(joint_evidence, str)
                or not joint_evidence.strip()
                or joint_evidence not in text
                or text.count(joint_evidence) != 1
                or _JOINT_WORD.search(joint_evidence) is None
                or not _names_mentioned(joint_evidence, joint_scope)):
            _fail("joint_scope lacks literal own-text participation evidence")
    elif joint_evidence is not None:
        _fail("joint_evidence requires a named joint_scope")
    if action in ("offer", "revise"):
        arrangement = obj["arrangement_type"]
        if arrangement in ("royalty", "contingent", "in_kind", "other"):
            if (obj["parse_status"] != "unsupported_arrangement"
                    or not isinstance(obj["unsupported_reason"], str)
                    or not obj["unsupported_reason"].strip()
                    or _UNSUPPORTED_TEXT[arrangement].search(text) is None
                    or _PROPOSAL_TEXT.search(text) is None):
                _fail("unsupported arrangement requires an explicit reason")
            raise UnsupportedArrangementError(obj["unsupported_reason"])
        if (arrangement != "cash_equity" or obj["parse_status"] != "ok"
                or obj["unsupported_reason"] is not None):
            _fail("cash-equity reply has inconsistent arrangement or parse status")
        if (type(obj["cash_usd"]) is not int or obj["cash_usd"] <= 0
                or type(obj["equity_pct"]) not in (int, float)
                or not math.isfinite(obj["equity_pct"])
                or not 0 < obj["equity_pct"] <= 100
                or not isinstance(obj["offer_evidence"], str)
                or not obj["offer_evidence"].strip()):
            _fail("generated offer has invalid cash, equity, or evidence")
        if action == "offer" and request["own_offer_id"] is not None:
            _fail("new offer would overwrite this investor's active offer")
        if action == "revise" and request["own_offer_id"] is None:
            _fail("revision has no active own offer")
        if action == "revise":
            prior = next(item for item in request["offers"]
                         if item["offer_id"] == request["own_offer_id"])
            if (obj["cash_usd"], float(obj["equity_pct"])) == (
                    prior["cash_usd"], float(prior["equity_pct"])):
                _fail("revision repeats the current offer terms")
    else:
        if any(obj[key] is not None for key in (
                "arrangement_type", "cash_usd", "equity_pct",
                "offer_evidence", "unsupported_reason")):
            _fail("non-offer reply carries transaction terms")
        if obj["parse_status"] != "ok" or joint_scope:
            _fail("non-offer reply has a transaction status or joint scope")
        if action == "withdraw" and request["own_offer_id"] is None:
            _fail("withdrawal has no active own offer")
    reply = TwinReply(text=text, action=action, cash_usd=obj["cash_usd"],
                      equity_pct=obj["equity_pct"],
                      offer_evidence=obj["offer_evidence"],
                      joint_scope=tuple(joint_scope), joint_evidence=joint_evidence,
                      evidence_ids=evidence_ids, peer_span_ids=peer_span_ids,
                      parse_status="parsed", request_id=request["request_id"],
                      seed=request["seed"],
                      model_identity=dict(request["model_identity"]))
    try:
        if action in ("offer", "revise"):
            _check_terms(reply)
        else:
            _check_nonoffer_action(action, text)
    except ValueError as exc:
        raise BackendContractError("public text does not support the generated action") from exc
    return reply


class SnapshotHistoricalRetriever:
    """Project a verified training snapshot into live, pre-target analogies.

    Only a committed *other-investor* public turn can anchor the query. The
    recent context and the target's own prior speech are drawn from the same
    chronological prefix. The snapshot's manifest digest pins every returned
    case to a particular training memory and embedding configuration.
    """

    def __init__(self, snapshot: KnowledgeSnapshot,
                 encoder: Callable[[str], Sequence[float]], *, k: int = 3):
        if (not isinstance(snapshot, KnowledgeSnapshot)
                or not callable(encoder)
                or type(k) is not int or not 0 <= k <= 3):
            _fail("snapshot, encoder, or retrieval budget is invalid")
        self.index_version = _hash(snapshot.manifest_sha256, "manifest_sha256")
        artifact_hashes = snapshot.manifest.get("artifact_sha256")
        if not isinstance(artifact_hashes, Mapping):
            _fail("snapshot has no encoder artifact registry")
        for name in ("embedding_model", "embedding_config"):
            declared = _hash(artifact_hashes.get(name), f"snapshot {name}")
            observed = _hash(getattr(encoder, f"{name}_sha256", None),
                             f"encoder {name}_sha256")
            if observed != declared:
                _fail(f"encoder {name} differs from the pinned snapshot")
        self.snapshot = snapshot
        self.encoder = encoder
        self.k = k
        self.stateless = bool(getattr(encoder, "stateless", False))

    @staticmethod
    def _query(investor_id: str, view: Mapping[str, Any]) -> dict[str, Any] | None:
        if (not isinstance(view, Mapping)
                or view.get("investor_id") != investor_id
                or not isinstance(view.get("turns"), list)):
            _fail("retrieval requires the target's public prefix")
        pitch_id = _nonempty(view.get("pitch_id"), "pitch_id")
        turns = view["turns"]
        if (type(view.get("visible_prior_turns")) is not int
                or view["visible_prior_turns"] != len(turns)):
            _fail("retrieval prefix length is inconsistent")
        for number, turn in enumerate(turns, 1):
            if (not isinstance(turn, Mapping)
                    or type(turn.get("turn")) is not int
                    or turn["turn"] != number
                    or turn.get("role") not in ("founder", "investor", "controller")
                    or not isinstance(turn.get("actor_id"), str)
                    or not turn["actor_id"]
                    or not isinstance(turn.get("text"), str)
                    or not turn["text"].strip()):
                _fail("retrieval prefix contains a malformed public turn")
        source = next((turn for turn in reversed(turns)
                       if turn["role"] == "investor"
                       and turn["actor_id"] != investor_id), None)
        if source is None:
            return None

        def project(turn: Mapping[str, Any]) -> dict[str, Any]:
            return {"turn": turn["turn"], "speaker": turn["actor_id"],
                    "text": turn["text"][:512]}

        query = {
            "case_id": pitch_id,
            "target_investor": investor_id,
            "source_investor": source["actor_id"],
            "source_quote": source["text"][:512],
            "source_turn": source["turn"],
            "target_turn": len(turns) + 1,
            "recent_context": [project(turn) for turn in turns[-12:]],
            "target_prior": [project(turn) for turn in turns
                             if turn["role"] == "investor"
                             and turn["actor_id"] == investor_id][-8:],
        }
        return visible_prefix(query)

    def __call__(self, investor_id: str,
                 public_view: Mapping[str, Any]) -> list[HistoricalCase]:
        query = self._query(investor_id, public_view)
        if query is None or self.k == 0:
            return []
        try:
            hits = retrieve_knowledge(self.snapshot, query, self.encoder,
                                      k=self.k)
        except (ValueError, TypeError, KeyError) as exc:
            raise BackendContractError("snapshot retrieval failed") from exc
        cases: list[HistoricalCase] = []
        for hit in hits:
            if not isinstance(hit, Mapping) or not isinstance(hit.get("example"), Mapping):
                _fail("snapshot retrieval returned a malformed example")
            example = hit["example"]
            try:
                safe = visible_prefix(example)
            except (ValueError, TypeError) as exc:
                raise BackendContractError("retrieved example has an unsafe prefix") from exc
            if (example.get("split") != "train"
                    or safe["target_investor"] != investor_id
                    or safe["case_id"] == query["case_id"]):
                _fail("retrieved example crosses the train, investor, or venture boundary")
            cases.append(HistoricalCase(
                case_id=_nonempty(example.get("id"), "historical row id"),
                investor_id=investor_id,
                venture_id=safe["case_id"],
                index_version=self.index_version,
                partition="train",
                source_turn=safe["source_turn"],
                target_turn=safe["target_turn"],
                public_prefix=canonical_json(safe),
                observed_reply=_nonempty(example.get("target_quote"),
                                         "historical observed reply"),
                relevance_score=_numeric(hit.get("score"), "relevance score"),
            ))
        return cases


class LocalTwinBackend:
    """Adapt a pinned local generator to ``schema.TwinBackend.respond``.

    The adapter holds only configuration and injected services. Session facts,
    peer speech, and offer ownership arrive afresh through ``public_view``.
    Inference or validation failures propagate before the panel commits a turn.
    """

    def __init__(self, generator: LocalTextGenerator, *, model_version: str,
                  retriever: HistoricalRetriever | None = None,
                  index_version: str | None = None, max_cases: int = 3,
                  base_seed: int = 0):
        if (not callable(getattr(generator, "generate", None))
                or not callable(getattr(generator, "artifact_identity", None))):
            _fail("generator must implement generate and artifact_identity")
        _nonempty(model_version, "model_version")
        pinned_index = getattr(retriever, "index_version", None)
        if index_version is None and retriever is not None and pinned_index is not None:
            index_version = pinned_index
        if ((retriever is None) != (index_version is None)
                or retriever is not None
                and (not isinstance(retriever, SnapshotHistoricalRetriever)
                     or not callable(retriever))
                or type(max_cases) is not int or not 0 <= max_cases <= 3
                or type(base_seed) is not int or base_seed < 0):
            _fail("retriever, index version, case budget, or seed is invalid")
        if index_version is not None:
            _hash(index_version, "index_version")
            if pinned_index is not None and pinned_index != index_version:
                _fail("retriever index differs from its pinned snapshot")
        prompt_path = Path(prompt_builder.__file__).resolve(strict=True)
        prompt_sha256 = _file_sha256(prompt_path)
        if prompt_sha256 != _PROMPT_IMPORT_SHA256:
            _fail("the executing prompt code changed after it was imported")
        if retriever is not None:
            expected_prompt = _hash(
                retriever.snapshot.manifest["artifact_sha256"].get("prompt_builder"),
                "snapshot prompt_builder")
            if prompt_sha256 != expected_prompt:
                _fail("the executing prompt builder differs from the pinned snapshot")
        self.generator = generator
        self.model_version = model_version
        self.retriever = retriever
        self.index_version = index_version
        self.prompt_path = prompt_path
        self.prompt_sha256 = prompt_sha256
        self.max_cases = max_cases
        self.base_seed = base_seed
        self.stateless = bool(getattr(generator, "stateless", False)
                              and (retriever is None
                                   or getattr(retriever, "stateless", False)))

    def policy_identity(self, investor_id: str) -> dict[str, str | None]:
        """Bind one investor's executing model, prompt, tokenizer, and index."""
        _nonempty(investor_id, "investor_id")
        if _file_sha256(self.prompt_path) != self.prompt_sha256:
            _fail("the executing prompt builder changed after session pinning")
        raw = self.generator.artifact_identity(investor_id)
        if not isinstance(raw, Mapping) or set(raw) != {
            "backbone_sha256", "adapter_sha256", "tokenizer_sha256"
        }:
            _fail("generator identity must name the backbone, adapter, and tokenizer")
        identity: dict[str, str | None] = {
            name: _hash(raw[name], name) for name in (
                "backbone_sha256", "adapter_sha256", "tokenizer_sha256")
        }
        snapshot = self.retriever.snapshot if self.retriever is not None else None
        files = snapshot.manifest["files"] if snapshot is not None else None
        identity.update({
            "prompt_sha256": self.prompt_sha256,
            "index_sha256": _hash(files["vectors"]["sha256"], "index_sha256")
            if files is not None else None,
            "snapshot_sha256": self.index_version,
            "embedding_model_sha256": _hash(
                snapshot.manifest["artifact_sha256"]["embedding_model"],
                "embedding_model_sha256") if snapshot is not None else None,
            "embedding_config_sha256": _hash(
                snapshot.manifest["artifact_sha256"]["embedding_config"],
                "embedding_config_sha256") if snapshot is not None else None,
            "model_version": self.model_version,
            "schema_version": SCHEMA_VERSION,
        })
        return identity

    def respond(self, investor_id: str,
                public_view: dict[str, Any]) -> TwinReply:
        view = _checked_view(investor_id, public_view)
        identity = self.policy_identity(investor_id)
        if view["policy_identity"] != identity:
            _fail("panel policy identity differs from the executing local twin")
        cases = []
        if self.retriever is not None:
            retrieved = self.retriever(investor_id, view)
            cases = _checked_cases(retrieved, investor_id=investor_id,
                                   pitch_id=view["pitch_id"],
                                   index_version=self.index_version,
                                   max_cases=self.max_cases)
        evidence_catalog, peer_span_catalog = _reference_catalogs(view, cases)
        payload = {"schema_version": SCHEMA_VERSION,
                   "model_identity": identity,
                   "index_version": self.index_version,
                   **view, "historical_cases": cases,
                   "evidence_catalog": evidence_catalog,
                   "peer_span_catalog": peer_span_catalog}
        try:
            material = canonical_json(payload)
        except (TypeError, ValueError) as exc:
            raise BackendContractError("request is not JSON-safe") from exc
        request_id = sha256(material.encode("utf-8")).hexdigest()
        payload["request_id"] = request_id
        seed_material = f"{self.base_seed}:{request_id}".encode("utf-8")
        seed = int.from_bytes(sha256(seed_material).digest()[:8], "big")
        payload["seed"] = seed
        messages = prompt_builder.build_messages(payload)
        request = GenerationRequest(messages=messages, request_id=request_id,
                                    session_id=view["session_id"],
                                    turn_id=view["turn_id"],
                                    investor_id=investor_id,
                                    prefix_hash=view["prefix_hash"],
                                    model_identity=identity,
                                    model_version=self.model_version,
                                    schema_version=SCHEMA_VERSION, seed=seed)
        raw = self.generator.generate(request)
        return _parse_reply(raw, payload)
