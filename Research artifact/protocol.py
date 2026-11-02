"""Auditable next-turn protocol for a text-only multi-investor twin.

This standard-library implementation defines the input boundary, peer masking,
public-state ownership, and five-label next-turn scoring. These historical
investor-action labels are distinct from the panel's live state-machine actions.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any


# Historical next-turn evaluation taxonomy, not the panel action vocabulary.
# In particular, "accept" is a historical investor label; a live panel deal
# requires an explicit founder decision recorded by panel.TwinPanel.
ACTIONS = ("offer", "counteroffer", "accept", "exit", "none")
CHANGES = ("change", "continuity", "unknown")
PEER_ROLES = frozenset(("investor", "judge", "shark", "mark",
                        "other_investor_or_guest"))
MASK = "[peer speech masked]"
FORBIDDEN_PROMPT_KEYS = frozenset((
    "completion", "gold", "labels", "after_state", "following_turns",
    "target_raw_text", "target_reply", "target_evidence", "own_action",
    "public_change", "cash_usd", "equity_pct",
))


def panel_action_to_component(
    action: str, *, actor_id: str | None = None,
    prior_offer_owner: str | None = None,
    terms_changed: bool = False, own_text_evidence: bool = False,
) -> str | None:
    """Map a live investor action to the historical next-turn label space.

    A live revision becomes a historical ``counteroffer`` only after the
    caller has checked ownership, changed terms, and an own-turn text span.
    ``withdraw`` has no unambiguous five-class label. Founder decisions are
    outside this investor-action adapter.
    """
    mapping = {
        "offer": "offer",
        "exit": "exit",
        "ask": "none",
        "comment": "none",
        "withdraw": None,
    }
    if action == "revise":
        if (actor_id and prior_offer_owner == actor_id and terms_changed
                and own_text_evidence):
            return "counteroffer"
        return None
    if action not in mapping:
        raise ValueError(f"unsupported panel investor action: {action}")
    return mapping[action]


def _audit_prompt_tree(value: Any, target_turn: int) -> None:
    """Reject target labels and future turns at any nesting level."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_PROMPT_KEYS:
                raise ValueError(f"target/label field in prompt: {key}")
            if key == "turn" and (type(child) is not int or child >= target_turn):
                raise ValueError("future or malformed turn in prompt")
            _audit_prompt_tree(child, target_turn)
    elif isinstance(value, list):
        for child in value:
            _audit_prompt_tree(child, target_turn)


def build_pair(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return full and peer-masked inputs for exactly the same target turn.

    The fixture has the target's future turn to test non-leakage. Only turns
    strictly earlier than target_turn are used to build either input.
    """
    required = {"case_id", "target_investor", "target_speaker_slot",
                "target_turn", "turns", "gold_action"}
    if not required <= set(case):
        raise ValueError("missing case fields")
    if case["gold_action"] not in ACTIONS or type(case["target_turn"]) is not int:
        raise ValueError("invalid target or gold action")
    turns = case["turns"]
    if not isinstance(turns, list) or not turns:
        raise ValueError("empty transcript")
    numbers = [turn.get("turn") for turn in turns]
    if (any(type(n) is not int for n in numbers)
            or numbers != sorted(set(numbers))):
        raise ValueError("turns must be unique and chronological")
    target = [turn for turn in turns if turn["turn"] == case["target_turn"]]
    if len(target) != 1 or str(target[0].get("speaker")) != case["target_speaker_slot"]:
        raise ValueError("target speaker/turn mismatch")
    prefix = [turn for turn in turns if turn["turn"] < case["target_turn"]]
    declared_slots = case.get("investor_speaker_slots", ())
    if (not isinstance(declared_slots, (list, tuple, set, frozenset))
            or any(not isinstance(slot, str) or not slot
                   for slot in declared_slots)):
        raise ValueError("investor_speaker_slots must be investor identifiers")
    investor_slots = set(declared_slots)
    target_prior = [turn for turn in prefix
                    if str(turn.get("speaker")) == case["target_speaker_slot"]]

    def make(masked: bool) -> dict[str, Any]:
        context = []
        for turn in prefix[-30:]:
            if not isinstance(turn.get("text"), str):
                raise ValueError("turn text must be a string")
            speaker = str(turn["speaker"])
            role = turn.get("role_hint", "unknown")
            is_peer = (speaker != case["target_speaker_slot"] and
                       (speaker in investor_slots or str(role).casefold() in PEER_ROLES))
            context.append({
                "turn": turn["turn"],
                "speaker_slot": "other_investor" if masked and is_peer else speaker,
                "role_hint": "other_investor" if masked and is_peer else role,
                "text": MASK if masked and is_peer else turn["text"],
            })
        payload = {
            "target_investor": case["target_investor"],
            "target_speaker_slot": case["target_speaker_slot"],
            "target_turn_to_predict": case["target_turn"],
            "recent_context": context,
            "target_prior": [{
                "turn": turn["turn"], "speaker_slot": str(turn["speaker"]),
                "role_hint": turn.get("role_hint", "unknown"), "text": turn["text"],
            } for turn in target_prior],
        }
        _audit_prompt_tree(payload, case["target_turn"])
        return payload

    full, masked = make(False), make(True)
    if len(full["recent_context"]) != len(masked["recent_context"]):
        raise AssertionError("mask changed the exposure window")
    for original, control in zip(full["recent_context"], masked["recent_context"]):
        if original["turn"] != control["turn"]:
            raise AssertionError("mask changed turn order")
        is_peer = (original["speaker_slot"] != case["target_speaker_slot"]
                   and (original["speaker_slot"] in investor_slots
                        or str(original["role_hint"]).casefold() in PEER_ROLES))
        if is_peer != (control["text"] == MASK):
            raise AssertionError("masking did not follow speaker ownership")
        if not is_peer and original != control:
            raise AssertionError("mask altered founder or target speech")
    return {"full": full, "masked": masked}


def chat_messages(payload: dict[str, Any], task: str) -> list[dict[str, str]]:
    """Build the two-message prompt structure for one task."""
    if task == "joint_v1":
        system = (
            "Predict the named investor's next PUBLIC utterance and investment "
            "action from only the chronological transcript prefix. Attribute "
            "each proposal to its speaker. Return one JSON object with "
            "own_action, cash_usd, equity_pct, public_change, reply. "
            "If prior public state is not established, public_change is unknown."
        )
    elif task == "action_v2":
        system = (
            "Predict the named investor's next PUBLIC investment action from "
            "only the chronological transcript prefix. Attribute each offer "
            "to its actual speaker. Return one JSON object whose sole key is "
            "own_action: offer, counteroffer, accept, exit, or none."
        )
    else:
        raise ValueError("unknown task")
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(
                payload, ensure_ascii=False, separators=(",", ":"))}]


@dataclass(frozen=True)
class PublicEvent:
    case_id: str
    speaker: str
    turn: int
    text: str
    action: str
    evidence_quote: str | None = None


class PublicLedger:
    """Replay verified own-speaker events; never transfer an offer to a peer."""

    def __init__(self, case_id: str):
        self.case_id = case_id
        self.states: dict[str, str] = {}
        self._last_turn = -1

    def apply(self, event: PublicEvent) -> str:
        if event.case_id != self.case_id or event.turn <= self._last_turn:
            raise ValueError("wrong case or nonchronological event")
        if event.action not in (*ACTIONS, "unknown"):
            raise ValueError("unknown action")
        if event.action in {"offer", "counteroffer", "accept", "exit"}:
            if not event.evidence_quote or event.text.count(event.evidence_quote) != 1:
                raise ValueError("new action needs exact own-turn evidence")
        state = self.states.get(event.speaker, "unknown")
        if event.action in {"offer", "counteroffer"}:
            state = "active_offer"
        elif event.action == "exit":
            state = "out"
        elif event.action == "accept":
            state = "investor_public_acceptance"
        self.states[event.speaker] = state
        self._last_turn = event.turn
        return state

    def state_of(self, speaker: str) -> str:
        return self.states.get(speaker, "unknown")


def score_confusion(matrix: dict[str, dict[str, int]]) -> dict[str, Any]:
    """Score historical five-label investor predictions, not panel outcomes.

    Absent classes have zero F1.
    """
    columns = (*ACTIONS, "invalid")
    if set(matrix) != set(ACTIONS):
        raise ValueError("expected all five gold classes")
    if any(set(matrix[action]) != set(columns)
           or any(type(matrix[action][column]) is not int
                  or matrix[action][column] < 0 for column in columns)
           for action in ACTIONS):
        raise ValueError("invalid confusion matrix")
    total = sum(sum(matrix[action].values()) for action in ACTIONS)
    correct = sum(matrix[action][action] for action in ACTIONS)
    per_class = {}
    for action in ACTIONS:
        tp = matrix[action][action]
        support = sum(matrix[action].values())
        predicted = sum(matrix[gold][action] for gold in ACTIONS)
        fp, fn = predicted - tp, support - tp
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        per_class[action] = {"support": support, "predicted": predicted,
                             "f1": f1}
    return {"n": total, "correct": correct,
            "accuracy": correct / total if total else None,
            "macro_f1": sum(x["f1"] for x in per_class.values()) / 5,
            "predicted_distribution": {
                action: per_class[action]["predicted"] for action in ACTIONS},
            "per_class": per_class}
