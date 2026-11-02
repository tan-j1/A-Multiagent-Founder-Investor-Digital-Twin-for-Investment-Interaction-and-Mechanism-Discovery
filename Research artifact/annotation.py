"""Source-grounded annotation records for text-only investor interaction.

The registry, directed links, reviewed target turns, and venture partition are
separate objects.  A reply is one training observation even when several
earlier investor statements link to it.  All character offsets refer to the
unmodified public transcript, never to a normalized model prompt.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import math
from typing import Any, Mapping, Sequence


QUALITY = frozenset({"observed", "unknown", "not_applicable", "indeterminate"})
REVIEW = frozenset({"retained", "revised", "rejected", "pending"})
SPLITS = frozenset({"fit", "retrieval", "tuning", "final_evaluation"})
RELATIONS = frozenset({
    "question_to_judge", "answer_to_judge", "agree", "challenge", "build_on",
    "competitive_offer", "joint_offer", "meta_comment", "uncertain",
})
UPTAKE = frozenset({
    "named_quote", "named_paraphrase", "topical_continuation",
    "no_discernible_uptake", "unknown",
})
INCREMENTS = frozenset({
    "new_venture_fact", "interpretation", "strategic_offer",
    "social_endorsement", "unknown",
})
HISTORICAL_ACTIONS = frozenset({
    "offer", "counteroffer", "accept", "exit", "none", "unknown",
})
STANCE = frozenset({
    "open", "favorable", "concern", "conditional", "offer_pending",
    "withdrawn", "exited", "unknown",
})
STANCE_DIRECTION = frozenset({
    "continuity", "refinement", "explicit_shift", "indeterminate",
})
TERMINAL = frozenset({"deal", "no_deal", "censored", "unsupported"})
FIELD_IDS = frozenset({
    "P01", "P02", *(f"T{i:02d}" for i in range(1, 8)),
    *(f"E{i:02d}" for i in range(1, 7)),
    *(f"S{i:02d}" for i in range(1, 6)),
    *(f"N{i:02d}" for i in range(1, 5)), "V01", "Q01",
})
FIELD_VOCABULARY: Mapping[str, frozenset[str]] = {
    "E02": RELATIONS, "E03": UPTAKE, "E04": INCREMENTS,
    "S01": STANCE, "S03": STANCE_DIRECTION, "S04": HISTORICAL_ACTIONS,
    "V01": TERMINAL,
}


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class RawTurn:
    venture_id: str
    episode_id: str
    turn: int
    speaker_id: str
    role: str
    text: str
    transcript_version: str
    source_sha256: str

    def __post_init__(self) -> None:
        if (not self.venture_id or not self.episode_id or not self.speaker_id
                or not self.transcript_version or type(self.turn) is not int
                or self.turn < 0 or not isinstance(self.text, str)):
            raise ValueError("invalid raw-turn identity")
        if self.role not in {"founder", "investor", "other", "unknown"}:
            raise ValueError("unknown speaker role")
        if (len(self.source_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.source_sha256)):
            raise ValueError("source_sha256 must be a lowercase SHA-256 digest")


class TranscriptRegistry:
    """Immutable-by-convention address book for one version per venture."""

    def __init__(self, turns: Sequence[RawTurn]):
        self.turns: dict[tuple[str, int], RawTurn] = {}
        versions: dict[str, tuple[str, str, str]] = {}
        for turn in turns:
            key = (turn.venture_id, turn.turn)
            if key in self.turns:
                raise ValueError(f"duplicate venture/turn: {key}")
            identity = (turn.episode_id, turn.transcript_version,
                        turn.source_sha256)
            if turn.venture_id in versions and versions[turn.venture_id] != identity:
                raise ValueError("a venture mixes transcript versions or source hashes")
            versions[turn.venture_id] = identity
            self.turns[key] = turn
        if not self.turns:
            raise ValueError("empty transcript registry")
        self.versions = versions

    def get(self, venture_id: str, turn: int) -> RawTurn:
        try:
            return self.turns[(venture_id, turn)]
        except KeyError as exc:
            raise ValueError("span or annotation names a missing raw turn") from exc

    def prefix(self, venture_id: str, target_turn: int) -> tuple[RawTurn, ...]:
        return tuple(sorted((row for (venture, number), row in self.turns.items()
                             if venture == venture_id and number < target_turn),
                            key=lambda row: row.turn))


@dataclass(frozen=True)
class EvidenceSpan:
    venture_id: str
    turn: int
    speaker_id: str
    start: int
    end: int
    quote: str
    transcript_version: str
    source_sha256: str

    def verify(self, registry: TranscriptRegistry) -> None:
        raw = registry.get(self.venture_id, self.turn)
        if (self.speaker_id != raw.speaker_id
                or self.transcript_version != raw.transcript_version
                or self.source_sha256 != raw.source_sha256
                or type(self.start) is not int or type(self.end) is not int
                or not 0 <= self.start < self.end <= len(raw.text)
                or raw.text[self.start:self.end] != self.quote):
            raise ValueError("evidence span does not match its attributed raw text")


@dataclass(frozen=True)
class CoderJudgment:
    coder_id: str
    quality: str
    value: Any
    evidence: tuple[EvidenceSpan, ...] = ()

    def verify(self, registry: TranscriptRegistry) -> None:
        if not self.coder_id or self.quality not in QUALITY:
            raise ValueError("invalid coder judgment")
        if self.quality != "observed" and self.value is not None:
            raise ValueError("nonobserved judgments must not invent a value")
        if self.quality == "observed" and self.value is None:
            raise ValueError("observed judgment needs a value")
        _canonical(self.value)
        for span in self.evidence:
            span.verify(registry)


def _span_iou(left: Sequence[EvidenceSpan],
              right: Sequence[EvidenceSpan]) -> float | None:
    if not left and not right:
        return None
    coordinates = {(s.venture_id, s.turn, p) for s in left
                   for p in range(s.start, s.end)}
    other = {(s.venture_id, s.turn, p) for s in right
             for p in range(s.start, s.end)}
    return len(coordinates & other) / len(coordinates | other)


@dataclass(frozen=True)
class FieldReview:
    field_id: str
    judgments: tuple[CoderJudgment, ...]
    reviewer_id: str
    decision: str
    final_quality: str
    final_value: Any
    final_evidence: tuple[EvidenceSpan, ...]
    disagreement: bool
    span_agreement: float | None
    numeric_term_difference: Mapping[str, float] | None
    adjudication: str | None

    def verify(self, registry: TranscriptRegistry) -> None:
        if (self.field_id not in FIELD_IDS or not self.reviewer_id
                or self.decision not in REVIEW or self.final_quality not in QUALITY
                or not self.judgments):
            raise ValueError("invalid field review")
        if len({j.coder_id for j in self.judgments}) != len(self.judgments):
            raise ValueError("coder IDs must be distinct for independent judgments")
        for judgment in self.judgments:
            judgment.verify(registry)
        if self.final_quality != "observed" and self.final_value is not None:
            raise ValueError("unknown or inapplicable field cannot contain a value")
        if self.final_quality == "observed" and self.final_value is None:
            raise ValueError("observed field has no value")
        if (self.final_quality == "observed" and self.field_id not in {"P01", "P02", "Q01"}
                and not self.final_evidence):
            raise ValueError("substantive observed label needs an exact source span")
        vocabulary = FIELD_VOCABULARY.get(self.field_id)
        if (self.final_quality == "observed" and vocabulary is not None
                and self.final_value not in vocabulary):
            raise ValueError(f"invalid {self.field_id} label")
        if self.decision in {"retained", "revised"} and self.disagreement and not self.adjudication:
            raise ValueError("disagreement requires an adjudication record")
        if self.span_agreement is not None and not 0 <= self.span_agreement <= 1:
            raise ValueError("span agreement must be in [0,1]")
        if self.numeric_term_difference is not None and any(
                not key or not math.isfinite(value) or value < 0
                for key, value in self.numeric_term_difference.items()):
            raise ValueError("numeric term differences must be finite and nonnegative")
        _canonical(self.final_value)
        for span in self.final_evidence:
            span.verify(registry)

    @property
    def analytic(self) -> bool:
        return self.decision in {"retained", "revised"}


def review_field(
    field_id: str, judgments: Sequence[CoderJudgment], *, reviewer_id: str,
    decision: str, final_quality: str, final_value: Any,
    final_evidence: Sequence[EvidenceSpan], adjudication: str | None = None,
) -> FieldReview:
    """Retain both independent codings and the measured reconciliation gap."""
    codings = tuple(judgments)
    if not codings:
        raise ValueError("a field needs at least one coder judgment")
    disagreement = len({(j.quality, _canonical(j.value)) for j in codings}) > 1
    overlap = (_span_iou(codings[0].evidence, codings[1].evidence)
               if len(codings) >= 2 else None)
    numeric_values: dict[str, list[float]] = {}
    for judgment in codings:
        if judgment.quality != "observed":
            continue
        values = (judgment.value if isinstance(judgment.value, Mapping)
                  else {"value": judgment.value})
        for key, value in values.items():
            if type(value) in (int, float):
                numeric_values.setdefault(str(key), []).append(float(value))
    numeric_difference = {
        key: max(values) - min(values)
        for key, values in numeric_values.items() if len(values) >= 2
    } or None
    return FieldReview(
        field_id, codings, reviewer_id, decision, final_quality, final_value,
        tuple(final_evidence), disagreement, overlap, numeric_difference,
        adjudication,
    )


@dataclass(frozen=True)
class DirectedPeerLink:
    link_id: str
    venture_id: str
    source_investor: str
    source_turn: int
    target_investor: str
    target_turn: int
    fields: Mapping[str, FieldReview]
    competing_source_ids: tuple[str, ...] = ()

    def verify(self, registry: TranscriptRegistry) -> None:
        if (not self.link_id or self.source_investor == self.target_investor
                or self.source_turn >= self.target_turn):
            raise ValueError("peer link must be directed and chronological")
        source = registry.get(self.venture_id, self.source_turn)
        target = registry.get(self.venture_id, self.target_turn)
        if (source.role != "investor" or target.role != "investor"
                or source.speaker_id != self.source_investor
                or target.speaker_id != self.target_investor):
            raise ValueError("peer link speaker attribution differs from transcript")
        if not {"E01", "E02", "E03", "E04", "E05"} <= self.fields.keys():
            raise ValueError("peer link misses required relation fields")
        for code, field_review in self.fields.items():
            if code not in {"E01", "E02", "E03", "E04", "E05"}:
                raise ValueError("peer link contains a nonrelation field")
            if field_review.field_id != code or not field_review.analytic:
                raise ValueError("unreviewed peer relation cannot enter analysis")
            field_review.verify(registry)
        directed = self.fields["E01"].final_evidence
        if (not any(span.turn == self.source_turn and span.speaker_id == self.source_investor
                    for span in directed)
                or not any(span.turn == self.target_turn and span.speaker_id == self.target_investor
                           for span in directed)):
            raise ValueError("directed link needs source and target evidence spans")
        for field_review in self.fields.values():
            if any(span.venture_id != self.venture_id or span.turn > self.target_turn
                   for span in field_review.final_evidence):
                raise ValueError("link evidence crosses target or venture boundary")


@dataclass(frozen=True)
class ReviewedTargetTurn:
    venture_id: str
    investor_id: str
    target_turn: int
    target_reply: EvidenceSpan
    fields: Mapping[str, FieldReview]
    incoming_link_ids: tuple[str, ...] = ()
    complete_eligible_prefix: bool = False
    response_opportunity: bool = True

    @property
    def key(self) -> tuple[str, str, int]:
        return self.venture_id, self.investor_id, self.target_turn

    def verify(self, registry: TranscriptRegistry) -> None:
        raw = registry.get(self.venture_id, self.target_turn)
        if raw.role != "investor" or raw.speaker_id != self.investor_id:
            raise ValueError("target turn is not attributable to named investor")
        self.target_reply.verify(registry)
        if (self.target_reply.turn != self.target_turn
                or self.target_reply.speaker_id != self.investor_id
                or self.target_reply.start != 0
                or self.target_reply.end != len(raw.text)):
            raise ValueError("target reply must cover exactly the investor's public turn")
        if len(set(self.incoming_link_ids)) != len(self.incoming_link_ids):
            raise ValueError("duplicate incoming link")
        for code, field_review in self.fields.items():
            if code not in FIELD_IDS or field_review.field_id != code:
                raise ValueError("target field ID mismatch")
            if not field_review.analytic:
                raise ValueError("unreviewed target label cannot enter analysis")
            field_review.verify(registry)
            if any(span.venture_id != self.venture_id or span.turn > self.target_turn
                   for span in field_review.final_evidence):
                raise ValueError("target label uses future or other-venture evidence")
        nonlink = self.fields.get("E06")
        if nonlink is not None and nonlink.final_quality == "observed":
            if (nonlink.final_value != "verified_no_incoming_link"
                    or self.incoming_link_ids or not self.complete_eligible_prefix):
                raise ValueError("nonlink requires complete reviewed prefix and no links")



@dataclass(frozen=True)
class ReviewedFounderChoice:
    venture_id: str
    turn: int
    move: str
    referenced_offer_id: str | None
    review: FieldReview

    def verify(self, registry: TranscriptRegistry) -> None:
        raw = registry.get(self.venture_id, self.turn)
        if (raw.role != "founder" or self.move not in
                {"accept", "reject", "counter", "undecided"}
                or self.review.field_id != "N04" or not self.review.analytic):
            raise ValueError("founder choice is not an adjudicated founder turn")
        self.review.verify(registry)
        if not any(span.venture_id == self.venture_id and span.turn == self.turn
                   and span.speaker_id == raw.speaker_id
                   for span in self.review.final_evidence):
            raise ValueError("founder choice lacks own-turn evidence")
        if self.move in {"accept", "reject", "counter"} and not self.referenced_offer_id:
            raise ValueError("founder decision or counter needs a referenced offer")


@dataclass(frozen=True)
class ReviewedVentureOutcome:
    venture_id: str
    status: str
    accepted_investor: str | None
    accepted_terms: Mapping[str, Any] | None
    review: FieldReview
    founder_choice_turn: int | None = None

    def verify(self, registry: TranscriptRegistry) -> None:
        if (self.venture_id not in registry.versions or self.status not in TERMINAL
                or self.review.field_id != "V01" or not self.review.analytic):
            raise ValueError("invalid adjudicated venture outcome")
        self.review.verify(registry)
        if self.review.final_quality != "observed":
            raise ValueError("terminal category must be reviewed as observed")
        coded = self.review.final_value
        if coded != self.status:
            raise ValueError("terminal status differs from reviewed code")
        if self.status == "deal":
            if not self.accepted_investor or self.accepted_terms is None:
                raise ValueError("deal requires accepted investor and terms")
            if self.founder_choice_turn is None:
                raise ValueError("deal requires an explicit founder choice turn")
            founder = registry.get(self.venture_id, self.founder_choice_turn)
            if founder.role != "founder":
                raise ValueError("accepted offer was not chosen by founder")
        elif (self.accepted_investor is not None or self.accepted_terms is not None):
            raise ValueError("nondeal outcome cannot carry accepted terms")


@dataclass(frozen=True)
class AnnotationCorpus:
    registry: TranscriptRegistry
    targets: tuple[ReviewedTargetTurn, ...]
    links: tuple[DirectedPeerLink, ...]
    venture_split: Mapping[str, str]
    founder_choices: tuple[ReviewedFounderChoice, ...] = ()
    outcomes: tuple[ReviewedVentureOutcome, ...] = ()
    corpus_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.targets:
            raise ValueError("annotation corpus has no reviewed target turns")
        if any(split not in SPLITS for split in self.venture_split.values()):
            raise ValueError("unknown venture partition")
        if set(self.registry.versions) != set(self.venture_split):
            raise ValueError("partition must assign every venture exactly once")
        keys: set[tuple[str, str, int]] = set()
        for target in self.targets:
            target.verify(self.registry)
            if target.key in keys:
                raise ValueError("duplicate venture/investor/target-turn training unit")
            keys.add(target.key)
        by_id: dict[str, DirectedPeerLink] = {}
        for link in self.links:
            link.verify(self.registry)
            if link.link_id in by_id:
                raise ValueError("duplicate directed link ID")
            if (link.venture_id, link.target_investor, link.target_turn) not in keys:
                raise ValueError("peer link has no reviewed target observation")
            by_id[link.link_id] = link
        for target in self.targets:
            for link_id in target.incoming_link_ids:
                link = by_id.get(link_id)
                if link is None or (link.venture_id, link.target_investor,
                                    link.target_turn) != target.key:
                    raise ValueError("incoming link does not point to target")
            expected = {link.link_id for link in self.links
                        if (link.venture_id, link.target_investor,
                            link.target_turn) == target.key}
            if expected != set(target.incoming_link_ids):
                raise ValueError("target omits a reviewed incoming link")
        choice_keys: set[tuple[str, int]] = set()
        for choice in self.founder_choices:
            choice.verify(self.registry)
            key = (choice.venture_id, choice.turn)
            if key in choice_keys:
                raise ValueError("duplicate founder choice turn")
            choice_keys.add(key)
        outcome_ventures: set[str] = set()
        for outcome in self.outcomes:
            outcome.verify(self.registry)
            if outcome.venture_id in outcome_ventures:
                raise ValueError("duplicate venture terminal observation")
            outcome_ventures.add(outcome.venture_id)
            if outcome.status == "deal" and not any(
                    choice.venture_id == outcome.venture_id
                    and choice.turn == outcome.founder_choice_turn
                    and choice.move == "accept"
                    for choice in self.founder_choices):
                raise ValueError("deal is not linked to reviewed founder acceptance")
        identity = {
            "raw_turns": [asdict(row) for row in sorted(
                self.registry.turns.values(),
                key=lambda x: (x.venture_id, x.turn))],
            "targets": [asdict(row) for row in sorted(
                self.targets, key=lambda x: x.key)],
            "links": [asdict(row) for row in sorted(
                self.links, key=lambda x: x.link_id)],
            "founder_choices": [asdict(row) for row in sorted(
                self.founder_choices, key=lambda x: (x.venture_id, x.turn))],
            "outcomes": [asdict(row) for row in sorted(
                self.outcomes, key=lambda x: x.venture_id)],
            "venture_split": dict(sorted(self.venture_split.items())),
        }
        object.__setattr__(self, "corpus_sha256",
                           sha256(_canonical(identity).encode("utf-8")).hexdigest())

    def target_weights(self) -> dict[tuple[str, str, int], float]:
        """One unit of reply weight per unique target, regardless of links."""
        return {target.key: 1.0 for target in self.targets}

    def link_weights(self) -> dict[str, float]:
        """A target's incoming edges share a total relation weight of one."""
        return {link_id: 1.0 / len(target.incoming_link_ids)
                for target in self.targets for link_id in target.incoming_link_ids}

    def partition(self, split: str) -> tuple[ReviewedTargetTurn, ...]:
        if split not in SPLITS:
            raise ValueError("unknown partition")
        return tuple(target for target in self.targets
                     if self.venture_split[target.venture_id] == split)
