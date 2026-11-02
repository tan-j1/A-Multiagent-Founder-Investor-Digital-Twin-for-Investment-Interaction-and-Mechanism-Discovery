"""Layered validation for fixed-prefix components and complete text pitches.

The units are reviewed target turns, ventures, and seeded whole-pitch
rollouts.  Unknown labels, censored sessions, and unsupported transactions
remain distinct throughout scoring.  The functions consume recorded
predictions and observed outcomes.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import math
import random
from typing import Any, Mapping, Sequence

from annotation import RELATIONS, TERMINAL


PANEL_ACTIONS = (
    "offer", "revision", "withdrawal", "exit", "no_new_action",
)
TRACE_CATEGORIES = (
    "question", "objection", "endorsement", "peer_reply",
    "competing_offer", "revision", "exit", "founder_counter",
)
COMPARISON_ARMS = (
    "interacting_twins", "no_peer_public_turns", "generic_role_switching",
    "prompt_baseline", "frequency_baseline",
)


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


@dataclass(frozen=True)
class ClassifiedRecord:
    venture_id: str
    gold: str
    predicted: str
    probabilities: Mapping[str, float] | None = None
    gold_span_supported: bool | None = None
    predicted_span_supported: bool | None = None


def classification_metrics(
    records: Sequence[ClassifiedRecord], labels: Sequence[str], *,
    unknown: Sequence[str] = ("unknown", "indeterminate"),
) -> dict[str, Any]:
    """Classwise precision, recall, F1, confusion, and explicit abstention."""
    names = tuple(labels)
    if not names or len(set(names)) != len(names):
        raise ValueError("metric label registry must be unique")
    unknown_set = set(unknown)
    if set(names) & unknown_set:
        raise ValueError("unknown labels must not be folded into evaluated classes")
    matrix = {gold: {pred: 0 for pred in (*names, *unknown)} for gold in names}
    unknown_gold = Counter()
    for record in records:
        if record.gold in unknown_set:
            unknown_gold[record.gold] += 1
            continue
        if record.gold not in names or record.predicted not in (*names, *unknown):
            raise ValueError("classification record uses an unregistered label")
        matrix[record.gold][record.predicted] += 1
    per_class: dict[str, dict[str, Any]] = {}
    for name in names:
        true_positive = matrix[name][name]
        support = sum(matrix[name].values())
        predicted = sum(matrix[gold][name] for gold in names)
        precision = _ratio(true_positive, predicted)
        recall = _ratio(true_positive, support)
        f1 = (2 * true_positive / (support + predicted)
              if support + predicted else 0.0)
        per_class[name] = {
            "support": support, "predicted": predicted,
            "precision": precision, "recall": recall, "f1": f1,
        }
    n = sum(sum(row.values()) for row in matrix.values())
    correct = sum(matrix[name][name] for name in names)
    return {
        "n_evaluated": n, "n_unknown_gold": dict(unknown_gold),
        "n_predicted_unknown": sum(matrix[gold][label] for gold in names
                                   for label in unknown),
        "accuracy": _ratio(correct, n),
        "macro_f1": sum(item["f1"] for item in per_class.values()) / len(names),
        "per_class": per_class, "confusion": matrix,
    }


def calibration_metrics(
    records: Sequence[ClassifiedRecord], labels: Sequence[str], *,
    bins: int = 10,
) -> dict[str, Any]:
    """Multiclass Brier and expected calibration error on observed gold labels."""
    names = tuple(labels)
    if bins < 1 or not names:
        raise ValueError("invalid calibration bins or labels")
    eligible = [row for row in records if row.gold in names
                and row.probabilities is not None]
    bucket = [[0, 0.0, 0.0] for _ in range(bins)]
    brier = 0.0
    for row in eligible:
        probabilities = row.probabilities
        if (set(probabilities) != set(names)
                or any(not math.isfinite(float(p)) or not 0 <= float(p) <= 1
                       for p in probabilities.values())
                or abs(sum(probabilities.values()) - 1.0) > 1e-6):
            raise ValueError("probabilities must form a complete simplex")
        confidence = max(probabilities.values())
        label = max(names, key=lambda name: probabilities[name])
        index = min(bins - 1, int(confidence * bins))
        bucket[index][0] += 1
        bucket[index][1] += confidence
        bucket[index][2] += int(label == row.gold)
        brier += sum((probabilities[name] - int(name == row.gold)) ** 2
                     for name in names)
    n = len(eligible)
    ece = (sum((count / n) * abs(total_confidence / count - correct / count)
               for count, total_confidence, correct in bucket if count)
           if n else None)
    return {"n": n, "brier": brier / n if n else None,
            "expected_calibration_error": ece,
            "bins": [{"n": count,
                      "mean_confidence": total_confidence / count if count else None,
                      "accuracy": correct / count if count else None}
                     for count, total_confidence, correct in bucket]}


def venture_cluster_interval(
    records: Sequence[ClassifiedRecord], labels: Sequence[str], *,
    replicates: int = 1000, seed: int = 0,
) -> tuple[float, float] | None:
    """Venture-resampled percentile interval for component macro-F1."""
    by_venture: dict[str, list[ClassifiedRecord]] = defaultdict(list)
    for row in records:
        by_venture[row.venture_id].append(row)
    ventures = sorted(by_venture)
    if len(ventures) < 2 or replicates < 2:
        return None
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        sample = [record for venture in rng.choices(ventures, k=len(ventures))
                  for record in by_venture[venture]]
        draws.append(classification_metrics(sample, labels)["macro_f1"])
    draws.sort()
    lower = draws[int(0.025 * (len(draws) - 1))]
    upper = draws[int(0.975 * (len(draws) - 1))]
    return lower, upper


def peer_relation_metrics(
    records: Sequence[ClassifiedRecord], *,
    bootstrap_replicates: int = 1000, seed: int = 0,
) -> dict[str, Any]:
    labels = tuple(sorted(RELATIONS - {"uncertain"}))
    score = classification_metrics(records, labels, unknown=("uncertain", "unknown"))
    score["venture_cluster_macro_f1_interval"] = venture_cluster_interval(
        records, labels, replicates=bootstrap_replicates, seed=seed)
    score["gold_span_support_rate"] = _ratio(
        sum(row.gold_span_supported is True for row in records),
        sum(row.gold_span_supported is not None for row in records))
    score["predicted_span_support_rate"] = _ratio(
        sum(row.predicted_span_supported is True for row in records),
        sum(row.predicted_span_supported is not None for row in records))
    return score


def public_stance_metrics(records: Sequence[ClassifiedRecord]) -> dict[str, Any]:
    return classification_metrics(records,
                                  ("continuity", "refinement", "explicit_shift"),
                                  unknown=("unknown", "indeterminate"))


def panel_action_metrics(records: Sequence[ClassifiedRecord]) -> dict[str, Any]:
    score = classification_metrics(records, PANEL_ACTIONS)
    score["calibration"] = calibration_metrics(records, PANEL_ACTIONS)
    return score


@dataclass(frozen=True)
class EconomicsRecord:
    venture_id: str
    gold_owner: str | None
    predicted_owner: str | None
    gold_cash_usd: int | None
    predicted_cash_usd: int | None
    gold_equity_pct: float | None
    predicted_equity_pct: float | None
    gold_joint_scope: tuple[str, ...] | None
    predicted_joint_scope: tuple[str, ...] | None
    gold_revision_link: str | None
    predicted_revision_link: str | None
    gold_supported: bool
    predicted_text_json_agree: bool | None


def economics_metrics(records: Sequence[EconomicsRecord]) -> dict[str, Any]:
    """Exact ownership and term agreement only on comparable supported forms."""
    comparable = [row for row in records if row.gold_supported]
    fields = {
        "owner": lambda row: row.gold_owner == row.predicted_owner,
        "cash_usd": lambda row: row.gold_cash_usd == row.predicted_cash_usd,
        "equity_pct": lambda row: row.gold_equity_pct == row.predicted_equity_pct,
        "joint_scope": lambda row: row.gold_joint_scope == row.predicted_joint_scope,
        "revision_link": lambda row: row.gold_revision_link == row.predicted_revision_link,
    }
    supported_agreement = [row.predicted_text_json_agree for row in comparable
                           if row.predicted_text_json_agree is not None]
    return {
        "n_total": len(records), "n_comparable": len(comparable),
        "exact": {name: _ratio(sum(check(row) for row in comparable), len(comparable))
                  for name, check in fields.items()},
        "text_json_agreement": _ratio(sum(supported_agreement),
                                      len(supported_agreement)),
        "n_unsupported_gold": len(records) - len(comparable),
    }


@dataclass(frozen=True)
class GroundingRecord:
    venture_id: str
    blind_relevance: float | None
    blind_actor_consistency: float | None
    quote_supported: bool | None
    founder_fact_supported: bool | None


def grounding_metrics(records: Sequence[GroundingRecord]) -> dict[str, Any]:
    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None
    relevance = [row.blind_relevance for row in records
                 if row.blind_relevance is not None]
    actor = [row.blind_actor_consistency for row in records
             if row.blind_actor_consistency is not None]
    quote = [row.quote_supported for row in records
             if row.quote_supported is not None]
    fact = [row.founder_fact_supported for row in records
            if row.founder_fact_supported is not None]
    if any(not math.isfinite(value) for value in (*relevance, *actor)):
        raise ValueError("blind ratings must be finite")
    return {
        "n": len(records), "blind_relevance_mean": mean(relevance),
        "blind_actor_consistency_mean": mean(actor),
        "quote_support_rate": mean(quote),
        "founder_fact_support_rate": mean(fact),
    }


@dataclass(frozen=True)
class TraceEvent:
    turn: int
    actor_id: str
    category: str

    def __post_init__(self) -> None:
        if (type(self.turn) is not int or self.turn < 0 or not self.actor_id
                or self.category not in TRACE_CATEGORIES):
            raise ValueError("invalid annotated trace event")


@dataclass(frozen=True)
class RolloutRecord:
    venture_id: str
    seed: int
    arm: str
    model_hash: str
    tokenizer_hash: str
    prompt_hash: str
    index_hash: str
    schema_version: str
    token_budget: int
    status: str
    accepted_investor: str | None = None
    accepted_cash_usd: int | None = None
    accepted_equity_pct: float | None = None
    trace: tuple[TraceEvent, ...] = ()
    trace_fidelity: float | None = None
    outcome_fidelity: float | None = None

    def __post_init__(self) -> None:
        if (not self.venture_id or type(self.seed) is not int
                or self.arm not in COMPARISON_ARMS
                or self.status not in TERMINAL or self.token_budget < 1):
            raise ValueError("invalid seeded rollout identity")
        if any(not value for value in (
                self.model_hash, self.tokenizer_hash, self.prompt_hash,
                self.index_hash, self.schema_version)):
            raise ValueError("rollout must pin all policy and index versions")
        if self.status == "deal" and self.accepted_investor is None:
            raise ValueError("deal lacks accepted investor")
        if self.status == "deal" and (
                type(self.accepted_cash_usd) is not int
                or self.accepted_cash_usd <= 0
                or type(self.accepted_equity_pct) not in (int, float)
                or not 0 < float(self.accepted_equity_pct) <= 100):
            raise ValueError("deal lacks supported cash-for-equity terms")
        if self.status != "deal" and any(value is not None for value in (
                self.accepted_investor, self.accepted_cash_usd,
                self.accepted_equity_pct)):
            raise ValueError("nondeal has accepted terms")
        if list(event.turn for event in self.trace) != sorted(
                set(event.turn for event in self.trace)):
            raise ValueError("trace events are not strictly ordered")
        for value in (self.trace_fidelity, self.outcome_fidelity):
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError("fidelity score must be in [0,1]")


def outcome_distribution(records: Sequence[RolloutRecord]) -> dict[str, Any]:
    """All scheduled seeds remain in the four-category denominator."""
    if not records:
        raise ValueError("outcome distribution needs scheduled rollouts")
    if len({(row.venture_id, row.arm, row.seed) for row in records}) != len(records):
        raise ValueError("duplicate venture/arm/seed rollout")
    counts = Counter(row.status for row in records)
    n = len(records)
    probability = {status: counts[status] / n for status in
                   ("deal", "no_deal", "censored", "unsupported")}
    completed = counts["deal"] + counts["no_deal"]
    return {"n_scheduled": n, "counts": dict(counts),
            "probability": probability,
            "completed_case_deal_rate": _ratio(counts["deal"], completed),
            "censored_rate": probability["censored"],
            "unsupported_rate": probability["unsupported"]}


def trace_distribution(records: Sequence[RolloutRecord]) -> dict[str, Any]:
    """Per-venture event-count distributions and chronology of interaction."""
    by_venture: dict[str, list[RolloutRecord]] = defaultdict(list)
    for row in records:
        by_venture[row.venture_id].append(row)
    result: dict[str, Any] = {}
    for venture, rows in sorted(by_venture.items()):
        counts = {category: Counter() for category in TRACE_CATEGORIES}
        first_positions = {category: [] for category in TRACE_CATEGORIES}
        adjacent = Counter()
        for row in rows:
            categories = [event.category for event in row.trace]
            local = Counter(categories)
            for category in TRACE_CATEGORIES:
                counts[category][local[category]] += 1
                if category in categories:
                    first_positions[category].append(categories.index(category))
            adjacent.update(zip(categories, categories[1:]))
        result[venture] = {
            "n_rollouts": len(rows),
            "event_count_distribution": {
                category: dict(sorted(distribution.items()))
                for category, distribution in counts.items()},
            "mean_first_order_position": {
                category: _ratio(sum(positions), len(positions))
                for category, positions in first_positions.items()},
            "adjacent_order_counts": {
                f"{left}>{right}": value
                for (left, right), value in sorted(adjacent.items())},
        }
    return result


@dataclass(frozen=True)
class ObservedOutcome:
    venture_id: str
    status: str
    accepted_investor: str | None = None
    accepted_cash_usd: int | None = None
    accepted_equity_pct: float | None = None
    comparable_terms: bool = False
    trace: tuple[TraceEvent, ...] = ()


def _longest_common_subsequence(left: Sequence[str], right: Sequence[str]) -> int:
    previous = [0] * (len(right) + 1)
    for item in left:
        current = [0] * (len(right) + 1)
        for index, other in enumerate(right, 1):
            current[index] = (previous[index - 1] + 1 if item == other
                              else max(previous[index], current[index - 1]))
        previous = current
    return previous[-1]


def trace_fidelity(
    simulated: Sequence[TraceEvent], observed: Sequence[TraceEvent],
) -> dict[str, float]:
    """Jointly compare event frequencies and their chronological order."""
    left = [f"{event.actor_id}:{event.category}" for event in simulated]
    right = [f"{event.actor_id}:{event.category}" for event in observed]
    if not left and not right:
        return {"count_similarity": 1.0, "order_similarity": 1.0,
                "composite": 1.0}
    a, b = Counter(left), Counter(right)
    count_similarity = 1 - sum(abs(a[key] - b[key]) for key in a.keys() | b.keys()) / (
        sum(a.values()) + sum(b.values()))
    order_similarity = _longest_common_subsequence(left, right) / max(
        len(left), len(right))
    return {"count_similarity": count_similarity,
            "order_similarity": order_similarity,
            "composite": (count_similarity + order_similarity) / 2}


def score_rollout_against_observed(
    rollout: RolloutRecord, observed: ObservedOutcome,
) -> RolloutRecord:
    """Attach prespecified whole-pitch trace and terminal fidelity scores."""
    if rollout.venture_id != observed.venture_id:
        raise ValueError("rollout and observed pitch name different ventures")
    trace_score = trace_fidelity(rollout.trace, observed.trace)["composite"]
    status_score = float(rollout.status == observed.status)
    if observed.status == "deal" and rollout.status == "deal":
        components = [status_score,
                      float(rollout.accepted_investor == observed.accepted_investor)]
        if observed.comparable_terms:
            components.append(float(
                rollout.accepted_cash_usd == observed.accepted_cash_usd
                and rollout.accepted_equity_pct == observed.accepted_equity_pct))
        outcome_score = sum(components) / len(components)
    else:
        outcome_score = status_score
    return replace(rollout, trace_fidelity=trace_score,
                   outcome_fidelity=outcome_score)


def outcome_fidelity(
    records: Sequence[RolloutRecord], observed: Mapping[str, ObservedOutcome],
) -> dict[str, Any]:
    """Full-schedule terminal calibration and conditional deal fidelity."""
    groups: dict[str, list[RolloutRecord]] = defaultdict(list)
    for row in records:
        groups[row.venture_id].append(row)
    brier: list[float] = []
    no_deal_brier: list[float] = []
    terminal_brier: list[float] = []
    deal_gap: list[float] = []
    completed_brier: list[float] = []
    investor_exact = []
    term_exact = []
    for venture, rollouts in groups.items():
        gold = observed.get(venture)
        if gold is None or gold.status not in TERMINAL:
            raise ValueError("observed venture status is missing")
        probabilities = {status: sum(row.status == status for row in rollouts)
                         / len(rollouts) for status in TERMINAL}
        brier.append((probabilities["deal"] - int(gold.status == "deal")) ** 2)
        no_deal_brier.append((probabilities["no_deal"]
                              - int(gold.status == "no_deal")) ** 2)
        terminal_brier.append(sum(
            (probabilities[status] - int(gold.status == status)) ** 2
            for status in TERMINAL))
        deal_gap.append(abs(probabilities["deal"] - int(gold.status == "deal")))
        completed = [row for row in rollouts
                     if row.status in {"deal", "no_deal"}]
        if gold.status in {"deal", "no_deal"} and completed:
            conditional_deal = sum(row.status == "deal" for row in completed) / len(completed)
            completed_brier.append((conditional_deal - int(gold.status == "deal")) ** 2)
        if gold.status == "deal":
            predicted_deals = [row for row in rollouts if row.status == "deal"]
            for row in predicted_deals:
                investor_exact.append(row.accepted_investor == gold.accepted_investor)
                if gold.comparable_terms:
                    term_exact.append(
                        row.accepted_cash_usd == gold.accepted_cash_usd
                        and row.accepted_equity_pct == gold.accepted_equity_pct)
    return {
        "n_ventures": len(groups),
        "deal_brier": _ratio(sum(brier), len(brier)),
        "no_deal_brier": _ratio(sum(no_deal_brier), len(no_deal_brier)),
        "terminal_multiclass_brier": _ratio(sum(terminal_brier), len(terminal_brier)),
        "deal_absolute_calibration_gap": _ratio(sum(deal_gap), len(deal_gap)),
        "completed_case_deal_brier": _ratio(sum(completed_brier), len(completed_brier)),
        "accepted_investor_accuracy_given_deal": _ratio(
            sum(investor_exact), len(investor_exact)),
        "exact_cash_equity_given_comparable_deal": _ratio(
            sum(term_exact), len(term_exact)),
        "distribution": outcome_distribution(records),
    }


def matched_baseline_comparison(
    records: Sequence[RolloutRecord], *, reference_arm: str = "interacting_twins",
) -> dict[str, Any]:
    """Compare the same venture/seed under a frozen model and token budget."""
    if reference_arm not in COMPARISON_ARMS:
        raise ValueError("unknown reference arm")
    groups: dict[tuple[str, int], dict[str, RolloutRecord]] = defaultdict(dict)
    for row in records:
        key = (row.venture_id, row.seed)
        if row.arm in groups[key]:
            raise ValueError("duplicate matched arm")
        groups[key][row.arm] = row
    expected_arms = {row.arm for row in records}
    contrasts: dict[str, list[tuple[RolloutRecord, RolloutRecord]]] = defaultdict(list)
    for arms in groups.values():
        if set(arms) != expected_arms or reference_arm not in arms:
            raise ValueError("matched seed lacks a prespecified comparison arm")
        reference = arms[reference_arm]
        identity = (reference.model_hash, reference.tokenizer_hash,
                    reference.schema_version, reference.token_budget)
        for name, row in arms.items():
            # Prompt and retrieval hashes may differ by arm; the shared model
            # identity and token budget are the matched experimental controls.
            if (row.model_hash, row.tokenizer_hash,
                    row.schema_version, row.token_budget) != identity:
                raise ValueError("matched arms differ in model or token budget")
            if name != reference_arm:
                contrasts[name].append((reference, row))
    result: dict[str, Any] = {}
    for name, pairs in sorted(contrasts.items()):
        trace_pairs = [(left.trace_fidelity, right.trace_fidelity)
                       for left, right in pairs
                       if left.trace_fidelity is not None
                       and right.trace_fidelity is not None]
        outcome_pairs = [(left.outcome_fidelity, right.outcome_fidelity)
                         for left, right in pairs
                         if left.outcome_fidelity is not None
                         and right.outcome_fidelity is not None]
        result[name] = {
            "n_matched_seeds": len(pairs),
            "mean_trace_fidelity_difference": _ratio(
                sum(left - right for left, right in trace_pairs), len(trace_pairs)),
            "mean_outcome_fidelity_difference": _ratio(
                sum(left - right for left, right in outcome_pairs),
                len(outcome_pairs)),
            "reference_distribution": outcome_distribution(
                [left for left, _ in pairs]),
            "baseline_distribution": outcome_distribution(
                [right for _, right in pairs]),
        }
    return result


@dataclass(frozen=True)
class SystemCheck:
    parse_success: bool
    invalid_action_rejected: bool | None
    offer_ownership_valid: bool
    replay_hash_agrees: bool
    retrieval_eligible: bool
    branch_reproducible: bool | None


def system_validity(records: Sequence[SystemCheck]) -> dict[str, Any]:
    def rate(field: str) -> float | None:
        values = [getattr(row, field) for row in records
                  if getattr(row, field) is not None]
        return _ratio(sum(values), len(values))
    return {field + "_rate": rate(field) for field in (
        "parse_success", "invalid_action_rejected", "offer_ownership_valid",
        "replay_hash_agrees", "retrieval_eligible", "branch_reproducible")}


@dataclass(frozen=True)
class ProspectiveRecord:
    venture_id: str
    panel_id: str
    founder_strategy: str
    source: str  # "human" or "simulation"
    terminal_status: str
    trace_categories: tuple[str, ...]

    def __post_init__(self) -> None:
        if (not self.venture_id or not self.panel_id or not self.founder_strategy
                or self.source not in {"human", "simulation"}
                or self.terminal_status not in TERMINAL
                or any(category not in TRACE_CATEGORIES
                       for category in self.trace_categories)):
            raise ValueError("invalid prospective panel record")


def _total_variation(left: Counter[str], right: Counter[str]) -> float | None:
    if not left or not right:
        return None
    a, b = sum(left.values()), sum(right.values())
    return 0.5 * sum(abs(left[key] / a - right[key] / b)
                     for key in set(left) | set(right))


def prospective_correspondence(
    records: Sequence[ProspectiveRecord],
    *, bootstrap_replicates: int = 1000, seed: int = 0,
) -> dict[str, Any]:
    """Compare human and simulated distributions within truthful strategies.

    Panel and venture IDs are preserved in the returned clustering inventory;
    neither is treated as an independent event or turn.
    """
    by_strategy: dict[str, dict[str, list[ProspectiveRecord]]] = defaultdict(
        lambda: {"human": [], "simulation": []})
    for row in records:
        by_strategy[row.founder_strategy][row.source].append(row)
    result = {}
    rng = random.Random(seed)
    for strategy, sources in sorted(by_strategy.items()):
        human, simulation = sources["human"], sources["simulation"]
        if not human or not simulation:
            raise ValueError("prospective strategy needs both human and simulation panels")
        human_outcome = Counter(row.terminal_status for row in human)
        simulated_outcome = Counter(row.terminal_status for row in simulation)
        human_trace = Counter(category for row in human
                              for category in row.trace_categories)
        simulated_trace = Counter(category for row in simulation
                                  for category in row.trace_categories)
        venture_groups: dict[str, dict[str, dict[str, list[ProspectiveRecord]]]] = (
            defaultdict(lambda: {"human": defaultdict(list),
                                 "simulation": defaultdict(list)}))
        for row in (*human, *simulation):
            venture_groups[row.venture_id][row.source][row.panel_id].append(row)
        ventures = sorted(venture_groups)
        intervals: dict[str, tuple[float, float] | None] = {
            "outcome": None, "trace": None}
        if len(ventures) >= 2 and bootstrap_replicates >= 2:
            draws: dict[str, list[float]] = {"outcome": [], "trace": []}
            for _ in range(bootstrap_replicates):
                sampled = {"human": [], "simulation": []}
                for venture in rng.choices(ventures, k=len(ventures)):
                    for source in ("human", "simulation"):
                        panels = venture_groups[venture][source]
                        panel_ids = sorted(panels)
                        if panel_ids:
                            for panel_id in rng.choices(panel_ids, k=len(panel_ids)):
                                sampled[source].extend(panels[panel_id])
                if not sampled["human"] or not sampled["simulation"]:
                    continue
                a, b = sampled["human"], sampled["simulation"]
                distance_outcome = _total_variation(
                    Counter(row.terminal_status for row in a),
                    Counter(row.terminal_status for row in b))
                distance_trace = _total_variation(
                    Counter(category for row in a
                            for category in row.trace_categories),
                    Counter(category for row in b
                            for category in row.trace_categories))
                if distance_outcome is not None:
                    draws["outcome"].append(distance_outcome)
                if distance_trace is not None:
                    draws["trace"].append(distance_trace)
            for key, values in draws.items():
                if values:
                    values.sort()
                    intervals[key] = (
                        values[int(0.025 * (len(values) - 1))],
                        values[int(0.975 * (len(values) - 1))])
        result[strategy] = {
            "outcome_total_variation": _total_variation(
                human_outcome, simulated_outcome),
            "trace_total_variation": _total_variation(
                human_trace, simulated_trace),
            "venture_panel_cluster_interval": intervals,
            "n_human_panels": len({(row.venture_id, row.panel_id)
                                   for row in human}),
            "n_simulated_panels": len({(row.venture_id, row.panel_id)
                                       for row in simulation}),
            "n_ventures": len({row.venture_id for row in (*human, *simulation)}),
        }
    return result
