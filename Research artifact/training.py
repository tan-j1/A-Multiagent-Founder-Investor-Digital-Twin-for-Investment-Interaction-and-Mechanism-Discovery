"""Investor-conditioned Qwen adaptation on reviewed text-only target turns.

The Qwen backbone is supplied by the caller and frozen.  One shared low-rank
adapter is fitted across investors; the named target investor, the
chronological public prefix, and that investor's earlier attributable speech
condition it through the prompt.  No row of the historical corpus can enter
twice merely because several peers spoke before its target turn.  Prompt
tokens are excluded from the autoregressive objective.  Action, terms, stance,
and consistency have independent availability masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from annotation import AnnotationCorpus, FieldReview
from prompt_builder import build_text_messages
from protocol import ACTIONS, _audit_prompt_tree, score_confusion


ACTION_INDEX = {name: index for index, name in enumerate(ACTIONS)}
STANCE_DIRECTIONS = ("continuity", "refinement", "explicit_shift")
STANCE_INDEX = {name: index for index, name in enumerate(STANCE_DIRECTIONS)}
QWEN_LINEAR_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class TrainingObservation:
    venture_id: str
    investor_id: str
    target_turn: int
    public_prefix: Mapping[str, Any]
    observed_reply: str
    incoming_link_ids: tuple[str, ...]
    action: str | None = None
    cash_usd: int | None = None
    equity_pct: float | None = None
    stance_direction: str | None = None
    own_action_evidence: str | None = None
    owner_supported: bool = False
    stance_supported: bool = False
    reply_weight: float = 1.0

    @property
    def key(self) -> tuple[str, str, int]:
        return self.venture_id, self.investor_id, self.target_turn

    def __post_init__(self) -> None:
        if (not self.venture_id or not self.investor_id
                or type(self.target_turn) is not int or self.target_turn < 0
                or not self.observed_reply or self.reply_weight != 1.0):
            raise ValueError("one reviewed target reply must carry exactly one unit weight")
        if self.action is not None and self.action not in ACTION_INDEX:
            raise ValueError("unobserved historical action must be masked")
        if self.stance_direction is not None and self.stance_direction not in STANCE_INDEX:
            raise ValueError("unsupported or indeterminate stance must be masked")
        if (self.cash_usd is None) != (self.equity_pct is None):
            raise ValueError("term objective requires both observed cash and equity")
        if self.cash_usd is not None and (
                type(self.cash_usd) is not int or self.cash_usd <= 0
                or type(self.equity_pct) not in (int, float)
                or not 0 < float(self.equity_pct) <= 100):
            raise ValueError("invalid observed terms")
        if self.own_action_evidence is not None and (
                not self.own_action_evidence
                or self.own_action_evidence not in self.observed_reply):
            raise ValueError("own-action evidence is not in the target reply")
        if self.owner_supported and (
                self.action not in {"offer", "counteroffer"}
                or self.own_action_evidence is None):
            raise ValueError("offer ownership needs supported own action")
        if self.stance_supported and self.stance_direction is None:
            raise ValueError("stance consistency needs reviewed own before/after speech")
        _audit_prompt_tree(dict(self.public_prefix), self.target_turn)


def _observed(review: FieldReview | None) -> Any | None:
    return (review.final_value if review is not None
            and review.analytic and review.final_quality == "observed" else None)


def observations_from_corpus(
    corpus: AnnotationCorpus, *, split: str = "fit",
    max_prefix_turns: int = 30,
) -> tuple[TrainingObservation, ...]:
    """Project reviewed targets into preturn inputs and separate completions.

    The prefix is assembled from the raw registry, never from annotated target
    fields.  Only target-turn S04/N02/S03 labels can supervise the auxiliary
    objectives; unknown and indeterminate fields produce absent masks.
    """
    if type(max_prefix_turns) is not int or max_prefix_turns <= 0:
        raise ValueError("invalid context window")
    result: list[TrainingObservation] = []
    for target in corpus.partition(split):
        prefix = corpus.registry.prefix(target.venture_id, target.target_turn)
        if not prefix:
            raise ValueError("target has no earlier public conversation")
        own_prior = [row for row in prefix if row.speaker_id == target.investor_id]
        payload = {
            "venture_id": target.venture_id,
            "target_investor": target.investor_id,
            "target_turn_to_predict": target.target_turn,
            "recent_context": [
                {"turn": row.turn, "speaker_slot": row.speaker_id,
                 "role_hint": row.role, "text": row.text}
                for row in prefix[-max_prefix_turns:]
            ],
            "target_prior": [
                {"turn": row.turn, "speaker_slot": row.speaker_id,
                 "role_hint": row.role, "text": row.text}
                for row in own_prior
            ],
        }
        action = _observed(target.fields.get("S04"))
        if action not in ACTION_INDEX:
            action = None
        stance = _observed(target.fields.get("S03"))
        if stance not in STANCE_INDEX:
            stance = None
        terms = _observed(target.fields.get("N02"))
        cash, equity = None, None
        if isinstance(terms, Mapping) and terms.get("form") == "cash_for_equity":
            cash, equity = terms.get("cash_usd"), terms.get("equity_pct")
            if cash is not None and equity is not None:
                evidence = target.fields["N02"].final_evidence
                own_spans = [span for span in evidence
                             if span.turn == target.target_turn
                             and span.speaker_id == target.investor_id]
                if not own_spans:
                    raise ValueError("observed numeric terms lack target's own-turn spans")
            else:
                cash, equity = None, None
        action_review = target.fields.get("S04")
        action_evidence = None
        if action is not None and action != "none" and action_review is not None:
            own_spans = [span for span in action_review.final_evidence
                         if span.turn == target.target_turn
                         and span.speaker_id == target.investor_id]
            if not own_spans:
                raise ValueError("historical action lacks target's own public evidence")
            action_evidence = own_spans[0].quote
        owner_supported = False
        if action in {"offer", "counteroffer"}:
            owner_review = target.fields.get("N01")
            owner_value = _observed(owner_review)
            if owner_value is not None:
                owner = (owner_value.get("owner") if isinstance(owner_value, Mapping)
                         else owner_value)
                if owner != target.investor_id:
                    raise ValueError("reviewed offer owner differs from target speaker")
                owner_supported = any(
                    span.turn == target.target_turn
                    and span.speaker_id == target.investor_id
                    for span in owner_review.final_evidence)
                if not owner_supported:
                    raise ValueError("offer ownership lacks target's own-turn span")
        stance_supported = False
        if stance is not None:
            prior_review = target.fields.get("S01")
            after_review = target.fields.get("S03")
            prior = _observed(prior_review)
            if prior is not None and after_review is not None:
                before_spans = [span for span in prior_review.final_evidence
                                if span.turn < target.target_turn
                                and span.speaker_id == target.investor_id]
                after_spans = [span for span in after_review.final_evidence
                               if span.turn == target.target_turn
                               and span.speaker_id == target.investor_id]
                stance_supported = bool(before_spans and after_spans)
            if not stance_supported:
                raise ValueError("stance direction lacks own-speaker before/after spans")
        result.append(TrainingObservation(
            target.venture_id, target.investor_id, target.target_turn,
            payload, target.target_reply.quote, target.incoming_link_ids,
            action=action, cash_usd=cash, equity_pct=equity,
            stance_direction=stance, own_action_evidence=action_evidence,
            owner_supported=owner_supported, stance_supported=stance_supported,
        ))
    if len({row.key for row in result}) != len(result):
        raise ValueError("duplicate target reply after projection")
    return tuple(result)


def assert_venture_separation(
    corpus: AnnotationCorpus, *partitions: str,
) -> None:
    """Verify that no venture appears in more than one requested partition."""
    if not partitions:
        partitions = ("fit", "retrieval", "tuning", "final_evaluation")
    seen: set[str] = set()
    for partition in partitions:
        ventures = {target.venture_id for target in corpus.partition(partition)}
        if seen & ventures:
            raise ValueError("a venture appears in more than one requested partition")
        seen |= ventures


@dataclass(frozen=True)
class TokenBatch:
    input_ids: Tensor
    attention_mask: Tensor
    labels: Tensor
    investor_index: Tensor
    prompt_last_index: Tensor
    reply_weight: Tensor
    action_label: Tensor
    stance_label: Tensor
    term_target: Tensor
    term_mask: Tensor
    consistency_mask: Tensor
    owner_mask: Tensor
    stance_consistency_mask: Tensor

    def to(self, device: torch.device | str) -> TokenBatch:
        return TokenBatch(**{name: getattr(self, name).to(device)
                             for name in self.__dataclass_fields__})


def tokenize_observations(
    observations: Sequence[TrainingObservation], tokenizer: Any,
    investor_to_index: Mapping[str, int], *, max_tokens: int,
) -> TokenBatch:
    """Create Qwen chat inputs with all prompt positions masked as -100."""
    if not observations or type(max_tokens) is not int or max_tokens <= 1:
        raise ValueError("empty batch or invalid token budget")
    if len({row.key for row in observations}) != len(observations):
        raise ValueError("one target reply appears more than once in a batch")
    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        pad_id = eos_id
    if pad_id is None:
        raise ValueError("tokenizer needs a pad or EOS token")
    encoded: list[tuple[list[int], list[int], TrainingObservation]] = []
    for row in observations:
        if row.investor_id not in investor_to_index:
            raise ValueError("target investor is outside the registered roster")
        messages = build_text_messages(row.public_prefix)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True)
        reply = tokenizer.encode(row.observed_reply, add_special_tokens=False)
        if eos_id is not None:
            reply = [*reply, eos_id]
        if not prompt or not reply or len(prompt) + len(reply) > max_tokens:
            raise ValueError("target exceeds the registered context budget")
        encoded.append((list(prompt), list(reply), row))
    width = max(len(prompt) + len(reply) for prompt, reply, _ in encoded)
    arrays: dict[str, list[Any]] = {name: [] for name in (
        "input_ids", "attention_mask", "labels", "investor_index",
        "prompt_last_index", "reply_weight", "action_label", "stance_label",
        "term_target", "term_mask", "consistency_mask", "owner_mask",
        "stance_consistency_mask")}
    for prompt, reply, row in encoded:
        sequence = [*prompt, *reply]
        padding = width - len(sequence)
        arrays["input_ids"].append(sequence + [pad_id] * padding)
        arrays["attention_mask"].append([1] * len(sequence) + [0] * padding)
        arrays["labels"].append([-100] * len(prompt) + reply + [-100] * padding)
        arrays["investor_index"].append(investor_to_index[row.investor_id])
        arrays["prompt_last_index"].append(len(prompt) - 1)
        arrays["reply_weight"].append(row.reply_weight)
        arrays["action_label"].append(ACTION_INDEX.get(row.action, -100))
        arrays["stance_label"].append(STANCE_INDEX.get(row.stance_direction, -100))
        arrays["term_target"].append([
            math.log1p(row.cash_usd) if row.cash_usd is not None else 0.0,
            float(row.equity_pct) / 100 if row.equity_pct is not None else 0.0,
        ])
        arrays["term_mask"].append([
            row.cash_usd is not None, row.equity_pct is not None,
        ])
        arrays["consistency_mask"].append(
            row.action is not None and row.own_action_evidence is not None)
        arrays["owner_mask"].append(row.owner_supported)
        arrays["stance_consistency_mask"].append(row.stance_supported)
    return TokenBatch(
        input_ids=torch.tensor(arrays["input_ids"], dtype=torch.long),
        attention_mask=torch.tensor(arrays["attention_mask"], dtype=torch.long),
        labels=torch.tensor(arrays["labels"], dtype=torch.long),
        investor_index=torch.tensor(arrays["investor_index"], dtype=torch.long),
        prompt_last_index=torch.tensor(arrays["prompt_last_index"], dtype=torch.long),
        reply_weight=torch.tensor(arrays["reply_weight"], dtype=torch.float32),
        action_label=torch.tensor(arrays["action_label"], dtype=torch.long),
        stance_label=torch.tensor(arrays["stance_label"], dtype=torch.long),
        term_target=torch.tensor(arrays["term_target"], dtype=torch.float32),
        term_mask=torch.tensor(arrays["term_mask"], dtype=torch.bool),
        consistency_mask=torch.tensor(arrays["consistency_mask"], dtype=torch.bool),
        owner_mask=torch.tensor(arrays["owner_mask"], dtype=torch.bool),
        stance_consistency_mask=torch.tensor(
            arrays["stance_consistency_mask"], dtype=torch.bool),
    )


class SharedLoRALinear(nn.Module):
    """Frozen linear projection plus one rank-r update shared by all investors."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if (not isinstance(base, nn.Linear) or rank < 1
                or alpha <= 0 or not 0 <= dropout < 1):
            raise ValueError("invalid LoRA specification")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor) -> Tensor:
        update = F.linear(F.linear(self.dropout(values), self.a), self.b)
        return self.base(values) + update * self.scale


class InvestorConditionedQwen(nn.Module):
    """Frozen Qwen backbone with one shared low-rank adapter and auxiliary heads.

    Investor conditioning enters through the prompt: the named target investor,
    the chronological public prefix, and that investor's earlier attributable
    speech.  ``investor_ids`` registers the roster scored by the offer-owner
    head.  ``target_modules`` are exact module-path suffixes; the default
    adapts every attention and MLP projection.  The caller supplies a locally
    loaded Qwen causal LM; this class makes no network or model-hub call.
    """

    def __init__(
        self, backbone: nn.Module, investor_ids: Sequence[str], *,
        target_modules: Sequence[str] = QWEN_LINEAR_MODULES,
        rank: int = 8, alpha: float = 16.0, dropout: float = 0.05,
    ):
        super().__init__()
        if not investor_ids or len(set(investor_ids)) != len(investor_ids):
            raise ValueError("investor roster must be unique")
        self.investor_ids = tuple(investor_ids)
        self.investor_to_index = {name: i for i, name in enumerate(investor_ids)}
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        names = [name for name, module in self.backbone.named_modules()
                 if isinstance(module, nn.Linear)
                 and any(name == suffix or name.endswith("." + suffix)
                         for suffix in target_modules)]
        if not names:
            raise ValueError("no requested Qwen linear projections were found")
        for name in names:
            parent_name, _, child_name = name.rpartition(".")
            parent = self.backbone.get_submodule(parent_name) if parent_name else self.backbone
            base = getattr(parent, child_name)
            setattr(parent, child_name, SharedLoRALinear(base, rank, alpha, dropout))
        self.adapted_modules = tuple(names)
        hidden_size = getattr(getattr(backbone, "config", None), "hidden_size", None)
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ValueError("Qwen backbone config must declare hidden_size")
        self.action_head = nn.Linear(hidden_size, len(ACTIONS))
        self.speech_action_head = nn.Linear(hidden_size, len(ACTIONS))
        self.stance_head = nn.Linear(hidden_size, len(STANCE_DIRECTIONS))
        self.stance_speech_head = nn.Linear(hidden_size, len(STANCE_DIRECTIONS))
        self.owner_head = nn.Linear(hidden_size, len(investor_ids))
        self.term_head = nn.Linear(hidden_size, 2)

    def forward(self, batch: TokenBatch) -> dict[str, Tensor]:
        output = self.backbone(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = output.hidden_states[-1]
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        prompt_state = hidden[rows, batch.prompt_last_index]
        completion_mask = batch.labels.ne(-100).unsqueeze(-1)
        reply_state = ((hidden * completion_mask).sum(dim=1)
                       / completion_mask.sum(dim=1).clamp_min(1))
        return {
            "token_logits": output.logits,
            "action_logits": self.action_head(prompt_state),
            "speech_action_logits": self.speech_action_head(reply_state),
            "stance_logits": self.stance_head(prompt_state),
            "stance_speech_logits": self.stance_speech_head(reply_state),
            "owner_logits": self.owner_head(reply_state),
            "term_prediction": self.term_head(prompt_state),
        }

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (parameter for parameter in self.parameters()
                if parameter.requires_grad)

    @torch.no_grad()
    def generate(
        self, input_ids: Tensor, attention_mask: Tensor,
        **generation_options: Any,
    ) -> Tensor:
        """Decode public text from an investor-conditioned prompt."""
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("generation inputs must have matching batch dimensions")
        return self.backbone.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            **generation_options)

    @torch.no_grad()
    def prompt_heads(
        self, input_ids: Tensor, attention_mask: Tensor,
    ) -> dict[str, Tensor]:
        """Read trained action, term, and stance heads before text decoding."""
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("prompt inputs must be a batched prefix")
        output = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, return_dict=True)
        last = attention_mask.long().sum(dim=1) - 1
        if (last < 0).any():
            raise ValueError("empty prompt")
        state = output.hidden_states[-1][
            torch.arange(input_ids.shape[0], device=input_ids.device), last]
        return {
            "action_logits": self.action_head(state),
            "stance_logits": self.stance_head(state),
            "term_prediction": self.term_head(state),
        }


ADAPTER_FORMAT = "shared_investor_conditioned_lora_v1"


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_adapter_bundle(
    model: InvestorConditionedQwen, path: Path | str, *,
    backbone_sha256: str, tokenizer_sha256: str,
) -> str:
    """Persist exactly the fitted shared LoRA matrices and auxiliary heads.

    The shared frozen backbone is deliberately absent from this checkpoint.
    The returned file digest is the adapter identity pinned by local inference.
    """
    for name, digest in (("backbone", backbone_sha256),
                         ("tokenizer", tokenizer_sha256)):
        if (len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError(f"{name} has no lowercase SHA-256 identity")
    trainable = {name for name, parameter in model.named_parameters()
                 if parameter.requires_grad}
    state = {name: value.detach().cpu().contiguous()
             for name, value in model.state_dict().items() if name in trainable}
    if set(state) != trainable:
        raise ValueError("adapter state differs from trainable parameter registry")
    bundle = {
        "format": ADAPTER_FORMAT,
        "investor_ids": model.investor_ids,
        "adapted_modules": model.adapted_modules,
        "rank": model.rank, "alpha": model.alpha, "dropout": model.dropout,
        "backbone_sha256": backbone_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "state": state,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, destination)
    return _sha256_file(destination)


def load_adapter_bundle(
    backbone: nn.Module, path: Path | str, *,
    expected_backbone_sha256: str, expected_tokenizer_sha256: str,
) -> InvestorConditionedQwen:
    """Reconstruct the fitted shared adapter and roster on a verified backbone."""
    bundle = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (not isinstance(bundle, dict) or bundle.get("format") != ADAPTER_FORMAT
            or bundle.get("backbone_sha256") != expected_backbone_sha256
            or bundle.get("tokenizer_sha256") != expected_tokenizer_sha256):
        raise ValueError("adapter bundle is not bound to these model artifacts")
    model = InvestorConditionedQwen(
        backbone, tuple(bundle["investor_ids"]),
        target_modules=tuple(bundle["adapted_modules"]),
        rank=int(bundle["rank"]), alpha=float(bundle["alpha"]),
        dropout=float(bundle["dropout"]),
    )
    expected = {name for name, parameter in model.named_parameters()
                if parameter.requires_grad}
    state = bundle.get("state")
    if not isinstance(state, dict) or set(state) != expected:
        raise ValueError("adapter checkpoint parameter registry differs")
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name in expected:
            tensor = state[name]
            if not isinstance(tensor, Tensor) or tensor.shape != parameters[name].shape:
                raise ValueError(f"adapter parameter {name} has wrong shape")
            parameters[name].copy_(tensor)
    return model


@dataclass(frozen=True)
class LossWeights:
    reply: float = 1.0
    action: float = 1.0
    terms: float = 1.0
    stance: float = 1.0
    consistency: float = 1.0

    def __post_init__(self) -> None:
        if any(not math.isfinite(value) or value < 0
               for value in (self.reply, self.action, self.terms,
                             self.stance, self.consistency)):
            raise ValueError("loss weights must be finite and nonnegative")


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    denominator = mask.to(values.dtype).sum()
    if denominator.item() == 0:
        return values.sum() * 0
    return (values * mask.to(values.dtype)).sum() / denominator


def compute_loss(
    outputs: Mapping[str, Tensor], batch: TokenBatch,
    weights: LossWeights = LossWeights(),
) -> tuple[Tensor, dict[str, Tensor]]:
    """Masked multiobjective loss with per-reply token-length normalization."""
    logits = outputs["token_logits"][:, :-1]
    target = batch.labels[:, 1:]
    token_loss = F.cross_entropy(
        logits.transpose(1, 2), target, reduction="none", ignore_index=-100)
    token_mask = target.ne(-100)
    per_reply = (token_loss * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp_min(1)
    reply = (per_reply * batch.reply_weight).sum() / batch.reply_weight.sum()

    action_mask = batch.action_label.ne(-100)
    action = _masked_mean(F.cross_entropy(
        outputs["action_logits"], batch.action_label.clamp_min(0),
        reduction="none"), action_mask)
    stance_mask = batch.stance_label.ne(-100)
    stance = _masked_mean(F.cross_entropy(
        outputs["stance_logits"], batch.stance_label.clamp_min(0),
        reduction="none"), stance_mask)
    term_delta = F.smooth_l1_loss(
        outputs["term_prediction"], batch.term_target, reduction="none")
    terms = _masked_mean(term_delta, batch.term_mask)

    structured = F.log_softmax(outputs["action_logits"], dim=-1)
    spoken = F.log_softmax(outputs["speech_action_logits"], dim=-1)
    symmetric_kl = 0.5 * (
        F.kl_div(structured, spoken.exp(), reduction="none").sum(dim=-1)
        + F.kl_div(spoken, structured.exp(), reduction="none").sum(dim=-1))
    consistency = _masked_mean(symmetric_kl, batch.consistency_mask)
    owner = _masked_mean(F.cross_entropy(
        outputs["owner_logits"], batch.investor_index, reduction="none"),
        batch.owner_mask)
    prior_stance = F.log_softmax(outputs["stance_logits"], dim=-1)
    spoken_stance = F.log_softmax(outputs["stance_speech_logits"], dim=-1)
    stance_kl = 0.5 * (
        F.kl_div(prior_stance, spoken_stance.exp(), reduction="none").sum(dim=-1)
        + F.kl_div(spoken_stance, prior_stance.exp(), reduction="none").sum(dim=-1))
    stance_speech_ce = F.cross_entropy(
        outputs["stance_speech_logits"], batch.stance_label.clamp_min(0),
        reduction="none")
    consistency = (consistency + owner
                   + _masked_mean(stance_kl + stance_speech_ce,
                                  batch.stance_consistency_mask))
    components = {"reply": reply, "action": action, "terms": terms,
                  "stance": stance, "consistency": consistency}
    total = sum(getattr(weights, name) * value
                for name, value in components.items())
    return total, components


def fit_epoch(
    model: InvestorConditionedQwen, batches: Iterable[TokenBatch],
    optimizer: torch.optim.Optimizer, *,
    weights: LossWeights = LossWeights(), max_grad_norm: float = 1.0,
) -> dict[str, float]:
    """One adapter update pass; caller selects weights on held-out ventures."""
    model.train()
    totals = {name: 0.0 for name in ("total", "reply", "action", "terms",
                                    "stance", "consistency")}
    count = 0
    device = next(model.parameters()).device
    for original in batches:
        batch = original.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, components = compute_loss(model(batch), batch, weights)
        loss.backward()
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(tuple(model.trainable_parameters()), max_grad_norm)
        optimizer.step()
        totals["total"] += float(loss.detach())
        for name, value in components.items():
            totals[name] += float(value.detach())
        count += 1
    if count == 0:
        raise ValueError("no adapter training batches")
    return {name: value / count for name, value in totals.items()}


@torch.no_grad()
def five_action_diagnostic(
    model: InvestorConditionedQwen, batches: Iterable[TokenBatch],
) -> dict[str, Any]:
    """Historical five-label action score, separate from live-panel actions."""
    matrix = {gold: {pred: 0 for pred in (*ACTIONS, "invalid")}
              for gold in ACTIONS}
    model.eval()
    device = next(model.parameters()).device
    for original in batches:
        batch = original.to(device)
        prediction = model(batch)["action_logits"].argmax(dim=-1)
        for gold, pred in zip(batch.action_label.tolist(), prediction.tolist()):
            if gold >= 0:
                matrix[ACTIONS[gold]][ACTIONS[pred]] += 1
    return score_confusion(matrix)
