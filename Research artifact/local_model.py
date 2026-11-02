"""Offline investor-conditioned Qwen adapter and BGE retrieval encoder.

The two interfaces verify content fingerprints before loading local model
artifacts. The panel and knowledge snapshot supply their own version pins;
this module supplies the executing weights and tokenizer identities.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from threading import RLock
from typing import Mapping, Sequence

from knowledge import artifact_sha256
from local_backend import BackendContractError, GenerationRequest
from panel import (EXIT_PATTERN, WITHDRAW_PATTERN, own_sentences, parse_terms,
                   proposal_sentences)
from prompt_builder import build_text_messages
from protocol import ACTIONS


def _verified_digest(path: Path | str, expected: str, name: str) -> str:
    if (not isinstance(expected, str) or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)):
        raise ValueError(f"{name} has no SHA-256 pin")
    observed = artifact_sha256(path)
    if observed != expected:
        raise ValueError(f"{name} differs from its pinned artifact")
    return observed


_JOINT_WORD = re.compile(r"\b(?:joint(?:ly)?|together|with|joining|partner(?:ing)?|"
                         r"co[- ]invest(?:ing)?|collaborat(?:e|ing))\b", re.I)
_UNSUPPORTED = (
    ("royalty", re.compile(r"\broyalt(?:y|ies)\b", re.I),
     "public text proposes a royalty arrangement"),
    ("contingent", re.compile(r"\b(?:contingent|subject to|provided that|if you)\b", re.I),
     "public text proposes a conditional arrangement"),
    ("in_kind", re.compile(r"\bin[- ]kind\b", re.I),
     "public text proposes in-kind consideration"),
)
MAX_LOG_CASH_GAP = 2.0
MAX_EQUITY_FRACTION_GAP = 0.25
RETRY_SAMPLING = {"do_sample": True, "temperature": 0.7, "top_p": 0.9}


class QwenInvestorGenerator:
    """Run the shared investor-conditioned Qwen adapter and assemble one reply.

    The adapter is trained to complete the target investor's public text from a
    prompt naming that investor and carrying its earlier attributable speech.
    Its auxiliary action and term heads are read on the same preturn prompt.
    The generated text is never edited: its action and terms are read from the
    investor's own sentences and must agree with the heads, or the reply is
    rejected with a typed error.  The first attempt decodes greedily; a retry
    samples with the request seed.
    """

    stateless = True

    def __init__(self, *, backbone_dir: Path | str, tokenizer_dir: Path | str,
                 adapter_bundle: Path | str, investor_ids: Sequence[str],
                 expected_backbone_sha256: str, expected_tokenizer_sha256: str,
                 expected_adapter_sha256: str, device: str,
                 max_new_tokens: int = 384):
        self.backbone_dir = Path(backbone_dir).resolve(strict=True)
        self.tokenizer_dir = Path(tokenizer_dir).resolve(strict=True)
        self.adapter_bundle = Path(adapter_bundle).resolve(strict=True)
        if (not self.backbone_dir.is_dir() or not self.tokenizer_dir.is_dir()
                or not self.adapter_bundle.is_file() or len(investor_ids) < 2
                or len(set(investor_ids)) != len(investor_ids)
                or not isinstance(device, str) or not device
                or type(max_new_tokens) is not int or max_new_tokens < 1):
            raise ValueError("Qwen artifact, investor roster, or generation limit is invalid")
        self.investor_ids = tuple(investor_ids)
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._identity = {
            "backbone_sha256": _verified_digest(
                self.backbone_dir, expected_backbone_sha256, "Qwen backbone"),
            "tokenizer_sha256": _verified_digest(
                self.tokenizer_dir, expected_tokenizer_sha256, "Qwen tokenizer"),
            "adapter_sha256": _verified_digest(
                self.adapter_bundle, expected_adapter_sha256, "adapter bundle"),
        }
        self._lock = RLock()
        self._model = None
        self._tokenizer = None

    def artifact_identity(self, investor_id: str) -> Mapping[str, str]:
        if investor_id not in self.investor_ids:
            raise ValueError("investor is outside the adapter roster")
        return dict(self._identity)

    def _load(self) -> None:
        if self._model is not None:
            return
        for name, path in (("backbone_sha256", self.backbone_dir),
                           ("tokenizer_sha256", self.tokenizer_dir),
                           ("adapter_sha256", self.adapter_bundle)):
            _verified_digest(path, self._identity[name], name)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from training import load_adapter_bundle

        tokenizer = AutoTokenizer.from_pretrained(
            str(self.tokenizer_dir), local_files_only=True, trust_remote_code=False)
        backbone = AutoModelForCausalLM.from_pretrained(
            str(self.backbone_dir), local_files_only=True, trust_remote_code=False)
        backbone.eval().to(self.device)
        model = load_adapter_bundle(
            backbone, self.adapter_bundle,
            expected_backbone_sha256=self._identity["backbone_sha256"],
            expected_tokenizer_sha256=self._identity["tokenizer_sha256"])
        if tuple(model.investor_ids) != self.investor_ids:
            raise ValueError("adapter bundle investor registry differs from panel roster")
        model.eval().to(self.device)
        self._tokenizer = tokenizer
        self._model = model

    @staticmethod
    def _references(text: str, payload: Mapping[str, object]) -> tuple[list[str], list[str]]:
        evidence: list[str] = []
        peers: list[str] = []
        for entry in payload["peer_span_catalog"]:
            investor_mention = re.search(
                rf"(?<!\w){re.escape(entry['investor_id'])}(?!\w)", text) is not None
            if (investor_mention
                    or len(entry["quote"]) >= 12 and entry["quote"] in text):
                peers.append(entry["peer_span_id"])
                evidence.append(f"turn:{entry['peer_span_id'].split(':')[1]}")
        for entry in payload["evidence_catalog"]:
            quote = entry["quote"]
            if (entry["evidence_id"] not in evidence and len(quote) >= 12
                    and quote in text):
                evidence.append(entry["evidence_id"])
        return evidence, peers

    @staticmethod
    def _joint(text: str, target: str, roster: Sequence[str]) -> tuple[list[str], str | None]:
        for candidate, own in own_sentences(text):
            if _JOINT_WORD.search(own):
                partners = [name for name in roster if name != target and re.search(
                    rf"(?<!\w){re.escape(name)}(?!\w)", candidate) is not None]
                if partners and text.count(candidate) == 1:
                    return partners, candidate
        return [], None

    def _assemble(self, raw_text: str, action_label: str, term_prediction,
                  payload: Mapping[str, object], request: GenerationRequest) -> str:
        """Classify one generated turn from its own sentences without editing it."""
        text = raw_text.strip()
        if not text:
            raise BackendContractError("investor adapter produced no public text")
        diagnostic_terms = [float(value) for value in term_prediction[0].tolist()]
        if len(diagnostic_terms) != 2 or not all(
                math.isfinite(value) for value in diagnostic_terms):
            raise BackendContractError("investor adapter produced invalid term diagnostics")
        own_offer = payload["own_offer_id"]
        arrangement = unsupported_reason = None
        cash = equity = offer_evidence = None
        sentences = [own for _, own in own_sentences(text)]
        proposals = proposal_sentences(text)
        if proposals:
            action = "revise" if own_offer else "offer"
            unsupported = next(((kind, reason) for kind, pattern, reason in _UNSUPPORTED
                                if any(pattern.search(own) for _, own in proposals)), None)
            try:
                stated = {terms for _, own in proposals for terms in parse_terms(own)}
            except ValueError as exc:
                raise BackendContractError(str(exc)) from exc
            if unsupported is not None:
                arrangement, unsupported_reason = unsupported
            elif not stated:
                arrangement = "other"
                unsupported_reason = "proposal lacks explicit cash-for-equity terms"
            elif len(stated) > 1:
                raise BackendContractError("generated text states conflicting terms")
            else:
                arrangement = "cash_equity"
                cash, equity = stated.pop()
                offer_evidence = next(literal for literal, own in proposals
                                      if parse_terms(own))
        elif any(EXIT_PATTERN.search(own) for own in sentences):
            action = "exit"
        elif any(WITHDRAW_PATTERN.search(own) for own in sentences):
            action = "withdraw"
        else:
            action = "ask" if "?" in text else "comment"

        if arrangement == "cash_equity":
            expected_action = "counteroffer" if action == "revise" else "offer"
            if action_label != expected_action:
                raise BackendContractError("action head disagrees with the spoken offer")
            if (abs(diagnostic_terms[0] - math.log1p(cash)) > MAX_LOG_CASH_GAP
                    or abs(diagnostic_terms[1] - equity / 100)
                    > MAX_EQUITY_FRACTION_GAP):
                raise BackendContractError("term head disagrees with the spoken terms")
        elif arrangement is None:
            if action == "exit" and action_label != "exit":
                raise BackendContractError("action head disagrees with the spoken exit")
            if action in ("ask", "comment") and action_label != "none":
                raise BackendContractError("action head disagrees with nontransactional text")
        joint_scope, joint_evidence = self._joint(
            text, request.investor_id, payload["investor_order"])
        if action not in ("offer", "revise"):
            joint_scope, joint_evidence = [], None
        evidence_ids, peer_span_ids = self._references(text, payload)
        return json.dumps({
            "request_id": request.request_id,
            "session_id": request.session_id,
            "turn_id": request.turn_id,
            "investor_id": request.investor_id,
            "prefix_hash": request.prefix_hash,
            "visible_prior_turns": payload["visible_prior_turns"],
            "own_offer_id": payload["own_offer_id"],
            "action": action,
            "text": text,
            "arrangement_type": arrangement,
            "cash_usd": cash,
            "equity_pct": equity,
            "offer_evidence": offer_evidence,
            "joint_scope": joint_scope,
            "joint_evidence": joint_evidence,
            "evidence_ids": evidence_ids,
            "peer_span_ids": peer_span_ids,
            "parse_status": "unsupported_arrangement" if unsupported_reason else "ok",
            "unsupported_reason": unsupported_reason,
        }, ensure_ascii=False, sort_keys=True)

    def generate(self, request: GenerationRequest) -> str:
        if not isinstance(request, GenerationRequest):
            raise TypeError("generation requires a versioned request")
        if (request.investor_id not in self.investor_ids
                or any(request.model_identity.get(key) != value
                       for key, value in self._identity.items())):
            raise ValueError("model request differs from the executing adapter")
        payload = json.loads(request.messages[1]["content"])
        text_messages = build_text_messages(payload)
        with self._lock:
            self._load()
            import torch

            assert self._tokenizer is not None and self._model is not None
            prompt = self._tokenizer.apply_chat_template(
                list(text_messages), tokenize=False, add_generation_prompt=True)
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                heads = self._model.prompt_heads(
                    inputs["input_ids"], inputs["attention_mask"])
                decoding = {"do_sample": False}
                if payload.get("attempt", 0):
                    torch.manual_seed(request.seed)
                    decoding = dict(RETRY_SAMPLING)
                output = self._model.generate(
                    inputs["input_ids"], inputs["attention_mask"],
                    **decoding, max_new_tokens=self.max_new_tokens,
                    pad_token_id=self._tokenizer.eos_token_id)
            generated = output[0, inputs["input_ids"].shape[1]:]
            raw_text = self._tokenizer.decode(generated, skip_special_tokens=True)
            action_index = int(heads["action_logits"].argmax(dim=-1)[0])
            return self._assemble(raw_text, ACTIONS[action_index],
                                  heads["term_prediction"], payload, request)


class BgeLocalEncoder:
    """Normalized local BGE text embeddings bound to snapshot artifact hashes."""

    stateless = True

    def __init__(self, *, model_dir: Path | str,
                 config_artifact: Path | str, model_sha256: str,
                 config_sha256: str, device: str):
        self.model_dir = Path(model_dir).resolve(strict=True)
        if not self.model_dir.is_dir() or not isinstance(device, str) or not device:
            raise ValueError("BGE local model directory or device is invalid")
        self.embedding_model_sha256 = _verified_digest(
            self.model_dir, model_sha256, "BGE embedding model")
        config_path = Path(config_artifact).resolve(strict=True)
        self.config_path = config_path
        self.embedding_config_sha256 = _verified_digest(
            config_path, config_sha256, "BGE embedding configuration")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if (not isinstance(config, dict)
                or type(config.get("max_length")) is not int
                or not 1 <= config["max_length"] <= 8192
                or config.get("pooling") not in ("cls", "mean")
                or config.get("normalize") is not True
                or not isinstance(config.get("query_prefix", ""), str)):
            raise ValueError("BGE embedding configuration is incomplete")
        self.max_length = config["max_length"]
        self.pooling = config["pooling"]
        self.query_prefix = config.get("query_prefix", "")
        self.device = device
        self._model = None
        self._tokenizer = None
        self._lock = RLock()

    def _load(self) -> None:
        if self._model is not None:
            return
        _verified_digest(self.model_dir, self.embedding_model_sha256,
                         "BGE embedding model")
        _verified_digest(self.config_path, self.embedding_config_sha256,
                         "BGE embedding configuration")
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir), local_files_only=True, trust_remote_code=False)
        model = AutoModel.from_pretrained(
            str(self.model_dir), local_files_only=True, trust_remote_code=False)
        model.eval().to(self.device)
        self._tokenizer = tokenizer
        self._model = model

    def __call__(self, text: str) -> Sequence[float]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("BGE query text is empty")
        with self._lock:
            self._load()
            import torch

            assert self._tokenizer is not None and self._model is not None
            inputs = self._tokenizer(self.query_prefix + text,
                                     max_length=self.max_length, truncation=True,
                                     return_tensors="pt").to(self.device)
            with torch.inference_mode():
                hidden = self._model(**inputs).last_hidden_state
                if self.pooling == "cls":
                    pooled = hidden[:, 0]
                else:
                    mask = inputs["attention_mask"].unsqueeze(-1)
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                vector = pooled[0].float()
                vector = vector / torch.linalg.vector_norm(vector).clamp(min=1e-12)
            result = vector.detach().cpu().tolist()
            if not all(math.isfinite(component) for component in result):
                raise ValueError("BGE encoder produced a nonfinite vector")
            return result
