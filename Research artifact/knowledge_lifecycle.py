"""Adjudicated ingest, partition audit, compressed BGE build, and publication.

The historical index is rebuilt in an immutable directory. Publishing changes
only a small active-snapshot pointer; existing sessions retain their loaded
snapshot and never write generated turns into historical retrieval memory.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence
from uuid import uuid4

from annotation import AnnotationCorpus, DirectedPeerLink, EvidenceSpan, FieldReview
from knowledge import (
    ARTIFACTS, CHANNELS, DATA_FILES, MEMORY_KEYS, VERSION, VECTOR_FORMAT,
    BoundEncoder, KnowledgeSnapshot, artifact_sha256, _audit_records, _canonical_bytes,
    _digest, _feature_texts, _ordered_id_digest, _query_vector,
    _sha256, _validate_adjudications, _validate_partitions,
    _validate_raw_registry, _validate_train, _write_npz, load_snapshot,
)


ACTIVE_POINTER = "active_snapshot.json"


def _span_record(span: EvidenceSpan) -> dict[str, Any]:
    return {
        "case_id": span.venture_id, "turn": span.turn,
        "speaker": span.speaker_id, "start": span.start,
        "end": span.end, "quote": span.quote,
        "transcript_version": span.transcript_version,
        "source_sha256": span.source_sha256,
    }


def _source_span(link: DirectedPeerLink) -> EvidenceSpan:
    for span in link.fields["E01"].final_evidence:
        if span.turn == link.source_turn and span.speaker_id == link.source_investor:
            return span
    raise ValueError(f"link {link.link_id} has no reviewed source span")


def _primary_link(links: Sequence[DirectedPeerLink]) -> DirectedPeerLink:
    priority = {
        "named_quote": 4, "named_paraphrase": 3,
        "topical_continuation": 2, "no_discernible_uptake": 1,
        "unknown": 0,
    }
    return sorted(links, key=lambda link: (
        -priority.get(link.fields["E03"].final_value, 0),
        -link.source_turn, link.link_id,
    ))[0]


def _review_metadata(fields: Mapping[str, FieldReview]) -> tuple[dict[str, str],
                                                               dict[str, Any],
                                                               dict[str, str]]:
    statuses = {code: review.final_quality for code, review in fields.items()}
    disagreements = {
        code: {"disagreement": review.disagreement,
               "span_agreement": review.span_agreement,
               "numeric_term_difference": review.numeric_term_difference}
        for code, review in fields.items()
    }
    adjudications = {code: review.adjudication for code, review in fields.items()
                     if review.adjudication is not None}
    return statuses, disagreements, adjudications


def corpus_snapshot_records(
    corpus: AnnotationCorpus, *, rights_permitted_cases: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]],
           list[dict[str, Any]], dict[str, Any]]:
    """Project human-reviewed target units into the retrieval release schema."""
    permitted = set(rights_permitted_cases)
    if len(permitted) != len(rights_permitted_cases):
        raise ValueError("rights-permitted venture IDs are duplicated")
    link_by_id = {link.link_id: link for link in corpus.links}
    prepared: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    retained_cases: set[str] = set()
    for target in corpus.partition("retrieval"):
        if not target.incoming_link_ids:
            continue
        if target.venture_id not in permitted:
            raise ValueError(f"retrieval venture {target.venture_id} lacks rights permission")
        links = [link_by_id[identity] for identity in target.incoming_link_ids]
        eligible = [link for link in links
                    if link.fields["E02"].final_quality == "observed"
                    and link.fields["E03"].final_quality == "observed"]
        if not eligible:
            continue
        primary = _primary_link(eligible)
        source_span = _source_span(primary)
        prefix = corpus.registry.prefix(target.venture_id, target.target_turn)
        context = [{"turn": turn.turn, "speaker": turn.speaker_id, "text": turn.text}
                   for turn in prefix[-4:]]
        own_prior = [{"turn": turn.turn, "speaker": turn.speaker_id, "text": turn.text}
                     for turn in prefix if turn.speaker_id == target.investor_id][-2:]
        episode, transcript_version, source_sha = corpus.registry.versions[target.venture_id]
        identity = f"{target.venture_id}|{target.investor_id}|{target.target_turn}"
        prepared.append({
            "id": identity, "case_id": target.venture_id, "split": "train",
            "target_investor": target.investor_id,
            "source_investor": primary.source_investor,
            "source_quote": source_span.quote,
            "recent_context": context, "target_prior": own_prior,
            "source_turn": primary.source_turn, "target_turn": target.target_turn,
            "target_quote": target.target_reply.quote,
            "label_relation": primary.fields["E02"].final_value,
            "label_effect": primary.fields["E03"].final_value,
            "episode_id": episode, "transcript_version": transcript_version,
            "source_sha256": source_sha,
            "source_link_ids": [link.link_id for link in links],
            "annotation_corpus_sha256": corpus.corpus_sha256,
        })
        field_reviews = dict(target.fields)
        field_reviews.update(primary.fields)
        statuses, disagreements, adjudication = _review_metadata(field_reviews)
        statuses.update(speaker="observed", source_span="observed",
                        target_span="observed", relation="observed", effect="observed")
        coder_ids = sorted({judgment.coder_id for review in field_reviews.values()
                            for judgment in review.judgments})
        reviewer_ids = sorted({review.reviewer_id for review in field_reviews.values()})
        reviews.append({
            "id": identity, "case_id": target.venture_id,
            "target_investor": target.investor_id,
            "target_turn": target.target_turn,
            "transcript_version": transcript_version,
            "review_decision": "accepted",
            "coder_ids": coder_ids,
            "reviewer_id": primary.fields["E02"].reviewer_id,
            "reviewer_ids": reviewer_ids,
            "field_status": statuses,
            "disagreements": disagreements,
            "adjudication": adjudication,
            "relation": primary.fields["E02"].final_value,
            "effect": primary.fields["E03"].final_value,
            "source_span": _span_record(source_span),
            "target_span": _span_record(target.target_reply),
            "source_links": [_span_record(_source_span(link)) for link in links],
            "source_link_ids": [link.link_id for link in links],
        })
        retained_cases.add(target.venture_id)
    if not prepared:
        raise ValueError("no reviewed retrieval observations passed the source-link gate")
    registry: list[dict[str, Any]] = []
    for case in sorted(retained_cases):
        episode, transcript_version, source_sha = corpus.registry.versions[case]
        turns = [
            {"turn": row.turn, "speaker": row.speaker_id, "text": row.text}
            for (venture, _), row in sorted(corpus.registry.turns.items())
            if venture == case
        ]
        registry.append({
            "case_id": case, "episode_id": episode,
            "transcript_version": transcript_version,
            "source_sha256": source_sha,
            "transcript_sha256": hashlib.sha256(_canonical_bytes(turns)).hexdigest(),
            "rights_permitted": True, "turns": turns,
        })
    partitions = {
        "version": "venture_partition_v1", "locked": True,
        "corpus_sha256": corpus.corpus_sha256,
        "partitions": {name: sorted(case for case, group in corpus.venture_split.items()
                                    if group == name)
                       for name in ("fit", "retrieval", "tuning", "final_evaluation")},
    }
    return prepared, reviews, registry, partitions


def build_snapshot_from_corpus(
    directory: Path | str, *, snapshot_id: str, corpus: AnnotationCorpus,
    rights_permitted_cases: Sequence[str],
    raw_source_files: Mapping[str, Path | str],
    artifacts: Mapping[str, Path | str], encoder: BoundEncoder,
) -> KnowledgeSnapshot:
    """Build a release directly from the human-reviewed interaction corpus."""
    prepared, reviews, registry, partitions = corpus_snapshot_records(
        corpus, rights_permitted_cases=rights_permitted_cases)
    return build_snapshot(
        directory, snapshot_id=snapshot_id, prepared_rows=prepared,
        adjudications=reviews, raw_registry=registry,
        raw_source_files=raw_source_files, split_manifest=partitions,
        artifacts=artifacts, encoder=encoder,
    )


def _plain(value: Any) -> Any:
    """Detach caller data and reject non-JSON numbers before sealing a release."""
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_canonical_bytes(value) + b"\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("wb") as stream:
        for row in rows:
            stream.write(_canonical_bytes(row) + b"\n")


def _verify_source_files(raw_registry: Sequence[dict[str, Any]],
                         raw_source_files: Mapping[str, Path | str]) -> None:
    cases = {row["case_id"] for row in raw_registry}
    if set(raw_source_files) != cases:
        raise ValueError("source files must correspond exactly to the raw-text registry")
    for row in raw_registry:
        path = Path(raw_source_files[row["case_id"]]).resolve(strict=True)
        if not path.is_file() or _sha256(path) != row["source_sha256"]:
            raise ValueError(f"raw source hash differs for venture {row['case_id']}")


def _encode_channels(rows: Sequence[Mapping[str, Any]],
                     encoder: BoundEncoder) -> tuple[dict[str, list[tuple[float, ...]]],
                                                     dict[str, int]]:
    vectors: dict[str, list[tuple[float, ...]]] = {name: [] for name in CHANNELS}
    dimensions: dict[str, int] = {}
    for row_index, row in enumerate(rows):
        texts = _feature_texts(row)
        for channel in CHANNELS:
            raw = list(encoder(texts[channel]))
            if not raw:
                raise ValueError(f"BGE returned an empty {channel} embedding")
            dimension = len(raw)
            if channel in dimensions and dimension != dimensions[channel]:
                raise ValueError(f"BGE {channel} dimension changed at row {row_index}")
            dimensions[channel] = dimension
            vectors[channel].append(_query_vector(lambda _text, values=raw: values,
                                                   texts[channel], dimension,
                                                   f"{channel}[{row_index}]"))
    if len(set(dimensions.values())) != 1:
        raise ValueError("all BGE channels must use one embedding dimension")
    return vectors, dimensions


def build_snapshot(
    directory: Path | str, *, snapshot_id: str,
    prepared_rows: Sequence[Mapping[str, Any]],
    adjudications: Sequence[Mapping[str, Any]],
    raw_registry: Sequence[Mapping[str, Any]],
    raw_source_files: Mapping[str, Path | str],
    split_manifest: Mapping[str, Any],
    artifacts: Mapping[str, Path | str],
    encoder: BoundEncoder,
) -> KnowledgeSnapshot:
    """Build one reviewed retrieval release and verify it before atomic placement.

    Each prepared row is one unique venture/target-investor/target-turn reply.
    Its reviewed source and target spans are checked against the registered raw
    transcript, while the four venture partitions are locked independently.
    """
    if not isinstance(snapshot_id, str) or not snapshot_id.strip():
        raise ValueError("snapshot_id is required")
    if set(artifacts) != set(ARTIFACTS):
        raise ValueError("BGE model, BGE config, and prompt builder artifacts are required")
    artifact_hashes = {}
    for name in ARTIFACTS:
        path = Path(artifacts[name]).resolve(strict=True)
        artifact_hashes[name] = artifact_sha256(path)
    for name in ("embedding_model", "embedding_config"):
        if getattr(encoder, f"{name}_sha256", None) != artifact_hashes[name]:
            raise ValueError(f"BGE {name} differs from indexed artifact")

    source = sorted((_plain(row) for row in prepared_rows),
                    key=lambda row: (row["case_id"], row["target_investor"],
                                     row["target_turn"], row["id"]))
    memory = [{key: row[key] for key in MEMORY_KEYS} for row in source]
    reviews = sorted((_plain(row) for row in adjudications), key=lambda row: row["id"])
    registry = sorted((_plain(row) for row in raw_registry), key=lambda row: row["case_id"])
    partitions = _plain(split_manifest)

    _validate_train(source)
    _validate_train(memory)
    _validate_partitions(partitions, memory)
    raw_by_case = _validate_raw_registry(registry, memory)
    _verify_source_files(registry, raw_source_files)
    _validate_adjudications(reviews, source, raw_by_case)
    vectors, dimensions = _encode_channels(memory, encoder)
    report = _audit_records(source, memory, reviews, registry, partitions, vectors)

    target = Path(directory).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"snapshot already exists: {target}")
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=target.parent))
    try:
        filenames = {
            "source_train": "source_train.jsonl",
            "memory": "memory.jsonl",
            "vectors": "vectors.npz",
            "adjudications": "adjudications.jsonl",
            "raw_registry": "raw_registry.jsonl",
            "split_manifest": "split_manifest.json",
            "audit_report": "audit_report.json",
        }
        _write_jsonl(stage / filenames["source_train"], source)
        _write_jsonl(stage / filenames["memory"], memory)
        _write_npz(stage / filenames["vectors"], vectors, dimensions)
        _write_jsonl(stage / filenames["adjudications"], reviews)
        _write_jsonl(stage / filenames["raw_registry"], registry)
        _write_json(stage / filenames["split_manifest"], partitions)
        _write_json(stage / filenames["audit_report"], report)
        manifest = {
            "version": VERSION,
            "snapshot_id": snapshot_id,
            "construction_split": "train",
            "vector_format": VECTOR_FORMAT,
            "row_count": len(memory),
            "case_count": len({row["case_id"] for row in memory}),
            "ordered_row_ids_sha256": _ordered_id_digest(memory),
            "embedding_dimensions": dimensions,
            "artifact_sha256": artifact_hashes,
            "files": {name: {"path": filenames[name],
                             "sha256": _sha256(stage / filenames[name])}
                      for name in DATA_FILES},
        }
        _write_json(stage / "manifest.json", manifest)
        load_snapshot(stage, artifacts=artifacts)
        if not stage.resolve().is_relative_to(target.parent.resolve()):
            raise ValueError("staged snapshot escaped its output directory")
        os.replace(stage, target)
    except BaseException:
        if stage.exists() and stage.resolve().is_relative_to(target.parent.resolve()):
            shutil.rmtree(stage)
        raise
    return load_snapshot(target, artifacts=artifacts)


def publish_snapshot(snapshot_directory: Path | str,
                     registry_directory: Path | str, *,
                     artifacts: Mapping[str, Path | str]) -> dict[str, Any]:
    """Atomically select a verified release for sessions created afterward."""
    snapshot = load_snapshot(snapshot_directory, artifacts=artifacts)
    registry = Path(registry_directory).resolve()
    registry.mkdir(parents=True, exist_ok=True)
    pointer = {
        "version": "active_knowledge_snapshot_v1",
        "snapshot_id": snapshot.manifest["snapshot_id"],
        "snapshot_directory": str(Path(snapshot_directory).resolve(strict=True)),
        "manifest_sha256": snapshot.manifest_sha256,
        "index_sha256": snapshot.manifest["files"]["vectors"]["sha256"],
        "embedding_model_sha256": snapshot.manifest["artifact_sha256"]["embedding_model"],
        "prompt_builder_sha256": snapshot.manifest["artifact_sha256"]["prompt_builder"],
    }
    temporary = registry / f".{ACTIVE_POINTER}.{uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(_canonical_bytes(pointer) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, registry / ACTIVE_POINTER)
    finally:
        if temporary.exists():
            temporary.unlink()
    return pointer


def load_active_snapshot(registry_directory: Path | str, *,
                         artifacts: Mapping[str, Path | str]) -> KnowledgeSnapshot:
    """Resolve the atomic pointer once when a new pitch session starts."""
    path = Path(registry_directory).resolve(strict=True) / ACTIVE_POINTER
    pointer = json.loads(path.read_text(encoding="utf-8"))
    if pointer.get("version") != "active_knowledge_snapshot_v1":
        raise ValueError("unsupported active snapshot pointer")
    snapshot = load_snapshot(pointer["snapshot_directory"], artifacts=artifacts)
    if (snapshot.manifest_sha256 != _digest(pointer.get("manifest_sha256"), "manifest_sha256")
            or snapshot.manifest["snapshot_id"] != pointer.get("snapshot_id")
            or snapshot.manifest["files"]["vectors"]["sha256"] != pointer.get("index_sha256")
            or snapshot.manifest["artifact_sha256"]["embedding_model"] != pointer.get("embedding_model_sha256")
            or snapshot.manifest["artifact_sha256"]["prompt_builder"] != pointer.get("prompt_builder_sha256")):
        raise ValueError("active pointer differs from its immutable snapshot")
    return snapshot
