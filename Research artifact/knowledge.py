"""Immutable, reviewed text memory for multi-investor pitch interaction.

Training-side cases are retained as local JSONL. Three aligned matrices of
normalized BGE embeddings are stored as compressed NumPy-compatible arrays.
The snapshot manifest binds records, partitions, adjudications, arrays, encoder,
and prompt artifacts. A live pitch reads one pinned snapshot throughout its run.
"""

from __future__ import annotations

from dataclasses import dataclass
import ast
import hashlib
import io
import json
import math
from pathlib import Path
import struct
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence
import zipfile


VERSION = "train_only_three_channel_v2"
VECTOR_FORMAT = "npz_float32_c_order_v1"
CHANNELS = ("source", "context", "prior")
WEIGHTS = {"source": 0.60, "context": 0.30, "prior": 0.10}
ARTIFACTS = ("embedding_model", "embedding_config", "prompt_builder")
DATA_FILES = (
    "source_train", "memory", "vectors", "adjudications", "raw_registry",
    "split_manifest", "audit_report",
)
PARTITIONS = ("fit", "retrieval", "tuning", "final_evaluation")
FIELD_STATES = ("observed", "unknown", "not_applicable", "indeterminate")
MEMORY_KEYS = (
    "id", "case_id", "split", "target_investor", "source_investor",
    "source_quote", "recent_context", "target_prior", "source_turn",
    "target_turn", "target_quote", "label_relation", "label_effect",
)
UNKNOWN_INVESTOR = "Unknown investor"


@dataclass(frozen=True)
class BoundEncoder:
    """Local BGE encoder callable pinned to the indexed model and configuration."""

    encode: Callable[[str], Sequence[float]]
    embedding_model_sha256: str
    embedding_config_sha256: str

    def __call__(self, text: str) -> Sequence[float]:
        return self.encode(text)


def _freeze(value: Any) -> Any:
    """Copy JSON-like data into an immutable, recursively sealed structure."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    """Return a detached JSON-like copy of one retrieved example."""
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class KnowledgeSnapshot:
    """A verified, read-only-in-use version of historical training cases."""

    manifest_sha256: str
    manifest: Mapping[str, Any]
    memory: tuple[Mapping[str, Any], ...]
    vectors: Mapping[str, tuple[tuple[float, ...], ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest", _freeze(self.manifest))
        object.__setattr__(self, "memory", _freeze(self.memory))
        object.__setattr__(self, "vectors", _freeze(self.vectors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_sha256(path: Path | str) -> str:
    """Hash a single artifact or an entire local model directory deterministically."""
    root = Path(path).resolve(strict=True)
    if root.is_file():
        return _sha256(root)
    if not root.is_dir():
        raise ValueError("artifact is neither a file nor a model directory")
    members = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
    if not members:
        raise ValueError("model directory is empty")
    files = [member for member in members if member.is_file()]
    if not files or any((member.is_symlink() and not member.is_file())
                        or (not member.is_file() and not member.is_dir())
                        for member in members):
        raise ValueError("model directory contains no files or unsupported entries")
    digest = hashlib.sha256(b"local-model-tree-v1\0")
    for member in files:
        relative = member.relative_to(root).as_posix().encode("utf-8")
        digest.update(struct.pack("<I", len(relative)))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256(member)))
    return digest.hexdigest()


def _digest(value: Any, field: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _member(directory: Path, spec: Any, field: str) -> Path:
    entry = _object(spec, field)
    name = entry.get("path")
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError(f"{field}.path must be relative")
    root = directory.resolve(strict=True)
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"{field}.path escapes or is not a file in snapshot")
    if _sha256(path) != _digest(entry.get("sha256"), f"{field}.sha256"):
        raise ValueError(f"{field} hash mismatch")
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(_object(json.loads(line), f"{path.name}:{number}"))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path.name}:{number}") from exc
    if not rows:
        raise ValueError(f"empty JSONL: {path.name}")
    return rows


def visible_prefix(row: Mapping[str, Any]) -> dict[str, Any]:
    """Copy only current-case fields available strictly before the target turn.

    Target replies, labels, later turns, and other unlisted keys are deliberately
    absent from the returned projection, including when present on an evaluation
    row. The case ID is retained solely for cross-venture exclusion.
    """
    if not isinstance(row, Mapping):
        raise ValueError("query must be a mapping")
    text_fields = ("case_id", "target_investor", "source_investor", "source_quote")
    for key in text_fields:
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"missing or blank {key}")
    source_turn, target_turn = row.get("source_turn"), row.get("target_turn")
    if (type(source_turn) is not int or type(target_turn) is not int
            or source_turn >= target_turn):
        raise ValueError("source turn must precede target turn")

    result: dict[str, Any] = {key: row[key] for key in text_fields}
    result["source_turn"] = source_turn
    result["target_turn"] = target_turn
    for field in ("recent_context", "target_prior"):
        value = row.get(field)
        if not isinstance(value, list):
            raise ValueError(f"{field} must be a list")
        turns: list[dict[str, Any]] = []
        preceding = -math.inf
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                raise ValueError(f"{field}[{index}] is not an object")
            turn = item.get("turn")
            if (type(turn) is not int or turn >= target_turn or turn < preceding
                    or not isinstance(item.get("speaker"), str)
                    or not isinstance(item.get("text"), str)):
                raise ValueError(f"{field}[{index}] crosses the target prefix or is malformed")
            if field == "target_prior" and item["speaker"] != row["target_investor"]:
                raise ValueError("target prior must be attributed to the target investor")
            preceding = turn
            turns.append({"turn": turn, "speaker": item["speaker"], "text": item["text"]})
        result[field] = turns
    return result


def _validate_train(rows: Sequence[dict[str, Any]]) -> None:
    seen: set[str] = set()
    seen_targets: set[tuple[str, str, int]] = set()
    for row in rows:
        if any(key not in row for key in MEMORY_KEYS):
            raise ValueError("training memory lacks a required field")
        identity = row["id"]
        if (row["split"] != "train" or not isinstance(identity, str)
                or not identity or identity in seen):
            raise ValueError("memory contains a non-training or duplicate row")
        seen.add(identity)
        visible_prefix(row)
        target_key = (row["case_id"], row["target_investor"], row["target_turn"])
        if target_key in seen_targets:
            raise ValueError("multiple peer links duplicate one target-turn observation")
        seen_targets.add(target_key)
        if (row["source_investor"] == UNKNOWN_INVESTOR
                or row["source_investor"] == row["target_investor"]):
            raise ValueError("training link must connect two named investors")
        for field in ("target_quote", "label_relation", "label_effect"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"training example {identity} lacks {field}")


def _vector(value: Any, dimension: int, field: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != dimension:
        raise ValueError(f"{field} has the wrong dimension")
    if any(type(component) not in (int, float) for component in value):
        raise ValueError(f"{field} contains a nonnumeric component")
    vector = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in vector):
        raise ValueError(f"{field} contains a nonfinite component")
    norm = math.sqrt(sum(component * component for component in vector))
    if not 0.99 <= norm <= 1.01:
        raise ValueError(f"{field} is not a normalized embedding")
    return vector


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _ordered_id_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(_canonical_bytes([row["id"] for row in rows])).hexdigest()


def _pack_npy(rows: Sequence[Sequence[float]], dimension: int) -> bytes:
    """Write a C-order little-endian float32 matrix in NumPy v1 format."""
    header = ("{'descr': '<f4', 'fortran_order': False, "
              f"'shape': ({len(rows)}, {dimension}), }}")
    pad = (16 - ((10 + len(header) + 1) % 16)) % 16
    encoded_header = (header + " " * pad + "\n").encode("latin1")
    if len(encoded_header) > 65535:
        raise ValueError("vector header exceeds NumPy v1 limit")
    stream = io.BytesIO()
    stream.write(b"\x93NUMPY\x01\x00")
    stream.write(struct.pack("<H", len(encoded_header)))
    stream.write(encoded_header)
    for index, row in enumerate(rows):
        checked = _vector(list(row), dimension, f"vector[{index}]")
        stream.write(struct.pack(f"<{dimension}f", *checked))
    return stream.getvalue()


def _write_npz(path: Path, vectors: Mapping[str, Sequence[Sequence[float]]],
               dimensions: Mapping[str, int]) -> None:
    """Store three independently compressed, NumPy-compatible matrices."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9, allowZip64=True) as archive:
        for channel in CHANNELS:
            info = zipfile.ZipInfo(f"{channel}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, _pack_npy(vectors[channel], dimensions[channel]),
                             compress_type=zipfile.ZIP_DEFLATED,
                             compresslevel=9)


def _read_npy(data: bytes, row_count: int, dimension: int,
              channel: str) -> tuple[tuple[float, ...], ...]:
    if len(data) < 10 or data[:8] != b"\x93NUMPY\x01\x00":
        raise ValueError(f"{channel} is not a NumPy v1 matrix")
    header_length = struct.unpack_from("<H", data, 8)[0]
    if len(data) < 10 + header_length:
        raise ValueError(f"{channel} header is truncated")
    try:
        header = ast.literal_eval(data[10:10 + header_length].decode("latin1"))
    except (SyntaxError, ValueError, UnicodeError) as exc:
        raise ValueError(f"{channel} header is malformed") from exc
    if (not isinstance(header, dict) or header.get("descr") != "<f4"
            or header.get("fortran_order") is not False
            or header.get("shape") != (row_count, dimension)):
        raise ValueError(f"{channel} matrix metadata differs from manifest")
    payload = data[10 + header_length:]
    if len(payload) != row_count * dimension * 4:
        raise ValueError(f"{channel} matrix length differs from manifest")
    components = (part[0] for part in struct.iter_unpack("<f", payload))
    rows = []
    for index in range(row_count):
        rows.append(_vector([next(components) for _ in range(dimension)],
                            dimension, f"{channel}[{index}]"))
    return tuple(rows)


def _read_npz(path: Path, row_count: int,
              dimensions: Mapping[str, int]) -> dict[str, tuple[tuple[float, ...], ...]]:
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(CHANNELS) or set(names) != {f"{c}.npy" for c in CHANNELS}:
            raise ValueError("vector archive does not contain the three channel matrices")
        matrices = {}
        for channel in CHANNELS:
            info = archive.getinfo(f"{channel}.npy")
            expected_max = 10 + 65535 + row_count * dimensions[channel] * 4
            if info.file_size > expected_max:
                raise ValueError(f"{channel} matrix exceeds declared shape")
            matrices[channel] = _read_npy(archive.read(info), row_count,
                                          dimensions[channel], channel)
    return matrices


def _validate_partitions(split_manifest: dict[str, Any],
                         memory: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    if split_manifest.get("version") != "venture_partition_v1" or split_manifest.get("locked") is not True:
        raise ValueError("venture partitions are not locked")
    partitions = _object(split_manifest.get("partitions"), "partitions")
    if set(partitions) != set(PARTITIONS):
        raise ValueError("venture partitions must separate fit, retrieval, tuning, and final evaluation")
    assignment: dict[str, str] = {}
    for group in PARTITIONS:
        cases = partitions[group]
        if not isinstance(cases, list) or any(not isinstance(case, str) or not case for case in cases):
            raise ValueError(f"{group} venture IDs are malformed")
        for case in cases:
            if case in assignment:
                raise ValueError(f"venture {case} appears in more than one partition")
            assignment[case] = group
    if not partitions["retrieval"]:
        raise ValueError("retrieval partition is empty")
    for row in memory:
        if assignment.get(row["case_id"]) != "retrieval":
            raise ValueError("memory contains a venture outside the retrieval training partition")
    return assignment


def _validate_raw_registry(raw_registry: Sequence[dict[str, Any]],
                           memory: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for entry in raw_registry:
        case = entry.get("case_id")
        if not isinstance(case, str) or not case or case in cases:
            raise ValueError("raw registry contains a missing or duplicate venture ID")
        for field in ("episode_id", "transcript_version"):
            if not isinstance(entry.get(field), str) or not entry[field].strip():
                raise ValueError(f"raw registry {case} lacks {field}")
        _digest(entry.get("source_sha256"), f"raw registry {case}.source_sha256")
        _digest(entry.get("transcript_sha256"), f"raw registry {case}.transcript_sha256")
        if entry.get("rights_permitted") is not True:
            raise ValueError(f"raw registry {case} is not rights-permitted")
        turns = entry.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"raw registry {case} has no ordered turns")
        prior = -math.inf
        observed: set[int] = set()
        for turn in turns:
            if (not isinstance(turn, dict) or type(turn.get("turn")) is not int
                    or turn["turn"] <= prior or turn["turn"] in observed
                    or not isinstance(turn.get("speaker"), str) or not turn["speaker"].strip()
                    or not isinstance(turn.get("text"), str)):
                raise ValueError(f"raw registry {case} has malformed turn order")
            observed.add(turn["turn"])
            prior = turn["turn"]
        if hashlib.sha256(_canonical_bytes(turns)).hexdigest() != entry["transcript_sha256"]:
            raise ValueError(f"raw registry {case} transcript hash mismatch")
        cases[case] = entry
    if {row["case_id"] for row in memory} - set(cases):
        raise ValueError("a retrieval venture is absent from the raw-text registry")
    return cases


def _span_text(span: Any, *, case: str, expected_turn: int,
               expected_speaker: str, turns: Mapping[int, dict[str, Any]],
               field: str) -> str:
    part = _object(span, field)
    turn = turns.get(expected_turn)
    if turn is None or part.get("turn") != expected_turn or turn["speaker"] != expected_speaker:
        raise ValueError(f"{field} turn or speaker differs from the transcript")
    if part.get("speaker") != expected_speaker or part.get("case_id") != case:
        raise ValueError(f"{field} attribution differs from the annotated row")
    start, end = part.get("start"), part.get("end")
    if (type(start) is not int or type(end) is not int or start < 0
            or end <= start or end > len(turn["text"])):
        raise ValueError(f"{field} offsets are outside the transcript turn")
    quote = turn["text"][start:end]
    if part.get("quote") != quote:
        raise ValueError(f"{field} quote is not recoverable at its character offsets")
    return quote


def _validate_adjudications(adjudications: Sequence[dict[str, Any]],
                            source: Sequence[dict[str, Any]],
                            registry: Mapping[str, dict[str, Any]]) -> None:
    if len(adjudications) != len(source):
        raise ValueError("adjudication count differs from training observations")
    by_id = {row.get("id"): row for row in adjudications}
    if len(by_id) != len(adjudications) or set(by_id) != {row["id"] for row in source}:
        raise ValueError("adjudication IDs do not align with training observations")
    for row in source:
        identity = row["id"]
        review = by_id[identity]
        raw = registry[row["case_id"]]
        if (review.get("case_id") != row["case_id"]
                or review.get("target_investor") != row["target_investor"]
                or review.get("target_turn") != row["target_turn"]):
            raise ValueError(f"adjudication {identity} has a different target key")
        if (row.get("episode_id") != raw["episode_id"]
                or row.get("transcript_version") != raw["transcript_version"]
                or row.get("source_sha256") != raw["source_sha256"]
                or review.get("transcript_version") != raw["transcript_version"]):
            raise ValueError(f"observation {identity} mixes transcript versions or sources")
        if review.get("review_decision") != "accepted":
            raise ValueError(f"observation {identity} has not passed human review")
        if (review.get("relation") != row["label_relation"]
                or review.get("effect") != row["label_effect"]):
            raise ValueError(f"observation {identity} differs from adjudicated interaction labels")
        coder_ids = review.get("coder_ids")
        if (not isinstance(coder_ids, list) or not coder_ids
                or len(set(coder_ids)) != len(coder_ids)
                or any(not isinstance(coder, str) or not coder.strip() for coder in coder_ids)
                or not isinstance(review.get("reviewer_id"), str)
                or not review["reviewer_id"].strip()):
            raise ValueError(f"observation {identity} lacks coder or reviewer provenance")
        if not isinstance(review.get("disagreements"), dict) or not isinstance(review.get("adjudication"), dict):
            raise ValueError(f"observation {identity} lacks disagreement or adjudication record")
        statuses = _object(review.get("field_status"), f"{identity}.field_status")
        for field in ("speaker", "source_span", "target_span", "relation", "effect"):
            if statuses.get(field) != "observed":
                raise ValueError(f"observation {identity} lacks reviewed {field}")
        if any(value not in FIELD_STATES for value in statuses.values()):
            raise ValueError(f"observation {identity} has an invalid field status")
        raw_turns = {turn["turn"]: turn for turn in raw["turns"]}
        source_quote = _span_text(review.get("source_span"), case=row["case_id"],
                                  expected_turn=row["source_turn"],
                                  expected_speaker=row["source_investor"],
                                  turns=raw_turns, field=f"{identity}.source_span")
        target_quote = _span_text(review.get("target_span"), case=row["case_id"],
                                  expected_turn=row["target_turn"],
                                  expected_speaker=row["target_investor"],
                                  turns=raw_turns, field=f"{identity}.target_span")
        if source_quote != row["source_quote"] or target_quote != row["target_quote"]:
            raise ValueError(f"observation {identity} quotes differ from reviewed spans")
        for field in ("recent_context", "target_prior"):
            for item in row[field]:
                original = raw_turns.get(item["turn"])
                if original is None or any(original[key] != item[key] for key in ("speaker", "text")):
                    raise ValueError(f"observation {identity} {field} differs from public transcript")
        links = review.get("source_links", [])
        if not isinstance(links, list):
            raise ValueError(f"observation {identity} source links are malformed")
        for index, link in enumerate(links):
            if not isinstance(link, dict) or type(link.get("turn")) is not int:
                raise ValueError(f"observation {identity} source link {index} is malformed")
            if link["turn"] >= row["target_turn"] or link.get("speaker") == row["target_investor"]:
                raise ValueError(f"observation {identity} has a future or own-speaker source link")
            _span_text(link, case=row["case_id"], expected_turn=link["turn"],
                       expected_speaker=link["speaker"], turns=raw_turns,
                       field=f"{identity}.source_links[{index}]")


def _audit_records(source: Sequence[dict[str, Any]], memory: Sequence[dict[str, Any]],
                   adjudications: Sequence[dict[str, Any]],
                   raw_registry: Sequence[dict[str, Any]],
                   split_manifest: dict[str, Any],
                   vectors: Mapping[str, Sequence[Sequence[float]]]) -> dict[str, Any]:
    _validate_train(source)
    _validate_train(memory)
    if len(source) != len(memory) or len(adjudications) != len(memory):
        raise ValueError("source, memory, and review counts differ")
    for index, (original, projected) in enumerate(zip(source, memory)):
        if {key: original[key] for key in MEMORY_KEYS} != projected:
            raise ValueError(f"memory row {index} differs from its reviewed source")
        hidden_variant = dict(original)
        hidden_variant.update(target_quote="hidden", label_relation="hidden",
                              label_effect="hidden", future_turns=["hidden"],
                              venture_outcome="hidden")
        if (visible_prefix(original) != visible_prefix(hidden_variant)
                or _feature_texts(visible_prefix(original))
                != _feature_texts(visible_prefix(hidden_variant))):
            raise ValueError(f"target information leaks into the query for row {index}")
    assignments = _validate_partitions(split_manifest, memory)
    registry = _validate_raw_registry(raw_registry, memory)
    _validate_adjudications(adjudications, source, registry)
    for channel in CHANNELS:
        if len(vectors[channel]) != len(memory):
            raise ValueError(f"{channel} vector count differs from reviewed rows")
    eligibility: dict[str, list[str]] = {}
    for row in memory:
        candidate_ids: list[str] = []
        for index in _eligible_indices(memory, row):
            example = memory[index]
            if (example["split"] != "train" or example["case_id"] == row["case_id"]
                    or example["target_investor"] != row["target_investor"]):
                raise ValueError("retrieval eligibility crosses a split, venture, or investor")
            candidate_ids.append(example["id"])
        eligibility[row["id"]] = candidate_ids
    return {
        "version": "knowledge_audit_v1", "status": "passed",
        "row_count": len(memory),
        "case_count": len({row["case_id"] for row in memory}),
        "reviewed_count": len(adjudications),
        "raw_registry_count": len(registry),
        "partition_case_counts": {name: list(assignments.values()).count(name) for name in PARTITIONS},
        "eligible_pair_count": sum(len(ids) for ids in eligibility.values()),
        "eligibility_by_target": eligibility,
        "ordered_row_ids_sha256": _ordered_id_digest(memory),
        "temporal_gate_passed": True,
        "target_query_invariance_passed": True,
        "same_investor_cross_venture_training_filter_passed": True,
    }


def load_snapshot(
    directory: Path | str, *, artifacts: Mapping[str, Path | str]
) -> KnowledgeSnapshot:
    """Load a reviewed, split-locked, hash-bound local retrieval snapshot."""
    root = Path(directory).resolve(strict=True)
    manifest_path = root / "manifest.json"
    manifest = _object(json.loads(manifest_path.read_text(encoding="utf-8")), "manifest")
    if manifest.get("version") != VERSION or manifest.get("construction_split") != "train":
        raise ValueError("unsupported or non-training memory snapshot")
    if not isinstance(manifest.get("snapshot_id"), str) or not manifest["snapshot_id"].strip():
        raise ValueError("snapshot lacks an immutable identity")
    if manifest.get("vector_format") != VECTOR_FORMAT:
        raise ValueError("unsupported compressed vector format")
    files = _object(manifest.get("files"), "files")
    if set(files) != set(DATA_FILES):
        raise ValueError("snapshot file registry is incomplete")
    paths = {name: _member(root, files[name], name) for name in files}

    recorded_artifacts = _object(manifest.get("artifact_sha256"), "artifact_sha256")
    if set(recorded_artifacts) != set(ARTIFACTS) or set(artifacts) != set(ARTIFACTS):
        raise ValueError("encoder and prompt artifacts must be supplied")
    for name in ARTIFACTS:
        path = Path(artifacts[name]).resolve(strict=True)
        if artifact_sha256(path) != _digest(recorded_artifacts[name], name):
            raise ValueError(f"{name} differs from snapshot manifest")

    source = _read_jsonl(paths["source_train"])
    memory = _read_jsonl(paths["memory"])
    adjudications = _read_jsonl(paths["adjudications"])
    raw_registry = _read_jsonl(paths["raw_registry"])
    split_manifest = _object(json.loads(paths["split_manifest"].read_text(encoding="utf-8")),
                             "split_manifest")
    audit_report = _object(json.loads(paths["audit_report"].read_text(encoding="utf-8")),
                           "audit_report")
    if (type(manifest.get("row_count")) is not int
            or type(manifest.get("case_count")) is not int
            or manifest["row_count"] <= 0 or manifest["case_count"] <= 0):
        raise ValueError("invalid snapshot row or venture count")
    if not (len(source) == len(memory) == len(adjudications) == manifest["row_count"]):
        raise ValueError("snapshot row count mismatch")
    if len({row["case_id"] for row in memory}) != manifest.get("case_count"):
        raise ValueError("snapshot venture count mismatch")
    if _ordered_id_digest(memory) != _digest(manifest.get("ordered_row_ids_sha256"),
                                            "ordered_row_ids_sha256"):
        raise ValueError("vector row identity alignment differs from manifest")

    declared = _object(manifest.get("embedding_dimensions"), "embedding_dimensions")
    if set(declared) != set(CHANNELS) or any(
        type(declared[name]) is not int or declared[name] <= 0 for name in CHANNELS
    ):
        raise ValueError("invalid embedding dimensions")
    vectors = _read_npz(paths["vectors"], len(memory), declared)
    recomputed = _audit_records(source, memory, adjudications, raw_registry,
                                split_manifest, vectors)
    if recomputed != audit_report:
        raise ValueError("stored knowledge audit differs from verified records")
    return KnowledgeSnapshot(
        manifest_sha256=_sha256(manifest_path),
        manifest=manifest,
        memory=tuple(memory),
        vectors=vectors,
    )


def _feature_texts(row: Mapping[str, Any]) -> dict[str, str]:
    source = row["source_quote"].strip()[:512]
    context = "\n".join(
        f"{turn['speaker']}: {turn['text']}" for turn in row["recent_context"][-4:]
    )[-1200:]
    prior = "\n".join(turn["text"] for turn in row["target_prior"][-2:])[-600:]
    return {"source": source, "context": context, "prior": prior}


def _query_vector(
    encoder: Callable[[str], Sequence[float]], text: str, dimension: int, field: str
) -> tuple[float, ...]:
    value = encoder(text)
    if len(value) != dimension or any(
        isinstance(component, (str, bytes, bool)) for component in value
    ):
        raise ValueError(f"query {field} embedding has the wrong type or dimension")
    try:
        vector = tuple(float(component) for component in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"query {field} embedding is not numeric") from exc
    if not all(math.isfinite(component) for component in vector):
        raise ValueError(f"query {field} embedding is not finite")
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0 or not math.isfinite(norm):
        raise ValueError(f"query {field} embedding is zero or nonfinite")
    return tuple(component / norm for component in vector)


def _check_encoder(snapshot: KnowledgeSnapshot,
                   encoder: Callable[[str], Sequence[float]]) -> None:
    expected = snapshot.manifest["artifact_sha256"]
    for name in ("embedding_model", "embedding_config"):
        attr = f"{name}_sha256"
        actual = getattr(encoder, attr, None)
        if actual != expected[name]:
            raise ValueError(f"online BGE {name} differs from indexed snapshot")


def _eligible_indices(memory: Sequence[Mapping[str, Any]],
                      query: Mapping[str, Any]) -> list[int]:
    return [
        index for index, example in enumerate(memory)
        if example["split"] == "train"
        and example["target_investor"] == query["target_investor"]
        and example["case_id"] != query["case_id"]
    ]


def retrieve(
    snapshot: KnowledgeSnapshot,
    query: Mapping[str, Any],
    encoder: Callable[[str], Sequence[float]],
    *,
    k: int = 3,
) -> list[dict[str, Any]]:
    """Rank same-investor, other-venture training analogies from a safe prefix.

    The three cosine channels have weights .60/.30/.10; absent context or
    candidate-attributed prior redistributes their weight over available channels. A
    matching known source investor adds .03. Ties resolve by memory ID, and
    each training venture can contribute at most one of the three examples.
    """
    if type(k) is not int or not 0 <= k <= 3:
        raise ValueError("k must be an integer from zero through three")
    _check_encoder(snapshot, encoder)
    item = visible_prefix(query)
    if k == 0 or item["target_investor"] == UNKNOWN_INVESTOR:
        return []
    eligible = _eligible_indices(snapshot.memory, item)
    if not eligible:
        return []

    active = {"source": WEIGHTS["source"]}
    if item["recent_context"]:
        active["context"] = WEIGHTS["context"]
    if item["target_prior"]:
        active["prior"] = WEIGHTS["prior"]
    texts = _feature_texts(item)
    dimensions = snapshot.manifest["embedding_dimensions"]
    query_vectors = {
        name: _query_vector(encoder, texts[name], dimensions[name], name)
        for name in active
    }
    ranked: list[tuple[float, str, int]] = []
    for index in eligible:
        example = snapshot.memory[index]
        candidate_active = {"source": WEIGHTS["source"]}
        if "context" in active and example["recent_context"]:
            candidate_active["context"] = WEIGHTS["context"]
        if "prior" in active and example["target_prior"]:
            candidate_active["prior"] = WEIGHTS["prior"]
        score = sum(
            weight * sum(a * b for a, b in zip(snapshot.vectors[name][index], query_vectors[name]))
            for name, weight in candidate_active.items()
        ) / sum(candidate_active.values())
        if (item["source_investor"] != UNKNOWN_INVESTOR
                and example["source_investor"] == item["source_investor"]):
            score += 0.03
        ranked.append((score, example["id"], index))
    ranked.sort(key=lambda part: (-part[0], part[1]))

    selected: list[dict[str, Any]] = []
    seen_ventures: set[str] = set()
    for score, _, index in ranked:
        example = snapshot.memory[index]
        if example["case_id"] in seen_ventures:
            continue
        seen_ventures.add(example["case_id"])
        selected.append({"score": round(score, 6), "example": _thaw(example)})
        if len(selected) == k:
            break
    return selected
