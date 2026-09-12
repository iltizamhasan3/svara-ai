"""Run the reproducible Week 4 sentence embedding and topic experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.ai.embeddings import (  # noqa: E402
    EMBEDDING_VERSION,
    EmbeddingConfig,
    SentenceTransformerEncoder,
)
from app.ai.evaluation import manual_review_rows, topic_metrics  # noqa: E402
from app.ai.segmentation import SEGMENTATION_VERSION, segment_rows  # noqa: E402
from app.ai.topic_model import TopicModelConfig, fit_topic_model  # noqa: E402


DEFAULT_INPUT = ROOT / "data/samples/igar/Rating_labeled_sample.csv"
DEFAULT_OUTPUT = ROOT / "artifacts/week4"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_input_rows(
    path: Path,
    *,
    text_column: str,
    delimiter: str | None = None,
    no_header: bool = False,
) -> list[dict[str, object]]:
    """Read a CSV/TSV text column without changing the source row order."""

    selected_delimiter = effective_delimiter(path, delimiter)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        if no_header:
            reader = csv.reader(handle, delimiter=selected_delimiter)
            return [
                {text_column: values[0]}
                for values in reader
                if values and any(value.strip() for value in values)
            ]

        reader = csv.DictReader(handle, delimiter=selected_delimiter)
        if not reader.fieldnames or text_column not in reader.fieldnames:
            available = ", ".join(reader.fieldnames or []) or "<none>"
            raise ValueError(f"text column {text_column!r} not found; available: {available}")
        return [dict(row) for row in reader]


def effective_delimiter(path: Path, delimiter: str | None = None) -> str:
    return delimiter or ("\t" if path.suffix.casefold() == ".tsv" else ",")


def display_path(path: Path) -> str:
    resolved = path.resolve()
    return str(resolved.relative_to(ROOT)) if resolved.is_relative_to(ROOT) else str(resolved)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_assignments(path: Path, units, topic_ids, probabilities) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_row_number", "unit_index", "text", "topic_id", "topic_probability"],
            lineterminator="\n",
        )
        writer.writeheader()
        for unit, topic_id, probability in zip(units, topic_ids, probabilities, strict=True):
            writer.writerow(
                {
                    "source_row_number": unit.source_row_number,
                    "unit_index": unit.unit_index,
                    "text": unit.text,
                    "topic_id": topic_id,
                    "topic_probability": "" if probability is None else f"{probability:.8f}",
                }
            )


def write_review_sheet(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "topic_id",
        "unit_count",
        "keywords",
        "representative_texts",
        "manual_judgment",
        "review_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def review_has_manual_entries(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        return any(
            row.get("topic_id") != "-1"
            and (row.get("manual_judgment", "").strip() or row.get("review_notes", "").strip())
            for row in rows
        )


def review_topic_ids(path: Path) -> set[int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            int(row["topic_id"])
            for row in rows
            if row.get("topic_id") not in {None, "", "-1"}
        }


def should_preserve_review(path: Path, topic_ids: set[int]) -> bool:
    """Keep completed judgments safe and reject stale topic worksheets."""

    if not review_has_manual_entries(path):
        return False
    existing_topic_ids = review_topic_ids(path)
    if existing_topic_ids != topic_ids:
        raise ValueError(
            "manual review topics do not match the new run; use a new output directory "
            "or pass --overwrite-review"
        )
    return True


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cluster_payload(result) -> dict[str, object]:
    return {
        "topic_version": result.config.version,
        "clusters": [
            {
                "topic_id": cluster.topic_id,
                "label": cluster.label,
                "unit_count": cluster.unit_count,
                "keywords": [keyword.model_dump() for keyword in cluster.keywords],
                "representative_texts": list(cluster.representative_texts),
            }
            for cluster in result.clusters
        ],
        "outlier": {
            "topic_id": -1,
            "unit_count": result.outlier_count,
            "included_as_cluster": False,
        },
        "warnings": list(result.warnings),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--text-column", default="content")
    parser.add_argument("--delimiter", default=None)
    parser.add_argument("--no-header", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-rows", type=int, default=30)
    parser.add_argument("--split-clauses", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--umap-n-neighbors", type=int, default=5)
    parser.add_argument("--umap-n-components", type=int, default=3)
    parser.add_argument("--umap-min-dist", type=float, default=0.0)
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=3)
    parser.add_argument("--hdbscan-min-samples", type=int, default=1)
    parser.add_argument("--min-topic-size", type=int, default=3)
    parser.add_argument("--top-n-words", type=int, default=5)
    parser.add_argument("--representative-texts", type=int, default=3)
    parser.add_argument(
        "--overwrite-review",
        action="store_true",
        help="replace a completed manual review worksheet explicitly",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = args.input_path.resolve()
    output_dir = args.output_dir.resolve()
    if args.max_rows < 1:
        raise ValueError("max_rows must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_input_rows(
        input_path,
        text_column=args.text_column,
        delimiter=args.delimiter,
        no_header=args.no_header,
    )[: args.max_rows]
    units = segment_rows(rows, args.text_column, split_clauses=args.split_clauses)
    texts = [unit.text for unit in units]

    embedding_defaults = EmbeddingConfig()
    embedding_config = EmbeddingConfig(
        model_name=args.model_name or embedding_defaults.model_name,
        revision=args.model_revision or embedding_defaults.revision,
        batch_size=args.batch_size,
    )
    embeddings = SentenceTransformerEncoder(embedding_config).encode(texts)

    topic_config = TopicModelConfig(
        random_state=args.random_state,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_n_components=args.umap_n_components,
        umap_min_dist=args.umap_min_dist,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        hdbscan_min_samples=args.hdbscan_min_samples,
        min_topic_size=args.min_topic_size,
        top_n_words=args.top_n_words,
        representative_texts=args.representative_texts,
    )
    result = fit_topic_model(texts, embeddings.vectors, config=topic_config)

    embedding_path = output_dir / "embeddings.npy"
    clusters_path = output_dir / "topic_clusters.json"
    assignments_path = output_dir / "topic_assignments.csv"
    metrics_path = output_dir / "topic_metrics.json"
    review_path = output_dir / "manual_evaluation.csv"
    manifest_path = output_dir / "experiment_manifest.json"
    preserve_review = (
        not args.overwrite_review
        and should_preserve_review(review_path, {cluster.topic_id for cluster in result.clusters})
    )
    np.save(embedding_path, embeddings.vectors)
    write_json(clusters_path, _cluster_payload(result))
    write_assignments(assignments_path, units, result.topic_ids, result.probabilities)
    write_json(metrics_path, topic_metrics(result))
    if not preserve_review:
        write_review_sheet(review_path, manual_review_rows(result))

    artifact_paths = {
        "embeddings": embedding_path,
        "topic_clusters": clusters_path,
        "topic_assignments": assignments_path,
        "topic_metrics": metrics_path,
        "manual_evaluation": review_path,
    }
    relative_paths = {name: display_path(path) for name, path in artifact_paths.items()}
    selected_delimiter = effective_delimiter(input_path, args.delimiter)
    manifest = {
        "manifest_version": 1,
        "experiment": "week4_topic_discovery",
        "dataset": {
            "path": str(input_path.relative_to(ROOT)) if input_path.is_relative_to(ROOT) else str(input_path),
            "sha256": sha256_file(input_path),
            "input_rows_selected": len(rows),
            "analysis_units": len(units),
            "max_rows": args.max_rows,
        },
        "input_parsing": {
            "text_column": args.text_column,
            "delimiter": selected_delimiter,
            "no_header": args.no_header,
        },
        "segmentation": {
            "version": SEGMENTATION_VERSION,
            "split_clauses": args.split_clauses,
        },
        "embedding": {
            "version": EMBEDDING_VERSION,
            **asdict(embedding_config),
            "shape": list(embeddings.vectors.shape),
            "dtype": str(embeddings.vectors.dtype),
        },
        "topic": {
            **asdict(topic_config),
            "keyword_method": "bertopic.c_tf_idf",
            "library": "bertopic",
        },
        "software": {
            package: _package_version(package)
            for package in (
                "bertopic",
                "hdbscan",
                "sentence-transformers",
                "umap-learn",
                "scikit-learn",
                "torch",
                "transformers",
            )
        },
        "results": topic_metrics(result),
        "artifact_paths": relative_paths,
        "artifact_checksums": {
            relative_paths[name]: sha256_file(path) for name, path in artifact_paths.items()
        },
        "manual_review": {
            "status": "preserved" if preserve_review else "template",
            "overwrite_requires_explicit_flag": True,
        },
        "status": "initial_cluster_experiment",
        "quality_note": "Unsupervised topic discovery; manual review is required before production use.",
    }
    write_json(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "results": manifest["results"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
