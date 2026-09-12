# SVARA AI — Week 4 Topic Discovery Baseline

Status: selesai untuk scope Project Lead + AI/NLP Engineer Week 4.

## Pipeline

The reproducible baseline is:

```text
feedback rows
→ sentence splitting
→ high-precision contrastive clause splitting
→ Sentence Transformer embeddings
→ UMAP
→ HDBSCAN
→ BERTopic topic assignments
→ c-TF-IDF keywords
→ representative texts
→ descriptive and manual evaluation
```

Sentence units retain `source_row_number` and a deterministic `unit_index`.
Outliers use `topic_id = -1` and are not emitted as normal topic clusters.

## Frozen baseline configuration

| Component | Configuration |
|---|---|
| Segmentation | `segmentation-v1`, clauses enabled for the initial run |
| Embedding model | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Embedding revision | `e8f8c211226b894fcb81acc59f3b34ba3efd5f42` |
| Embedding output | 384-dimensional, float32, normalized, CPU batch size 8 |
| Topic version | `topic-v1` |
| UMAP | neighbors 5, components 3, min distance 0.0, cosine metric, seed 42 |
| HDBSCAN | min cluster size 3, min samples 1, Euclidean metric |
| BERTopic | min topic size 3, probabilities enabled |
| Keywords | BERTopic `ClassTfidfTransformer(reduce_frequent_words=True)`, top 5 words |

The exact dependency versions and checksums are recorded in
[`artifacts/week4/experiment_manifest.json`](../artifacts/week4/experiment_manifest.json).

## Initial experiment

The run uses the tracked 30-row IGAR sample as a domain-shaped smoke dataset,
not as a representative production corpus. Clause splitting produced 55
analysis units and BERTopic produced four non-outlier clusters:

| Topic | Units | Initial interpretation | Manual verdict |
|---:|---:|---|---|
| 0 | 20 | Account data and date-of-birth editing | useful |
| 1 | 14 | Mixed generic praise, login fragments, and COVID text | not useful |
| 2 | 14 | Vaccine/certificate concerns with mixed context | partially useful |
| 3 | 7 | Short acknowledgements and greetings | not useful |

Diagnostics are in [`topic_metrics.json`](../artifacts/week4/topic_metrics.json):

- outlier rate: `0.0000`;
- topic diversity: `1.0000` over the 20 generated keyword slots;
- four clusters require qualitative review.

The metrics are descriptive only. Topic discovery is unsupervised and has no
aspect ground-truth labels in this sample. The manual review worksheet is
[`manual_evaluation.csv`](../artifacts/week4/manual_evaluation.csv).

## Artifacts

- [`embeddings.npy`](../artifacts/week4/embeddings.npy)
- [`topic_assignments.csv`](../artifacts/week4/topic_assignments.csv)
- [`topic_clusters.json`](../artifacts/week4/topic_clusters.json)
- [`topic_metrics.json`](../artifacts/week4/topic_metrics.json)
- [`manual_evaluation.csv`](../artifacts/week4/manual_evaluation.csv)
- [`experiment_manifest.json`](../artifacts/week4/experiment_manifest.json)

## Reproduction

From the repository root, install the pinned Week 4 AI extra and run:

```bash
cd backend
python -m pip install -e '.[ai]'
cd ..
backend/.venv/bin/python scripts/run_week4_topic_discovery.py \
  --input-path data/samples/igar/Rating_labeled_sample.csv \
  --text-column content \
  --output-dir artifacts/week4 \
  --max-rows 30 \
  --split-clauses \
  --batch-size 8 \
  --umap-n-neighbors 5 \
  --umap-n-components 3 \
  --hdbscan-min-cluster-size 3 \
  --hdbscan-min-samples 1 \
  --min-topic-size 3 \
  --top-n-words 5
```

## Limitations and next tuning pass

- The sample is intentionally small; the result validates the pipeline and
  artifact contract, not production topic quality.
- Topic 1 and Topic 3 should not be presented as actionable aspects.
- The next experiment should add Indonesian stopword handling, filter or route
  very short generic units to outliers, and compare settings on a larger
  feedback corpus.
- Backend inference and dashboard integration remain outside the Project Lead
  + AI/NLP Week 4 scope.
