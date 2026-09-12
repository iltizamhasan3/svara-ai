# SVARA AI — Week 5 Sentiment Inference

Status: selesai untuk scope Project Lead + AI/NLP Engineer Week 5.

## Scope yang diselesaikan

- preprocessing inference memakai `preprocessing-v1` yang sama dengan training;
- bundle IndoBERT divalidasi sebelum dipakai dan dideskripsikan dengan export manifest;
- batch inference CPU mempertahankan urutan, duplicate, confidence, probability map, dan source row number;
- inference eksternal dijalankan pada sample IGAR yang dilacak checksum-nya;
- error classification dibuat sebagai bucket review deterministik.

Integrasi persistence, endpoint dashboard, dan penggantian `MockAnalysisPipeline` pada backend tetap berada pada scope Backend + Data Engineer di tracker Week 5.

## Model export dan inference contract

- Model: `indobenchmark/indobert-base-p1`
- Immutable revision: `c2cd0b51ddce6580eb35263b39b0a1e5fb0a39e2`
- Model version: `sentiment-model-v1`
- Preprocessing: `preprocessing-v1`
- Label order: `positive=0`, `neutral=1`, `negative=2`
- Runtime: CPU, local bundle, `max_length=128`, default batch size `8`
- Release: [`ai-model-v1.0.0`](https://github.com/iltizamhasan3/svara-ai/releases/tag/ai-model-v1.0.0)

Model weights tidak diduplikasi ke source repository karena ukurannya sekitar 498 MB. Export manifest mencatat file inference wajib, ukuran, checksum, mapping label, dan release asset:

- [`sentiment_model_export_manifest.json`](../artifacts/week5/sentiment_model_export_manifest.json)
- [`model_bundle.py`](../backend/app/ai/model_bundle.py)
- [`export_sentiment_model.py`](../scripts/export_sentiment_model.py)

Loader menggunakan `local_files_only=True` secara default agar inference tidak diam-diam mengunduh atau mengganti model.

## Preprocessing

`prepare_inference_rows` melakukan Unicode NFKC dan normalisasi whitespace seperti pipeline training. Punctuation, emoji, slang, dan sinyal sentimen dipertahankan. Row dengan text kosong dilewati; duplicate tidak dihapus agar jumlah prediksi tetap sesuai input. Setiap hasil membawa nomor row satu-based dari CSV.

## External validation pada IGAR

Input yang dipakai:

- file: `data/samples/igar/Rating_labeled_sample.csv`;
- checksum SHA-256: `f8966d3b15fd69906b803ce341bd0185761911e374de481a98bb7baeaf04bb88`;
- 30 row, masing-masing 10 `Negative`, 10 `Neutral`, dan 10 `Positive`;
- text column: `content`;
- gold label: `labelScoreBase`.

IGAR hanya digunakan sebagai external/domain validation dan tidak digunakan untuk training, checkpoint selection, atau tuning.

### Hasil

| Metric | Value |
| --- | ---: |
| Accuracy | 0.6667 |
| Macro precision | 0.6349 |
| Macro recall | 0.6667 |
| Macro F1 | 0.6205 |

Confusion matrix menggunakan urutan `positive`, `neutral`, `negative`:

| Actual \\ Predicted | positive | neutral | negative |
| --- | ---: | ---: | ---: |
| positive | 10 | 0 | 0 |
| neutral | 2 | 2 | 6 |
| negative | 0 | 2 | 8 |

Temuan utama:

- seluruh 10 row positive diprediksi benar;
- negative terdeteksi 8 dari 10 row;
- neutral menjadi kelas terlemah: recall `0.20`, F1 `0.2857`;
- error terbanyak adalah `neutral → negative` sebanyak 6 row;
- 10 dari 30 row salah (`error_rate=0.3333`).

`igar_metrics.json` juga menyimpan konfigurasi efektif (`batch_size`, `max_length`, threads, column mapping, deterministic flag) dan versi runtime agar hasil dapat direproduksi. Export manifest adalah trust anchor; exporter menolak overwrite kecuali flag release eksplisit digunakan.

### Error analysis

| Review bucket | Count |
| --- | ---: |
| mixed_signal | 3 |
| short_text | 2 |
| question_or_request | 2 |
| long_text | 1 |
| other | 2 |

Bucket tersebut adalah heuristik triage berbasis teks untuk membantu review manual, bukan penjelasan kausal model. Confidence tinggi juga tidak dianggap sebagai bukti prediksi benar. Sample IGAR kecil dan label rating-derived dapat mengandung noise atau ketidaksesuaian dengan definisi sentiment SmSA; hasil ini tidak boleh digeneralisasi sebagai performa produksi.

## Reproduksi

Dari root repository, setelah bundle model tersedia secara lokal:

```bash
backend/.venv/bin/python scripts/run_week5_igar_inference.py \
  --batch-size 8 \
  --max-length 128 \
  --torch-threads 4
```

Output tersedia di [`artifacts/week5/igar/`](../artifacts/week5/igar/):

- `igar_predictions.csv`: semua row dengan label, confidence, probability map, dan correctness;
- `igar_metrics.json`: metrics, classification report, confusion matrix, checksum, dan metadata model;
- `igar_error_analysis.csv`: hanya row yang salah;
- `igar_error_analysis.json`: ringkasan bucket, confusion pair, dan contoh error.

## Batasan lanjutan

Hasil Week 5 belum berarti milestone CSV → Sentiment Analysis → Dashboard selesai karena backend belum mengganti mock pipeline, menyimpan prediksi, atau menyediakan endpoint dashboard. Itu merupakan pekerjaan lintas role pada bagian tracker berikutnya.
