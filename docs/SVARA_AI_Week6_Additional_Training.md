# SVARA AI — Week 6 Additional Training and IGAR Boundary

Status: selesai untuk scope Project Lead + AI/NLP Engineer. Bobot model kandidat tetap lokal dan tidak dimasukkan ke source repository.

## Keputusan utama

IGAR dikunci sebagai external test-only. Tidak ada `labelScoreBase`, `score`, prediksi, atau metrik IGAR yang dipakai untuk training, checkpoint selection, learning-rate selection, thresholding, atau pemilihan kandidat.

Untuk meningkatkan relevansi domain, eksperimen memakai sumber terpisah: [Indonesian Google Play Review](https://huggingface.co/datasets/jakartaresearch/google-play-review). Metadata dataset mencantumkan bahasa Indonesia, sentiment classification, Google Play reviews, dan lisensi CC BY 4.0.

## Data preparation

Sumber raw yang dipakai untuk training tambahan adalah `train.csv` (7,028 rows). Sumber publik hanya menyediakan label binary `pos`/`neg`; target tiga kelas diturunkan secara eksplisit dari rating:

| Published stars | Training label |
| --- | --- |
| 1–2 | `negative` |
| 3 | `neutral` |
| 4–5 | `positive` |

Quality and leakage guards:

- 1 row missing (`N/A`) dibuang;
- 800 duplicate rows dibuang;
- 19 normalized-text groups dengan derived label yang konflik (1,056 rows) dibuang seluruhnya;
- 16 rows yang overlap dengan SmSA dibuang;
- 478 rows yang exact-text overlap dengan IGAR dibuang hanya sebagai guard kebocoran;
- IGAR dipakai pada tahap ini hanya untuk text-key overlap guard; field label/score tidak dikonsumsi dan prediction/metric IGAR tidak pernah dibuat;
- split akhir Google Play Review: 3,741 train, 467 validation, 469 test;
- split dibuat stratified, seed `42`, dan ketiga split disjoint.

Raw source dan processed payload di-ignore oleh Git. Provenance, URL, license, size, checksum, schema, dan aturan label ada di [`week6_training_sources.json`](../data/manifests/week6_training_sources.json). Reproduksi preparation:

```bash
backend/.venv/bin/python scripts/download_google_play_review_training.py
backend/.venv/bin/python scripts/prepare_week6_google_play.py
```

## Candidate experiment

Kandidat dimulai dari `artifacts/week3/model-v1`, membekukan encoder, dan hanya melatih classifier head:

- training source: Google Play Review train split, 3,741 rows;
- epoch: `1.0`;
- learning rate: `1e-4`;
- batch size: `64`;
- max sequence length: `128`;
- seed: `42`;
- device: CPU, PyTorch threads `4`;
- checkpoint selection tetap memakai SmSA validation, bukan IGAR;
- class-weighted run dicoba terpisah dan tidak dipilih.

## Model selection evidence

### SmSA validation regression check

| Model | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| v1 baseline | 0.937302 | 0.910796 |
| Google Play unweighted | 0.938095 | 0.911100 |
| Google Play class-weighted | 0.937302 | 0.912405 |

### Google Play validation

| Model | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| v1 baseline | 0.766595 | 0.549131 |
| Google Play unweighted | **0.798715** | **0.556349** |
| Google Play class-weighted | 0.777302 | 0.546841 |

### Google Play confirmation test

| Model | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| v1 baseline | 0.769723 | 0.575428 |
| Google Play unweighted | **0.814499** | **0.598803** |
| Google Play class-weighted | 0.795309 | 0.587914 |

Pada confirmation test, neutral F1 naik dari `0.1319` menjadi `0.1791`, sementara positive F1 naik dari `0.8701` menjadi `0.9038`. Candidate unweighted dipilih sebagai kandidat domain-app-review; weighted candidate ditolak karena macro-F1 Google lebih rendah.

Angka di atas adalah validasi non-IGAR dan tidak boleh disebut sebagai performa IGAR. Candidate juga belum otomatis menggantikan release `ai-model-v1.0.0`.

## Reproduksi evaluasi non-IGAR

Evaluator terpisah menolak path yang terlihat seperti IGAR dan memvalidasi model bundle melalui export manifest. Untuk baseline:

```bash
backend/.venv/bin/python scripts/evaluate_non_igar_tsv.py \
  --input data/processed/week6/google_play_review/google_play_review_test.tsv \
  --model-dir artifacts/week3/model-v1 \
  --output /tmp/svara-google-test-baseline.json \
  --dataset-name "Indonesian Google Play Review test" \
  --batch-size 128 --max-length 128 --torch-threads 4
```

Untuk candidate lokal, gunakan `model-v1` dan `export_manifest.json` di folder candidate yang dihasilkan oleh eksperimen training:

```bash
backend/.venv/bin/python scripts/evaluate_non_igar_tsv.py \
  --input data/processed/week6/google_play_review/google_play_review_test.tsv \
  --model-dir artifacts/week6/google-play-head-candidate-lr1e-4/model-v1 \
  --export-manifest artifacts/week6/google-play-head-candidate-lr1e-4/export_manifest.json \
  --output /tmp/svara-google-test-candidate.json \
  --dataset-name "Indonesian Google Play Review test" \
  --batch-size 128 --max-length 128 --torch-threads 4
```

Perintah IGAR tetap berada di evaluator Week 5 dan hanya boleh dijalankan sebagai final external test setelah model candidate disetujui untuk release. Tidak ada hasil IGAR baru yang dipakai dalam laporan Week 6 ini.

## Known limitation and next gate

Label neutral Google Play Review adalah derived/weak label dan jumlahnya kecil (`23` rows pada masing-masing holdout). Hasil ini menunjukkan peningkatan yang konsisten pada domain app-review proxy, tetapi belum membuktikan generalisasi ke seluruh IGAR. Gate berikutnya adalah review model artifact, keputusan release, lalu satu kali external test IGAR yang tidak digunakan untuk tuning.
