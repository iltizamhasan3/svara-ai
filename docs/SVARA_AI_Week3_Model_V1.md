# SVARA AI — Sentiment Model v1

Status: selesai untuk scope Project Lead + AI/NLP Engineer Week 3.

## Model dan data

- Base model: [`indobenchmark/indobert-base-p1`](https://huggingface.co/indobenchmark/indobert-base-p1)
- Immutable Hugging Face revision: `c2cd0b51ddce6580eb35263b39b0a1e5fb0a39e2`
- Dataset: IndoNLU SmSA
- Preprocessing: `preprocessing-v1`
- Label order: `positive=0`, `neutral=1`, `negative=2`
- Leakage-safe rows: `10,919` train, `1,260` validation, `500` official test
- Test evaluation: belum dijalankan; test split tetap held-out dan memerlukan freeze gate eksplisit
- IGAR: external/domain-validation sample only, tidak dipakai untuk training atau tuning

## Konfigurasi training

- Epochs: `1`
- Max sequence length: `128`
- Train batch size: `16`
- Validation batch size: `32`
- Gradient accumulation: `1` (effective batch size `16`)
- Learning rate: `2e-5`
- Weight decay: `0.01`
- Seed: `42`
- PyTorch threads: `4`
- Device: CPU
- Best checkpoint selection: validation `macro_f1`

## Validation result

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9373015873 |
| Macro Precision | 0.9255246214 |
| Macro Recall | 0.8990797466 |
| Macro F1 | 0.9107956226 |

Confusion matrix menggunakan urutan label `positive`, `neutral`, `negative`:

| Actual \\ Predicted | positive | neutral | negative |
| --- | ---: | ---: | ---: |
| positive | 705 | 7 | 23 |
| neutral | 11 | 104 | 16 |
| negative | 19 | 3 | 372 |

## Artefak

- [`metrics.json`](../artifacts/week3/metrics.json)
- [`classification_report.json`](../artifacts/week3/classification_report.json)
- [`confusion_matrix.json`](../artifacts/week3/confusion_matrix.json)
- [`confusion_matrix.csv`](../artifacts/week3/confusion_matrix.csv)
- [`confusion_matrix.png`](../artifacts/week3/confusion_matrix.png)
- [`experiment_manifest.json`](../artifacts/week3/experiment_manifest.json)
- Model dan tokenizer tersimpan lokal di `artifacts/week3/model-v1/` dan sengaja di-ignore karena ukuran bobot sekitar 498 MB. Manifest menyimpan path, checkpoint identifier, dan checksum.

## Reproduksi

Dari root repository, install optional AI dependencies lalu jalankan:

```bash
backend/.venv/bin/python scripts/train_indobert.py \
  --epochs 1 \
  --train-batch-size 16 \
  --eval-batch-size 32 \
  --gradient-accumulation-steps 1 \
  --torch-threads 4 \
  --output-dir artifacts/week3
```

Runner memvalidasi checksum raw split, menghapus overlap train terhadap validation/test, menegakkan split disjointness, menyimpan model terbaik, dan menghasilkan metrik validation secara deterministik. Gunakan `--allow-test-evaluation` hanya setelah split manifest diberi freeze condition eksplisit.
