# SVARA AI — AI Contract Week 1

Dokumen ini adalah kontrak kanonik antara AI/NLP layer dan backend. JSON Schema yang dapat divalidasi berada di [`contracts/ai-result.schema.json`](../contracts/ai-result.schema.json).

## Keputusan baseline

| Area | Keputusan Week 1 |
|---|---|
| Sentiment | Tiga label: `positive`, `neutral`, `negative`. Model target adalah IndoBERT yang di-fine-tune pada SmSA. Checkpoint final ditetapkan setelah eksperimen Week 2–3. |
| External evaluation | IGAR dipakai sebagai external/domain validation setelah baseline SmSA tersedia; tidak digunakan untuk tuning confirmation set. |
| Unit | Sentence splitting adalah default. Clause splitting sederhana dapat ditambahkan jika pola campuran membutuhkan resolusi lebih kecil. |
| Topic | Sentence Transformer + BERTopic dengan UMAP, HDBSCAN, dan c-TF-IDF. Output diperlakukan sebagai candidate aspect/topic, bukan full ABSA. |
| Outlier | `topic_id = -1` untuk unit yang tidak masuk cluster. Outlier tidak dibuat sebagai `topic_clusters` biasa. |
| Assignment | MVP menyimpan satu topic utama per analysis unit. Keterbatasan multi-aspect dicatat untuk post-MVP. |
| Overall metric | Prediksi dan persentase dihitung pada level analysis unit. `total_feedback` tetap ditampilkan sebagai jumlah row input dan `total_units` ditampilkan sebagai denominator analitik. |
| Trend | Agregasi harian berdasarkan tanggal valid dalam timezone UTC. Row dengan tanggal invalid tidak masuk trend dan menghasilkan warning. |
| Dataset limit | Hard limit konfigurasi MVP: 10.000 feedback per analysis. |
| Versioning | Simpan `model_name@revision`, `preprocessing-vN`, dan `topic-vN` pada metadata analysis. |
| LLM | Hanya menerima aggregate snapshot; kegagalan LLM tidak menggagalkan core analysis. |

## Canonical unit result

```json
{
  "unit_id": "00000000-0000-4000-8000-000000000001",
  "source_row_number": 1,
  "text": "otp tidak masuk",
  "sentiment": "negative",
  "confidence": 0.94,
  "probabilities": {
    "positive": 0.02,
    "neutral": 0.04,
    "negative": 0.94
  },
  "topic_id": 1,
  "keywords": [
    {"keyword": "otp", "weight": 0.182, "rank": 1}
  ]
}
```

`unit_id` digunakan untuk traceability dari hasil model ke `analysis_units`. `source_row_number` membantu debugging sebelum persistence database tersedia.

## Model interface

```python
class AnalysisPipeline(Protocol):
    def run(self, rows: list[dict[str, object]], feedback_column: str) -> list[AIResult]: ...
```

`MockAnalysisPipeline` adalah adapter Week 1 yang deterministik. Ia hanya membuktikan bentuk output dan alur integrasi; hasilnya tidak boleh dipakai sebagai metrik model produksi.

## Quality gates Week 2–3

- label mapping dataset ditulis eksplisit;
- split train/validation/test bebas leakage dan duplicate;
- Macro F1 serta metrik per kelas dilaporkan;
- confusion matrix dan contoh error dianalisis;
- topic dievaluasi dengan outlier rate, keyword usefulness, dan review manual;
- confirmation set tidak digunakan untuk memilih hyperparameter.

## Referensi dataset/model

- SmSA: [IndoNLP/nusa-crowd dataset adapter](https://github.com/IndoNLP/nusa-crowd/blob/master/nusacrowd/nusa_datasets/smsa/smsa.py)
- IndoNLU paper: [ACL Anthology](https://aclanthology.org/2020.aacl-main.85/)
- IGAR: [DOI 10.1016/j.dib.2026.112708](https://doi.org/10.1016/j.dib.2026.112708)
