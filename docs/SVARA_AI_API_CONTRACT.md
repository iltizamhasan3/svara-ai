# SVARA AI — API Contract Week 1

Base URL: `/api/v1`

API ini menjadi sumber kebenaran untuk frontend dan backend. Supabase Auth adalah jalur authentication production. Skeleton Week 1 menggunakan header `X-User-ID` berisi UUID sebagai development stub agar ownership dapat dites tanpa Supabase.

## Error envelope

```json
{
  "error": {
    "code": "validation_error",
    "message": "feedback_column is required",
    "details": []
  }
}
```

Status umum: `400` input bisnis tidak valid, `401` authentication tidak ada/invalid, `404` resource tidak ditemukan atau bukan milik user, `409` state transition tidak valid, `413` file/dataset melebihi limit, `422` payload gagal validasi schema, dan `500` unexpected error.

## Endpoints

### `GET /health`

Public liveness check.

Response `200`:

```json
{"status": "ok", "service": "svara-ai-api", "version": "0.1.0"}
```

### `POST /datasets/upload`

Auth required. Multipart field: `file`.

Rules:

- hanya `.csv` dan `.xlsx`;
- file tidak boleh kosong;
- dataset maksimal 10.000 row;
- response berisi metadata, nama kolom, dan maksimal lima row preview;
- Week 1 menyimpan file di memory; Supabase Storage private digunakan pada tahap berikutnya.

Response `201`:

```json
{
  "dataset_id": "uuid",
  "filename": "reviews.csv",
  "file_type": "csv",
  "row_count": 3,
  "column_count": 2,
  "columns": ["review", "date"],
  "preview": [{"review": "...", "date": "2026-08-01"}]
}
```

### `POST /analyses`

Auth required. `feedback_column` wajib; `date_column` optional dan harus merupakan nama kolom yang tersedia.

Request:

```json
{
  "dataset_id": "uuid",
  "name": "Review Agustus",
  "feedback_column": "review",
  "date_column": "date"
}
```

Response `201`:

```json
{"analysis_id": "uuid", "status": "pending"}
```

### `POST /analyses/{analysis_id}/start`

Auth required. Memindahkan `pending` atau `failed` menjadi `processing` dan menjalankan worker mock pada skeleton.

Response `202`:

```json
{"analysis_id": "uuid", "status": "processing"}
```

Start ulang analysis yang sedang `processing` menghasilkan `409`. Analysis `completed` bersifat idempotent dan mengembalikan status completed.

### `GET /analyses/{analysis_id}/status`

Auth required. Dipoll frontend setiap 2–5 detik sampai status `completed` atau `failed`.

Response `200`:

```json
{
  "analysis_id": "uuid",
  "status": "processing",
  "progress": 55,
  "stage": "topic_discovery",
  "error_message": null
}
```

### `GET /analyses/{analysis_id}/dashboard`

Auth required. Tersedia saat core analysis `completed`. LLM insight dapat berstatus `failed` atau `skipped` tanpa membuat dashboard gagal.

Response mengikuti [`contracts/dashboard.schema.json`](../contracts/dashboard.schema.json).

### `GET /analyses?page=1&limit=20`

Auth required. Mengembalikan analysis milik current user saja. `page` minimal 1 dan `limit` berada pada rentang 1–100.

Response `200`:

```json
{
  "items": [],
  "page": 1,
  "limit": 20,
  "total": 0
}
```

## State machine

```text
pending → processing → completed
                   ↘ failed → processing (retry)
```

Pada production, `BackgroundTasks` harus diganti durable queue/worker. Skeleton Week 1 sengaja menggunakan background task in-process untuk membuktikan contract dan polling.
