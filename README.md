# SVARA AI

SVARA AI adalah fondasi *Feedback Intelligence and Decision Support System* untuk mengubah feedback teks berbahasa Indonesia menjadi analisis sentiment, candidate aspect/topic, issue, dan insight yang dapat dibaca melalui dashboard.

Repository ini baru menyelesaikan fondasi Minggu 1. Runtime AI pada tahap ini menggunakan `MockAnalysisPipeline` yang deterministik agar kontrak dan alur lintas layer dapat diuji tanpa mengunduh model besar. IndoBERT, Sentence Transformer, dan BERTopic akan diintegrasikan pada tahap eksperimen berikutnya.

## Struktur

```text
contracts/       JSON Schema dan fixture lintas layanan
backend/         FastAPI API, mock pipeline, in-memory repository, tests
frontend/        Next.js App Router + Tailwind wireframe runnable
supabase/        konfigurasi lokal, migration awal, seed scenario
docs/            PRD, architecture, schema, contract, wireframe, test plan
scripts/         pemeriksaan fondasi Week 1
```

## Prasyarat

- Python 3.11+
- Node.js 20+
- npm 10+
- Supabase CLI hanya diperlukan untuk menjalankan migration secara lokal

## Menjalankan backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
uvicorn app.main:app --reload
```

Untuk mengaktifkan parser XLSX, install extra data: `python -m pip install -e '.[dev,data]'`.

API tersedia di `http://127.0.0.1:8000/api/v1` dan OpenAPI di `/docs`.

## Menjalankan frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend tersedia di `http://localhost:3000`. Salin `frontend/.env.example` menjadi `.env.local` jika ingin mengganti URL backend.

## Verifikasi Week 1

```bash
python3 scripts/check_week1.py
cd backend && python -m pytest
cd frontend && npm run typecheck && npm run build
```

Migration Supabase dijalankan terpisah karena membutuhkan PostgreSQL/Supabase lokal:

```bash
supabase start
supabase db reset
```

## Status keamanan

Header `X-User-ID` hanya merupakan authentication stub untuk development dan test. Production harus menggantinya dengan validasi Supabase JWT. File upload pada skeleton disimpan di memory; private Supabase Storage menjadi pekerjaan implementasi berikutnya.

## Dokumen fondasi

- [AI contract](docs/SVARA_AI_AI_CONTRACT.md)
- [API contract](docs/SVARA_AI_API_CONTRACT.md)
- [Acceptance criteria dan test plan](docs/SVARA_AI_ACCEPTANCE_CRITERIA.md)
- [Wireframes](docs/SVARA_AI_WIREFRAMES.md)
- [Development guide](docs/SVARA_AI_DEVELOPMENT_GUIDE.md)
