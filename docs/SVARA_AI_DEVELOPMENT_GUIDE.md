# SVARA AI — Development Guide Week 1

## Environment

| Component | Baseline |
|---|---|
| Python | 3.11+ |
| Node.js | 20+ |
| Backend | FastAPI + Pydantic |
| Frontend | Next.js App Router + React + Tailwind |
| Database target | PostgreSQL via Supabase |
| Auth target | Supabase Auth |
| Local API auth stub | `X-User-ID` UUID; development/test only |

Parser XLSX dan Pandas berada pada optional dependency extra `data`; quality gate Week 1 tidak membutuhkan model atau dependency AI berat.

## Branching strategy

Branch feature dari `main` dengan pola:

```text
feat/<scope>-<short-name>
fix/<scope>-<short-name>
docs/<scope>-<short-name>
test/<scope>-<short-name>
```

Pull request wajib memiliki ringkasan perubahan, command test, dan catatan migration apabila ada.

## Ownership

- Project Lead + AI/NLP: `contracts/`, `backend/app/ai/`, model decision record, evaluation artifact.
- Backend + Data: `backend/app/api/`, `backend/app/services/`, `backend/app/repositories/`, `supabase/`.
- Frontend + UI/UX: `frontend/`, wireframe dan design token.
- Integration + QA: `backend/tests/`, `contracts/fixtures/`, acceptance matrix, smoke checks.

## Environment variables

Backend memakai default in-memory pada Week 1. Production/Supabase akan membutuhkan:

```text
APP_ENV=development
MAX_DATASET_ROWS=10000
MAX_UPLOAD_BYTES=10485760
MAX_CSV_FIELD_LENGTH=100000
SUPABASE_URL=
SUPABASE_ANON_KEY=
SUPABASE_SERVICE_ROLE_KEY=  # backend only, never expose to frontend
```

Frontend:

```text
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000/api/v1
```

## Quality gates

```bash
python3 scripts/check_week1.py
cd backend && python -m pytest -q
cd frontend && npm run typecheck && npm run build
```

## Known Week 1 boundaries

- Upload belum memakai Supabase Storage.
- Repository masih in-memory.
- Mock pipeline bukan model produksi.
- `BackgroundTasks` bukan durable queue.
- Supabase migration membutuhkan target Supabase/PostgreSQL dengan `auth.users`.
