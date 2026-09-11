# SVARA AI — Acceptance Criteria dan Test Plan Week 1

## Week 1 acceptance

### Product dan AI

- Problem statement, MVP scope, non-goals, dan user flow tersedia di PRD.
- Pipeline ditulis eksplisit: preprocessing → unit splitting → sentiment/embedding → topic discovery → aggregation.
- LLM hanya menerima aggregate snapshot.
- AI contract memiliki satu schema kanonik, fixture valid, dan fixture invalid.
- Sentiment memakai tiga label yang konsisten.
- Outlier topic memakai `topic_id = -1`.
- Denominator, trend, dataset limit, model version, dan fallback telah diputuskan.

### Backend dan data

- FastAPI dapat start dan menyediakan OpenAPI.
- Endpoint health, upload, create/start analysis, status, dashboard, dan history terdaftar.
- Pydantic memvalidasi request/response.
- Mock pipeline menghasilkan output deterministic sesuai AI contract.
- Migration awal memuat rantai dataset → feedback item → analysis unit → sentiment/topic → metrics/insight.
- Ownership dataset dan analysis tidak dapat dilewati oleh user lain.

### Frontend dan UX

- Next.js App Router dan Tailwind terpasang.
- Wireframe upload mencakup file selection, preview, mapping, validation, loading, dan start.
- Wireframe dashboard mencakup overview, sentiment, topic, issue, trend optional, dan insight failure state.
- TypeScript memakai tipe dashboard yang sama dengan API contract.

### QA dan workflow

- Environment, branch naming, dan command test terdokumentasi.
- Test mencakup payload invalid, malformed UUID, ownership, dan flow mock end-to-end.
- Week 1 milestone dapat diverifikasi dari artefak repo, bukan hanya checkbox.

## Test matrix

| Area | Skenario | Expected |
|---|---|---|
| Contract | confidence < 0 atau > 1 | Ditolak validation error |
| Contract | sentiment di luar tiga label | Ditolak validation error |
| Contract | topic id < -1 | Ditolak validation error |
| Upload | extension selain CSV/XLSX | Ditolak `400` |
| Upload | file kosong / tanpa row | Ditolak `400` |
| Analysis | feedback column tidak ada | Ditolak `400` |
| Analysis | UUID malformed | Ditolak `422` |
| Ownership | user B membaca analysis user A | `404` |
| Lifecycle | pending → processing → completed | Status dan dashboard tersedia |
| LLM fallback | insight gagal | Core dashboard tetap tersedia |
| Frontend | typecheck dan production build | Lulus tanpa model AI |
| Database | migration dan RLS definitions | Struktur dan policy terdeteksi |

## Command verification

```bash
python3 scripts/check_week1.py
cd backend && python -m pytest -q
cd frontend && npm run typecheck
cd frontend && npm run build
```
