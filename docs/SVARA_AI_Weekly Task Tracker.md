# SVARA AI — Weekly Task Tracker

**Target utama:** MVP selesai dan *freeze* pada **Minggu 10**  
**Minggu 11–14:** evaluasi, perbaikan, dokumentasi, dan presentasi

> Status implementasi: Week 1 foundation selesai. Artefak yang dapat diverifikasi tersedia di `contracts/`, `backend/`, `frontend/`, `supabase/`, dan `scripts/check_week1.py`.

### Pembagian Role
1. **Project Lead + AI/NLP Engineer**
2. **Backend + Data Engineer**
3. **Frontend + UI/UX Developer**
4. **Integration + Quality Assurance Engineer**

---

# Minggu 1 — Project Foundation & Contract

### Project Lead + AI/NLP Engineer
- [x] Finalisasi *problem statement* SVARA AI
- [x] Finalisasi scope MVP
- [x] Menentukan pipeline AI/NLP
- [x] Mempelajari IndoNLU SmSA dan IGAR
- [x] Menentukan pendekatan IndoBERT untuk sentiment analysis
- [x] Menentukan pendekatan BERTopic untuk automatic aspect discovery
- [x] Menentukan output contract AI

**Output AI Contract:**
```json
{
  "unit_id": "uuid",
  "source_row_number": 1,
  "text": "otp tidak masuk",
  "sentiment": "negative",
  "confidence": 0.92,
  "probabilities": {
    "positive": 0.02,
    "neutral": 0.06,
    "negative": 0.92
  },
  "topic_id": 2,
  "keywords": [
    {"keyword": "otp", "weight": 0.182, "rank": 1},
    {"keyword": "login", "weight": 0.164, "rank": 2}
  ]
}
```

### Backend + Data Engineer
- [x] Setup project FastAPI
- [x] Menentukan struktur folder backend
- [x] Mendesain database awal
- [x] Menentukan struktur dataset internal
- [x] Membuat rancangan endpoint API
- [x] Menentukan format komunikasi backend dengan AI

### Frontend + UI/UX Developer
- [x] Setup Next.js/React
- [x] Setup Tailwind CSS
- [x] Membuat design system sederhana
- [x] Membuat wireframe
- [x] Mendesain halaman Upload Dataset
- [x] Mendesain halaman Dashboard

### Integration + QA Engineer
- [x] Menyusun API contract frontend-backend
- [x] Menyusun acceptance criteria MVP
- [x] Menentukan skenario testing utama
- [x] Setup repository dan branching strategy
- [x] Menentukan development environment

### Milestone Minggu 1
- [x] Scope MVP terkunci
- [x] User flow disepakati
- [x] AI contract disepakati
- [x] API contract disepakati
- [x] Wireframe tersedia

---

# Minggu 2 — Dataset & Upload Pipeline

### Project Lead + AI/NLP Engineer
- [ ] Download IndoNLU SmSA
- [ ] Download/sample IGAR
- [ ] Melakukan EDA SmSA
- [ ] Mengecek distribusi label
- [ ] Membangun preprocessing awal
- [ ] Menyiapkan notebook eksperimen IndoBERT

### Backend + Data Engineer
- [ ] Implementasi upload CSV
- [ ] Implementasi upload XLSX
- [ ] Validasi ekstensi file
- [ ] Parsing data dengan Pandas
- [ ] Menampilkan metadata dataset
- [ ] Implementasi dataset preview
- [ ] Menangani missing value dasar

### Frontend + UI/UX Developer
- [ ] Implementasi halaman upload
- [ ] Drag & drop file
- [ ] Dataset preview table
- [ ] Error state file tidak valid
- [ ] Loading state upload
- [ ] Implementasi tampilan metadata dataset

### Integration + QA Engineer
- [ ] Test CSV valid
- [ ] Test XLSX valid
- [ ] Test file kosong
- [ ] Test format file tidak didukung
- [ ] Test dataset dengan missing values
- [ ] Dokumentasikan bug

### Milestone Minggu 2
- [ ] **CSV/XLSX → Upload → Preview berhasil end-to-end**

---

# Minggu 3 — Sentiment Model & Column Mapping

### Project Lead + AI/NLP Engineer
- [ ] Fine-tuning IndoBERT dengan SmSA
- [ ] Evaluasi validation set
- [ ] Menghasilkan Accuracy
- [ ] Menghasilkan Precision
- [ ] Menghasilkan Recall
- [ ] Menghasilkan Macro F1
- [ ] Membuat confusion matrix
- [ ] Menyimpan model versi pertama

### Backend + Data Engineer
- [ ] Implementasi column mapping
- [ ] Mapping kolom feedback
- [ ] Mapping kolom tanggal opsional
- [ ] Simpan konfigurasi dataset
- [ ] Membuat analysis job
- [ ] Membuat mock AI inference endpoint

### Frontend + UI/UX Developer
- [ ] Halaman column mapping
- [ ] Dropdown pemilihan kolom feedback
- [ ] Dropdown tanggal opsional
- [ ] Preview hasil mapping
- [ ] Tombol Start Analysis
- [ ] Halaman processing/loading

### Integration + QA Engineer
- [ ] Test mapping kolom benar
- [ ] Test mapping kolom salah
- [ ] Test dataset tanpa tanggal
- [ ] Test dataset dengan tanggal
- [ ] Test mock analysis flow

### Milestone Minggu 3
- [ ] Sentiment model v1 tersedia
- [ ] Upload → Mapping → Start Analysis berjalan

---

# Minggu 4 — Automatic Aspect Discovery

### Project Lead + AI/NLP Engineer
- [ ] Implementasi sentence/clause splitting
- [ ] Menentukan Sentence Transformer
- [ ] Menghasilkan text embeddings
- [ ] Eksperimen BERTopic
- [ ] Konfigurasi UMAP
- [ ] Konfigurasi HDBSCAN
- [ ] Menghasilkan cluster awal
- [ ] Menghasilkan keyword dengan c-TF-IDF
- [ ] Mengevaluasi kualitas cluster secara manual

### Backend + Data Engineer
- [ ] Membuat service untuk AI inference
- [ ] Integrasi mock aspect result
- [ ] Menyiapkan struktur penyimpanan topic/aspect
- [ ] Menyimpan topic ID
- [ ] Menyimpan keyword per cluster
- [ ] Menyimpan confidence sentiment

### Frontend + UI/UX Developer
- [ ] Membuat dashboard dengan mock data
- [ ] Overall Sentiment chart
- [ ] Summary cards
- [ ] Placeholder aspect/topic cluster
- [ ] Placeholder top issues

### Integration + QA Engineer
- [ ] Test AI output schema
- [ ] Test backend menerima mock AI result
- [ ] Test dashboard dengan mock API
- [ ] Validasi konsistensi contract antar-layer

### Milestone Minggu 4
- [ ] IndoBERT berjalan
- [ ] BERTopic menghasilkan cluster
- [ ] Dashboard mock sudah terlihat

---

# Minggu 5 — Sentiment End-to-End Integration

### Project Lead + AI/NLP Engineer
- [ ] Finalisasi preprocessing sentiment
- [ ] Export model untuk inference
- [ ] Membuat fungsi batch inference
- [ ] Test inference pada data IGAR
- [ ] Analisis error classification

### Backend + Data Engineer
- [ ] Integrasi IndoBERT sebenarnya
- [ ] Batch processing feedback
- [ ] Simpan prediction per feedback
- [ ] Hitung total positive
- [ ] Hitung total neutral
- [ ] Hitung total negative
- [ ] Hitung persentase sentiment
- [ ] Endpoint dashboard sentiment

### Frontend + UI/UX Developer
- [ ] Hubungkan dashboard ke API asli
- [ ] Overall sentiment visualization
- [ ] Total feedback card
- [ ] Positive/Neutral/Negative cards
- [ ] Loading state analysis

### Integration + QA Engineer
- [ ] Test CSV → IndoBERT → Database
- [ ] Cocokkan hasil model dengan API
- [ ] Test dataset kecil
- [ ] Test dataset menengah
- [ ] Logging error

### Milestone Minggu 5
- [ ] **CSV → Sentiment Analysis → Dashboard bekerja end-to-end**

> Jika milestone ini belum tercapai, lakukan evaluasi scope sebelum menambah fitur.

---

# Minggu 6 — Aspect, Topic & Issue Analysis

### Project Lead + AI/NLP Engineer
- [ ] Finalisasi BERTopic pipeline
- [ ] Menghasilkan topic cluster otomatis
- [ ] Menghasilkan top keywords per cluster
- [ ] Integrasi c-TF-IDF
- [ ] Eksperimen KeyBERT jika diperlukan
- [ ] Menentukan representative feedback
- [ ] Menggabungkan sentiment dengan topic cluster

### Backend + Data Engineer
- [ ] Integrasi BERTopic ke pipeline
- [ ] Simpan topic assignment
- [ ] Simpan keywords
- [ ] Hitung jumlah feedback per topic
- [ ] Hitung sentiment per topic
- [ ] Ranking topic dominan
- [ ] Ranking negative issues

### Frontend + UI/UX Developer
- [ ] Aspect/topic cluster section
- [ ] Sentiment per aspect
- [ ] Top Issues component
- [ ] Representative keywords
- [ ] Tooltip/detail cluster

### Integration + QA Engineer
- [ ] Test BERTopic integration
- [ ] Test topic dengan dataset berbeda
- [ ] Cek apakah cluster kosong ditangani
- [ ] Test hasil sentiment per aspect
- [ ] Verifikasi ranking issue

### Milestone Minggu 6
- [ ] **Sentiment + Automatic Aspect Discovery + Top Issues berjalan**

---

# Minggu 7 — Statistics, Trend & Dashboard Completion

### Project Lead + AI/NLP Engineer
- [ ] Evaluasi hasil cluster
- [ ] Test pipeline pada domain berbeda
- [ ] Evaluasi eksternal sentiment menggunakan IGAR
- [ ] Dokumentasi hasil eksperimen
- [ ] Freeze metode inti AI

### Backend + Data Engineer
- [ ] Trend sentiment berdasarkan tanggal
- [ ] Grouping per periode
- [ ] Filtering sentiment
- [ ] Filtering aspect/topic
- [ ] Ranking issue
- [ ] Dashboard aggregation API
- [ ] Optimasi query database

### Frontend + UI/UX Developer
- [ ] Sentiment Trend chart
- [ ] Sentiment per Aspect chart
- [ ] Top Issues panel
- [ ] Filter dashboard
- [ ] Empty state jika tanggal tidak tersedia
- [ ] Dashboard responsive

### Integration + QA Engineer
- [ ] End-to-end test dashboard
- [ ] Test dataset tanpa tanggal
- [ ] Test dataset dengan tanggal
- [ ] Test filter
- [ ] Test beberapa ukuran layar

### Milestone Minggu 7
- [ ] **Dashboard utama SVARA AI selesai**

Dashboard minimal:
- [ ] Total feedback
- [ ] Overall sentiment
- [ ] Sentiment trend
- [ ] Aspect/topic clusters
- [ ] Sentiment per aspect
- [ ] Top issues

---

# Minggu 8 — LLM Insight Summary

### Project Lead + AI/NLP Engineer
- [ ] Menentukan data yang dikirim ke LLM
- [ ] Membuat prompt insight summary
- [ ] Pastikan raw dataset tidak dikirim seluruhnya
- [ ] Test beberapa variasi output
- [ ] Membatasi panjang hasil
- [ ] Menambahkan guardrail prompt

### Backend + Data Engineer
- [ ] Membuat LLM service
- [ ] Kirim data agregat ke LLM
- [ ] Simpan insight summary
- [ ] Handling API timeout
- [ ] Handling LLM failure
- [ ] Fallback ketika LLM tidak tersedia

### Frontend + UI/UX Developer
- [ ] AI Insight card
- [ ] Loading state insight
- [ ] Error/fallback state
- [ ] Integrasi insight ke dashboard

### Integration + QA Engineer
- [ ] Test insight dengan beberapa dataset
- [ ] Test kondisi LLM gagal
- [ ] Cek bahwa dashboard tetap bekerja tanpa LLM
- [ ] Cek konsistensi summary dengan statistik

### Milestone Minggu 8
- [ ] **Full end-to-end SVARA AI tersedia**

```text
Upload
→ Mapping
→ Analyze
→ Sentiment
→ Aspect
→ Issue
→ Statistics
→ LLM Summary
→ Dashboard
```

---

# Minggu 9 — Stabilization & System Testing

### Project Lead + AI/NLP Engineer
- [ ] Final evaluation sentiment
- [ ] Final external test IGAR
- [ ] Analisis confusion matrix
- [ ] Dokumentasi model
- [ ] Freeze parameter AI

### Backend + Data Engineer
- [ ] Performance improvement
- [ ] Batch optimization
- [ ] Error handling
- [ ] Database cleanup
- [ ] Logging
- [ ] Security dasar upload file

### Frontend + UI/UX Developer
- [ ] UI polishing
- [ ] Responsive design
- [ ] Loading states
- [ ] Error states
- [ ] Empty states
- [ ] Chart readability

### Integration + QA Engineer
- [ ] Functional testing
- [ ] Integration testing
- [ ] Regression testing
- [ ] UAT internal
- [ ] Buat daftar bug berdasarkan prioritas
- [ ] Test demo dataset

### Dataset Testing Minimum
- [ ] 100 rows
- [ ] 1.000 rows
- [ ] 5.000 rows
- [ ] dataset tanpa tanggal
- [ ] dataset dengan tanggal
- [ ] missing text
- [ ] duplicate feedback
- [ ] emoji
- [ ] typo/slang

### Milestone Minggu 9
- [ ] Tidak ada bug kritis
- [ ] Prototype siap menuju freeze

---

# Minggu 10 — MVP Freeze & Deployment

### Project Lead + AI/NLP Engineer
- [ ] Freeze model sentiment
- [ ] Freeze BERTopic configuration
- [ ] Freeze prompt LLM
- [ ] Siapkan hasil evaluasi model
- [ ] Siapkan dataset demo

### Backend + Data Engineer
- [ ] Final backend deployment
- [ ] Production environment
- [ ] Database production
- [ ] Final API documentation
- [ ] Performance check

### Frontend + UI/UX Developer
- [ ] Deploy frontend
- [ ] Final dashboard polishing
- [ ] Final responsive check
- [ ] Siapkan demo flow

### Integration + QA Engineer
- [ ] Production smoke test
- [ ] Final UAT
- [ ] Test seluruh user flow
- [ ] Final regression test
- [ ] Backup demo scenario

### Milestone Minggu 10
- [ ] **MVP FREEZE**
- [ ] Prototype deployed
- [ ] Siap didemonstrasikan
- [ ] Tidak menambah core feature baru

---

# Minggu 11 — Improvement

### Semua Role
- [ ] Review masukan dosen/pengguna
- [ ] Prioritaskan improvement yang tidak mengubah scope
- [ ] Perbaiki model jika ditemukan masalah penting
- [ ] Optimasi performa
- [ ] Polishing UI
- [ ] Tambahkan export jika waktu memungkinkan

### Milestone
- [ ] Prototype versi improvement

---

# Minggu 12 — Final Evaluation & UAT

### Project Lead + AI/NLP Engineer
- [ ] Finalisasi hasil evaluasi AI
- [ ] Interpretasi hasil eksperimen

### Backend + Data Engineer
- [ ] Bug fixing final
- [ ] Final performance testing

### Frontend + UI/UX Developer
- [ ] Final usability improvement
- [ ] Final screenshot sistem

### Integration + QA Engineer
- [ ] UAT eksternal
- [ ] Final test report
- [ ] Final bug verification

### Milestone
- [ ] Sistem final tervalidasi

---

# Minggu 13 — Documentation

### Project Lead + AI/NLP Engineer
- [ ] Menulis problem, metode, AI pipeline
- [ ] Menulis hasil evaluasi model
- [ ] Menulis limitations

### Backend + Data Engineer
- [ ] Dokumentasi architecture
- [ ] Dokumentasi database
- [ ] Dokumentasi API

### Frontend + UI/UX Developer
- [ ] Dokumentasi user flow
- [ ] Screenshot fitur
- [ ] Dokumentasi dashboard

### Integration + QA Engineer
- [ ] Dokumentasi testing
- [ ] UAT report
- [ ] Deployment guide

### Milestone
- [ ] Laporan final selesai

---

# Minggu 14 — Presentation & Final Demo

### Project Lead + AI/NLP Engineer
- [ ] Finalisasi slide
- [ ] Menjelaskan problem dan innovation
- [ ] Menjelaskan AI/NLP pipeline

### Backend + Data Engineer
- [ ] Menjelaskan arsitektur backend/data
- [ ] Backup deployment

### Frontend + UI/UX Developer
- [ ] Menjalankan demo aplikasi
- [ ] Menjelaskan dashboard

### Integration + QA Engineer
- [ ] Menjelaskan testing dan feasibility
- [ ] Menyiapkan backup video demo
- [ ] Test seluruh perangkat presentasi

### Semua Anggota
- [ ] Rehearsal presentasi
- [ ] Rehearsal demo
- [ ] Siapkan jawaban pertanyaan dosen
- [ ] Final submission

---

# Checkpoint Utama

- [ ] **Minggu 1:** Scope + contract selesai
- [ ] **Minggu 2:** Upload CSV/XLSX selesai
- [ ] **Minggu 3:** Sentiment model v1 selesai
- [ ] **Minggu 5:** CSV → Sentiment → Dashboard selesai
- [ ] **Minggu 6:** Aspect discovery + issue selesai
- [ ] **Minggu 7:** Dashboard utama selesai
- [ ] **Minggu 8:** Full pipeline termasuk LLM selesai
- [ ] **Minggu 9:** Testing & stabilization
- [ ] **Minggu 10:** **MVP FREEZE**
- [ ] **Minggu 11–14:** improvement, evaluasi, dokumentasi, final demo

## Definition of Done MVP

SVARA AI dianggap mencapai MVP apabila pengguna sudah dapat menjalankan alur berikut tanpa bantuan developer:

```text
Upload CSV/XLSX
        ↓
Preview Dataset
        ↓
Mapping Kolom
        ↓
Start Analysis
        ↓
Sentiment Analysis
        ↓
Automatic Aspect Discovery
        ↓
Issue Extraction
        ↓
Statistics & Trend
        ↓
AI Insight Summary
        ↓
Dashboard
```

Semua fitur di luar alur tersebut dianggap sebagai **fitur tambahan** dan tidak boleh menghambat target MVP pada minggu ke-10.
