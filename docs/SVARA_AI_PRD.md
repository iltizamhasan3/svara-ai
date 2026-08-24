# SVARA AI Product Requirements Document

**Document:** Product Requirements Document  
**Product:** SVARA AI  
**Version:** 1.0  
**Status:** Development Baseline  
**Last Updated:** 24 August 2026  

---

## 1. Product Overview

**SVARA AI: Sistem Visualisasi dan Analitik Respons Aspirasi berbasis Artificial Intelligence** adalah prototipe *web-based Feedback Intelligence and Decision Support System* yang mengubah kumpulan feedback berbentuk teks menjadi informasi terstruktur untuk mendukung evaluasi layanan.

Pengguna mengunggah dataset berformat CSV atau XLSX, memilih kolom yang berisi feedback dan tanggal jika tersedia, kemudian sistem menjalankan analisis sentimen, penemuan kelompok aspek secara otomatis, ekstraksi isu, agregasi statistik, analisis tren, dan penyusunan ringkasan insight.

SVARA AI tidak mengharuskan pengguna menentukan daftar aspek secara manual. Kelompok aspek ditemukan secara otomatis dari pola semantik dalam dataset menggunakan sentence embedding dan BERTopic. Hasil cluster direpresentasikan menggunakan kata kunci utama dari c-TF-IDF atau KeyBERT.

Large Language Model atau LLM hanya digunakan pada tahap akhir untuk menghasilkan **Insight Summary** dari hasil analisis yang sudah teragregasi. LLM bukan mesin klasifikasi sentimen, bukan pembentuk cluster, dan bukan sumber utama hasil analisis.

---

## 2. Problem Statement

Instansi pemerintah, kampus, dan penyedia layanan digital menerima feedback pengguna melalui survei, ulasan aplikasi, formulir evaluasi, dan kanal pengaduan. Ketika jumlah feedback mencapai ratusan atau ribuan, analisis manual menjadi lambat, sulit dilakukan secara konsisten, dan sering kali hanya menghasilkan informasi berupa rating agregat.

Padahal, satu feedback dapat memuat beberapa opini sekaligus. Pengguna dapat menilai satu bagian layanan secara positif dan bagian lain secara negatif. Kondisi ini menyebabkan isu dominan, pola keluhan, dan perubahan sentimen sulit dikenali secara cepat.

SVARA AI dirancang untuk mengubah data feedback tidak terstruktur menjadi informasi yang dapat dibaca melalui dashboard, sehingga pengguna dapat melihat distribusi sentimen, kelompok aspek yang dominan, isu utama, tren, dan ringkasan hasil analisis.

---

## 3. Product Goals

SVARA AI memiliki tujuan utama sebagai berikut:

1. Mengurangi kebutuhan membaca feedback satu per satu.
2. Mengklasifikasikan sentimen feedback ke dalam Positive, Neutral, dan Negative.
3. Menemukan kelompok aspek atau topik dominan secara otomatis tanpa taxonomy manual.
4. Menghubungkan hasil sentiment dengan kelompok aspek pada tingkat sentence atau clause.
5. Menemukan isu dan kata kunci yang paling sering muncul.
6. Menyajikan hasil analisis melalui dashboard yang mudah dibaca.
7. Menampilkan tren sentimen jika dataset menyediakan informasi tanggal.
8. Menghasilkan ringkasan insight menggunakan LLM berdasarkan data agregat.
9. Mendukung penggunaan lintas domain selama input utama berupa feedback teks berbahasa Indonesia.

---

## 4. Non-Goals

Fitur berikut tidak termasuk scope MVP:

- Scraping Google Play, YouTube, X, Instagram, atau media sosial lainnya.
- Monitoring feedback secara real-time.
- Integrasi langsung dengan kanal pengaduan eksternal.
- Pelatihan Large Language Model dari awal.
- LLM untuk klasifikasi setiap feedback.
- LLM untuk penamaan aspek.
- Manual taxonomy management yang kompleks.
- Automatic policy recommendation.
- Automatic response generation kepada pengguna.
- Multi-tenant enterprise administration.
- Role Based Access Control kompleks.
- Mobile native application.
- Real-time collaborative dashboard.

Fitur tersebut dapat menjadi pengembangan setelah MVP.

---

## 5. Target Users

### 5.1 Service Evaluator

Pengguna yang bertanggung jawab mengevaluasi kualitas layanan berdasarkan feedback pengguna.

Contoh:

- unit pelayanan publik,
- pengelola aplikasi,
- unit evaluasi kampus,
- customer experience team,
- organisasi yang mengelola survei.

### 5.2 Analyst

Pengguna yang ingin memahami pola dalam dataset feedback tanpa membangun pipeline NLP sendiri.

### 5.3 Management or Decision Maker

Pengguna yang terutama membaca hasil dashboard dan Insight Summary untuk memahami isu yang membutuhkan perhatian.

---

## 6. Primary User Flow

```text
Login
  ↓
Create New Analysis
  ↓
Upload CSV/XLSX
  ↓
Preview Dataset
  ↓
Map Feedback Column
  ↓
Map Date Column (Optional)
  ↓
Start Analysis
  ↓
Processing
  ↓
Dashboard
  ↓
Review Insight
```

---

## 7. End-to-End Analysis Pipeline

```text
CSV / XLSX
    ↓
File Validation
    ↓
Column Mapping
    ↓
Preprocessing
    ↓
Sentence / Clause Splitting
    ↓
┌───────────────────────────────┐
│                               │
▼                               ▼
Sentiment Analysis         Automatic Aspect Discovery
IndoBERT                   Sentence Transformer
                                ↓
                             BERTopic
                                ↓
                             HDBSCAN
                                ↓
                             c-TF-IDF
│                               │
└───────────────┬───────────────┘
                ↓
        Aspect-Level Sentiment
                ↓
          Issue Extraction
                ↓
      Statistics and Trends
                ↓
         LLM Insight Summary
                ↓
             Dashboard
```

### Important Methodological Note

BERTopic menghasilkan semantic topic clusters. Dalam SVARA AI, cluster tersebut digunakan sebagai **candidate aspects** atau kelompok aspek yang berulang dalam feedback.

SVARA AI tidak mengklaim BERTopic sebagai model Aspect-Based Sentiment Analysis penuh. Sentiment pada kelompok aspek diperoleh dengan menghubungkan hasil sentiment sentence atau clause dengan cluster semantik yang ditemukan.

---

## 8. Functional Requirements

### FR-01 Authentication

Sistem harus memungkinkan pengguna masuk ke aplikasi sebelum menggunakan fitur analisis.

**MVP implementation:** Supabase Auth.

**Acceptance criteria:**

- Pengguna dapat login.
- Pengguna hanya dapat melihat analysis miliknya sendiri.
- Session pengguna tetap tersedia selama masih valid.

---

### FR-02 Create Analysis

Pengguna harus dapat membuat satu analysis baru.

Data minimal:

- analysis name,
- dataset file,
- feedback column,
- optional date column.

---

### FR-03 Upload Dataset

Sistem harus menerima:

- `.csv`
- `.xlsx`

Sistem harus menolak:

- file type selain format yang didukung,
- file kosong,
- dataset tanpa row,
- dataset yang gagal diparse.

**MVP recommended limit:** maksimal 10.000 feedback per analysis.

Batas dapat disesuaikan berdasarkan kemampuan deployment.

---

### FR-04 Dataset Preview

Setelah upload, sistem harus menampilkan preview dataset.

Minimal:

- nama kolom,
- beberapa row awal,
- total row,
- jumlah kolom.

Tujuannya agar pengguna dapat memastikan dataset yang dipilih sudah benar sebelum analisis.

---

### FR-05 Column Mapping

Pengguna harus memilih satu kolom sebagai:

**Required**
- Feedback Text

**Optional**
- Date

Sistem harus mencegah analisis apabila feedback column belum dipilih.

---

### FR-06 Preprocessing

Sistem harus melakukan preprocessing sebelum inference.

Minimal:

- handle null value,
- trim whitespace,
- cleaning dasar karakter,
- normalisasi line break,
- duplicate handling sesuai konfigurasi,
- sentence atau clause segmentation.

Preprocessing tidak boleh menghilangkan informasi penting yang diperlukan model sentiment.

---

### FR-07 Sentiment Classification

Setiap unit analisis harus memiliki:

- sentiment label,
- confidence score.

Label MVP:

- `positive`
- `neutral`
- `negative`

Model utama:

**IndoBERT yang di-fine-tune menggunakan IndoNLU SmSA.**

---

### FR-08 Automatic Aspect Discovery

Sistem harus menemukan kelompok aspek secara otomatis dari feedback.

Pipeline:

```text
Text Unit
  ↓
Sentence Embedding
  ↓
BERTopic
  ↓
UMAP
  ↓
HDBSCAN
  ↓
c-TF-IDF
  ↓
Topic Cluster
```

Hasil cluster minimal memiliki:

- topic ID,
- jumlah unit,
- top keywords,
- representative text.

Topic outlier dari HDBSCAN dapat memiliki nilai `-1`.

---

### FR-09 Aspect Display Label

SVARA AI tidak menggunakan LLM untuk penamaan aspek.

Nama yang ditampilkan pada dashboard dibuat dari top keywords.

Contoh:

```text
otp / login / verifikasi
loading / lambat / crash
fitur / menu / fungsi
```

Default MVP:

**3 keyword dengan bobot tertinggi dari setiap cluster.**

---

### FR-10 Aspect-Level Sentiment

Sistem harus menghubungkan hasil sentiment dengan topic cluster.

Contoh:

```text
"OTP tidak pernah masuk"
sentiment = negative
topic = otp / login / verifikasi
```

Backend kemudian menghitung distribusi sentiment untuk setiap topic.

---

### FR-11 Issue Extraction

Sistem harus menghasilkan isu atau kata kunci representatif.

Pendekatan:

- c-TF-IDF,
- KeyBERT jika diperlukan,
- frequency count sebagai pendukung.

Hasil minimal:

- issue phrase atau keyword,
- frequency,
- topic,
- dominant sentiment jika relevan.

---

### FR-12 Statistical Aggregation

Backend harus menghitung:

- total feedback,
- total analysis units,
- positive count,
- neutral count,
- negative count,
- sentiment percentage,
- feedback count per topic,
- sentiment per topic,
- ranking topic,
- top negative issues.

---

### FR-13 Trend Analysis

Jika date column tersedia, sistem harus menghasilkan trend sentiment berdasarkan waktu.

MVP aggregation:

- daily atau
- weekly

tergantung rentang tanggal dan jumlah data.

Jika tidak ada date column, seluruh komponen trend harus disembunyikan atau menampilkan *Not Available*.

---

### FR-14 Dashboard

Dashboard MVP harus memiliki:

#### A. Overview Cards

- Total Feedback
- Positive %
- Neutral %
- Negative %

#### B. Overall Sentiment

Visualisasi distribusi sentiment.

#### C. Sentiment Trend

Hanya jika date column tersedia.

#### D. Aspect or Topic Clusters

Menampilkan:

- keyword label,
- jumlah mention,
- dominant sentiment.

#### E. Sentiment per Aspect

Menampilkan Positive, Neutral, dan Negative untuk setiap topic cluster.

#### F. Top Issues

Menampilkan isu yang paling dominan, terutama isu dengan sentiment negatif.

#### G. AI Insight Summary

Menampilkan ringkasan bahasa natural dari hasil agregasi.

---

### FR-15 Insight Summary

LLM hanya menerima **data agregat**, bukan seluruh raw feedback.

Contoh input:

```json
{
  "total_feedback": 5000,
  "sentiment": {
    "positive": 52,
    "neutral": 17,
    "negative": 31
  },
  "top_topics": [
    {
      "keywords": ["otp", "login", "verifikasi"],
      "negative_percentage": 71
    }
  ],
  "top_issues": [
    "otp tidak masuk",
    "gagal login",
    "loading lambat"
  ]
}
```

LLM menghasilkan ringkasan singkat yang menjelaskan:

- kecenderungan sentiment secara keseluruhan,
- cluster dengan masalah terbesar,
- isu yang dominan,
- perubahan tren jika tersedia.

LLM tidak diperbolehkan membuat angka baru yang tidak terdapat pada input.

---

### FR-16 Analysis Status

Setiap analysis memiliki status:

```text
pending
processing
completed
failed
```

Frontend harus dapat melakukan polling status selama proses berjalan.

Jika gagal, user harus menerima error message yang dapat dipahami.

---

### FR-17 Analysis History

Pengguna dapat melihat daftar analysis sebelumnya.

Minimal:

- analysis name,
- dataset name,
- created date,
- status,
- total feedback.

---

### FR-18 Export

**MVP optional, post-freeze preferred.**

Jika tersedia, sistem dapat mengekspor:

- hasil analysis dalam CSV,
- hasil per feedback,
- topic assignment,
- sentiment result.

PDF report bukan prioritas MVP.

---

## 9. Dashboard Information Architecture

```text
Dashboard
│
├── Analysis Header
│   ├── Analysis Name
│   ├── Dataset Name
│   └── Analysis Date
│
├── Overview
│   ├── Total Feedback
│   ├── Positive
│   ├── Neutral
│   └── Negative
│
├── Overall Sentiment
│
├── Sentiment Trend
│
├── Aspect / Topic Analysis
│   ├── Topic Keywords
│   ├── Topic Size
│   └── Sentiment Distribution
│
├── Top Issues
│
└── AI Insight Summary
```

---

## 10. Suggested Dashboard Layout

```text
┌───────────────────────────────────────────────────────────┐
│ SVARA AI                           Dataset: Review App     │
├───────────────────────────────────────────────────────────┤
│ Total       Positive        Neutral          Negative     │
│ 5,000         52%             17%              31%        │
├──────────────────────────┬────────────────────────────────┤
│ Overall Sentiment        │ Sentiment Trend                │
│ [Chart]                  │ [Chart]                        │
├──────────────────────────┴────────────────────────────────┤
│ Sentiment per Aspect / Topic                              │
│                                                          │
│ OTP / Login / Verifikasi               71% Negative      │
│ Loading / Lambat / Crash               63% Negative      │
│ Fitur / Menu / Fungsi                  68% Positive      │
├──────────────────────────┬────────────────────────────────┤
│ Top Issues               │ Topic Clusters                 │
│ OTP gagal                │ OTP / Login / Verifikasi       │
│ Gagal login              │ Loading / Lambat / Crash       │
│ Loading lambat           │ Fitur / Menu / Fungsi          │
├──────────────────────────┴────────────────────────────────┤
│ AI Insight Summary                                        │
│ Mayoritas feedback memiliki sentimen positif...           │
└───────────────────────────────────────────────────────────┘
```

---

## 11. Data Inputs

### Required

Dataset harus memiliki minimal satu kolom teks.

Contoh:

```csv
review
"aplikasinya membantu tetapi otp tidak pernah masuk"
"tampilannya bagus dan mudah digunakan"
"sering crash ketika membuka menu pembayaran"
```

### Optional

Dataset dapat memiliki kolom:

```csv
review,date
"otp tidak masuk","2026-08-01"
"loading sangat lambat","2026-08-02"
```

Kolom selain hasil mapping dapat disimpan sebagai metadata jika diperlukan.

---

## 12. Data Outputs

Contoh internal result:

```json
{
  "feedback_id": "uuid",
  "text": "otp tidak pernah masuk",
  "units": [
    {
      "text": "otp tidak pernah masuk",
      "sentiment": "negative",
      "confidence": 0.94,
      "topic_id": 3,
      "topic_keywords": [
        "otp",
        "login",
        "verifikasi"
      ]
    }
  ]
}
```

---

## 13. Non-Functional Requirements

### NFR-01 Usability

User harus dapat menyelesaikan:

```text
Upload → Mapping → Analyze → Dashboard
```

tanpa bantuan developer.

### NFR-02 Performance

Target MVP:

- preview file: kurang dari 5 detik untuk dataset normal,
- API dashboard: kurang dari 2 detik setelah analysis selesai,
- analysis dapat berjalan sebagai background job.

Inference time tergantung ukuran dataset dan infrastructure.

### NFR-03 Reliability

Jika LLM tidak tersedia:

- analysis utama tetap completed,
- dashboard tetap dapat digunakan,
- AI Insight menampilkan status unavailable.

### NFR-04 Security

- Setiap user hanya dapat membaca analysis miliknya sendiri.
- File upload divalidasi.
- Password tidak disimpan oleh aplikasi sendiri jika menggunakan Supabase Auth.
- API key LLM hanya disimpan pada server.
- Raw file tidak dikirim ke LLM.

### NFR-05 Privacy

Dataset user dapat mengandung feedback sensitif.

MVP harus:

- membatasi akses berdasarkan user,
- tidak mengekspos original file secara public,
- menggunakan private storage bucket,
- tidak mengirim seluruh feedback ke LLM.

### NFR-06 Reproducibility

Setiap analysis menyimpan informasi:

- sentiment model version,
- topic pipeline version,
- preprocessing version,
- prompt version untuk insight jika digunakan.

---

## 14. Technology Stack

### Frontend

- Next.js
- React
- Tailwind CSS
- Chart library seperti Recharts

### Backend

- FastAPI
- Python
- Pandas
- Pydantic

### NLP / AI

- Hugging Face Transformers
- IndoBERT
- IndoNLU SmSA
- Sentence Transformers
- BERTopic
- UMAP
- HDBSCAN
- c-TF-IDF
- KeyBERT optional
- External LLM API hanya untuk Insight Summary

### Database and Auth

- PostgreSQL
- Supabase
- Supabase Auth
- Supabase Storage

---

## 15. MVP Definition of Done

SVARA AI dianggap mencapai MVP jika pengguna dapat menjalankan alur berikut tanpa bantuan developer:

```text
Login
  ↓
Upload CSV/XLSX
  ↓
Preview
  ↓
Column Mapping
  ↓
Start Analysis
  ↓
Sentiment Analysis
  ↓
Automatic Aspect Discovery
  ↓
Issue Extraction
  ↓
Statistics and Trend
  ↓
Insight Summary
  ↓
Dashboard
```

Dengan kondisi:

- file berhasil diupload,
- model sentiment menghasilkan output,
- cluster topic berhasil dibuat,
- statistik dashboard sesuai hasil,
- dashboard tetap berfungsi jika LLM gagal,
- analysis history dapat dibuka kembali.

---

## 16. MVP Acceptance Checklist

- [ ] Login berjalan.
- [ ] CSV dapat diupload.
- [ ] XLSX dapat diupload.
- [ ] Dataset preview bekerja.
- [ ] Feedback column dapat dipilih.
- [ ] Date column bersifat optional.
- [ ] Analysis job dapat dibuat.
- [ ] Sentiment prediction tersimpan.
- [ ] BERTopic menghasilkan topic cluster.
- [ ] Top keywords tersimpan.
- [ ] Sentiment per topic dapat dihitung.
- [ ] Top issues dapat ditampilkan.
- [ ] Trend tersedia untuk dataset bertanggal.
- [ ] Dashboard menggunakan API aktual.
- [ ] Insight Summary dapat dihasilkan.
- [ ] Dashboard tidak gagal jika LLM gagal.
- [ ] Analysis history tersedia.
- [ ] User tidak dapat membaca analysis milik user lain.

---

## 17. Success Metrics

### Model Metrics

Sentiment classification:

- Accuracy
- Precision
- Recall
- Macro F1
- Confusion Matrix

External testing dapat menggunakan IGAR.

### Topic Discovery Evaluation

Karena automatic aspect discovery bersifat unsupervised, evaluasi tidak menggunakan accuracy secara langsung tanpa ground-truth aspect label.

Metode evaluasi MVP:

- qualitative inspection,
- representative feedback inspection,
- cluster coherence jika digunakan,
- outlier ratio,
- topic diversity.

### Product Metrics

Untuk prototype:

- successful analysis rate,
- failed analysis rate,
- processing time,
- dashboard load time,
- UAT task completion.

---

## 18. Key Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| BERTopic menghasilkan cluster kurang jelas | Topic sulit diinterpretasi | Tuning min topic size, embedding model, preprocessing, tampilkan representative text |
| Banyak outlier HDBSCAN | Banyak feedback tidak masuk topic | Evaluasi parameter dan ukuran dataset |
| Sentiment mixed dalam satu feedback | Hasil sentiment tidak akurat per aspect | Sentence atau clause splitting sebelum analysis |
| Dataset terlalu kecil | Topic discovery tidak stabil | Berikan minimum recommended rows dan warning |
| Dataset terlalu besar | Processing lama | Batch inference dan configurable row limit |
| LLM timeout | Insight tidak tersedia | Insight dibuat optional, dashboard tetap completed |
| API contract berubah | Integrasi terlambat | Freeze contract sejak awal dan gunakan mock data |
| Model terlalu berat untuk deployment | Server lambat | Gunakan model caching dan dedicated backend service |

---

## 19. Development Priority

### P0: Required for MVP

- Authentication
- Upload
- Preview
- Column Mapping
- Sentiment
- Automatic Aspect Discovery
- Topic Keywords
- Sentiment per Topic
- Statistics
- Dashboard
- Analysis Job Status

### P1: Strongly Recommended

- Trend
- Top Issues
- Analysis History
- LLM Insight Summary
- Representative Feedback

### P2: Post-MVP

- Export
- Advanced filtering
- PDF report
- Advanced topic management
- Social data integration

---

## 20. Scope Freeze

Core feature baru tidak ditambahkan setelah **Week 10 MVP Freeze**.

Week 11 sampai Week 14 difokuskan pada:

- bug fixing,
- model evaluation,
- usability improvement,
- performance improvement,
- UAT,
- documentation,
- presentation.
