# SVARA AI — Wireframes Week 1

Wireframe ini menjadi acuan layout awal, bukan final visual design.

## Upload Dataset — `/analyses/new`

```text
┌────────────────────────────────────────────────────────────────────┐
│ SVARA AI                                      [user] [Sign out]      │
├────────────────────────────────────────────────────────────────────┤
│ New analysis                                                        │
│ Analisis feedback layanan dalam beberapa langkah.                  │
│                                                                    │
│ ┌────────────────────────────────────────────────────────────────┐ │
│ │  Upload CSV / XLSX                                             │ │
│ │  Drag and drop file atau [Choose file]                         │ │
│ │  Maksimal 10.000 feedback                                     │ │
│ └────────────────────────────────────────────────────────────────┘ │
│                                                                    │
│ Analysis name [________________________________________]           │
│ Feedback column [review                         v]                 │
│ Date column     [date                           v] optional        │
│                                                                    │
│ Dataset preview                                                    │
│ ┌────────────────────────────────────────────────────────────────┐ │
│ │ review                                      │ date             │ │
│ │ aplikasinya bagus ...                      │ 2026-08-01       │ │
│ └────────────────────────────────────────────────────────────────┘ │
│                                                   [Start analysis] │
└────────────────────────────────────────────────────────────────────┘
```

State wajib: empty, dragging, upload loading, invalid file, preview ready, mapping validation error, dan upload success.

## Dashboard — `/analyses/{analysisId}`

```text
┌────────────────────────────────────────────────────────────────────┐
│ SVARA AI / Review Agustus                         [Back] [Export]   │
│ Status: completed                                                   │
├──────────────┬──────────────┬──────────────┬──────────────────────┤
│ Total units  │ Positive     │ Neutral      │ Negative             │
│ 5,000        │ 52.0%        │ 17.0%        │ 31.0%                │
├────────────────────────────────────┬───────────────────────────────┤
│ Overall sentiment                   │ Sentiment trend (optional)    │
│ [bar / donut chart]                 │ [daily line chart]            │
├────────────────────────────────────┼───────────────────────────────┤
│ Candidate aspects / topics          │ Sentiment per aspect           │
│ [keywords + size + outlier]        │ [stacked bar chart]            │
├────────────────────────────────────┴───────────────────────────────┤
│ Top issues                                                            │
│ 1. otp tidak masuk                              256 mentions         │
│ 2. loading lambat                               184 mentions         │
├────────────────────────────────────────────────────────────────────┤
│ AI insight                                                           │
│ Ringkasan berbasis aggregate metrics. [loading / failed fallback]    │
└────────────────────────────────────────────────────────────────────┘
```

Dashboard harus tetap menampilkan core metrics apabila LLM insight berstatus `failed` atau `skipped`.
