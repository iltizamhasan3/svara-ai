"use client";

import {ChangeEvent, useState} from "react";

const previewRows = [
  {review: "aplikasinya bagus tetapi otp tidak masuk", date: "2026-08-01"},
  {review: "loading sangat lambat dan sering crash", date: "2026-08-02"},
  {review: "fiturnya lengkap dan mudah digunakan", date: "2026-08-03"},
];

export default function NewAnalysisPage() {
  const [filename, setFilename] = useState<string | null>(null);

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    setFilename(event.target.files?.[0]?.name ?? null);
  }

  return (
    <main className="min-h-screen bg-canvas px-6 py-8 text-ink">
      <div className="mx-auto max-w-5xl">
        <header className="flex items-center justify-between">
          <a href="/" className="text-xl font-bold tracking-tight">SVARA<span className="text-brand-600">.</span></a>
          <a href="/" className="text-sm font-semibold text-muted hover:text-ink">Cancel</a>
        </header>

        <section className="mt-14 max-w-2xl">
          <p className="eyebrow">New analysis</p>
          <h1 className="mt-3 text-4xl font-bold tracking-tight">Upload your feedback dataset.</h1>
          <p className="mt-4 text-lg leading-8 text-muted">Pastikan kolom feedback dan tanggal sudah mudah dikenali sebelum analisis dimulai.</p>
        </section>

        <section className="panel mt-10 p-6 sm:p-8">
          <label htmlFor="dataset-file" className="flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed border-brand-200 bg-brand-50/60 px-6 py-12 text-center transition hover:border-brand-500">
            <span className="flex h-12 w-12 items-center justify-center rounded-full bg-white text-2xl text-brand-600 shadow-sm">↑</span>
            <span className="mt-4 text-lg font-semibold">{filename ?? "Drop CSV or XLSX here"}</span>
            <span className="mt-2 text-sm text-muted">or choose a file · maximum 10.000 feedback</span>
            <input id="dataset-file" type="file" accept=".csv,.xlsx" className="sr-only" onChange={handleFileChange} />
          </label>

          <div className="mt-8 grid gap-5 sm:grid-cols-3">
            <label className="text-sm font-semibold sm:col-span-2">
              Analysis name
              <input defaultValue="Review Agustus" className="mt-2 w-full rounded-xl border border-slate-300 px-4 py-3 font-normal outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-100" />
            </label>
            <div className="hidden sm:block" />
            <label className="text-sm font-semibold">
              Feedback column
              <select defaultValue="review" className="mt-2 w-full rounded-xl border border-slate-300 bg-white px-4 py-3 font-normal outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-100">
                <option value="review">review</option>
              </select>
            </label>
            <label className="text-sm font-semibold">
              Date column <span className="font-normal text-muted">(optional)</span>
              <select defaultValue="date" className="mt-2 w-full rounded-xl border border-slate-300 bg-white px-4 py-3 font-normal outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-100">
                <option value="date">date</option>
                <option value="none">No date</option>
              </select>
            </label>
          </div>

          <div className="mt-8 overflow-hidden rounded-2xl border border-slate-200">
            <div className="flex items-center justify-between bg-slate-50 px-4 py-3">
              <div>
                <p className="text-sm font-semibold">Dataset preview</p>
                <p className="mt-1 text-xs text-muted">3 rows · 2 columns</p>
              </div>
              <span className="rounded-full bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-700">ready</span>
            </div>
            <div className="overflow-x-auto">
              <table className="min-w-full text-left text-sm">
                <thead className="border-t border-slate-200 text-xs uppercase tracking-wide text-muted">
                  <tr><th className="px-4 py-3 font-semibold">review</th><th className="px-4 py-3 font-semibold">date</th></tr>
                </thead>
                <tbody>
                  {previewRows.map((row) => <tr key={row.date} className="border-t border-slate-100"><td className="max-w-md px-4 py-3">{row.review}</td><td className="px-4 py-3 text-muted">{row.date}</td></tr>)}
                </tbody>
              </table>
            </div>
          </div>

          <div className="mt-8 flex items-center justify-between gap-4">
            <p className="text-sm text-muted">Analysis berjalan sebagai job dan dapat dipantau dari dashboard.</p>
            <button type="button" className="shrink-0 rounded-xl bg-brand-600 px-5 py-3 text-sm font-semibold text-white hover:bg-brand-700">Start analysis</button>
          </div>
        </section>
      </div>
    </main>
  );
}
