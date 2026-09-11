export default function HomePage() {
  return (
    <main className="min-h-screen bg-canvas px-6 py-8 text-ink">
      <div className="mx-auto flex max-w-6xl flex-col gap-16">
        <header className="flex items-center justify-between">
          <a href="/" className="text-xl font-bold tracking-tight">
            SVARA<span className="text-brand-600">.</span>
          </a>
          <a href="/login" className="text-sm font-semibold text-muted hover:text-ink">
            Sign in
          </a>
        </header>

        <section className="grid items-center gap-10 lg:grid-cols-[1.1fr_0.9fr]">
          <div>
            <p className="eyebrow">Feedback intelligence</p>
            <h1 className="mt-4 max-w-3xl text-5xl font-bold leading-[1.04] tracking-[-0.04em] sm:text-6xl">
              Turn every response into a clearer next step.
            </h1>
            <p className="mt-6 max-w-xl text-lg leading-8 text-muted">
              SVARA AI membantu tim membaca sentiment, menemukan candidate aspect, dan memprioritaskan issue dari feedback berbahasa Indonesia.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <a href="/analyses/new" className="rounded-full bg-brand-600 px-6 py-3 text-sm font-semibold text-white transition hover:bg-brand-700">
                Mulai analisis
              </a>
              <a href="/analyses/demo" className="rounded-full border border-slate-300 bg-white px-6 py-3 text-sm font-semibold text-ink transition hover:border-brand-500">
                Lihat dashboard demo
              </a>
            </div>
          </div>

          <div className="panel relative overflow-hidden p-6">
            <div className="absolute right-0 top-0 h-32 w-32 rounded-full bg-brand-100 blur-2xl" />
            <div className="relative">
              <div className="flex items-center justify-between">
                <div>
                  <p className="eyebrow">Latest analysis</p>
                  <h2 className="mt-2 text-2xl font-bold">Review Agustus</h2>
                </div>
                <span className="rounded-full bg-brand-100 px-3 py-1 text-xs font-semibold text-brand-700">completed</span>
              </div>
              <div className="mt-8 grid grid-cols-3 gap-3">
                {[
                  ["Positive", "52%", "bg-emerald-50 text-emerald-700"],
                  ["Neutral", "17%", "bg-amber-50 text-amber-700"],
                  ["Negative", "31%", "bg-rose-50 text-rose-700"],
                ].map(([label, value, tone]) => (
                  <div key={label} className={`rounded-xl p-3 ${tone}`}>
                    <p className="text-xs font-medium">{label}</p>
                    <p className="mt-2 text-xl font-bold">{value}</p>
                  </div>
                ))}
              </div>
              <div className="mt-6 rounded-xl bg-slate-50 p-4">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-semibold">Top issue</span>
                  <span className="text-muted">256 mentions</span>
                </div>
                <p className="mt-2 font-medium">OTP tidak masuk</p>
                <div className="mt-4 h-2 overflow-hidden rounded-full bg-slate-200">
                  <div className="h-full w-[72%] rounded-full bg-brand-500" />
                </div>
              </div>
            </div>
          </div>
        </section>

        <footer className="border-t border-slate-200 py-6 text-sm text-muted">
          Week 1 foundation · Next.js + FastAPI + Supabase-ready
        </footer>
      </div>
    </main>
  );
}
