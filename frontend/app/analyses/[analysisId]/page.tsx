import type {DashboardResponse} from "../../../lib/types";

const demoDashboard: DashboardResponse = {
  analysis: {id: "demo", name: "Review Agustus", status: "completed"},
  overview: {
    total_feedback: 5000,
    total_units: 5000,
    positive: 2600,
    neutral: 850,
    negative: 1550,
    positive_percentage: 52,
    neutral_percentage: 17,
    negative_percentage: 31,
  },
  topics: [
    {topic_id: 1, label: "OTP / LOGIN / VERIFIKASI", unit_count: 730, sentiment: {positive: 12, neutral: 17, negative: 71}},
    {topic_id: 2, label: "PERFORMA / LOADING / CRASH", unit_count: 540, sentiment: {positive: 18, neutral: 14, negative: 68}},
  ],
  top_issues: [
    {text: "otp tidak masuk", frequency: 256, topic_id: 1},
    {text: "loading lambat", frequency: 184, topic_id: 2},
    {text: "verifikasi gagal", frequency: 139, topic_id: 1},
  ],
  trend: [],
  insight: {status: "skipped", summary: null},
};

function SentimentBar({dashboard}: {dashboard: DashboardResponse}) {
  return (
    <div className="mt-6 space-y-3">
      {[
        ["Positive", dashboard.overview.positive_percentage, "bg-emerald-500"],
        ["Neutral", dashboard.overview.neutral_percentage, "bg-amber-400"],
        ["Negative", dashboard.overview.negative_percentage, "bg-rose-500"],
      ].map(([label, value, color]) => (
        <div key={label as string}>
          <div className="mb-1 flex justify-between text-sm"><span className="font-medium">{label}</span><span className="text-muted">{value}%</span></div>
          <div className="h-2 rounded-full bg-slate-100"><div className={`h-2 rounded-full ${color}`} style={{width: `${value}%`}} /></div>
        </div>
      ))}
    </div>
  );
}

export default async function AnalysisDashboardPage({params}: {params: Promise<{analysisId: string}>}) {
  const {analysisId} = await params;
  const dashboard = {...demoDashboard, analysis: {...demoDashboard.analysis, id: analysisId}};

  return (
    <main className="min-h-screen bg-canvas px-6 py-8 text-ink">
      <div className="mx-auto max-w-6xl">
        <header className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-4"><a href="/" className="text-xl font-bold tracking-tight">SVARA<span className="text-brand-600">.</span></a><span className="text-slate-300">/</span><span className="text-sm font-medium text-muted">Analysis dashboard</span></div>
          <a href="/analyses/new" className="rounded-full border border-slate-300 bg-white px-4 py-2 text-sm font-semibold hover:border-brand-500">New analysis</a>
        </header>

        <section className="mt-14 flex flex-wrap items-end justify-between gap-6">
          <div><p className="eyebrow">Completed analysis</p><h1 className="mt-3 text-4xl font-bold tracking-tight">{dashboard.analysis.name}</h1><p className="mt-3 text-sm text-muted">Core metrics siap dibaca · analysis id {dashboard.analysis.id}</p></div>
          <span className="rounded-full bg-brand-100 px-3 py-1.5 text-xs font-semibold text-brand-700">{dashboard.analysis.status}</span>
        </section>

        <section className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {[
            ["Total feedback", dashboard.overview.total_feedback.toLocaleString("id-ID"), "rows"],
            ["Total units", dashboard.overview.total_units.toLocaleString("id-ID"), "analysis units"],
            ["Positive", `${dashboard.overview.positive_percentage}%`, `${dashboard.overview.positive.toLocaleString("id-ID")} units`],
            ["Negative", `${dashboard.overview.negative_percentage}%`, `${dashboard.overview.negative.toLocaleString("id-ID")} units`],
          ].map(([label, value, caption]) => <div key={label} className="panel p-5"><p className="text-sm font-medium text-muted">{label}</p><p className="mt-3 text-3xl font-bold tracking-tight">{value}</p><p className="mt-2 text-xs text-muted">{caption}</p></div>)}
        </section>

        <section className="mt-4 grid gap-4 lg:grid-cols-2">
          <div className="panel p-6"><div className="flex items-center justify-between"><div><p className="eyebrow">Overview</p><h2 className="mt-2 text-xl font-bold">Overall sentiment</h2></div><span className="text-xs text-muted">unit-level</span></div><SentimentBar dashboard={dashboard} /></div>
          <div className="panel p-6"><p className="eyebrow">Trend</p><h2 className="mt-2 text-xl font-bold">Sentiment trend</h2><div className="mt-6 flex min-h-28 items-center justify-center rounded-xl border border-dashed border-slate-300 bg-slate-50 text-sm text-muted">No date trend in this Week 1 mock</div></div>
        </section>

        <section className="mt-4 grid gap-4 lg:grid-cols-[1.1fr_0.9fr]">
          <div className="panel p-6"><p className="eyebrow">Discovery</p><h2 className="mt-2 text-xl font-bold">Candidate aspects</h2><div className="mt-5 space-y-3">{dashboard.topics.map((topic) => <div key={topic.topic_id} className="rounded-xl border border-slate-200 p-4"><div className="flex items-center justify-between gap-3"><span className="font-semibold">{topic.label}</span><span className="text-sm text-muted">{topic.unit_count} units</span></div><div className="mt-3 flex h-2 overflow-hidden rounded-full bg-slate-100"><div className="bg-emerald-500" style={{width: `${topic.sentiment.positive}%`}} /><div className="bg-amber-400" style={{width: `${topic.sentiment.neutral}%`}} /><div className="bg-rose-500" style={{width: `${topic.sentiment.negative}%`}} /></div></div>)}</div></div>
          <div className="panel p-6"><p className="eyebrow">Prioritize</p><h2 className="mt-2 text-xl font-bold">Top issues</h2><div className="mt-5 space-y-4">{dashboard.top_issues.map((issue, index) => <div key={issue.text} className="flex items-start gap-3"><span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-brand-50 text-xs font-bold text-brand-700">{index + 1}</span><div className="min-w-0 flex-1"><p className="font-semibold">{issue.text}</p><p className="mt-1 text-sm text-muted">{issue.frequency} mentions</p></div></div>)}</div></div>
        </section>

        <section className="panel mt-4 flex flex-wrap items-center justify-between gap-5 p-6"><div><p className="eyebrow">Optional generative layer</p><h2 className="mt-2 text-xl font-bold">AI insight summary</h2><p className="mt-2 text-sm text-muted">LLM belum diaktifkan pada skeleton Week 1. Core dashboard tetap tersedia.</p></div><span className="rounded-full bg-slate-100 px-3 py-1.5 text-xs font-semibold text-muted">{dashboard.insight.status}</span></section>
      </div>
    </main>
  );
}
