export default function LoginPage() {
  return (
    <main className="flex min-h-screen items-center justify-center bg-canvas px-6 py-12 text-ink">
      <section className="panel w-full max-w-md p-8">
        <p className="eyebrow">SVARA AI</p>
        <h1 className="mt-3 text-3xl font-bold tracking-tight">Welcome back</h1>
        <p className="mt-3 text-sm leading-6 text-muted">Authentication akan menggunakan Supabase Auth pada tahap integrasi.</p>
        <div className="mt-8 space-y-4">
          <label className="block text-sm font-semibold">
            Email
            <input type="email" placeholder="you@example.com" className="mt-2 w-full rounded-xl border border-slate-300 px-4 py-3 font-normal outline-none transition focus:border-brand-500 focus:ring-2 focus:ring-brand-100" />
          </label>
          <label className="block text-sm font-semibold">
            Password
            <input type="password" placeholder="••••••••" className="mt-2 w-full rounded-xl border border-slate-300 px-4 py-3 font-normal outline-none transition focus:border-brand-500 focus:ring-2 focus:ring-brand-100" />
          </label>
          <button type="button" className="w-full rounded-xl bg-brand-600 px-4 py-3 text-sm font-semibold text-white hover:bg-brand-700">
            Continue
          </button>
        </div>
        <a href="/" className="mt-6 block text-center text-sm font-semibold text-muted hover:text-ink">Back to home</a>
      </section>
    </main>
  );
}
