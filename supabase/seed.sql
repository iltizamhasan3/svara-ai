-- Local seed scenario. Create a confirmed Supabase Auth user with email
-- demo@svara.local first; this seed intentionally never creates passwords.

insert into public.profiles (id, display_name)
select id, 'SVARA Demo User'
from auth.users
where email = 'demo@svara.local'
on conflict (id) do update set display_name = excluded.display_name;

with owner as (
  select id from auth.users where email = 'demo@svara.local' limit 1
)
insert into public.datasets (
  user_id, original_filename, storage_path, file_type, row_count,
  column_count, columns_json, file_size_bytes, content_hash
)
select
  owner.id,
  'seed_feedback.csv',
  owner.id::text || '/seed/seed_feedback.csv',
  'csv',
  3,
  2,
  '["review", "date"]'::jsonb,
  null,
  'week1-seed'
from owner
where not exists (
  select 1 from public.datasets d
  where d.user_id = owner.id and d.content_hash = 'week1-seed'
);

with seeded_dataset as (
  select id
  from public.datasets
  where content_hash = 'week1-seed'
  order by created_at desc
  limit 1
), seed_rows(source_row_number, raw_text, feedback_date) as (
  values
    (1, 'aplikasinya bagus tetapi otp tidak masuk', '2026-08-01'::timestamptz),
    (2, 'loading sangat lambat dan sering crash', '2026-08-02'::timestamptz),
    (3, 'fiturnya lengkap dan mudah digunakan', '2026-08-03'::timestamptz)
)
insert into public.feedback_items (dataset_id, source_row_number, raw_text, normalized_text, feedback_date)
select seeded_dataset.id, seed_rows.source_row_number, seed_rows.raw_text, lower(seed_rows.raw_text), seed_rows.feedback_date
from seeded_dataset cross join seed_rows
on conflict (dataset_id, source_row_number) do nothing;
