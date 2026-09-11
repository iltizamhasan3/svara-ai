create extension if not exists pgcrypto;

create type analysis_status as enum ('pending', 'processing', 'completed', 'failed');
create type sentiment_label as enum ('positive', 'neutral', 'negative');
create type insight_status as enum ('pending', 'completed', 'failed', 'skipped');

create table public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  display_name text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table public.datasets (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references public.profiles(id) on delete cascade,
  original_filename text not null,
  storage_path text not null,
  file_type text not null check (file_type in ('csv', 'xlsx')),
  row_count integer,
  column_count integer,
  columns_json jsonb,
  file_size_bytes bigint,
  content_hash text,
  created_at timestamptz not null default now(),
  unique (id, user_id)
);

create table public.analyses (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references public.profiles(id) on delete cascade,
  dataset_id uuid not null,
  name text not null,
  feedback_column text not null,
  date_column text,
  status analysis_status not null default 'pending',
  current_stage text,
  progress smallint check (progress between 0 and 100),
  sentiment_model_name text,
  sentiment_model_version text,
  embedding_model_name text,
  topic_model_version text,
  preprocessing_version text,
  topic_config jsonb,
  error_message text,
  started_at timestamptz,
  completed_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (id, user_id),
  unique (id, dataset_id),
  constraint analyses_dataset_owner_fk
    foreign key (dataset_id, user_id)
    references public.datasets(id, user_id)
    on delete cascade
);

create table public.feedback_items (
  id uuid primary key default gen_random_uuid(),
  dataset_id uuid not null references public.datasets(id) on delete cascade,
  source_row_number integer not null,
  raw_text text,
  normalized_text text,
  feedback_date timestamptz,
  metadata jsonb,
  created_at timestamptz not null default now(),
  unique (dataset_id, source_row_number),
  unique (id, dataset_id)
);

create table public.analysis_units (
  id uuid primary key default gen_random_uuid(),
  analysis_id uuid not null references public.analyses(id) on delete cascade,
  dataset_id uuid not null,
  feedback_item_id uuid not null references public.feedback_items(id) on delete cascade,
  unit_index integer not null,
  raw_text text not null,
  normalized_text text,
  created_at timestamptz not null default now(),
  unique (analysis_id, feedback_item_id, unit_index),
  unique (id, analysis_id),
  constraint analysis_units_analysis_dataset_fk
    foreign key (analysis_id, dataset_id)
    references public.analyses(id, dataset_id)
    on delete cascade,
  constraint analysis_units_feedback_dataset_fk
    foreign key (feedback_item_id, dataset_id)
    references public.feedback_items(id, dataset_id)
    on delete cascade
);

create table public.sentiment_predictions (
  id uuid primary key default gen_random_uuid(),
  analysis_unit_id uuid not null unique references public.analysis_units(id) on delete cascade,
  label sentiment_label not null,
  confidence double precision not null check (confidence >= 0 and confidence <= 1),
  positive_probability double precision check (positive_probability between 0 and 1),
  neutral_probability double precision check (neutral_probability between 0 and 1),
  negative_probability double precision check (negative_probability between 0 and 1),
  model_version text,
  created_at timestamptz not null default now()
);

create table public.topic_clusters (
  id uuid primary key default gen_random_uuid(),
  analysis_id uuid not null references public.analyses(id) on delete cascade,
  topic_id integer not null,
  display_label text,
  unit_count integer not null default 0,
  representative_texts jsonb,
  created_at timestamptz not null default now(),
  unique (analysis_id, topic_id),
  unique (id, analysis_id)
);

create table public.topic_keywords (
  id uuid primary key default gen_random_uuid(),
  topic_cluster_id uuid not null references public.topic_clusters(id) on delete cascade,
  keyword text not null,
  weight double precision,
  rank integer not null,
  created_at timestamptz not null default now(),
  unique (topic_cluster_id, rank)
);

create table public.unit_topic_assignments (
  id uuid primary key default gen_random_uuid(),
  analysis_unit_id uuid not null unique,
  analysis_id uuid not null,
  topic_cluster_id uuid,
  topic_probability double precision check (topic_probability between 0 and 1),
  created_at timestamptz not null default now(),
  constraint unit_topic_assignments_unit_analysis_fk
    foreign key (analysis_unit_id, analysis_id)
    references public.analysis_units(id, analysis_id)
    on delete cascade,
  constraint unit_topic_assignments_topic_analysis_fk
    foreign key (topic_cluster_id, analysis_id)
    references public.topic_clusters(id, analysis_id)
    on delete cascade
);

create table public.issues (
  id uuid primary key default gen_random_uuid(),
  analysis_id uuid not null references public.analyses(id) on delete cascade,
  topic_cluster_id uuid references public.topic_clusters(id) on delete cascade,
  issue_text text not null,
  frequency integer not null default 0 check (frequency >= 0),
  score double precision,
  dominant_sentiment sentiment_label,
  created_at timestamptz not null default now(),
  constraint issues_topic_analysis_fk
    foreign key (topic_cluster_id, analysis_id)
    references public.topic_clusters(id, analysis_id)
    on delete cascade
);

create table public.analysis_metrics (
  id uuid primary key default gen_random_uuid(),
  analysis_id uuid not null unique references public.analyses(id) on delete cascade,
  total_feedback integer not null default 0 check (total_feedback >= 0),
  total_units integer not null default 0 check (total_units >= 0),
  positive_count integer not null default 0 check (positive_count >= 0),
  neutral_count integer not null default 0 check (neutral_count >= 0),
  negative_count integer not null default 0 check (negative_count >= 0),
  positive_percentage double precision not null default 0 check (positive_percentage between 0 and 100),
  neutral_percentage double precision not null default 0 check (neutral_percentage between 0 and 100),
  negative_percentage double precision not null default 0 check (negative_percentage between 0 and 100),
  topic_metrics jsonb not null default '[]'::jsonb,
  trend_metrics jsonb not null default '[]'::jsonb,
  calculated_at timestamptz not null default now()
);

create table public.ai_insights (
  id uuid primary key default gen_random_uuid(),
  analysis_id uuid not null unique references public.analyses(id) on delete cascade,
  status insight_status not null default 'pending',
  provider text,
  model_name text,
  prompt_version text,
  input_snapshot jsonb,
  summary text,
  error_message text,
  generated_at timestamptz,
  created_at timestamptz not null default now()
);

create index idx_datasets_user_id on public.datasets(user_id);
create index idx_datasets_created_at on public.datasets(created_at desc);
create index idx_analyses_user_id on public.analyses(user_id);
create index idx_analyses_dataset_id on public.analyses(dataset_id);
create index idx_analyses_status on public.analyses(status);
create index idx_analyses_created_at on public.analyses(created_at desc);
create index idx_feedback_items_dataset_id on public.feedback_items(dataset_id);
create index idx_feedback_items_feedback_date on public.feedback_items(feedback_date);
create index idx_analysis_units_analysis_id on public.analysis_units(analysis_id);
create index idx_analysis_units_feedback_item_id on public.analysis_units(feedback_item_id);
create index idx_sentiment_predictions_label on public.sentiment_predictions(label);
create index idx_topic_clusters_analysis_id on public.topic_clusters(analysis_id);
create index idx_topic_keywords_cluster on public.topic_keywords(topic_cluster_id);
create index idx_unit_topic_assignments_cluster on public.unit_topic_assignments(topic_cluster_id);
create index idx_unit_topic_assignments_analysis on public.unit_topic_assignments(analysis_id);
create index idx_issues_analysis_id on public.issues(analysis_id);
create index idx_issues_topic_cluster_id on public.issues(topic_cluster_id);
create index idx_issues_frequency on public.issues(frequency desc);

alter table public.profiles enable row level security;
alter table public.datasets enable row level security;
alter table public.analyses enable row level security;
alter table public.feedback_items enable row level security;
alter table public.analysis_units enable row level security;
alter table public.sentiment_predictions enable row level security;
alter table public.topic_clusters enable row level security;
alter table public.topic_keywords enable row level security;
alter table public.unit_topic_assignments enable row level security;
alter table public.issues enable row level security;
alter table public.analysis_metrics enable row level security;
alter table public.ai_insights enable row level security;

create policy "Users can view own profile"
on public.profiles for select using (auth.uid() = id);

create policy "Users can view own datasets"
on public.datasets for select using (auth.uid() = user_id);

create policy "Users can insert own datasets"
on public.datasets for insert with check (auth.uid() = user_id);

create policy "Users can view own analyses"
on public.analyses for select using (auth.uid() = user_id);

create policy "Users can insert own analyses"
on public.analyses for insert
with check (
  auth.uid() = user_id
  and exists (
    select 1 from public.datasets d
    where d.id = dataset_id and d.user_id = auth.uid()
  )
);

comment on table public.analysis_units is 'Accessed through backend service role after parent analysis ownership has been verified.';
comment on table public.ai_insights is 'LLM input_snapshot must contain aggregate data only; never raw uploaded feedback.';
