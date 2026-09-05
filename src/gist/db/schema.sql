-- Gist web-demo persistence (Neon Postgres + pgvector).
--
-- What lives here and what does not: this schema holds everything *derived*
-- from a video — metadata, transcript, per-candidate embeddings, and the chat
-- history. The source media stays on disk with only its path recorded, because
-- a single 61-minute source is ~756 MB against ~1 MB of derived data, and
-- Postgres cannot serve HTTP byte ranges, so a blob column would break video
-- scrubbing in the browser and force a full download before playback.
--
-- The embeddings are the reason this schema exists at all. Encoding frames and
-- audio windows is query-independent and expensive (Whisper alone runs minutes
-- on an hour-long video); scoring them against a query is neither. Persisting
-- the encoder output once turns every later query into a vector comparison,
-- which is what makes "ingest first, ask afterwards" viable as a UX.

create extension if not exists vector;

-- ---------------------------------------------------------------- videos ----

create table if not exists videos (
    id                  uuid primary key default gen_random_uuid(),
    youtube_url         text        not null unique,
    youtube_id          text,
    title               text        not null,
    duration_seconds    double precision not null,
    thumbnail_url       text,
    -- Path to the retained source on local disk. Kept because evidence clips
    -- are cut per query with ffmpeg, and which span to cut is not known until
    -- the selector has run.
    source_path         text,
    -- pending -> ingesting -> ready | failed
    status              text        not null default 'pending',
    status_detail       text,
    progress            double precision not null default 0,
    frame_count         integer     not null default 0,
    audio_window_count  integer     not null default 0,
    transcript          text,
    error               text,
    created_at          timestamptz not null default now(),
    ingested_at         timestamptz
);

create index if not exists videos_status_idx     on videos (status);
create index if not exists videos_created_at_idx on videos (created_at desc);

-- ---------------------------------------------------------------- frames ----

create table if not exists frames (
    id                bigserial primary key,
    video_id          uuid not null references videos (id) on delete cascade,
    frame_index       integer not null,
    timestamp_seconds double precision not null,
    asset_path        text not null,
    ocr_text          text,
    scene_start_seconds double precision,
    scene_end_seconds   double precision,
    -- openai/clip-vit-base-patch32 projection.
    embedding         vector(512),
    unique (video_id, frame_index)
);

create index if not exists frames_video_time_idx on frames (video_id, timestamp_seconds);

-- --------------------------------------------------------- audio windows ----

create table if not exists audio_windows (
    id              bigserial primary key,
    video_id        uuid not null references videos (id) on delete cascade,
    window_index    integer not null,
    start_seconds   double precision not null,
    end_seconds     double precision not null,
    transcript_text text,
    asset_path      text,
    -- laion/clap-htsat-unfused projection. Null when the window was scored by
    -- Whisper transcript alone rather than by CLAP.
    embedding       vector(512),
    unique (video_id, window_index)
);

create index if not exists audio_windows_video_time_idx
    on audio_windows (video_id, start_seconds);

-- --------------------------------------------------------- conversations ----

create table if not exists conversations (
    id         uuid primary key default gen_random_uuid(),
    video_id   uuid not null references videos (id) on delete cascade,
    title      text,
    created_at timestamptz not null default now()
);

create index if not exists conversations_video_idx
    on conversations (video_id, created_at desc);

create table if not exists messages (
    id              uuid primary key default gen_random_uuid(),
    conversation_id uuid not null references conversations (id) on delete cascade,
    role            text not null check (role in ('user', 'assistant')),
    query           text,
    answer          text,
    answer_provider text,
    -- The compressed evidence set, the run metrics, and the cut clip spans.
    -- Stored as JSONB rather than normalized: they are written once, read
    -- whole, and their shape tracks CompressionResponse rather than this schema.
    selected_evidence jsonb,
    metrics           jsonb,
    clips             jsonb,
    created_at      timestamptz not null default now()
);

create index if not exists messages_conversation_idx
    on messages (conversation_id, created_at);
