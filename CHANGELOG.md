# Changelog

## Unreleased

## 0.2

raghilda 0.2 expands the package from the core RAG workflow into a more
complete toolkit for building and maintaining retrieval stores. The release
adds crawl and ingest APIs with caching and concurrency, a Cloudflare-backed
crawler for JavaScript-rendered sites, a PostgreSQL store backend, and NVIDIA
NIM embedding support.

### Added

- Added `raghilda.crawl`, including `CrawlScope`, `FetchedSource`,
  `DirectoryCrawler`, `WebCrawler`, and `CloudflareCrawler`, for discovering
  directory, web, and Cloudflare sources and converting them to markdown
  documents.
- Added `BaseStore.ingest()` and `IngestSummary` for bulk document ingestion
  with optional document preparation, parallel writes, and inserted, replaced,
  and skipped counts.
- Added crawler caching so repeated or interrupted crawls can reuse fetched and
  converted content.
- Added `CloudflareCrawler` for crawling and converting JavaScript-rendered
  sites through Cloudflare's Browser Rendering API.
- Added `PostgreSQLStore`, backed by `psycopg2` and `pgvector`, with full-text
  search, vector search, combined retrieval, attributes, and HNSW index support.
- Added `EmbeddingNVIDIA` for NVIDIA NIM embeddings, including differentiated
  query and document input types and rate-limit backoff.
- Added user guide pages for quickstart, crawling and ingestion, Cloudflare
  crawling, and chatlas integration.

### Changed

- `CrawlScope.include_patterns` and `CrawlScope.exclude_patterns` now use one
  glob-style pattern syntax across `DirectoryCrawler`, `WebCrawler`, and
  `CloudflareCrawler`.
- Existing regex strings passed to `WebCrawler` or `DirectoryCrawler` should be
  rewritten as globs or passed as compiled `re.Pattern` objects.
- Reorganized the user guide onboarding pages and refreshed the README.

### Fixed

- Fixed sitemap URL extraction so each `<loc>` entry is collected as one URL.
- Improved DuckDB BM25 retrieval errors when the index has not been built or
  has become stale.
