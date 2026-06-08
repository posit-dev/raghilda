# Changelog

## Unreleased

### Added

- Added `raghilda.crawl`, including `CrawlScope`, `FetchedSource`,
  `DirectoryCrawler`, `WebCrawler`, and `CloudflareCrawler`, for discovering
  directory, web, and Cloudflare sources and converting them to markdown
  documents.
- Added `BaseStore.ingest()` and `IngestSummary` for bulk document ingestion
  with optional document preparation, parallel writes, and inserted, replaced,
  and skipped counts.

### Fixed

- Fixed sitemap URL extraction so each `<loc>` entry is collected as one URL.
