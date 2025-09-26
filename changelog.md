# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]
- Planned: richer feedback export (PDF/CSV).
- Planned: admin dashboard for reviewing sessions.

## [2.1] - 2025-09-26
### Added
- Health endpoints:
  - `/healthz` → returns `{ok: true}` for Render health checks.
  - `HEAD /` → responds 200 (avoids 405 logs).
  - `/favicon.ico` → returns favicon if present, otherwise 204.
- Session-level recommendations logic (`finalize_session_recommendations`).

### Changed
- Consolidated scoring into `compute_scores()` (LLM + fallback).
- Prompt loading is session-aware (avoids repeats).
- Slimmed inline comments → replaced with top-level docstring and section headers.
- Cleaned up duplicate helpers and unused imports.

### Fixed
- Removed Unicode arrow (`→`) that caused SyntaxError.
- Silence 404 spam from favicon requests.

## [2.0] - 2025-09-24
### Added
- Full adaptive evaluation flow:
  - Starts at A1 and auto-advances only on passing.
  - Second-chance retry if fail.
  - Session-aware prompt rotation.
  - Per-category score chips.
  - Turn-by-turn + final recommendations.
- Static UI served via `/web`.

---
