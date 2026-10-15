# Public portfolio working rules

- Keep ITT denominators as all assigned customers. Never condition on post-treatment outcomes.
- Train preprocessing on training rows only; keep train/validation/test separate.
- Never tune on the test results or represent exploratory analysis as prospective confirmation.
- Numbers in reports must come from computed structured results, not free LLM prose.
- The default demo is synthetic, offline and has no database or API requirements.
- Do not upload secrets, raw datasets, row-level predictions, model files or third-party papers.
- Product SQL and packaged SQL resources must agree.
- Run `python -m pytest` for this public snapshot. Full PostgreSQL/API acceptance is separate.
- Preserve existing outputs; write each new run to a new artifacts directory.
