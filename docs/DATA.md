# Event data contract

Provide one Parquet file containing event-level rows. The CLI never downloads a dataset.

| Column | Meaning |
| --- | --- |
| `userId` | Nonnegative integer ID or its string representation; null/blank anonymous IDs are discarded |
| `ts` | Event time: epoch milliseconds or a UTC-naive timestamp |
| `registration` | Registration time, in the same timestamp formats |
| `page` | Action, including `NextSong` and `Cancellation Confirmation` |
| `song`, `artist` | Nullable listening identifiers |
| `sessionId` | Nullable session identifier, scoped within each user |
| `length` | Nullable numeric listening duration |
| `gender` | Original feature encodes `M`; other/missing values map to zero |
| `level` | Subscription tier; `paid` is the paid tier |
| `status` | HTTP status; non-200 and missing values count as errors |

Normalize timezone-aware timestamps to UTC and remove the timezone before writing Parquet. The cutoff accepts timezone-aware values and is normalized to UTC. Numeric timestamps are interpreted as **milliseconds**, never seconds.

An eligible user must have an observation at or before the cutoff and no prior cancellation confirmation. The target is any cancellation in `(cutoff, cutoff + horizon]`. The latest event in the file must reach the horizon end. This global coverage guard does **not** establish complete follow-up for every user; verify collection continuity and censoring separately.

Event order in the file is irrelevant. For equal timestamps, latest-action features use a deterministic lexicographic page/tier tie-break; this cannot reconstruct an unavailable within-timestamp event order. Missing page categories produce zero-count columns. Null numeric aggregates and undefined ratios become zero. Registration gaps therefore require a separate data-quality review.

The pipeline uses all available pre-cutoff history, not a fixed-length rolling window. In forward evaluation, an existing user may occur in both time cohorts: this is a longitudinal evaluation of future behavior, not an unseen-user evaluation.

Do not commit private event logs, per-user predictions or trained models. The checked-in notebook has its original outputs removed; the aggregate ROC figure and result manifest preserve the historical evidence.
