# Project guidance

## Application purpose and goals

MeteoInterpolator is a Python application for turning meteorological station observations into spatial grids over a digital elevation model (DEM). It combines data ingestion and storage, temporal resampling, elevation-based models, optional residual interpolation using distance fields, and cross-validation. A FastAPI service exposes station data and submits interpolation requests; results are written as Zarr datasets with spatial and processing metadata.

The development goals are scientifically consistent predictions, reproducible outputs, extensible data providers and interpolation models, and a clear separation between numerical methods, orchestration, and I/O. Preserve these goals when extending the application.

Current scope: the coordinator supports `tair_2m` interpolation at hourly or daily frequency. Other meteorological variables exist in ingestion and schemas; this does not imply end-to-end interpolation support. `src/datagaps.py` is a placeholder, not an implemented gap-filling feature. Update these notes when those capabilities change.

## Code responsibilities

| Location | Responsibility |
| --- | --- |
| `src/api/` | FastAPI endpoints and request/response schemas. |
| `src/config.py`, `src/runtime.py` | Configuration validation, component construction, and shared resource initialization. |
| `src/coordinator.py` | Request preparation, job creation, run lifecycle, and coordination of result writing. |
| `src/worker.py` | One interpolation job: cross-validation, prediction, status, and result metadata. |
| `src/interpolate/` | Interpolation composition, regions, and cross-validation. |
| `src/interpolate/models/` | Numerical models and fitted-model interfaces. |
| `src/interpolate/distance/` | Distance fields and distance calculator implementations. |
| `src/meteo/` | Provider adapters, ingestion, retrieval, sensor mapping, and meteorological resampling. |
| `src/domain/` | DEM and meteorological data objects and data validation schemas. |
| `src/database/` | SQLAlchemy models and database operations. |
| `src/execute/` | Job executor interface and serial execution. |
| `src/array/`, `src/write.py` | Array caching and CRS utilities; result serialization and publication. |
| `scripts/` | Operational entry points and exploratory scripts. |
| `tests/` | Unit and workflow tests. |
| `alembic/` | Database migrations. |

## Rules for changes

- Put reusable application logic in the appropriate `src/` module. Keep scripts focused on calling that logic and tests in `tests/`.
- Extend existing components before introducing parallel implementations or new architectural layers. Keep unrelated refactoring out of focused changes.
- Keep HTTP handling in `api/`, request orchestration in the coordinator, per-job processing in the worker, and numerical methods in `interpolate/`.
- Add models through `InterpolationModel` and `FittedInterpolationModel`; add distance calculators through `BaseDistanceCalculator`; add providers through `BaseMeteoHandler`. Follow their existing registry/factory patterns and ensure new implementations are imported so registration occurs.
- Reuse configuration models, domain validators, and existing timezone, CRS, and serialization helpers. Follow the surrounding code's conventions.
- Preserve existing API, configuration, and output contracts unless the requested change requires updating them. Update affected tests and examples together with such changes.

## Data and execution conventions

- Preserve timezone-aware timestamps through request and observation processing. Use existing helpers for UTC normalization and output serialization; retain the documented timezone meaning when storing timezone-naive array times.
- Preserve station IDs as strings. Align observations, training arrays, and distance fields by IDs and named coordinates, not assumed row order. Training arrays use the `id` dimension.
- Station elevations from the database/catalog are authoritative for training and station validation; request preparation does not overwrite them with DEM samples. Prediction grid elevations still come from the DEM.
- Preserve CRS metadata and explicitly transform coordinates when needed. Keep station locations, DEM grids, prediction targets, and distance fields spatially consistent.
- Keep validation at the existing data/model boundaries. Do not bypass finite-value, coordinate, dimension, or station-alignment checks to make an input pass.
- Preserve the distinction between completed, skipped, and failed jobs. Insufficient observations currently produce a skipped job.
- Runs share runtime resources and are serialized by the coordinator. Preserve its locking, cancellation, and shutdown behavior when changing execution; concurrency changes require checking resource ownership.
- Runtime initialization loads cached distance fields when required. Preserve the separate precomputation workflow in `scripts/precompute_distance_fields.py`.
- Write completed results through `src/write.py`. Preserve request-specific output directories, UTC filenames, CRS and processing metadata, and staging before publication. Keep large arrays out of retained run summaries.

## Development and verification

Run commands from the repository root. Python 3.13 or newer is required; dependencies are managed with `uv`.

- Set up or synchronize the environment: `uv sync --group dev`.
- Manage dependencies with `uv`; keep `pyproject.toml` and `uv.lock` consistent.
- Run relevant tests after behavior changes: `uv run pytest tests/test_interpolator.py` (replace with the affected test files).
- For changes spanning shared contracts or the full workflow, run `uv run pytest`.
- Add or update tests for changed behavior, especially numerical results, ID alignment, timezone/CRS handling, and output metadata. Use existing tests as examples; use temporary files and isolated data for test I/O.
- Documentation-only changes need a content/link review rather than a test run.
- Report what changed, which checks ran, and any remaining limitations. Do not claim checks passed if they were not run.

## Maintaining this guidance

Keep this file concise and tied to the actual code. Update it when responsibilities or contracts change. Put detailed architecture explanations in project documentation and link them here. Treat current limitations as context, not as a prohibition on explicitly requested features.
