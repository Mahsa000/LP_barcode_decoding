# API Reference

← [README](../README.md) · [Pipeline](pipeline.md) · [Design](design.md)

All endpoints are served by FastAPI at `:8000` and proxied through Node.js at `:3006`.  
Interactive docs available at `http://localhost:8000/docs` (Swagger UI) and `/redoc`.

---

## Ingestion & analysis

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| `POST` | `/analyze-raw-data` | `{ "data_path": str }` | Peak finding, line fitting, and spectral decomposition on raw `.map.lase` files |
| `POST` | `/load-and-filter-data` | `{ "data_path": str, "group": str, "area_code": int }` | Load filtered 4D data; run or reload CC analysis; returns sample list + histogram URLs |

---

## Visualization

| Method | Endpoint | Params | Description |
|--------|----------|--------|-------------|
| `GET` | `/` | — | Main control panel (HTML) |
| `GET` | `/line-peaks-view` | `sample`, `label` | Per-label view: voxel heat-maps per LID, overlaid spectra, MID colour coding (HTML) |
| `GET` | `/plot-multi-sample-iter` | `sample`, `label` | Cross-sample barcode tracking: same CC matched across consecutive measurements (HTML) |
| `POST` | `/plot-same-cc-across-samples` | `{ "sample": str, "label": int }` | Generate matched CC plot for a given start sample + label; returns cached HTML path |

---

## Label / barcode management

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| `POST` | `/load-label-data` | `{ "sample": str, "label": int }` | Load `pks`/`lns`/`mlt` for `(sample, label)` into server memory |
| `GET` | `/get-label-data` | `sample`, `label` | Return in-memory barcode data as JSON |
| `POST` | `/apply-label-changes` | `{ "sample": str, "label": int }` | Persist in-memory edits to per-label pickle; invalidate plot cache |
| `POST` | `/discard-label-changes` | `{ "sample": str, "label": int }` | Revert all in-memory edits for a label |
| `POST` | `/commit-label-changes` | `{ "sample": str, "label": int }` | Merge in-memory label back into the main data store |
| `POST` | `/finalize-sample` | `{ "sample": str }` | Concatenate all per-label pickles into one final pickle for a sample |

---

## LID operations (all with undo)

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| `POST` | `/split-lid-dbscan` | `{ "sample": str, "label": int, "lid": int }` | DBSCAN split of one LID into spatially/spectrally distinct sub-clusters |
| `POST` | `/undo-split-lid` | `{ "sample": str, "label": int, "lid": int }` | Revert a split; restores original LID |
| `POST` | `/delete-lid` | `{ "sample": str, "label": int, "lid": int }` | Mark a LID as dropped (`LIDS.DRP`) |
| `POST` | `/undo-delete-lid` | `{ "sample": str, "label": int, "lid": int }` | Restore a deleted LID |
| `POST` | `/merge-lids` | `{ "sample": str, "label": int, "lids": [int, ...] }` | Merge two or more LIDs into one |
| `POST` | `/undo-merge-lids` | `{ "sample": str, "label": int, "lid": int }` | Revert a merge; restores all original LIDs |

---

## MID operations (all with undo)

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| `POST` | `/assign-submid` | `{ "sample": str, "label": int, "lids": [int, ...] }` | Assign a new MID to a subset of LIDs within a CC |
| `POST` | `/undo-submid` | `{ "sample": str, "label": int, "mid": int }` | Revert a sub-MID assignment |

---

## Enumeration

| Method | Endpoint | Params | Description |
|--------|----------|--------|-------------|
| `GET` | `/get-samples` | — | List all loaded sample names |
| `GET` | `/list-labels` | `sample` | List all CC label indices for a sample |
| `GET` | `/list-lids` | `sample`, `label` | List all LIDs for a `(sample, label)` |

---

## Response conventions

- All mutation endpoints return `{ "status": "ok" }` on success or `{ "status": "error", "detail": str }` on failure
- Pydantic models enforce request validation; invalid payloads return HTTP 422
- Plot endpoints return `{ "url": str }` pointing to the cached HTML/PNG on disk
- The Node.js proxy forwards `/plots/*` path prefix directly to the FastAPI static file handler
