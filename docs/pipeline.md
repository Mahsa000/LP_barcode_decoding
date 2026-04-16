# Pipeline — Stage-by-Stage Reference

← [README](../README.md) · [API](api.md) · [Design](design.md)

---

## Data model

Every `.map.lase` file is an HDF5 container. The pipeline extracts three typed Pandas DataFrames per sample group:

| Table | Index | Key columns | Physical meaning |
|-------|-------|-------------|-----------------|
| `pks` | row int | `i`, `j`, `k`, `a`, `wl`, `E`, `fwhm`, `ph`, `ispt`, `lid` | Fitted emission peaks; spatial voxel (i,j,k), amplitude, wavelength, photon energy, linewidth, photon count, spectrum index, line ID |
| `lns` | `lid` (uint64) | `wl`, `E`, `dE`, `ph`, `n`, `peri`, `mid` | Aggregated spectral lines (one per LP); energy ± uncertainty, photon count, spatial period, multiplet ID |
| `mlt` | `mid` (uint64) | `i`, `j`, `k`, `n` | Barcode units; spatial centroid, multiplicity `n` |

**LID (Line ID)** — unique identifier for one LP emission line. After correct curation: one physical LP = one LID.

**MID (Multiplet ID)** — unique identifier for one barcode (cell or bead). Barcode = ordered set of emission energies of all LIDs belonging to a MID.

IDs use a 64-bit sentinel scheme (`LIDS.NON`, `LIDS.FLT`, `LIDS.UNC`, `LIDS.DRP`) to represent un-assigned, filtered, uncertain, and dropped states without nulls.

---

## Stage 1 — 4D LASE data ingestion

`data_loader.py::load_and_filter_data()`

- Scans a folder for `*.map.lase` files
- Selects analysis groups (e.g. `grp_0`) from HDF5 structure
- Applies area-code filtering on the 5-column coordinate table `(i, j, k, area, group_id)`
- Returns per-group dicts: spectra array `spts` (shape: n_spectra × n_wavelengths), coordinates `crds`, and the three DataFrames

---

## Stage 2 — Connected component (CC) detection

`data_analysis_v2.py::DataProcessor.analyze_connected_components()`

Projects 4D confocal stack data (x, y, z, λ) into a 2D spatial intensity image:

```python
image[i, j] += sum(ph)   # peak photon counts summed across z and λ
```

`scipy.ndimage.label` with full 8-connectivity identifies contiguous LP-signal regions. `skimage.measure.regionprops_table` extracts per-CC geometry: `area`, `centroid`, `bbox`, `perimeter`, `major_axis_length`, `minor_axis_length`.

Each CC is a candidate cell or bead. In sparse data, one CC = one barcode. In crowded confocal data, one CC may contain many aggregated barcodes — this is what stages 4 and 5 correct.

---

## Stage 3 — Cross-sample spectral matching

`data_analysis_v2.py::DataProcessor.match_consecutive_samples()`

For consecutive sample pairs, computes Pearson correlation between mean spectral profiles of spatially neighboring CCs:

```python
corr = pearsonr(preprocess(mean(ref_spectra)), preprocess(mean(cmp_spectra)))
```

Spectral preprocessing (`utilities.py`):
```python
def preprocess_spectrum(spectrum):
    background_removed = spectrum - median_filter(spectrum, size=101)
    return (background_removed - mean(background_removed)) / std(background_removed)
```

Scalability: tile-based bounding-box pruning (3×3 spatial tile neighborhoods) reduces O(N²) candidate pairs by ~100×. `joblib.Parallel` (thread backend) distributes tile comparisons.

Output: N_ref × N_cmp Pearson score matrix saved per sample pair.

---

## Stage 4 — Interactive QC curation

A REST API + browser UI exposes a split–merge–delete workflow for correcting pipeline errors manually, used to generate ground-truth labels for algorithm development.

**LID-level operations:**
- `split-lid-dbscan` — DBSCAN on spatial + spectral features to separate two merged LPs
- `merge-lids` — combine two LIDs into one (dual-mode LP pairs)
- `delete-lid` — mark a side-lobe artefact or noise peak as dropped

**MID-level operations:**
- `assign-submid` — subdivide an over-merged CC into separate barcode MIDs
- `undo-*` — full undo for every operation

Per-label edits are persisted as small `<sample>_<label>.pkl` diff files. The full dataset is never re-serialized on each edit.

---

## Stage 5 — Statistics-driven automated correction

Built on top of the curation API, the automated engine applies physics-aware priors to recover barcodes without manual intervention.

### LID-level corrections

1. **Side-lobe pruning** — removes peaks failing an amplitude-offset threshold (~6.5% of lines removed in high-density datasets)
2. **Weirdness scoring** — each LID receives a robust z-score from voxel-area and peak-count anomaly; LIDs with score `P > 0.8` are flagged
3. **Conditional k-means / spectral GMM splitter** — flagged LIDs are split when a two-component Gaussian mixture with spectral gap > 2.0 nm is confirmed (~8.9% of lines split)

### MID-level corrections

1. **HDBSCAN reclustering** — density-based re-partitioning of surviving LIDs; discovers barcode count automatically from local density; up to 59% of benchmark MIDs are re-clustered
2. **IoU-based noise reinjection** — isolated noise LIDs with spatial IoU > 0.20 to the nearest MID are reabsorbed
3. **Tiny-cluster absorption** — clusters of size ≤ 3 with IoU > 0.33 are merged into neighbouring MIDs

### Quantitative results

| Dataset | Lines processed | LIDs pruned | LIDs split | MIDs re-clustered | Barcodes (baseline) | Barcodes (post) | Gain |
|---------|----------------|-------------|------------|-------------------|---------------------|-----------------|------|
| 40k quartets | 198,033 | 6.5% | 8.9% | 41.9% | 28,686 | 45,192 | **+58%** |
| 16k quartets | 81,035 | 1.7% | 8.1% | 59.4% | 11,266 | 16,503 | **+46%** |

Cross-sample match improvement at Q = 0.7: **+24–27%** with **+1.7–1.8 pp accuracy**.

---

## Stage 6 — Bayesian flow-to-map matching

Probabilistic matcher for pairing barcodes between independent measurements (confocal imaging vs. flow cytometry, or repeated runs).

1. Line centroids compared in energy space with nearest-difference alignment within window `Δε`
2. Log-likelihood ratio computed from a calibrated 2D lookup table (energy difference × line energy) derived from empirical true/false alignment distributions
3. Per-pair match probability `Q_single` computed from `(n_matched, n_max, n_outliers)`
4. All candidate pairs embedded in a bipartite graph with explicit "miss" nodes; belief-propagation inference selects a globally consistent one-to-one assignment
5. Matches accepted above posterior threshold `Q_thresh`; accuracy–recovery trade-off characterised by sweeping `Q_thresh ∈ [0.5, 0.95]`

---

## Storage layout

```
<dataset_folder>/
├── connected_component_data.pkl        # per-CC geometry + peaks (all samples)
├── all_matrices.pkl                    # N_ref × N_cmp Pearson score matrices
├── all_subcorrelations.pkl             # per-CC best-match records
├── all_spatial_info.pkl                # bounding-box metadata
├── all_label_matches.pkl               # cross-sample barcode chain
├── saved_changes/
│   ├── id_counters.json                # monotonic LID/MID counters (uint64)
│   ├── <sample>_changes_diff.pkl       # per-sample curation diff history
│   └── label_data/
│       └── <sample>_<label>.pkl        # per-label pks/lns/mlt snapshots
└── plots/
    └── cc_matching_plots/
        └── cached_plots/               # HTML + PNG per (sample, label)
```
