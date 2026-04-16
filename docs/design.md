# Engineering Design Decisions

← [README](../README.md) · [Pipeline](pipeline.md) · [API](api.md)

---

## Why per-label pickle diffs instead of re-serializing the full dataset?

Full datasets exceed 15 GB. Re-writing the master pickle on every curation edit would make the browser UI unusable. Instead, each edit writes only a small `<sample>_<label>.pkl` diff (a few KB). The main data store is only updated when the user explicitly calls `/commit-label-changes` or `/finalize-sample`. This keeps every interactive operation under 100 ms regardless of dataset size.

---

## Why tile-based pruning for cross-sample CC matching?

Naïve O(N²) comparison of all CCs in sample A against all CCs in sample B is prohibitive at scale (a single high-density scan may have 50,000+ CCs). CCs are bucketed into a spatial tile grid; only CCs in 3×3 tile neighborhoods are considered as candidates. This reduces the comparison space by ~100× for typical scan densities while preserving all physically plausible matches (two CCs that represent the same barcode must be spatially proximate).

---

## Why HDBSCAN over k-means for MID reclustering?

k-means requires specifying the number of clusters upfront and is sensitive to initialisation. In crowded confocal data, a single connected component may contain anywhere from 1 to 30+ merged barcodes — the correct number is unknown until after correction. HDBSCAN discovers cluster count automatically from local density, handles noise points as outliers rather than forcing them into a cluster, and is robust to the irregular spatial distributions of LP signals in aggregated CCs.

---

## Why DBSCAN (not HDBSCAN) for individual LID splitting?

LID splitting is a finer-grained operation on a small set of peaks within one LID (typically 5–50 points). At this scale, DBSCAN with a fixed `eps` and `min_samples = 2` is faster and more predictable than HDBSCAN. The split is only accepted when a spectral GMM additionally confirms a two-component mixture with gap > 2.0 nm, preventing false splits on legitimate dense clusters.

---

## Why median filter for spectral background removal?

LP emission spectra sit on a slowly-varying photoluminescence background whose shape varies across samples and pump conditions. A large-kernel median filter (`size = 101`) removes this background without any fitted model or hyperparameters, making the z-scored residual profiles consistent across heterogeneous acquisition conditions. This is important for Pearson correlation matching: a fitted polynomial baseline can distort spectral shapes and produce false high correlations.

---

## Why Cython for scoring and decomposition?

Peak scoring and spectral decomposition iterate over large peak arrays with conditional branching (e.g. threshold checks, nearest-neighbour lookups, sparse index operations) that NumPy cannot vectorise. The Cython modules (`scoring/functions.pyx`, `decomposing/functions.pyx`) use:

- Typed C loops over numpy array buffers (no Python overhead per iteration)
- Custom C argsort (`c_argsort.h`) — avoids Python sort overhead on small arrays called millions of times
- uthash C hash-maps (`c_dict.h`) — O(1) LID/MID lookups replacing Python dict overhead in tight loops
- Sparse COO/CSR arrays for memory-efficient storage of large, mostly-empty correlation matrices

Net speedup: 10–50× over equivalent pure Python, which makes the difference between an interactive UI and a tool that stalls for minutes per label.

---

## Why Node.js as a proxy layer rather than serving FastAPI directly?

- Static files (HTML, JS, cached plots) are served efficiently by Express without touching the Python process
- The proxy decouples the frontend port (`:3006`) from the API port (`:8000`), allowing each to be scaled or replaced independently
- Node.js handles connection keep-alive and request queuing, protecting the single-threaded Uvicorn worker from thundering-herd effects during plot generation

---

## Why server-side plot generation instead of client-side rendering?

Generating plots in Matplotlib/Plotly on the server and caching them as HTML/PNG avoids sending large raw datasets (pks DataFrames with millions of rows) to the browser. Cached plots are invalidated only when a label edit is applied, so repeated navigation between labels is instant. This design also means the frontend stays as plain HTML/JS with no heavy charting dependencies.

---

## ID sentinel scheme

Using four reserved uint64 values (`NON`, `FLT`, `UNC`, `DRP`) instead of nullable integer columns:

- Eliminates Pandas nullable integer overhead on large DataFrames
- Allows bitwise ID masking in Cython loops without None checks
- Makes filtered/unassigned states explicit and distinguishable (an unassigned LID is different from a filtered one)
