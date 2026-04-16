# LP_barcode_decoding

Production FastAPI + Node.js platform for 4D spectral deconvolution and laser-particle barcode decoding at scale. Processes 10M+ emission spectra, decodes 40,000+ optical barcodes per run, and ships as a Dockerized async service with a full REST API and interactive browser UI.

**Stack:** Python · FastAPI · Pydantic · Uvicorn · Node.js · Docker · Cython · NumPy · SciPy · scikit-learn · HDBSCAN · h5py · Joblib · Plotly

---

## Quickstart

```bash
# Backend (FastAPI)
cd FastAPI
pip install -r requirements.txt
cd lase_analysis/c_code/utils && python setup.py build_ext --inplace && pip install . && cd ../../..
uvicorn app3_march5:app --host 0.0.0.0 --port 8000

# Frontend proxy (Node.js)
cd ..
npm install && node server.js   # → http://localhost:3006
```

**Docker:**
```bash
cd FastAPI
docker build -t lp-barcode .
docker run -p 8000:8000 -v /path/to/data:/data lp-barcode
```

---

## Architecture

```
Browser (HTML/JS)
    │ HTTP
Node.js / Express  :3006   ← static files + proxy
    │ HTTP
FastAPI / Uvicorn  :8000   ← REST API + analysis engine
    │
lase_analysis/             ← core Python library
    ├── data_loader.py         HDF5 ingestion, group filtering
    ├── data_analysis_v2.py    4D CC detection, cross-sample matching
    ├── visualization_cc_v2.py server-side plot generation, HTML cache
    ├── utilities.py           spectral preprocessing
    └── c_code/                Cython + C extensions (10–50× speedup)
```

Input files are `.map.lase` — HDF5-backed 4D confocal stacks (x, y, z, λ) produced by LASE spectroscopy instruments.

---

## What it does

Laser particles (LPs) are microscopic lasers attached to cells as optical barcodes. Each cell carries a unique set of sub-nanometre emission lines that serves as its identifier. This platform decodes those barcodes from confocal image stacks and matches them across independent measurements to track cells over time.

The core engineering challenge: in dense imaging conditions, signals from multiple barcodes overlap spatially and must be separated algorithmically. The statistics-driven correction engine recovers **+42–59% more valid barcodes** and **+24–27% more correct cross-sample matches** compared to the baseline pipeline.

---

## Key capabilities

| Capability | Implementation |
|---|---|
| 4D spectral data ingestion | HDF5 reader with group-aware filtering; typed Pandas DataFrames (`pks`, `lns`, `mlt`) |
| Connected component detection | `scipy.ndimage.label` + `skimage.regionprops_table` on projected intensity images |
| Cross-sample barcode matching | Tile-pruned O(N²→N) Pearson correlation with `joblib.Parallel` |
| Automated barcode correction | Side-lobe pruning → z-score anomaly detection → DBSCAN/GMM splitting → HDBSCAN reclustering → IoU reinjection |
| Bayesian flow-to-map matching | Bipartite belief-propagation graph; calibrated log-likelihood lookup; tunable accuracy–recovery threshold |
| Interactive QC curation | Full split / merge / delete / undo REST API with per-label diff persistence |
| Performance | Cython kernels with C argsort and uthash hash-maps; native C loops replacing NumPy where branching prevents vectorisation |
| Deployment | Docker image; Uvicorn ASGI; Node.js proxy; disk-backed plot cache with cache invalidation on edit |

---

## Performance

| Dataset | Barcodes recovered (baseline) | Barcodes recovered (this pipeline) | Gain |
|---|---|---|---|
| 16k LP quartets | 11,266 | 16,503 | **+46%** |
| 40k LP quartets | 28,686 | 45,192 | **+58%** |

Cross-sample match improvement at confidence threshold Q = 0.7: **+24–27%** with **+1.7–1.8 pp accuracy**.

---

## Data model

Three typed DataFrames per sample, keyed by 64-bit sentinel IDs:

| Table | Indexed by | Represents |
|---|---|---|
| `pks` | row | Fitted emission peaks — spatial voxel (i,j,k), wavelength, amplitude, photon count |
| `lns` | `lid` (uint64) | Spectral lines — one per LP; energy ± uncertainty, photon count, barcode assignment |
| `mlt` | `mid` (uint64) | Barcodes — spatial centroid, multiplicity (lines per barcode) |

→ See [docs/pipeline.md](docs/pipeline.md) for full schema and pipeline stages.  
→ See [docs/api.md](docs/api.md) for complete API reference.  
→ See [docs/design.md](docs/design.md) for engineering design decisions.

---

## Dependencies

**Python:** `fastapi` `pydantic` `uvicorn` `pandas` `numpy` `scipy` `scikit-image` `scikit-learn` `hdbscan` `matplotlib` `plotly` `h5py` `cython` `joblib`

**Node.js:** `express` `http-proxy-middleware` `node-fetch` `dotenv`

---

## Citation

```
Zarei M., Martino N., Yun A. "Integrated 4-D Spatial–Spectral Deconvolution
Pipeline for Robust Laser-Particle Barcode Decoding in Confocal Imaging."
(In preparation, 2025)

Martino N. et al. "Large-scale combinatorial optical barcoding of cells with
laser particles." Light: Science & Applications 14, 148 (2025).
https://doi.org/10.1038/s41377-025-01809-x
```

---

## License

All rights reserved. Contact the repository owner for usage terms.
