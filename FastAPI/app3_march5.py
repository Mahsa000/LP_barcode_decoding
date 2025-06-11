# april 11 5 pm
import logging
import os
import copy
import pickle
import json
import threading
from pathlib import Path
from typing import Optional, List, Any, Dict, Tuple

import pandas as pd
import numpy as np
import matplotlib
from starlette.routing import Mount

matplotlib.use('Agg')  # Headless backend for matplotlib
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from collections import defaultdict

# -- Local imports from your project --
from lase_analysis.data_loader import load_raw_files, load_and_filter_data
from lase_analysis.data_analysis_v2 import DataProcessor
from lase_analysis.visualization_cc_v2 import (
    plot_same_cc_across_samples,
    plot_multi_sample_comparison,
    visualize_line_peaks,
    plot_label_spectra
)

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_processing.log')
    ]
)
logger = logging.getLogger(__name__)

# ---------------- FastAPI ----------------
app = FastAPI(
    title="LASE Analysis API (Unified, Interactive)",
    description="A FastAPI backend for advanced LID + MID splitting, etc.",
    version="3.0.0"
)

# ---------------- Global Paths (base) ----------------
BASE_OUTPUT_FOLDER = Path("/Users/mahsazarei/LASE_VIS")
BASE_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

ANALYSIS_OUTPUT_FOLDER: Path = BASE_OUTPUT_FOLDER
SAVE_FILENAME = "connected_component_data.pkl"
CC_SAVE_PATH: Path = ANALYSIS_OUTPUT_FOLDER / SAVE_FILENAME
MODIFIED_SAVE_PATH = ANALYSIS_OUTPUT_FOLDER / "connected_component_data_modified.pkl"

PLOT_OUTPUT_FOLDER: Path = ANALYSIS_OUTPUT_FOLDER / "plots" / "cc_matching_plots"
PLOT_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
LID_PLOTS_FOLDER: Path = PLOT_OUTPUT_FOLDER / "lid_plots"
CHANGES_FOLDER: Path = ANALYSIS_OUTPUT_FOLDER / "saved_changes"
CHANGES_FOLDER.mkdir(parents=True, exist_ok=True)

LABELS_DATA_FOLDER = CHANGES_FOLDER / "label_data"
LABELS_DATA_FOLDER.mkdir(parents=True, exist_ok=True)
ID_COUNTER_FILE = CHANGES_FOLDER / "id_counters.json"


from collections import defaultdict
import shutil
# This is your "cache root" folder. You can keep it inside your normal PLOT_OUTPUT_FOLDER,
# or store it somewhere else. We'll define a new one for clarity:
CACHE_BASE_FOLDER: Path = PLOT_OUTPUT_FOLDER / "cached_plots"# = Path("cached_plots")  # e.g. /Users/you/LASE_VIS/cached_plots
CACHE_BASE_FOLDER.mkdir(parents=True, exist_ok=True)
CACHED_LABEL_PLOTS: Dict[Tuple[str,int], str] = {}


def get_label_cache_folder(
    sample_name: str,
    label_num: int,
    filter_key: Optional[str] = None
) -> Path:
    """
    Return the cache folder for a particular (sample, label, filter).
    If filter_key is None, this is just the unfiltered cache folder.
    """
    subpath = f"{sample_name}__label_{label_num}"
    if filter_key:
        subpath += "__" + filter_key
    return CACHE_BASE_FOLDER / subpath

def clear_label_cache(
    sample_name: str,
    label_num: int,
    filter_key: Optional[str] = None
) -> None:
    """
    Remove *all* cached artifacts (images AND HTML) for a given sample_name/label.
    
    - If filter_key is given (and non-empty), only cache folders matching
      sample__label_<n>__<filter_key>* are removed.
    - Otherwise, every cache sub-folder whose name starts with sample__label_<n>
      is removed.
    
    This will rmtree the entire folder (so you never leave behind the HTML
    wrapper or any nested images), and then it will also unlink any stray
    `cached_single_label.html` that might exist.
    """
    # 1) Build the base glob prefix
    base = f"{sample_name}__label_{label_num}"
    if filter_key:
        base = f"{base}__{filter_key}"

    # 2) Find all matching cache directories
    cache_base = CACHE_BASE_FOLDER  # your global
    pattern = f"{base}*"
    dirs = list(cache_base.glob(pattern))

    # 3) Delete each directory wholesale
    for d in dirs:
        if d.is_dir():
            try:
                logger.info(f"Clearing cache directory: {d}")
                shutil.rmtree(d, ignore_errors=False)
            except Exception as e:
                logger.warning(f"Failed to delete cache directory {d}: {e}")

    # 4) In case any HTML wrapper lives *inside* or *beside* those folders,
    #    remove them explicitly.
    #    (Most of the time step 3 already got them, but this catches stray files.)
    html_pattern = f"{pattern}/cached_single_label.html"
    for html_path in cache_base.glob(html_pattern):
        if html_path.is_file():
            try:
                logger.info(f"Removing stale HTML cache: {html_path}")
                html_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove HTML cache {html_path}: {e}")

# ---------------- Global Data Structures ----------------
CONNECTED_COMPONENT_DATA: Dict[str, Any] = {}
CONNECTED_COMPONENT_DATA_ORIG: Dict[str, Any] = {}
CONNECTED_COMPONENT_DATA_MODIFIED: Dict[str, Any] = {}

ALL_MATRICES: Dict[Tuple[str, str], Any] = {}
ALL_SUBCORRELATIONS: Dict[Tuple[str, str], Any] = {}
ALL_LABEL_MATCHES_ORIG: Dict[Tuple[str, int], List[Tuple[str, int]]] = {}
ALL_LABEL_MATCHES_MODIFIED: Dict[Tuple[str, int], List[Tuple[str, int]]] = {}
CURRENT_FILTERED_DATA: Dict[str, Any] = {}

# ---------------- Global Data Structures (NEW) ----------------
LABEL_MEMORY: Dict[Tuple[str, int], Dict[str, pd.DataFrame]] = {}

data_lock = threading.Lock()

@app.on_event("startup")
def startup_load_counters():
    load_id_counters()  # Ensure you have a function load_id_counters() defined somewhere

# ---------------- Pydantic Models ----------------
class AnalyzeRawParams(BaseModel):
    raw_data_path: str
    area_selection: Optional[int] = None
    digit_prefix: Optional[str] = None
    output_cc_path: Optional[str] = None
    force_refresh: Optional[bool] = False

class LoadFilterParams(BaseModel):
    analyzed_data_path: str
    area_selection: Optional[int] = None
    digit_prefix: Optional[str] = None
    output_cc_path: str
    force_refresh: Optional[bool] = False
    groups: Optional[List[str]] = None                  # ← new!

class PlotSameCCRequest(BaseModel):
    start_sample: str
    target_label: int

class SplitLidDBSCANRequest(BaseModel):
    sample_name: str
    target_label: int
    old_lid: int
    eps: float = 0.5
    min_samples: int = 1

class DeleteLidRequest(BaseModel):
    sample_name: str
    target_label: int
    lid_to_delete: int

class UndoDeleteLidRequest(BaseModel):
    sample_name: str
    target_label: int
    lid_to_restore: int

class MaybeLidRequest(BaseModel):
    sample_name: str
    target_label: int
    lid_to_maybe: int
    reason: str

class UndoSplitLidRequest(BaseModel):
    sample_name: str
    target_label: int
    old_lid: int

# ---------------- Helper: Normalize Sample IDs ----------------
def normalize_sample_id(sample_id: str) -> str:
    """
    Normalize a sample identifier by stripping any trailing ".lase" from the filename.
    For example: "250207 - QRT 20X 200pJ A00.map.lase" => "250207 - QRT 20X 200pJ A00.map"
    """
    sample_id = sample_id.strip()
    if sample_id.endswith(".lase"):
        sample_id = sample_id[:-5]
    return sample_id

# ---------------- Utility: Re-match for MODIFIED ----------------
def generate_label_matches_for_modified():
    from lase_analysis.data_analysis_v2 import DataProcessor
    processor = DataProcessor(output_folder=str(ANALYSIS_OUTPUT_FOLDER), padding=10)
    processor.connected_component_data = CONNECTED_COMPONENT_DATA_MODIFIED
    processor.filtered_data_storage = CURRENT_FILTERED_DATA  # Ensure consistency!
    processor.match_consecutive_samples()
    processor.generate_all_label_matches()
    global ALL_LABEL_MATCHES_MODIFIED
    ALL_LABEL_MATCHES_MODIFIED = processor.all_label_matches
    logger.info("[generate_label_matches_for_modified] Updated.")

# ---------------- Helper: set_user_folder ----------------
def set_user_folder(subfolder: str) -> None:
    global ANALYSIS_OUTPUT_FOLDER, CC_SAVE_PATH, PLOT_OUTPUT_FOLDER, LID_PLOTS_FOLDER, CACHE_BASE_FOLDER, CHANGES_FOLDER, LABELS_DATA_FOLDER, ID_COUNTER_FILE

    if not subfolder.strip():
        user_folder = BASE_OUTPUT_FOLDER / "default_dataset"
    else:
        user_folder = BASE_OUTPUT_FOLDER / subfolder.strip()
    user_folder.mkdir(parents=True, exist_ok=True)
    ANALYSIS_OUTPUT_FOLDER = user_folder
    CC_SAVE_PATH = ANALYSIS_OUTPUT_FOLDER / SAVE_FILENAME

    PARENT_PLOTS_FOLDER = ANALYSIS_OUTPUT_FOLDER / "plots"
    PARENT_PLOTS_FOLDER.mkdir(parents=True, exist_ok=True)

    new_routes = []
    for route in app.router.routes:
        if hasattr(route, "path") and route.path == "/plots":
            continue
        new_routes.append(route)
    app.router.routes = new_routes
    app.mount("/plots", StaticFiles(directory=str(PARENT_PLOTS_FOLDER)), name="plots")

    PLOT_OUTPUT_FOLDER = PARENT_PLOTS_FOLDER / "cc_matching_plots"
    PLOT_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    LID_PLOTS_FOLDER = PARENT_PLOTS_FOLDER / "lid_plots"
    LID_PLOTS_FOLDER.mkdir(parents=True, exist_ok=True)

    CACHE_BASE_FOLDER = PARENT_PLOTS_FOLDER / "cached_plots"# = Path("cached_plots")  # e.g. /Users/you/LASE_VIS/cached_plots
    CACHE_BASE_FOLDER.mkdir(parents=True, exist_ok=True)

    
    CHANGES_FOLDER = ANALYSIS_OUTPUT_FOLDER / "saved_changes"
    CHANGES_FOLDER.mkdir(parents=True, exist_ok=True)
    
    LABELS_DATA_FOLDER = CHANGES_FOLDER / "label_data"
    LABELS_DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    
    ID_COUNTER_FILE = CHANGES_FOLDER / "id_counters.json"
    logger.info(f"User folder set to: {ANALYSIS_OUTPUT_FOLDER}")


# april 22 analysis_params
@app.post("/analyze-raw-data")
def analyze_raw_lase(params: AnalyzeRawParams):
    raw_path = Path(params.raw_data_path).expanduser().resolve()
    if not raw_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {raw_path}")
    
    from lase_analysis.data_loader import read_lasefile
    raw_files = load_raw_files(raw_path)
    if not raw_files:
        raise HTTPException(status_code=400, detail="No .map.lase files found in the folder.")

    processor = DataProcessor(output_folder=".", padding=10)
    default_analysis_params = {
        'popt': {'pool': True, 'chunk': 100000, 'wdw': 3.0, 'method': 2},
        'lopt': {
            'pool': True, 'a': None, 'wl': None, 'fwhm': None,
            'E': [750, 1100], 'fwhmE': [0.2, 5.0],
            'ph': None, 'i': None, 'j': None, 'k': None, 'bad': None,
            'eps': 3, 'min_samples': 2, 'scale': [1, 1, 1, 0.5]
        },
        'ropt': {
            'multiplets': {'lns_dist': 5, 'min_overlap': 0.33},
            'splittable_lines': {
                'min_peri': 8, 'max_peri': None,
                'mass_shift': 4.5, 'lapl_sigma': 1.3,
                'dr_close': 3.0, 'dE_close': 0.3, 'dE_seed': 0.25, 'dE_reg': 0.5,
                'area_small': 6, 'area_left': 10, 'dE_left': 0.33
            }
        }
    }

    try:
        results = processor.analyze_data(raw_files, default_analysis_params)
        msg = f"Analyzed {len(results)} raw files. Each .map.lase now has 'base' analysis saved."
        logger.info(msg)
        return {"status": "success", "message": msg, "files": list(results.keys())}
    except Exception as e:
        logger.error(f"Error analyzing raw data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def create_baseline_label_files(force=False):
    """
    Iterate over all (sample, label) in CONNECTED_COMPONENT_DATA_MODIFIED,
    extract pks/lns/mlt (but not raw spectra),
    and save each label's small pickle in LABELS_DATA_FOLDER IF not already existing.
    If force=True, overwrites existing files.
    """
    for sample_name, sample_data in CONNECTED_COMPONENT_DATA_MODIFIED.items():
        props_list = sample_data.get("props", [])
        pks_all = sample_data.get("pks", pd.DataFrame())

        for comp in props_list:
            label_val = comp["label"]
            lns_df = comp.get("lns", pd.DataFrame())
            mlt_df = comp.get("mlt", pd.DataFrame())
            lids_for_label = lns_df.index.unique()
            pks_df = pks_all[pks_all["lid"].isin(lids_for_label)].copy()

            file_path = LABELS_DATA_FOLDER / f"{sample_name}_{label_val}.pkl"
            
            # If file already exists and force=False, skip
            if not force and file_path.exists():
                # Already have a baseline or modified label file, do not overwrite
                continue

            # Otherwise, create/overwrite with baseline data
            data_to_save = {
                "pks": pks_df,
                "lns": lns_df,
                "mlt": mlt_df
            }
            with open(file_path, "wb") as f:
                pickle.dump(data_to_save, f)


from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

def generate_multiplicity_histogram_by_sample(all_data: Dict[str, Any],
                                              output_folder: Path,
                                              max_mult: int = 20) -> str:
    """
    For each sample in all_data, this helper iterates over all the labels 
    (stored in sample_dict['props']) and extracts the multiplicity values from
    the 'mlt' DataFrame (column "n"). It then creates a bar chart showing the 
    frequency distribution of these multiplicities. Multiplicities 1..max_mult are 
    shown individually while any multiplicity greater than max_mult is grouped 
    in the final bin.
    
    The function creates one subplot (row) per sample and saves the composite 
    figure as a PNG in output_folder.

    Returns the filename, e.g. "multiplicity_histogram_by_sample.png".
    """
    sample_names = list(all_data.keys())
    n_samples = len(sample_names)
    
    # Create a figure with one row per sample.
    # Adjust the figure height if there are many samples.
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 4 * n_samples), squeeze=False)
    
    for idx, sample in enumerate(sample_names):
        ax = axes[idx][0]
        # Get the sample data (should have "props")
        sample_dict = all_data[sample]
        all_ns = []  # this will hold all multiplicity values from all labels for this sample
        for comp in sample_dict.get("props", []):
            mlt_df = comp.get("mlt", None)
            if mlt_df is None or "n" not in mlt_df.columns:
                continue
            all_ns.extend(mlt_df["n"].tolist())
        
        # Count the frequency distribution for this sample.
        freq = Counter(all_ns)
        # Create bins: 1 .. max_mult and a final bin for values > max_mult.
        bins = list(range(1, max_mult + 1)) + [max_mult + 1]
        counts = [freq.get(i, 0) for i in range(1, max_mult + 1)]
        counts.append(sum(v for m, v in freq.items() if m > max_mult))
        
        ax.bar(bins, counts)
        # Add count text on top of each bar
        for x, y in zip(bins, counts):
            ax.text(x, y, str(y), ha="center", va="bottom", fontsize=16)
        ax.set_xlabel("Multiplicity", fontsize=18)
        ax.set_ylabel("Frequency", fontsize=18)
        ax.set_title(f"Sample: {sample} (Total MIDs = {len(all_ns)})")
        ax.set_xticks(bins)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

        ax.set_xticklabels([str(i) for i in bins[:-1]] + [f"{max_mult}+"], rotation=0)
    
    fig.tight_layout()
    fname = "multiplicity_histogram_by_sample.png"
    outpath = output_folder / fname
    # output_folder.mkdir(parents=True, exist_ok=True)  # Ensure directory exists!
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    return fname


import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from pathlib import Path
from collections import Counter

def generate_energy_histogram_by_sample(all_data: Dict[str, any],
                                        output_folder: Path) -> str:
    """
    For each sample in all_data, this helper iterates over all the components 
    (stored in sample_dict['props']) and extracts the energy (E) values from the 
    lns DataFrame (column "E"). It then creates a histogram showing the frequency 
    distribution of these energy values for each sample. The x-axis is limited 
    from 750 to 1100 eV, with the label "E(ev)", and the y-axis is labeled "Frequency." 
    The title includes the sample name and total lns count.
    
    The function creates one subplot (row) per sample and saves the composite 
    figure as a PNG in output_folder.

    Returns the filename, e.g. "energy_histogram_by_sample.png".
    """
    sample_names = list(all_data.keys())
    n_samples = len(sample_names)
    
    # Create a figure with one row per sample.
    fig, axes = plt.subplots(n_samples, 1, figsize=(8, 4 * n_samples), squeeze=False)
    
    for idx, sample in enumerate(sample_names):
        ax = axes[idx][0]
        sample_dict = all_data[sample]
        all_E = []  # hold all energy values for this sample
        
        for comp in sample_dict.get("props", []):
            lns_df = comp.get("lns", None)
            if lns_df is None or "E" not in lns_df.columns:
                continue
            # Safely convert and append energy values
            try:
                energy_values = [float(val) for val in lns_df["E"].tolist()]
            except Exception as e:
                # Skip component if conversion fails
                continue
            all_E.extend(energy_values)
        
        if not all_E:
            ax.text(0.5, 0.5, "No Energy Data", transform=ax.transAxes, ha="center", fontsize=16)
            ax.set_title(f"Sample: {sample} (Total LIDs = 0)", fontsize=16)
            continue
        
        # Create bins from 750 to 1100 eV; 20 bins by default
        bins = np.linspace(750, 1150, num=55)
        ax.hist(all_E, bins=bins, edgecolor='black')
        ax.set_xlim(750, 1100)
        ax.set_xlabel("E(ev)", fontsize=16)
        ax.set_ylabel("Frequency", fontsize=16)
        ax.set_title(f"Sample: {sample} (Total LIDs = {len(all_E)})", fontsize=16)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
    
    fig.tight_layout()
    fname = "energy_histogram_by_sample.png"
    outpath = output_folder / fname
    # output_folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return fname
# april 23
import re, pickle
from pathlib import Path
import pandas as pd

# match:
#   1) sample.map_1234.pkl
#   2) sample.map_grp_0_1234.pkl
#   3) sample.map_grp_0_1234_5678.pkl
PAT = re.compile(
    r'^(?P<sample>.+?\.map)' +
    r'(?:_grp_(?P<grp>[01]))?' +
    r'_(?P<label>\d+)' +
    r'(?:_(?P<lid>\d+))?' +
    r'\.pkl$'
)

def _load_pickle(p: Path):
    with p.open("rb") as f:
        return pickle.load(f)

def _recalc_mlt(lns: pd.DataFrame) -> pd.DataFrame:
    if lns.empty or "mid" not in lns.columns:
        return pd.DataFrame()
    grp = lns.groupby("mid")
    return (
        grp.agg(i=("i","mean"),
                j=("j","mean"),
                k=("k","mean"),
                n=("mid","size"))
           .reset_index()
    )

def build_cc_data_from_folder(filtered_data_storage: dict,
                              base_output_folder: Path) -> dict:
    folder = Path(base_output_folder) / "saved_changes" / "label_data"
    if not folder.exists():
        raise FileNotFoundError(f"Missing folder: {folder}")

    # buckets[(sample, grp, label)] = {"lns": [...], "pks": [...], "mlt": [...]}
    buckets = {}

    for pkl_path in folder.glob("*.pkl"):
        m = PAT.match(pkl_path.name)
        if not m:
            # you can optionally log: logger.warning(f"Skipping {pkl_path.name}")
            continue

        sample = m.group("sample")
        grp    = f"grp_{m.group('grp')}" if m.group("grp") else None
        label  = int(m.group("label"))
        lid    = m.group("lid") and int(m.group("lid"))

        blob = _load_pickle(pkl_path)
        lns  = blob.get("lns", pd.DataFrame())
        pks  = blob.get("pks", pd.DataFrame())
        mlt  = blob.get("mlt", pd.DataFrame())

        key = (sample, grp, label)
        buk = buckets.setdefault(key, {"lns": [], "pks": [], "mlt": []})
        buk["lns"].append(lns)
        buk["pks"].append(pks)
        if not mlt.empty:
            buk["mlt"].append(mlt)

    # assemble final dict
    cc_data = {}
    for (sample, grp, label), buk in buckets.items():
        lns = pd.concat(buk["lns"], ignore_index=False)
        if lns.empty:
            continue

        pks = pd.concat(buk["pks"], ignore_index=True)
        if buk["mlt"]:
            mlt = pd.concat(buk["mlt"], ignore_index=True)
        else:
            mlt = _recalc_mlt(lns)

        comp = {
            "sample_id":   sample,
            "group_name":  grp,
            "label":       label,
            "lns":         lns,
            "pks":         pks,
            "mlt":         mlt,
            # carry ispt if present
            **({"ispt_values": list(pks["ispt"].unique())}
               if "ispt" in pks.columns else {})
        }

        samp = cc_data.setdefault(sample, {"props": [], "_pks_list": []})
        samp["props"].append(comp)
        samp["_pks_list"].append(pks)

    # build global peaks DataFrame
    for sample, samp in cc_data.items():
        samp["pks"] = pd.concat(samp.pop("_pks_list"), ignore_index=True)

    total_samples = len(cc_data)
    total_ccs     = sum(len(v["props"]) for v in cc_data.values())
    print(f"Rebuilt CONNECTED_COMPONENT_DATA: {total_samples} samples, {total_ccs} CCs total")

    return cc_data


# ---------------- Endpoint: load-and-filter-data ----------------
@app.post("/load-and-filter-data")
def load_and_filter_endpoint(params: LoadFilterParams, request: Request):
    """
    This endpoint loads the previously analyzed data if present, or re-runs the analysis if forced.
    It also ensures that the 'filtered_data_storage' is stored in app.state for later lazy raw spectra retrieval.
    """
    global CURRENT_FILTERED_DATA
    global CONNECTED_COMPONENT_DATA, CONNECTED_COMPONENT_DATA_ORIG, CONNECTED_COMPONENT_DATA_MODIFIED
    global ALL_LABEL_MATCHES_ORIG, ALL_LABEL_MATCHES_MODIFIED, ALL_MATRICES, ALL_SUBCORRELATIONS, ALL_SPATIAL_INFO
    
    set_user_folder(params.output_cc_path)
    logger.info(f"ANALYSIS_OUTPUT_FOLDER = {ANALYSIS_OUTPUT_FOLDER}")
    logger.info(f"CC_SAVE_PATH         = {CC_SAVE_PATH}")

    data_path = Path(params.analyzed_data_path).expanduser().resolve()
    if not data_path.exists():
        raise HTTPException(status_code=404, detail=f"Analyzed data path not found: {data_path}")

    # Always load (or re-load) the original filtered data.
    from lase_analysis.data_loader import load_and_filter_data
    try:
        filtered_data = load_and_filter_data(
            data_folder=data_path,
            area_code=params.area_selection,
            digit_prefix=params.digit_prefix,
            groups= params.groups   # ← here!
        )
        if not filtered_data:
            raise HTTPException(status_code=400, detail="No data found or missing .map.lase analysis.")

        # Set the filtered data (keys are already normalized in load_and_filter_data)
        CURRENT_FILTERED_DATA = filtered_data
        request.app.state.filtered_data_storage = filtered_data
    except Exception as e:
        logger.error(f"Error loading original data for lazy retrieval: {e}")
        raise HTTPException(status_code=500, detail=str(e))


    # If an existing CC file is present and not forced, load it.
    logger.info(f"Fast-load check: path={CC_SAVE_PATH} exists={CC_SAVE_PATH.exists()} force_refresh={params.force_refresh}")
    if CC_SAVE_PATH.exists() and not params.force_refresh:
        try:
            with open(CC_SAVE_PATH, "rb") as f:
                CONNECTED_COMPONENT_DATA = pickle.load(f)
            logger.info(f"Loaded CC data from {CC_SAVE_PATH}")
            # new:
            # CONNECTED_COMPONENT_DATA = build_cc_data_from_folder(
            # request.app.state.filtered_data_storage,
            # base_output_folder=ANALYSIS_OUTPUT_FOLDER
            # )
            # logger.info("Rebuilt CONNECTED_COMPONENT_DATA from label_data folder")

            files_to_load = {
                "ALL_MATRICES": "all_matrices.pkl",
                "ALL_SUBCORRELATIONS": "all_subcorrelations.pkl",
                "ALL_SPATIAL_INFO": "all_spatial_info.pkl",
                "ALL_LABEL_MATCHES_ORIG": "all_label_matches.pkl"
            }
            for var, filename in files_to_load.items():
                file_path = ANALYSIS_OUTPUT_FOLDER / filename
                if file_path.exists():
                    with open(file_path, "rb") as f:
                        loaded = pickle.load(f)
                    if var == "ALL_MATRICES":
                        ALL_MATRICES = loaded
                    elif var == "ALL_SUBCORRELATIONS":
                        ALL_SUBCORRELATIONS = loaded
                    elif var == "ALL_SPATIAL_INFO":
                        ALL_SPATIAL_INFO = loaded
                    elif var == "ALL_LABEL_MATCHES_ORIG":
                        ALL_LABEL_MATCHES_ORIG = loaded
                    logger.info(f"Loaded {var} from {file_path}")
                else:
                    logger.warning(f"File {filename} not found in {ANALYSIS_OUTPUT_FOLDER}.")

            CONNECTED_COMPONENT_DATA_ORIG = copy.deepcopy(CONNECTED_COMPONENT_DATA)
            CONNECTED_COMPONENT_DATA_MODIFIED = copy.deepcopy(CONNECTED_COMPONENT_DATA)
            ALL_LABEL_MATCHES_MODIFIED = copy.deepcopy(ALL_LABEL_MATCHES_ORIG)

            create_baseline_label_files(force=False)
            logger.info("Created baseline label files for all labels (loaded from existing data).")

            sample_names = list(CONNECTED_COMPONENT_DATA.keys())
            msg = f"[load-and-filter-data] Loaded existing analysis with {len(sample_names)} samples."
            logger.info(msg)
            # generate the histogram after loading/filtering the data
            hist_fname = generate_multiplicity_histogram_by_sample(
                CONNECTED_COMPONENT_DATA_ORIG,   # or _MODIFIED, whichever you prefer
                PLOT_OUTPUT_FOLDER,              # your ANALYSIS_OUTPUT_FOLDER/"plots"/"cc_matching_plots"
                max_mult=20
            )
            hist_url = f"/plots/cc_matching_plots/{hist_fname}"

            # NEW: Generate the energy histogram plot
            energy_hist_fname = generate_energy_histogram_by_sample(
                CONNECTED_COMPONENT_DATA_ORIG,  
                PLOT_OUTPUT_FOLDER
            )
            energy_hist_url = f"/plots/cc_matching_plots/{energy_hist_fname}"

            # Then include both URLs in your returned JSON:
            return {
                "status": "success",
                "message": msg,
                "samples": sample_names,
                "histogram_url": hist_url,
                "energy_histogram_url": energy_hist_url
            }


        except Exception as e:
            logger.error(f"Error loading from disk: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Otherwise, re-run the analysis from scratch.
    else:
        try:
            from lase_analysis.data_analysis_v2 import DataProcessor
            processor = DataProcessor(output_folder=str(ANALYSIS_OUTPUT_FOLDER), padding=10)
            # Process all samples (this will create connected_component_data.pkl)
            CONNECTED_COMPONENT_DATA = processor.process_all_samples(CURRENT_FILTERED_DATA, CC_SAVE_PATH)
            CONNECTED_COMPONENT_DATA_ORIG = copy.deepcopy(CONNECTED_COMPONENT_DATA)
            CONNECTED_COMPONENT_DATA_MODIFIED = copy.deepcopy(CONNECTED_COMPONENT_DATA)

            # Create a new instance for matching and ensure filtered_data_storage is set!
            processor_orig = DataProcessor(output_folder=str(ANALYSIS_OUTPUT_FOLDER), padding=10)
            processor_orig.connected_component_data = CONNECTED_COMPONENT_DATA_ORIG
            processor_orig.filtered_data_storage = CURRENT_FILTERED_DATA  # Critical line
            processor_orig.match_consecutive_samples()
            processor_orig.generate_all_label_matches()

            ALL_LABEL_MATCHES_ORIG = processor_orig.all_label_matches
            ALL_MATRICES = processor_orig.all_matrices
            ALL_SUBCORRELATIONS = processor_orig.all_subcorrelations
            ALL_SPATIAL_INFO = processor_orig.all_spatial_info

            generate_label_matches_for_modified()
            processor_orig.save_all_data()

            create_baseline_label_files(force=True)
            logger.info("Created baseline label files for all labels (new analysis).")

            # sample_names = list(CONNECTED_COMPONENT_DATA.keys())
            sample_names = sorted(CONNECTED_COMPONENT_DATA.keys())
            msg = f"[load-and-filter-data] New CC analysis generated with {len(sample_names)} samples."
            logger.info(msg)
            # NEW: Generate histogram and set the URL
            hist_fname = generate_multiplicity_histogram_by_sample(
                CONNECTED_COMPONENT_DATA_ORIG,  # or _MODIFIED if preferred
                PLOT_OUTPUT_FOLDER,
                max_mult=20
            )
            hist_url = f"/plots/cc_matching_plots/{hist_fname}"
            logger.info(f"Histogram generated at: {hist_url}")
            
            # NEW: Generate the energy histogram plot
            energy_hist_fname = generate_energy_histogram_by_sample(
                CONNECTED_COMPONENT_DATA_ORIG,  
                PLOT_OUTPUT_FOLDER
            )
            energy_hist_url = f"/plots/cc_matching_plots/{energy_hist_fname}"

            # Then include both URLs in your returned JSON:
            return {
                "status": "success",
                "message": msg,
                "samples": sample_names,
                "histogram_url": hist_url,
                "energy_histogram_url": energy_hist_url
            }

        except Exception as e:
            logger.error(f"Error in load-and-filter-data: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
def main_index_page():
    sample_options = []
    if CONNECTED_COMPONENT_DATA_MODIFIED:
        for s in CONNECTED_COMPONENT_DATA_MODIFIED.keys():
            sample_options.append(f"<option value='{s}'>{s}</option>")

    html_page = f"""
    <html>
    <head>
      <title>LASE Main Page</title>
      <style>
        body {{
          font-family: sans-serif; margin:20px;
        }}
        select, input {{
          margin:8px; padding:4px;
        }}
      </style>
    </head>
    <body>
      <h1>CC Analysis Control Panel</h1>
      <form action="/analyze-and-match-cc" method="post">
        <button name="skip_analysis" value="true">Reload Data (skip_analysis=true)</button>
        <button name="skip_analysis" value="false">Force Re-analyze (skip_analysis=false)</button>
      </form>
      <hr/>
      <h2>Show Matched Connected Component</h2>
      <form action="/show-matched-cc" method="get" target="_blank" style="border:1px solid #ccc; padding:20px;">
        <label>Sample Name:</label>
        <select name="sample_name">
          {''.join(sample_options)}
        </select><br/>
        <label>Label Index:</label>
        <input type="number" name="label_index" value="0" /><br/>
        <button type="submit">Show Matched CC</button>
      </form>
      <hr/>
      <h2>Iterate Over Labels (Multi-ID Spatial Plot)</h2>
      <form action="/plot-multi-sample-iter" method="get" target="_blank" style="border:1px solid #ccc; padding:20px;">
        <label>Sample Name:</label>
        <select name="sample_name">
          {''.join(sample_options)}
        </select><br/>
        <label>Label Index:</label>
        <input type="number" name="label_index" value="0" /><br/>
        <label>Max # of Samples:</label>
        <input type="number" name="max_samples" value="8" /><br/>
        <label>Min # of MIDs/Label:</label>
        <input type="text" name="min_mids_label" value="all" /><br/>
        <button type="submit">Generate Multi-ID Spatial Plot</button>
      </form>
    </body>
    </html>
    """
    return HTMLResponse(html_page)

#----------
@app.post("/plot-same-cc-across-samples")
def plot_same_cc_across_samples_endpoint(req: PlotSameCCRequest):
    """
    Plot the same CC across consecutive samples. Saves to PLOT_OUTPUT_FOLDER.
    Returns JSON with figure_path = /plots/....
    """
    if not CONNECTED_COMPONENT_DATA:
        raise HTTPException(status_code=400, detail="No data loaded. Run /load-and-filter-data first.")
    
    plot_dir = PLOT_OUTPUT_FOLDER  # <--- Ensure this matches the user folder route
    os.makedirs(plot_dir, exist_ok=True)

    try:
        plot_same_cc_across_samples(
            connected_component_data=CONNECTED_COMPONENT_DATA,
            filtered_data_storage=CURRENT_FILTERED_DATA,  # NEW parameter passed here
            save_directory=plot_dir,
            all_matrices=ALL_MATRICES,
            all_subcorrelations=ALL_SUBCORRELATIONS,
            start_sample=req.start_sample,
            target_label=req.target_label
        )
        fname = f"cc_starting_with_sample_{req.start_sample}_label_{req.target_label}.png"

        # subfolder = f"{sample_name}__label_{label_num}"
        # if filter_key:
        #     subfolder += "__" + filter_key

        # response = {
        #     "status": "success",
        #     "message": "Plot created successfully.",
        #     "figure_path": f"/plots/cached_plots/{subfolder}/{fname}"
        # }

        response = {
            "status": "success",
            "message": "Plot created successfully.",
            "figure_path": f"/plots/cc_matching_plots/{fname}"  # This ensures the front end can load /plots/xxx
        }
        return response
    except Exception as e:
        logger.error("[plot-same-cc-across-samples] Error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))





# new_lid_counter = 1_000_000
# new_mid_counter = 10_000_000
# Global variables
new_lid_counter = 100_000_000
new_mid_counter = 100_000_000

def load_id_counters():
    """
    Loads the counters from a JSON file if it exists.
    Otherwise, uses default values (100,000,000 and 100,000,000).
    """
    global new_lid_counter, new_mid_counter
    if os.path.exists(ID_COUNTER_FILE):
        with open(ID_COUNTER_FILE, "r") as f:
            counters = json.load(f)
        new_lid_counter = counters.get("lid_counter", 100_000_000)
        new_mid_counter = counters.get("mid_counter", 100_000_000)
    else:
        new_lid_counter = 100_000_000
        new_mid_counter = 100_000_000

def save_id_counters():
    """
    Saves the current ID counters to disk so that
    after a restart, we know what was the last used ID.
    """
    global new_lid_counter, new_mid_counter
    counters = {
        "lid_counter": new_lid_counter,
        "mid_counter": new_mid_counter
    }
    with open(ID_COUNTER_FILE, "w") as f:
        json.dump(counters, f, indent=2)

# ---- ID Generators ----
def get_new_orig_lid() -> int:
    global new_lid_counter
    new_lid_counter += 1
    new_lid = new_lid_counter
    save_id_counters()  # Persist after every increment
    return new_lid

def get_new_orig_mid() -> int:
    global new_mid_counter
    new_mid_counter += 1
    new_mid = new_mid_counter
    save_id_counters()  # Persist after every increment
    return new_mid



# ---- Utility: ensure_orig_lid/peak_id ----
def ensure_orig_lid(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "orig_lid" not in df.columns:
        df["orig_lid"] = df.index
    return df

def ensure_orig_peak_id(pks_df: pd.DataFrame) -> pd.DataFrame:
    pks_df = pks_df.copy()
    if "orig_peak_id" not in pks_df.columns:
        pks_df["orig_peak_id"] = pks_df.index
    return pks_df


## resolve the NaN error but not used in current apply_label_changes
def compute_label_diff(orig_df: pd.DataFrame, mod_df: pd.DataFrame, id_col="orig_lid"):
    """
    Same as your original, plus dropping NaNs before setting the index, so no KeyError.
    """
    if orig_df.empty and mod_df.empty:
        return {
            "deleted_lids": [],
            "added_rows": pd.DataFrame(columns=[id_col]),
            "updated_rows": pd.DataFrame(columns=[id_col])
        }
    if orig_df.empty:
        return {
            "deleted_lids": [],
            "added_rows": mod_df.copy(),
            "updated_rows": pd.DataFrame(columns=mod_df.columns)
        }
    if mod_df.empty:
        if id_col not in orig_df.columns:
            return {
                "deleted_lids": [],
                "added_rows": pd.DataFrame(columns=orig_df.columns),
                "updated_rows": pd.DataFrame(columns=orig_df.columns)
            }
        return {
            "deleted_lids": orig_df[id_col].unique().tolist(),
            "added_rows": pd.DataFrame(columns=orig_df.columns),
            "updated_rows": pd.DataFrame(columns=orig_df.columns)
        }

    if id_col not in orig_df.columns or id_col not in mod_df.columns:
        return {
            "deleted_lids": [],
            "added_rows": pd.DataFrame(columns=mod_df.columns),
            "updated_rows": pd.DataFrame(columns=mod_df.columns)
        }

    # Drop NaNs in the ID column so we never pass them to set_index
    orig_df = orig_df.dropna(subset=[id_col])
    mod_df  = mod_df.dropna(subset=[id_col])

    orig_map = orig_df.set_index(id_col, drop=False)
    mod_map  = mod_df.set_index(id_col, drop=False)

    deleted_ids = list(orig_map.index.difference(mod_map.index))
    new_ids     = list(mod_map.index.difference(orig_map.index))

    # Filter out any leftover NaNs or weird values
    deleted_ids = [x for x in deleted_ids if pd.notnull(x)]
    new_ids     = [x for x in new_ids if pd.notnull(x)]

    updated_rows_list = []
    common_ids = orig_map.index.intersection(mod_map.index)
    for cid in common_ids:
        row_o = orig_map.loc[cid]
        row_m = mod_map.loc[cid]
        if not row_o.equals(row_m):
            updated_rows_list.append(row_m)

    updated_rows_df = pd.DataFrame(updated_rows_list)
    if not updated_rows_df.empty:
        updated_rows_df.reset_index(drop=True, inplace=True)

    added_rows_df = pd.DataFrame(columns=mod_df.columns)
    if new_ids:
        added_rows_df = mod_map.loc[new_ids].copy().reset_index(drop=True)

    return {
        "deleted_lids": deleted_ids,
        "added_rows":   added_rows_df,
        "updated_rows": updated_rows_df
    }

# ---- Utility: apply_structured_diff_in_memory ----
def apply_structured_diff_in_memory(owner_dict: dict, diff: dict, key_name: str, id_col: str = "orig_lid"):
    if diff is None:
        return
    if key_name not in owner_dict or not isinstance(owner_dict[key_name], pd.DataFrame):
        owner_dict[key_name] = pd.DataFrame(columns=[id_col])

    df = owner_dict[key_name].copy()
    if id_col == "orig_lid":
        df = ensure_orig_lid(df)
    elif id_col == "orig_peak_id":
        df = ensure_orig_peak_id(df)

    deleted = diff.get("deleted_lids", [])
    if deleted and (id_col in df.columns):
        df = df[~df[id_col].isin(deleted)]

    added_rows = diff.get("added_rows", pd.DataFrame())
    if not added_rows.empty:
        if id_col == "orig_lid":
            added_rows = ensure_orig_lid(added_rows)
        elif id_col == "orig_peak_id":
            added_rows = ensure_orig_peak_id(added_rows)
        df = pd.concat([df, added_rows], ignore_index=True)

    updated_rows = diff.get("updated_rows", pd.DataFrame())
    if not updated_rows.empty and (id_col in updated_rows.columns):
        for _, row in updated_rows.iterrows():
            the_id = row[id_col]
            mask = (df[id_col] == the_id)
            for col in row.index:
                if col == id_col:
                    continue
                df.loc[mask, col] = row[col]

    owner_dict[key_name] = df

############## Load a Label ##############
@app.post("/load-label-data")
def load_label_data(payload: dict):
    """
    Loads pks/lns/mlt for (sample_name,label) from a small file in label_data/
    If not found, fallback to big structure once, then store a file.
    Then puts them in LABEL_MEMORY for ephemeral editing.
    """
    sample_name = payload.get("sample_name")
    label_val = payload.get("label")
    if not sample_name or label_val is None:
        raise HTTPException(status_code=400, detail="sample_name and label are required.")

    file_path = LABELS_DATA_FOLDER / f"{sample_name}_{label_val}.pkl"
    with data_lock:
        if file_path.exists():
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            pks_label = data["pks"].copy()
            lns_label = data["lns"].copy()
            mlt_label = data["mlt"].copy()
        else:
            # fallback: read from CONNECTED_COMPONENT_DATA_MODIFIED
            sample_data = CONNECTED_COMPONENT_DATA_MODIFIED.get(sample_name, {})
            if not sample_data:
                raise HTTPException(status_code=404, detail=f"No data for sample={sample_name} in modified data.")
            props_list = sample_data.get("props", [])
            comp = next((c for c in props_list if c["label"] == label_val), None)
            if not comp:
                raise HTTPException(status_code=404, detail=f"No label={label_val} in sample {sample_name}.")

            lns_label = comp.get("lns", pd.DataFrame())
            mlt_label = comp.get("mlt", pd.DataFrame())
            all_peaks = sample_data.get("pks", pd.DataFrame())
            lids = lns_label.index.unique()
            pks_label = all_peaks[all_peaks["lid"].isin(lids)].copy()

            # Write a new baseline file so next time it's quick
            data_save = {"pks": pks_label, "lns": lns_label, "mlt": mlt_label}
            with open(file_path, "wb") as f:
                pickle.dump(data_save, f)

        LABEL_MEMORY[(sample_name, label_val)] = {
            "pks": pks_label,
            "lns": lns_label,
            "mlt": mlt_label
        }

    return {
        "status": "ok",
        "message": f"Loaded data for {sample_name}:{label_val} into memory.",
        "num_lines": len(lns_label),
        "num_multiplets": len(mlt_label),
        "num_peaks": len(pks_label)
    }



# april 12, above is working one
@app.post("/apply-label-changes")
def apply_label_changes(payload: Dict[str, Any]):
    """
    Commits in-memory changes for each label in sample_name by:
     1. Saving the small .pkl files for each label in LABELS_DATA_FOLDER.
     2. Saving a diff file in CHANGES_FOLDER.
     3. Removing in-memory changes from LABEL_MEMORY.
     4. Clearing the cached plots/HTML for that label so that the next time the label is viewed, 
        the plots are re-generated with the updated data.

    This avoids having to re-save the full 15GB pickle on every change and ensures that
    navigation (e.g. Next/Prev) is fast because cached plots are used until new changes are applied.
    """
    sample_name = payload.get("sample_name")
    user_comment = payload.get("comment", "")
    if not sample_name:
        return {"status": "error", "message": "sample_name is required."}

    with data_lock:
        # Get all labels with in-memory changes for the given sample.
        labels_for_sample = [lbl for (s, lbl) in LABEL_MEMORY.keys() if s == sample_name]
        if not labels_for_sample:
            return {"status": "error", "message": f"No in-memory changes for sample {sample_name}"}

        diff_file = CHANGES_FOLDER / f"{sample_name}_changes_diff.pkl"
        if diff_file.exists():
            with open(diff_file, "rb") as f:
                existing_diffs = pickle.load(f)
        else:
            existing_diffs = {}

        updated_labels = []
        # Loop through each label in the sample
        for lbl in labels_for_sample:
            subset = LABEL_MEMORY[(sample_name, lbl)]
            pks_df = subset["pks"]
            lns_df = subset["lns"]
            mlt_df = subset["mlt"]

            # Save the updated data for the label to a small file.
            file_path = LABELS_DATA_FOLDER / f"{sample_name}_{lbl}.pkl"
            data_to_save = {
                "pks": pks_df,
                "lns": lns_df,
                "mlt": mlt_df
            }
            with open(file_path, "wb") as f:
                pickle.dump(data_to_save, f)

            # Record diff metadata (e.g. user comment) if needed.
            existing_diffs[lbl] = {"user_comment": user_comment}

            # Remove the in-memory changes now that they are saved.
            del LABEL_MEMORY[(sample_name, lbl)]
            updated_labels.append(lbl)

            # Clear the cache for this label so that next time the updated figures are generated.
            clear_label_cache(sample_name, lbl)

        # Save the updated diff file.
        with open(diff_file, "wb") as f:
            pickle.dump(existing_diffs, f)

    return {
        "status": "ok",
        "message": f"Applied changes for {sample_name}, labels={updated_labels}. "
                   f"Wrote each label to label_data/. Diffs in {diff_file.name}. "
                   f"Cache cleared for these labels."
    }



@app.post("/discard-label-changes")
def discard_label_changes(payload: Dict[str, Any]):
    """
    Remove (sample_name, label) from LABEL_MEMORY and also remove
    any corresponding entries from the changes file.
    This effectively reverts all local edits for that label.
    """
    sample_name = payload.get("sample_name")
    label_val = payload.get("label")
    if not sample_name or label_val is None:
        return {"status": "error", "message": "sample_name and label are required."}

    with data_lock:
        # Remove in-memory changes
        if (sample_name, label_val) in LABEL_MEMORY:
            del LABEL_MEMORY[(sample_name, label_val)]
        else:
            return {
                "status": "error",
                "message": f"No in-memory data to discard for {sample_name}:{label_val}."
            }
        
        # Remove any diff entries for this label from the changes file
        changes_file = os.path.join(CHANGES_FOLDER, f"{sample_name}_changes.pkl")
        if os.path.exists(changes_file):
            with open(changes_file, "rb") as f:
                diff_dict = pickle.load(f)
            if label_val in diff_dict:
                del diff_dict[label_val]
            with open(changes_file, "wb") as f:
                pickle.dump(diff_dict, f)

    return {
        "status": "ok",
        "message": f"Discarded in-memory edits and cleared diff data for {sample_name}:{label_val}."
    }
############## (Optional) Final Combination ##############
@app.post("/finalize-sample")
def finalize_sample(payload: dict):
    """
    Once all labeling is done for sample_name,
    combine each label .pkl into 3 dataframes: all_pks, all_lns, all_mlt.
    Store them in one final file or return them, your choice.
    """
    sample_name = payload.get("sample_name")
    if not sample_name:
        return {"status":"error","message":"sample_name is required"}

    label_files = LABELS_DATA_FOLDER.glob(f"{sample_name}_*.pkl")
    pks_list, lns_list, mlt_list = [], [], []
    for lf in label_files:
        with open(lf,"rb") as f:
            data = pickle.load(f)
        pks_list.append(data["pks"])
        lns_list.append(data["lns"])
        mlt_list.append(data["mlt"])

    all_pks = pd.concat(pks_list, ignore_index=True)
    all_lns = pd.concat(lns_list, ignore_index=False)
    all_mlt = pd.concat(mlt_list, ignore_index=False)

    final_file = ANALYSIS_OUTPUT_FOLDER / f"{sample_name}_final.pkl"
    with open(final_file,"wb") as f:
        pickle.dump({"pks":all_pks, "lns":all_lns, "mlt":all_mlt}, f)

    return {
        "status":"ok",
        "message":f"Final data => {final_file.name}, pks={len(all_pks)}, lns={len(all_lns)}, mlt={len(all_mlt)}"
    }
from copy import deepcopy

# new 2
# ---------------------------------------------------------------------------
#  FINAL, BULLET-PROOF merge_label_memory_for_plot
# ---------------------------------------------------------------------------
from copy import deepcopy
import pandas as pd
import numpy as np

def merge_label_memory_for_plot(sample_name: str,
                                label_val:   int,
                                data_source: dict) -> dict:
    """
    Return a copy of `data_source` in which the single label (sample_name,label_val)
    is replaced by the in-memory version stored in LABEL_MEMORY, **always leaving
    DataFrames – never dicts – in 'lns', 'mlt' and 'pks'.**
    """
    key = (sample_name, label_val)
    if key not in LABEL_MEMORY:
        return data_source         # nothing to merge

    subset = LABEL_MEMORY[key]     # {'lns': DF, 'mlt': DF, 'pks': DF, …}
    assert isinstance(subset["lns"], pd.DataFrame), "subset['lns'] must be a DataFrame"

    merged = deepcopy(data_source)           # cheap enough for one sample
    samp   = merged[sample_name]             # the dict for that sample

    # ── 1. overwrite / add the component inside samp["props"] ─────────────────
    in_props = False
    for comp in samp["props"]:
        if comp["label"] == label_val:
            comp["lns"] = subset["lns"].copy()
            comp["mlt"] = subset["mlt"].copy()
            in_props = True
            break
    if not in_props:
        samp["props"].append({
            "label": label_val,
            "lns":   subset["lns"].copy(),
            "mlt":   subset["mlt"].copy()
        })

    # ── 2. rebuild sample-level lns (drop old lids, append new) ───────────────
    old_lids = subset["lns"].index.astype(int).unique()

    original_lns = samp.get("lns", pd.DataFrame())
    new_lns      = pd.concat(
        [original_lns.drop(old_lids, errors="ignore"),
         subset["lns"]],
        axis=0, sort=False
    )
    new_lns.index      = new_lns.index.astype(int)
    new_lns.index.name = "lid"

    # *** safety net ***
    assert isinstance(new_lns, pd.DataFrame), "samp['lns'] became non-DataFrame!"
    samp["lns"] = new_lns

    # ── 3. rebuild sample-level pks similarly ────────────────────────────────
    original_pks = samp.get("pks", pd.DataFrame())
    samp["pks"]  = pd.concat(
        [original_pks[~original_pks["lid"].isin(old_lids)],
         subset["pks"]],
        ignore_index=True
    )

    # ── 4. finally recalc mlt so it matches lns exactly ──────────────────────
    assert isinstance(samp["lns"], pd.DataFrame), f"samp['lns'] became {type(samp['lns'])}"
    # recalc_mlt_for_label(samp)
    normalize_cc_tables(subset)
    # samp["mlt"] = recalc_mlt_for_label(samp["lns"])

    # recalc_mlt_for_label(samp)     # this now receives a valid DataFrame

    return merged



@app.get("/get-label-data")
def get_label_data(sample_name: str, label: int):
    """
    Return the small in-memory data for (sample_name, label).
    """
    with data_lock:
        subset = LABEL_MEMORY.get((sample_name, label))
        if not subset:
            raise HTTPException(status_code=404, detail=f"No label data in memory for {sample_name}:{label}")

        lns_json = subset["lns"].to_dict(orient="records")
        mlt_json = subset["mlt"].to_dict(orient="records")
        pks_json = subset["pks"].to_dict(orient="records")
        return {
            "status": "ok",
            "lns": lns_json,
            "mlt": mlt_json,
            "pks": pks_json
        }

# --------------------------------------------------------------------------
# NEW: "Commit" the local changes back to the big data store
# --------------------------------------------------------------------------
@app.post("/commit-label-changes")
def commit_label_changes(payload: dict):
    """
    Once user is done editing (sample_name, label) in memory, 
    we merge it back into the big data and remove from LABEL_MEMORY.
    """
    sample_name = payload.get("sample_name")
    label_val = payload.get("label")
    if (sample_name, label_val) not in LABEL_MEMORY:
        raise HTTPException(status_code=404, detail=f"No label data in memory for {sample_name}:{label_val}")

    with data_lock:
        subset = LABEL_MEMORY[(sample_name, label_val)]
        # 1) "lns" => we remove old lines for that label from the big data, then append the new subset["lns"]
        #    same for "mlt", "pks" as needed. 
        #    This is a placeholder approach that depends on how you store your big data

        # For demonstration, let's do:
        # remove old from CONNECTED_COMPONENT_DATA_MODIFIED, then add new lines
        big_sample = CONNECTED_COMPONENT_DATA_MODIFIED.get(sample_name, {})
        props = big_sample.get("props", [])
        for comp in props:
            if comp.get("label") == label_val:
                # Overwrite the lns, mlt in the big data:
                comp["lns"] = subset["lns"].copy()
                comp["mlt"] = subset["mlt"].copy()
                # For peaks, we assume they're in big_sample["pks"], so let's handle that
        # Now handle pks:
        all_peaks = big_sample.get("pks", pd.DataFrame())
        # remove any peaks belonging to the old lids in that label
        old_lids = subset["lns"].index.unique()
        all_peaks = all_peaks[~all_peaks["lid"].isin(old_lids)]
        # then append new pks
        all_peaks = pd.concat([all_peaks, subset["pks"]], ignore_index=True)
        big_sample["pks"] = all_peaks

        # store back
        CONNECTED_COMPONENT_DATA_MODIFIED[sample_name] = big_sample

        # 2) optionally save the big data to disk if you want:
        # with open(MODIFIED_SAVE_PATH, "wb") as f:
        #     pickle.dump(CONNECTED_COMPONENT_DATA_MODIFIED, f)

        # 3) remove from LABEL_MEMORY
        del LABEL_MEMORY[(sample_name, label_val)]

        return {"status": "ok", "message": f"Committed changes for {sample_name}:{label_val}"}



# ------------ constant for E ↔︎ λ conversion -------------------------------
from scipy.constants import h, c, e
KE      = 1e12 * h * c / e          # nm → meV
FWHME0  = 0.615

# ---------------------------------------------------------------------------
def build_line_from_peaks(pks: pd.DataFrame) -> dict:
    """
    Re-compute all per-line numeric metrics from the peaks that belong
    to that line.  Returns *plain* python scalars, never numpy types
    (avoids later dtype surprises).
    """
    if pks.empty:
        raise ValueError("No peaks given to build_line_from_peaks()")

    # weighted mean energy (same algorithm you already use)
    weights = 1.0 / np.square(0.1 + np.abs(pks["fwhmE"] - FWHME0))
    E0      = np.average(pks["E"], weights=weights)

    return {
        "i"   : float(np.average(pks["i"], weights=pks["a"])),
        "j"   : float(np.average(pks["j"], weights=pks["a"])),
        "k"   : float(np.average(pks["k"], weights=pks["a"])),
        "a"   : float(pks["a"].sum()),
        "wl"  : float(KE / E0),
        "dwl" : float(pks["wl"].std(ddof=0)),
        "E"   : float(E0),
        "dE"  : float(pks["E"].std(ddof=0)),
        "ph"  : float(pks["ph"].sum()),
        "n"   : int(len(pks)),
        "peri": float(
            pks["i"].max() - pks["i"].min()
          + pks["j"].max() - pks["j"].min()
        ),
        # “mid” is filled in the calling code
    }


# ---------------------------------------------------------------------------
def recalc_mlt_for_label(lns: pd.DataFrame) -> pd.DataFrame:
    """
    Return an `mlt` dataframe (index = mid, int64) rebuilt from *lns*.
    """
    if lns.empty or "mid" not in lns.columns:
        return pd.DataFrame(dtype=float)

    rows = (
        lns
        .groupby("mid", dropna=True)
        .agg({
            "i": "mean",
            "j": "mean",
            "k": "mean",
            "mid": "size"          # will be renamed → n
        })
        .rename(columns={"mid": "n"})
        .reset_index()
        .set_index("mid")
        .astype({"n": "int64"})
    )

    # ensure index is int64 (Int64Index, not float64)
    rows.index = rows.index.astype("int64")
    return rows


# ---------------------------------------------------------------------------
def normalize_cc_tables(comp: Dict[str, pd.DataFrame]) -> None:
    """
    Make *comp["lns"]* and *comp["mlt"]* conform to the invariants.
    (Modifies *comp* in-place.)
    """
    if "lns" in comp and not comp["lns"].empty:
        lns = comp["lns"].copy()
        # 1)  force the index to be int64 and unique
        lns.index = lns.index.astype("int64")
        lns.index.name = "lid"        #    ← **make the index name** "lid"
        lns.sort_index(inplace=True)
        comp["lns"] = lns

    # 2)  rebuild mlt so its index is the MID (int64)
    comp["mlt"] = recalc_mlt_for_label(comp["lns"])


@app.post("/split-lid-dbscan")
def split_lid_dbscan_endpoint(req: Dict[str, Any]):
    sample_name = req.get("sample_name")
    label = req.get("target_label")
    old_lid = req.get("old_lid")
    eps = float(req.get("eps", 0.5))
    min_samples = int(req.get("min_samples", 1))
    split_comment = req.get("split_comment", "")

    with data_lock:
        subset = LABEL_MEMORY.get((sample_name, label))
        if not subset:
            raise HTTPException(
                status_code=404,
                detail=f"No in-memory data for {sample_name}:{label}"
            )
        lns_df = subset["lns"]
        pks_df = subset["pks"]

        if old_lid not in lns_df.index:
            raise HTTPException(
                status_code=404,
                detail=f"LID {old_lid} not found for {sample_name}:{label}"
            )

        old_row = lns_df.loc[old_lid]
        sub_peaks = pks_df[pks_df["lid"] == old_lid]
        if sub_peaks.empty:
            return {"status": "error", "message": "No peaks for old_lid; cannot split."}

        from sklearn.cluster import DBSCAN
        X = sub_peaks[["wl", "fwhm"]].values
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels_ = db.fit_predict(X)
        unique_clusters = set(labels_) - {-1}

        if len(unique_clusters) <= 1:
            return {"status": "error", "message": "DBSCAN found <=1 cluster; no split."}

        # ---- Backup in subset memory (optional) ----
        if "split_backups" not in subset:
            subset["split_backups"] = {}
        subset["split_backups"][old_lid] = old_row.copy()

        # Remove old lid from lines
        lns_df = lns_df.drop(old_lid)

        # We'll track newly created lids so we can "undo" if needed
        new_lines = []
        new_lids_created = []

        for cluster_id in sorted(unique_clusters):
            c_idx = sub_peaks.index[labels_ == cluster_id]
            if len(c_idx) == 0:
                continue

            new_lid = get_new_orig_lid()
            # new_line = old_row.copy()
            # new_line.name = new_lid
            # new_line["orig_lid"] = new_lid
            # new_line["split_comment"] = split_comment
            # print(">>> split-lid-dbscan: sub_peaks['wl'] values:\n", sub_peaks["wl"].head())
            # print(">>> any non-null wl in sub_peaks?", sub_peaks["wl"].notna().any())

            # new_line["wl"] = sub_peaks.loc[c_idx, "wl"].mean()

            # # Reassign peaks
            # pks_df.loc[c_idx, "lid"] = new_lid
            # new_lines.append(new_line)
            # new_lids_created.append(new_lid)
            # april 25
            pks_df.loc[c_idx, "lid"] = new_lid            # 1.  re-tag the peaks

            # 2.  build a *fresh* line record from those peaks
            stats = build_line_from_peaks(pks_df.loc[c_idx])
            # dump out wl & E so you can see them in your console/log
            print(f">>> split-lid-dbscan: created LID={new_lid}  wl={stats['wl']:.3f}  E={stats['E']:.3f}")

            new_line = pd.Series(stats, name=int(new_lid))
            new_line["mid"]         = int(old_row["mid"])
            new_line["orig_lid"]    = int(new_lid)
            new_line["split_comment"] = split_comment

            new_lines.append(new_line)
            new_lids_created.append(new_lid)

        # Add the new lines to lns
        # lns_df = pd.concat([lns_df, pd.DataFrame(new_lines)], ignore_index=False)


        if new_lines:                                   # <-- append *all* of them
            new_df = pd.DataFrame(new_lines)
            new_df.index = new_df.index.astype("int64") # crucial
            lns_df = pd.concat([lns_df, new_df], sort=False)

        subset["lns"] = lns_df
        assert isinstance(subset["lns"], pd.DataFrame), "Programming error: lns must stay a DataFrame"
        subset["pks"] = pks_df
        # recalc_mlt_for_label(subset)
        # *** one single call that enforces both invariants ***
        normalize_cc_tables(subset)
        # april 25, bad fix
        # subset["lns"] = _populate_missing_line_metrics(
        # subset["lns"], subset["pks"]
    # )

        # ---- Record in changes file ----
        changes_file = os.path.join(CHANGES_FOLDER, f"{sample_name}_changes.pkl")
        if os.path.exists(changes_file):
            with open(changes_file, "rb") as f:
                diff_dict = pickle.load(f)
        else:
            diff_dict = {}

        # Make sure to initialize label dict
        if label not in diff_dict:
            diff_dict[label] = {}

        # Make sure each subkey is initialized
        if "split_backups" not in diff_dict[label]:
            diff_dict[label]["split_backups"] = {}
        if "split_new_lids" not in diff_dict[label]:
            diff_dict[label]["split_new_lids"] = {}

        # Save old lid backup
        diff_dict[label]["split_backups"][old_lid] = old_row.copy()
        # Also store which new lids replaced old_lid
        diff_dict[label]["split_new_lids"][old_lid] = new_lids_created

        with open(changes_file, "wb") as f:
            pickle.dump(diff_dict, f)

        # ── NEW: clear *that* label’s cache folder ───────────────
        clear_label_cache(sample_name, label)
        # ───────────────────────────────────────────────────────────
        return {
            "status": "ok",
            "message": f"Split LID {old_lid} into {len(unique_clusters)} lines. Created {new_lids_created}"
        }

@app.post("/undo-split-lid")
def undo_split_lid_endpoint(req: Dict[str, Any]):
    sample_name = req.get("sample_name")
    label = req.get("target_label")
    old_lid = req.get("old_lid")

    with data_lock:
        # 1) Load subset
        subset = LABEL_MEMORY.get((sample_name, label))
        if not subset:
            raise HTTPException(
                status_code=404,
                detail=f"No in-memory data for {sample_name}:{label}"
            )
        lns_df = subset["lns"]
        pks_df = subset["pks"]

        # 2) Load changes
        changes_file = os.path.join(CHANGES_FOLDER, f"{sample_name}_changes.pkl")
        if not os.path.exists(changes_file):
            raise HTTPException(status_code=400, detail="No changes file found; cannot undo.")

        with open(changes_file, "rb") as f:
            diff_dict = pickle.load(f)

        if label not in diff_dict or "split_backups" not in diff_dict[label] \
           or "split_new_lids" not in diff_dict[label]:
            raise HTTPException(status_code=400, detail="No split info to undo for this label.")

        if old_lid not in diff_dict[label]["split_backups"]:
            raise HTTPException(status_code=400, detail=f"No backup for old_lid={old_lid} in splits.")

        old_row = diff_dict[label]["split_backups"][old_lid]
        new_lids = diff_dict[label]["split_new_lids"][old_lid]

        # 3) Remove new lids from lns, revert their peaks to old_lid
        for nlid in new_lids:
            if nlid in lns_df.index:
                lns_df = lns_df.drop(nlid)
            pks_df.loc[pks_df["lid"] == nlid, "lid"] = old_lid

        # 4) Re-insert old lid row
        if old_lid not in lns_df.index:
            # Insert it
            lns_df = pd.concat([lns_df, pd.DataFrame([old_row], index=[old_lid])], ignore_index=False)

        subset["lns"] = lns_df
        subset["pks"] = pks_df
        # recalc_mlt_for_label(subset)
        normalize_cc_tables(subset)


        # 5) Remove the split info from diff_dict if desired
        diff_dict[label]["split_backups"].pop(old_lid, None)
        diff_dict[label]["split_new_lids"].pop(old_lid, None)

        with open(changes_file, "wb") as f:
            pickle.dump(diff_dict, f)

        # ── NEW: clear *that* label’s cache folder ───────────────
        clear_label_cache(sample_name, label)
        # ───────────────────────────────────────────────────────────

        return {
            "status": "ok",
            "message": f"Undo split for old_lid={old_lid}; removed {new_lids} and restored {old_lid}"
        }


# --------------------------------------------------------------------------
# EXAMPLE: Merge LIDs from in-memory data
# --------------------------------------------------------------------------
@app.post("/delete-lid")
def delete_lid_endpoint(req: Dict[str, Any]):
    sample_name = req.get("sample_name")
    label = req.get("label")
    lid_to_delete = req.get("lid_to_delete")
    reason = req.get("reason", "")

    with data_lock:
        subset = LABEL_MEMORY.get((sample_name, label))
        if not subset:
            raise HTTPException(
                status_code=404,
                detail=f"No in-memory data for (sample={sample_name}, label={label})."
            )
        lns_df = subset["lns"]
        pks_df = subset["pks"]

        if lid_to_delete not in lns_df.index:
            raise HTTPException(
                status_code=400,
                detail=f"LID {lid_to_delete} not found for (sample={sample_name}, label={label})."
            )

        # If the lid is already removed from the DataFrame, skip
        # (But usually we'd see it not in index anyway.)
        row_backup = lns_df.loc[lid_to_delete].copy()
        pks_backup = pks_df[pks_df["lid"] == lid_to_delete].copy()

        # Remove from memory
        lns_df = lns_df.drop(lid_to_delete)
        pks_df = pks_df[pks_df["lid"] != lid_to_delete]

        subset["lns"] = lns_df
        subset["pks"] = pks_df
        # recalc_mlt_for_label(subset)
        normalize_cc_tables(subset)

        # ---- Store in changes file ----
        changes_file = os.path.join(CHANGES_FOLDER, f"{sample_name}_changes.pkl")
        if os.path.exists(changes_file):
            with open(changes_file, "rb") as f:
                diff_dict = pickle.load(f)
        else:
            diff_dict = {}

        if label not in diff_dict:
            diff_dict[label] = {}
        if "deleted_lids" not in diff_dict[label]:
            diff_dict[label]["deleted_lids"] = []
        if "delete_backups" not in diff_dict[label]:
            diff_dict[label]["delete_backups"] = {}
        if "delete_backups_pks" not in diff_dict[label]:
            diff_dict[label]["delete_backups_pks"] = {}

        # If it's already in deleted_lids, skip to avoid duplicates
        if lid_to_delete not in diff_dict[label]["deleted_lids"]:
            diff_dict[label]["deleted_lids"].append(lid_to_delete)
            row_backup["delete_reason"] = reason
            diff_dict[label]["delete_backups"][lid_to_delete] = row_backup
            diff_dict[label]["delete_backups_pks"][lid_to_delete] = pks_backup

            with open(changes_file, "wb") as f:
                pickle.dump(diff_dict, f)

        # ── NEW: clear *that* label’s cache folder ───────────────
        clear_label_cache(sample_name, label)
        # ───────────────────────────────────────────────────────────

        return {
            "status": "ok",
            "message": f"Deleted LID {lid_to_delete}. reason={reason}"
        }


@app.post("/undo-delete-lid")
def undo_delete_lid_endpoint(req: Dict[str, Any]):
    sample_name = req.get("sample_name")
    label = req.get("label")
    lid_to_restore = req.get("lid_to_restore")

    with data_lock:
        subset = LABEL_MEMORY.get((sample_name, label))
        if not subset:
            raise HTTPException(status_code=404, detail="No in-memory data for that sample/label.")

        changes_file = os.path.join(CHANGES_FOLDER, f"{sample_name}_changes.pkl")
        if not os.path.exists(changes_file):
            raise HTTPException(status_code=400, detail="No changes file found; cannot undo delete.")

        with open(changes_file, "rb") as f:
            diff_dict = pickle.load(f)

        # Validate structure
        if label not in diff_dict:
            raise HTTPException(status_code=400, detail="No label in diff_dict to undo.")
        if "deleted_lids" not in diff_dict[label] or "delete_backups" not in diff_dict[label] \
           or "delete_backups_pks" not in diff_dict[label]:
            raise HTTPException(status_code=400, detail="No delete info found to undo for this label.")

        # Check if that lid is indeed in the deleted list
        if lid_to_restore not in diff_dict[label]["deleted_lids"]:
            raise HTTPException(status_code=400, detail=f"LID {lid_to_restore} was not in deleted_lids.")

        row_backup = diff_dict[label]["delete_backups"][lid_to_restore]
        pks_backup = diff_dict[label]["delete_backups_pks"][lid_to_restore]

        lns_df = subset["lns"]
        pks_df = subset["pks"]

        # Re-insert
        if lid_to_restore not in lns_df.index:
            row_backup_df = pd.DataFrame([row_backup], index=[lid_to_restore])
            lns_df = pd.concat([lns_df, row_backup_df], ignore_index=False)

        pks_df = pd.concat([pks_df, pks_backup], ignore_index=False)

        subset["lns"] = lns_df
        subset["pks"] = pks_df
        # recalc_mlt_for_label(subset)
        normalize_cc_tables(subset)

        # Remove from changes
        diff_dict[label]["deleted_lids"].remove(lid_to_restore)
        diff_dict[label]["delete_backups"].pop(lid_to_restore, None)
        diff_dict[label]["delete_backups_pks"].pop(lid_to_restore, None)

        with open(changes_file, "wb") as f:
            pickle.dump(diff_dict, f)

                # ── NEW: clear *that* label’s cache folder ───────────────
        clear_label_cache(sample_name, label)
        # ───────────────────────────────────────────────────────────

        return {
            "status": "ok",
            "message": f"Successfully restored LID {lid_to_restore}"
        }



@app.post("/merge-lids")
def merge_lids(req: Dict[str, Any]):
    sample_name = req.get("sample_name")
    label = req.get("target_label")
    lids_to_merge = req.get("lids", [])
    merge_comment = req.get("merge_comment", "")

    if not sample_name or not lids_to_merge:
        return {"status": "error", "message": "Must provide sample_name and lids array."}

    with data_lock:
        subset = LABEL_MEMORY.get((sample_name, label))
        if not subset:
            raise HTTPException(
                status_code=404,
                detail=f"No in-memory data for {sample_name}:{label}."
            )

        lns_df = subset["lns"]
        pks_df = subset["pks"]

        not_found = [lid for lid in lids_to_merge if lid not in lns_df.index]
        if not_found:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot merge. LIDs {not_found} are not in the in-memory subset."
            )

        subset_df = lns_df.loc[lids_to_merge]
        if subset_df.empty:
            return {"status": "error", "message": f"No matching lines found for {lids_to_merge}"}

        old_mids = subset_df["mid"].dropna().unique()
        if len(old_mids) == 0:
            chosen_mid = np.nan
        else:
            mid_counts = subset_df.groupby("mid").size().sort_values(ascending=False)
            chosen_mid = mid_counts.index[0]

   
        # april 25
                # 1) re-tag all the peaks into the new LID
        new_lid = get_new_orig_lid()
        pks_df.loc[pks_df["lid"].isin(lids_to_merge), "lid"] = new_lid

        # 2) build a fresh line record from those merged peaks
        merged_peaks = pks_df[pks_df["lid"] == new_lid]
        stats = build_line_from_peaks(merged_peaks)
        # ── DEBUG: print the new wl and E for verification ────────────────
        print(f">>> merge-lids: created LID={new_lid}  wl={stats['wl']:.3f}  E={stats['E']:.3f}")
        # ──────────────────────────────────────────────────────────────────

        # 3) assemble the new LNS row
        new_line = pd.Series(stats, name=new_lid)
        new_line["mid"]           = chosen_mid
        new_line["merge_comment"] = merge_comment
        new_line["orig_lid"]      = new_lid

        # 4) drop the old lids and append the newly computed one
        old_lines_backup = subset_df.copy()
        lns_df = lns_df.drop(lids_to_merge)
        lns_df = pd.concat([lns_df, pd.DataFrame([new_line])], axis=0)

        # 5) commit back into memory and recompute multiplets
        subset["lns"] = lns_df
        subset["pks"] = pks_df
        # recalc_mlt_for_label(subset)
        normalize_cc_tables(subset)


        # Changes file
        changes_file = os.path.join(CHANGES_FOLDER, f"{sample_name}_changes.pkl")
        if os.path.exists(changes_file):
            with open(changes_file, "rb") as f:
                diff_dict = pickle.load(f)
        else:
            diff_dict = {}

        if label not in diff_dict:
            diff_dict[label] = {}
        if "merged_lids" not in diff_dict[label]:
            diff_dict[label]["merged_lids"] = []
        if "merge_backups" not in diff_dict[label]:
            diff_dict[label]["merge_backups"] = {}
        if "merge_backup_pks" not in diff_dict[label]:
            diff_dict[label]["merge_backup_pks"] = {}

        # This records (old lids -> new lid) plus a backup of old lines
        diff_dict[label]["merged_lids"].append((lids_to_merge, new_lid))

        # Store lines + peaks backup for an undo operation
        diff_dict[label]["merge_backups"][new_lid] = old_lines_backup
        diff_dict[label]["merge_backup_pks"][new_lid] = pks_df[pks_df["lid"] == new_lid].copy()

        with open(changes_file, "wb") as f:
            pickle.dump(diff_dict, f)

                # ── NEW: clear *that* label’s cache folder ───────────────
        clear_label_cache(sample_name, label)
        # ───────────────────────────────────────────────────────────
        return {
            "status": "ok",
            "message": f"Merged lids {lids_to_merge} => new lid {new_lid}, mid={chosen_mid}"
        }

@app.post("/undo-merge-lids")
def undo_merge_lids_endpoint(req: Dict[str, Any]):
    sample_name = req.get("sample_name")
    label = req.get("target_label")
    new_lid = req.get("merged_lid")  # The newly created lid

    with data_lock:
        subset = LABEL_MEMORY.get((sample_name, label))
        if not subset:
            raise HTTPException(status_code=404, detail="No in-memory data for that sample/label.")

        changes_file = os.path.join(CHANGES_FOLDER, f"{sample_name}_changes.pkl")
        if not os.path.exists(changes_file):
            raise HTTPException(status_code=400, detail="No changes file found; cannot undo merge.")

        with open(changes_file, "rb") as f:
            diff_dict = pickle.load(f)

        if label not in diff_dict or "merge_backups" not in diff_dict[label]:
            raise HTTPException(status_code=400, detail="No merge info to undo.")

        if new_lid not in diff_dict[label]["merge_backups"]:
            raise HTTPException(status_code=400, detail=f"No backup found for new_lid={new_lid}.")

        # The old lines backup
        old_lines_backup = diff_dict[label]["merge_backups"][new_lid]
        # Possibly you'd want separate backups for each old lid's peaks,
        # but this sample code lumps them together.
        new_lid_pks_backup = diff_dict[label]["merge_backup_pks"].get(new_lid, pd.DataFrame())

        lns_df = subset["lns"]
        pks_df = subset["pks"]

        # Remove the merged line
        if new_lid in lns_df.index:
            lns_df = lns_df.drop(new_lid)

        # Any peaks that belong to new_lid need to revert to their original lids
        # But we didn't store the "original" lids. So if you want perfect undo,
        # you must store a mapping of each peak to its old lid. That is more involved.
        # For now, we'll just remove those peaks from pks, re-insert the old lines' peaks
        # from new_lid_pks_backup if you wish. Or skip if you can't restore precisely.

        # Re-insert old lines
        # old_lines_backup is a DataFrame indexed by old lids
        lns_df = pd.concat([lns_df, old_lines_backup], ignore_index=False)

        # If you want to do something with peaks, you'd store them per-lid. This is simplified
        # Recalc
        subset["lns"] = lns_df
        subset["pks"] = pks_df
        # recalc_mlt_for_label(subset)
        normalize_cc_tables(subset)

        # Remove the entry from changes
        diff_dict[label]["merge_backups"].pop(new_lid, None)
        diff_dict[label]["merge_backup_pks"].pop(new_lid, None)
        # Also remove the (lids, new_lid) tuple from merged_lids if you want:
        merged_list = diff_dict[label].get("merged_lids", [])
        merged_list = [pair for pair in merged_list if pair[1] != new_lid]
        diff_dict[label]["merged_lids"] = merged_list

        with open(changes_file, "wb") as f:
            pickle.dump(diff_dict, f)
                # ── NEW: clear *that* label’s cache folder ───────────────
        clear_label_cache(sample_name, label)
        # ───────────────────────────────────────────────────────────
        return {
            "status": "ok",
            "message": f"Undo merge for new_lid={new_lid} completed."
        }

# --------------------------------------------------------------------------
# EXAMPLE: Assign Sub-MID from in-memory data
# --------------------------------------------------------------------------
@app.post("/assign-submid")
def assign_submid_endpoint(req: dict):
    sample_name = req.get("sample_name")
    label = req.get("target_label")
    lids = req.get("lids", [])
    user_mid_str = req.get("new_sub_mid", "new")
    comment = req.get("comment", "")

    with data_lock:
        subset = LABEL_MEMORY.get((sample_name, label))
        if not subset:
            raise HTTPException(status_code=404, detail=f"No in-memory data for {sample_name}:{label}")

        lns_df = subset["lns"]

        # For undo, we must store the old mid
        # So let's gather old mids for these lids
        old_mids = lns_df.loc[lids, "mid"].copy() if len(lids) > 0 else pd.Series(dtype=object)

        if user_mid_str.lower() == "new":
            new_mid = get_new_orig_mid()
        else:
            try:
                new_mid = int(user_mid_str)
            except:
                new_mid = get_new_orig_mid()

        row_mask = lns_df.index.isin(lids)
        lns_df.loc[row_mask, "mid"] = new_mid
        lns_df.loc[row_mask, "assign_comment"] = comment

        subset["lns"] = lns_df
        # recalc_mlt_for_label(subset)
        normalize_cc_tables(subset)

        # store changes in changes_file
        changes_file = os.path.join(CHANGES_FOLDER, f"{sample_name}_changes.pkl")
        if os.path.exists(changes_file):
            with open(changes_file, "rb") as f:
                diff_dict = pickle.load(f)
        else:
            diff_dict = {}

        if label not in diff_dict:
            diff_dict[label] = {}

        # always initialize the subkey
        if "assign_submid" not in diff_dict[label]:
            diff_dict[label]["assign_submid"] = []

        # We store a record with old mids
        # Convert old_mids (Series) to a dict: { lid: old_mid }
        old_mids_dict = old_mids.to_dict()
        diff_dict[label]["assign_submid"].append({
            "lids": lids,
            "old_mids": old_mids_dict,
            "new_mid": new_mid,
            "comment": comment
        })

        with open(changes_file, "wb") as f:
            pickle.dump(diff_dict, f)
                # ── NEW: clear *that* label’s cache folder ───────────────
        clear_label_cache(sample_name, label)
        # ───────────────────────────────────────────────────────────
        return {
            "status": "ok",
            "message": f"Reassigned lids {lids} to new mid {new_mid}. comment={comment}"
        }

@app.post("/undo-submid")
def undo_submid_endpoint(req: dict):
    sample_name = req.get("sample_name")
    label = req.get("target_label")

    with data_lock:
        subset = LABEL_MEMORY.get((sample_name, label))
        if not subset:
            raise HTTPException(status_code=404, detail="No in-memory data for that sample/label.")

        changes_file = os.path.join(CHANGES_FOLDER, f"{sample_name}_changes.pkl")
        if not os.path.exists(changes_file):
            raise HTTPException(status_code=400, detail="No changes file found; cannot undo submid.")

        with open(changes_file, "rb") as f:
            diff_dict = pickle.load(f)

        if label not in diff_dict or "assign_submid" not in diff_dict[label]:
            raise HTTPException(status_code=400, detail="No submid assignments found to undo.")

        # We'll pop the last assignment from the list (LIFO). Or you could pass an index in the request
        if len(diff_dict[label]["assign_submid"]) == 0:
            return {"status": "error", "message": "No submid changes to undo."}

        last_record = diff_dict[label]["assign_submid"].pop()
        lids = last_record["lids"]
        old_mids = last_record["old_mids"]  # { lid: old_mid }

        lns_df = subset["lns"]
        # Revert each lid to its old mid
        for lid in lids:
            if lid in lns_df.index and lid in old_mids:
                lns_df.loc[lid, "mid"] = old_mids[lid]

        subset["lns"] = lns_df
        # recalc_mlt_for_label(subset)
        normalize_cc_tables(subset)

        # Save changes
        with open(changes_file, "wb") as f:
            pickle.dump(diff_dict, f)
                # ── NEW: clear *that* label’s cache folder ───────────────
        clear_label_cache(sample_name, label)
        # ───────────────────────────────────────────────────────────
        return {
            "status": "ok",
            "message": f"Reverted submid assignment for lids={lids}, restored old mids."
        }


# --------------------------------------------------------------------------
# Done. The above endpoints let you do small in-memory changes. 
# --------------------------------------------------------------------------

# -------------- /get-samples --------------
@app.get("/get-samples", response_class=JSONResponse)
def get_samples():
    """
    Returns a list of available sample names, sorted numerically.
    """
    samples = list(CONNECTED_COMPONENT_DATA_ORIG.keys())
    def numeric_sort_key(sname: str) -> int:
        import re
        match_ = re.match(r"(\d+)", sname)
        if match_:
            return int(match_.group(1))
        return 999999
    samples.sort(key=numeric_sort_key)
    return {"samples": samples}

@app.get("/list-labels")
def list_labels(sample_name: str):
    """
    Lists all 'label' values for a given sample in CONNECTED_COMPONENT_DATA_MODIFIED.
    """
    if not CONNECTED_COMPONENT_DATA_MODIFIED:
        raise HTTPException(status_code=400, detail="No data loaded.")
    if sample_name not in CONNECTED_COMPONENT_DATA_MODIFIED:
        raise HTTPException(status_code=404, detail=f"Sample '{sample_name}' not found.")
    props = CONNECTED_COMPONENT_DATA_MODIFIED[sample_name].get("props", [])
    # Each component in 'props' has a 'label' key.
    # Return them all as a simple list.
    return {"labels": [p["label"] for p in props]}

@app.get("/list-lids")
def list_lids(sample_name: str, target_label: int):
    """
    Lists the LIDs (line IDs) for a specific (sample_name, target_label).
    This reads from the big data structure CONNECTED_COMPONENT_DATA_MODIFIED,
    finds the component with 'label' == target_label, and returns the unique LIDs.
    """
    if not CONNECTED_COMPONENT_DATA_MODIFIED:
        raise HTTPException(status_code=400, detail="No data loaded.")
    if sample_name not in CONNECTED_COMPONENT_DATA_MODIFIED:
        raise HTTPException(status_code=404, detail=f"Sample '{sample_name}' not found.")
    props = CONNECTED_COMPONENT_DATA_MODIFIED[sample_name].get("props", [])
    target_comp = next((c for c in props if c["label"] == target_label), None)
    if not target_comp:
        raise HTTPException(status_code=404, detail=f"Label={target_label} not found.")
    # target_comp["lns"] is a DataFrame whose index is LID
    lids_ = list(target_comp["lns"].index.unique())
    return {"lids": lids_}



@app.get("/line-peaks-view", response_class=HTMLResponse)
def line_peaks_view(
    sample_name: str,
    target_label: int,
    specific_lid: Optional[int] = None,
    use_modified_data: bool = True
):
    """
    Renders an HTML page displaying the line-peaks plot for the specified sample/label/lid.
    With ephemeral merges from LABEL_MEMORY, so we see local splits/merges.
    """
    # 1) Base data selection
    if use_modified_data:
        base_data = CONNECTED_COMPONENT_DATA_MODIFIED
        dtype = "modified"
    else:
        base_data = CONNECTED_COMPONENT_DATA_ORIG
        dtype = "original"

    if sample_name not in base_data:
        raise HTTPException(status_code=404, detail=f"Sample '{sample_name}' not found in {dtype} data.")

    # 2) Ephemeral merge
    data_for_plot = merge_label_memory_for_plot(sample_name, target_label, base_data)

    # If a specific lid is requested and it's not found in data_for_plot,
    # fall back to the modified data.
    if specific_lid is not None:
        found = False
        for comp in data_for_plot[sample_name].get("props", []):
            if comp.get("label") == target_label:
                if specific_lid in comp.get("lns", pd.DataFrame()).index:
                    found = True
                    break
        if not found:
            # The user tried to view a LID that doesn't exist in ephemeral data
            # => return a 404 or some "LID is gone" message.
            raise HTTPException(
                status_code=404,
                detail=f"LID={specific_lid} has been deleted or not found in ephemeral data."
            )
        # if not found and not use_modified_data:
        #     data_for_plot = merge_label_memory_for_plot(sample_name, target_label, CONNECTED_COMPONENT_DATA_MODIFIED)
        #     dtype = "modified"

    # 3) Now do the normal logic
    props = data_for_plot[sample_name].get("props", [])
    target_comp = next((c for c in props if c.get("label") == target_label), None)
    if not target_comp:
        raise HTTPException(
            status_code=404,
            detail=f"Label {target_label} not found in sample {sample_name} ({dtype} + memory)."
        )

    # 4) Generate the line-peaks visualization from the ephemeral data
    try:
        interactive_html = visualize_line_peaks(
            connected_component_data=data_for_plot,  # use ephemeral
            filtered_data_storage=CURRENT_FILTERED_DATA,  # NEW parameter added
            ref_name=sample_name,
            target_label=target_label,
            wllims=[1200, 1500],
            plot_fwhm=True,
            specific_lid=specific_lid,
            output_folder=LID_PLOTS_FOLDER
        )
    except KeyError:
        logger.exception(f"[line-peaks-view] KeyError LID={specific_lid} in ephemeral {dtype} dataset")
        raise HTTPException(status_code=404, detail=f"LID={specific_lid} not found in ephemeral {dtype} data.")
    except Exception as e:
        logger.exception("[line-peaks-view] error in visualize_line_peaks.")
        raise HTTPException(status_code=500, detail=str(e))

    # 5) Build your HTML
    if specific_lid is not None:
        plot_fn = f"line_peaks_{sample_name}_label_{target_label}_lid_{specific_lid}.png"
    else:
        plot_fn = f"line_peaks_{sample_name}_label_{target_label}.png"
    main_plot_path = LID_PLOTS_FOLDER / plot_fn
    if not main_plot_path.exists():
        raise HTTPException(status_code=404, detail=f"Line peaks plot not found: {plot_fn}")


    # The final HTML with the same JavaScript (Delete, Undo, Split, Maybe, etc.)
    html_ = f"""
    <html>
    <head>
      <title>Line Peaks {dtype}</title>
      <style>
        body {{
          margin:0;
          padding:0;
          font-family: sans-serif;
        }}
        .plot-container {{
          position: relative;
          display: inline-block;
        }}
        .top-right-buttons {{
          position: absolute;
          top: 10px;
          right: 10px;
          background-color: #fff;
          border: 1px solid #ccc;
          padding: 10px;
          border-radius: 4px;
        }}
        button {{
          margin: 4px;
          padding: 6px 12px;
          border: none;
          border-radius: 3px;
          background-color: #007BFF;
          color: white;
          cursor: pointer;
        }}
        button:hover {{
          background-color: #0056b3;
        }}
        select, input[type="text"], input[type="number"] {{
          margin: 4px 0;
          padding: 4px;
          font-size: 14px;
        }}
      </style>
      <script>
        // Toggle additional reason input if "child" or "other" is selected.
        function toggleReasonInput() {{
            var sel = document.getElementById("reasonSelect");
            var extra = document.getElementById("reasonInput");
            if(sel.value === "child" || sel.value === "other") {{
                extra.style.display = "inline-block";
            }} else {{
                extra.style.display = "none";
            }}
        }}
        // Construct reason text
        function getReason() {{
            var sel = document.getElementById("reasonSelect").value;
            var extra = document.getElementById("reasonInput").value;
            if(sel === "child") {{
                return extra ? "Child " + extra + " nm apart" : "Child";
            }} else if(sel === "other") {{
                return extra ? extra : "Other";
            }} else {{
                return sel;
            }}
        }}
        // Eps for splitting
        function getEps() {{
            return parseFloat(document.getElementById("epsInput").value) || 0.5;
        }}
        function deleteLid() {{
          if(!confirm('Delete LID={specific_lid}?')) return;
          const payload = {{
            sample_name:'{sample_name}', 
            target_label:{target_label}, 
            lid_to_delete:{specific_lid},
            reason: getReason()
          }};
          fetch('/delete-lid', {{
            method:'POST',
            headers:{{'Content-Type':'application/json'}},
            body:JSON.stringify(payload)
          }})
          .then(r=>r.json())
          .then(d=>{{ alert('Delete:'+JSON.stringify(d)); window.location.reload(); }})
          .catch(e=>alert('Error:'+e));
        }}
        function undoDeleteLid() {{
          const payload = {{
            sample_name:'{sample_name}', 
            label:{target_label}, 
            lid_to_restore:{specific_lid}
          }};
          fetch('/undo-delete-lid', {{
            method:'POST',
            headers:{{'Content-Type':'application/json'}},
            body:JSON.stringify(payload)
          }})
          .then(r=>r.json())
          .then(d=>{{ alert('Undo Delete:'+JSON.stringify(d)); window.location.reload(); }})
          .catch(e=>alert('Error:'+e));
        }}
        function splitLid() {{
          if(!confirm('Split LID={specific_lid}?')) return;
          const payload = {{
            sample_name:'{sample_name}', 
            target_label:{target_label}, 
            old_lid:{specific_lid},
            eps: getEps(),
            min_samples: 1,
            split_comment: getReason()
          }};
          fetch('/split-lid-dbscan', {{
            method:'POST',
            headers:{{'Content-Type':'application/json'}},
            body:JSON.stringify(payload)
          }})
          .then(r=>r.json())
          .then(d=>{{ alert('Split:'+JSON.stringify(d)); window.location.reload(); }})
          .catch(e=>alert('Error:'+e));
        }}
        function undoSplitLid() {{
          if(!confirm('Undo split LID={specific_lid}?')) return;
          const payload = {{
            sample_name:'{sample_name}', 
            target_label:{target_label}, 
            old_lid:{specific_lid}
          }};
          fetch('/undo-split-lid', {{
            method:'POST',
            headers:{{'Content-Type':'application/json'}},
            body:JSON.stringify(payload)
          }})
          .then(r=>r.json())
          .then(d=>{{ alert('Undo Split:'+JSON.stringify(d)); window.location.reload(); }})
          .catch(e=>alert('Error:'+e));
        }}
        function maybeLid() {{
          const payload = {{
            sample_name:'{sample_name}', 
            target_label:{target_label}, 
            lid_to_maybe:{specific_lid},
            reason: getReason()
          }};
          fetch('/maybe-lid', {{
            method:'POST',
            headers:{{'Content-Type':'application/json'}},
            body:JSON.stringify(payload)
          }})
          .then(r=>r.json())
          .then(d=>{{ alert('Maybe:'+JSON.stringify(d)); window.location.reload(); }})
          .catch(e=>alert('Error:'+e));
        }}
      </script>
    </head>
    <body>
      <h2 style="margin:10px;">
        Line Peaks View ({dtype} dataset)<br>
        Sample={sample_name}, Label={target_label}, LID={specific_lid}
      </h2>
      <div class="plot-container" style="margin:10px;">
        <img src="/plots/lid_plots/{plot_fn}" style="max-width:800px;" />
        <div class="top-right-buttons">
          <button onclick="deleteLid()">Delete</button>
          <button onclick="undoDeleteLid()">Undo Delete</button>
          <button onclick="splitLid()">Split</button>
          <button onclick="undoSplitLid()">Undo Split</button>
          <button onclick="maybeLid()">Maybe</button>
          <br/>
          <label>Reason:</label>
          <select id="reasonSelect" onchange="toggleReasonInput()">
            <option value="">--Select Reason--</option>
            <option value="child">Child</option>
            <option value="split">Split</option>
            <option value="noise">Noise</option>
            <option value="other">Other</option>
          </select>
          <input id="reasonInput" type="text" placeholder="Enter details" style="display:none;" />
          <br/>
          <label>Eps:</label>
          <input id="epsInput" type="number" placeholder="Enter eps" value="0.5" step="0.1" />
        </div>
      </div>
      {interactive_html if interactive_html else ""}
    </body>
    </html>
    """
    return HTMLResponse(html_)



#     return HTMLResponse(html_out)


from collections import defaultdict
import math

def cached_plot_single_label_mode(
    sample_name: str,
    label_index: int,
    props: List[dict],
    base_data: dict,
    label_matches: dict,
    dataset_type: str,
    max_samples: int,
    use_modified_data: bool,
    save_changes: bool = False,
    min_mid_count: Optional[int] = None,
    desired_mid: Optional[int] = None,
    min_mids_per_label: Optional[float] = None,
    filter_key: Optional[str] = None
) -> HTMLResponse:
    """
    Identical to your 'plot_single_label_mode', except it first checks 
    a disk cache. If it finds 'cached_single_label.html' for this label, 
    it returns it immediately. Otherwise, it runs the normal steps, 
    then writes the results to the cache folder.

    So next/prev, or repeated visits to the same label, are instant.
    """

    # Validate label_index
    if label_index < 0 or label_index >= len(props):
        return HTMLResponse(
            f"<h3>label_index={label_index} out of range [0..{len(props) - 1}].</h3>",
            status_code=400
        )

    label_num = props[label_index]["label"]

    
    # 1) validate
    if label_index < 0 or label_index >= len(props):
        return HTMLResponse("<h3>label_index out of range</h3>", status_code=400)
    label_num = props[label_index]["label"]

    # ─── live‐edit bypass ─── if in-memory edits exist, clear its cache folder ───
    if (sample_name, label_num) in LABEL_MEMORY:
        clear_label_cache(sample_name, label_num, filter_key)

    # 2) locate cache
    label_cache_folder = get_label_cache_folder(sample_name, label_num, filter_key)
    label_cache_folder.mkdir(parents=True, exist_ok=True)
    html_cache_file = label_cache_folder / "cached_single_label.html"
    rel_cache_folder = f"cached_plots/{sample_name}__label_{label_num}"
    if filter_key:
        rel_cache_folder += f"__{filter_key}"

    # 3) if it's on disk now, serve it
    # if html_cache_file.exists():
    #     return HTMLResponse(html_cache_file.read_text(encoding="utf-8"))
    if html_cache_file.exists():
        html = html_cache_file.read_text(encoding="utf-8")
        resp = HTMLResponse(html)
        # force the browser to always re-download—even for the “cached” copy
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Pragma"]        = "no-cache"
        resp.headers["Expires"]       = "0"
        return resp

    # ------------------------------------------------------------------
    # 2) If no cached file => do the normal heavy-lifting
    # ------------------------------------------------------------------
    data_for_plot = merge_label_memory_for_plot(sample_name, label_num, base_data)

    # Label matches, truncated by max_samples if needed
    sample_label_matches = label_matches.get((sample_name, label_num), [])
    if len(sample_label_matches) > (max_samples - 1):
        sample_label_matches = sample_label_matches[: (max_samples - 1)]

    # Create the LID images (Option B) by calling your plot_multi_sample_comparison
    clickable_areas = plot_multi_sample_comparison(
        ref_sample=sample_name,
        ref_label=label_num,
        sample_label_matches=sample_label_matches,
        connected_component_data=data_for_plot,
        filtered_data_storage=CURRENT_FILTERED_DATA,  # your global or state
        plot_output_folder=str(label_cache_folder), #PLOT_OUTPUT_FOLDER
        plot_filename=f"iter_{dataset_type}_{sample_name}_{label_num}.png",
        samples_per_row=2,
        max_sample_rows=3,
        columns_per_sample=4,
        save_individual_subplots=True,
        target_label=label_num
    )

    # Build the HTML table of sub-lids (this is your original code). 
    sample_label_list = [(sample_name, label_num)]
    for (s, l, _) in sample_label_matches:
        sample_label_list.append((s, l))

    def build_single_label_multi_sample_html(clickable_areas, sample_label_list):
        block_data = defaultdict(lambda: defaultdict(list))
        encountered = []
        for (s, l) in sample_label_list:
            if (s, l) not in encountered:
                encountered.append((s, l))

        for area in clickable_areas:
            s_ = area["sample"]
            l_ = area["label"]
            mid_key = area.get("sub_mid") if area.get("sub_mid") is not None else area["mid"]
            block_data[(s_, l_)][mid_key].append(area)

        blocks_html = []
        for (smp, lab) in encountered:
            mids_map = block_data[(smp, lab)]
            if not mids_map:
                blocks_html.append(f"<div>No sub-lids in {smp}</div>")
                continue

            mid_order = []
            for mk, arrs in mids_map.items():
                mid_wls = [a["mid_wl"] for a in arrs if a.get("mid_wl") is not None]
                mm = min(mid_wls) if mid_wls else 99999
                mid_order.append((mk, mm))
            mid_order.sort(key=lambda x: x[1])

            table_rows = []
            for (mid_k, _) in mid_order:
                arrs = mids_map[mid_k]
                arrs_sorted = sorted(arrs, key=lambda x: x["lid_wl"])
                row_cols = []
                for info in arrs_sorted:
                    lid_ = info["lid"]
                    wl_ = info["lid_wl"]
                    fname_ = info.get("image_path", "")
                    import os
                    fname_short = os.path.basename(fname_)

                    left_panel = f"""
                    <div style="display:flex; flex-direction:column; margin-right:0px; text-align:left;">
                        <button style="font-size:12px; padding:2px 6px; width:50px;"
                            onclick="openLinePeaksView('{smp}', {lab}, {lid_}, {str(use_modified_data).lower()})">
                            Select
                        </button>
                        <div style="font-size:12px;">wl={wl_:.1f}</div>

                        <!-- sub-mid input -->
                        <input type="text" id="submid_{smp}_{lab}_{lid_}" 
                                style="width:50px; font-size:12px; margin-top:3px;" 
                                placeholder="sub-mid?" />
                        <button style="font-size:12px; padding:2px 6px; margin-top:3px; width:50px;"
                                onclick="assignSubMid('{smp}', {lab}, {lid_}, 'submid_{smp}_{lab}_{lid_}')">
                            Assign
                        </button>
                        <button style="font-size:12px; padding:2px 6px; margin-top:3px; width:50px;"
                        onclick="deleteLid('{smp}', {lab}, {lid_})">
                        Del LID
                        </button>
                        <button style="font-size:12px; padding:2px 6px; margin-top:3px; width:50px;"
                                onclick="mergeLids('{smp}', {lab}, {lid_})">
                            Merge
                        </button>
                        <button style="font-size:12px; padding:2px 6px; margin-top:3px; width:50px;"
                                onclick="undoSubMid('{smp}', {lab}, {lid_})">
                            Undo SM
                        </button>
                    </div>
                    """


                    fname_       = info.get("image_path", "")
                    fname_short  = os.path.basename(fname_)
                    # determine the file’s actual location on disk:
                    full_path    = Path(label_cache_folder) / fname_short
                    # use its mtime to force the browser to reload after each regen:
                    ver          = int(full_path.stat().st_mtime)


                    right_panel = f"""
                    <div style="flex-grow:1;">
                        <img src="/plots/{rel_cache_folder}/{fname_short}?v={ver}"
                            style="max-width:150px;" />
                    </div>
                    """
                    # right_panel = f"""
                    # <div style="flex-grow:1;">
                    #     <img src="/plots/{rel_cache_folder}/{fname_short}" style="max-width:150px;" />
                    # </div>
                    # """

                    col_ = f"""
                    <td style="border:1px solid #ccc; text-align:center; padding:0px;">
                        <div style="display:flex; flex-direction:row; align-items:center;">
                        {left_panel}
                        {right_panel}
                        </div>
                    </td>
                    """
                    row_cols.append(col_)
                table_rows.append(f"<tr>{''.join(row_cols)}</tr>")

            table_html = f"""
            <table style="border-collapse:collapse; margin:5px;">
                <thead>
                <tr>
                    <th colspan="999" style="background:#f0f0f0; text-align:center; border:2px solid #333;">
                    {smp} (label={lab})
                    </th>
                </tr>
                </thead>
                <tbody>
                {''.join(table_rows)}
                </tbody>
            </table>
            """
            blocks_html.append(f"""
            <div style="display:inline-block; margin-right:20px; vertical-align:top;">
                {table_html}
            </div>
            """)
        return f"""
        <div style="white-space:nowrap; overflow-x:auto; border:1px solid #ccc; padding:0;">
            {"".join(blocks_html)}
        </div>
        """

    big_html_block = build_single_label_multi_sample_html(clickable_areas, sample_label_list)

    # Plot the label spectra via your function
    # (unchanged)
    spectra_filename = plot_label_spectra(
        data=base_data,
        filtered_data_storage=CURRENT_FILTERED_DATA,
        sample_name=sample_name,
        label_num=label_num,
        output_folder=label_cache_folder#PLOT_OUTPUT_FOLDER
    )

    # Build nav links
    nav_params = f"&use_modified_data={str(use_modified_data).lower()}&max_samples={max_samples}"
    if min_mid_count is not None:
        nav_params += f"&min_mid_count={min_mid_count}"
    if desired_mid is not None:
        nav_params += f"&desired_mid={desired_mid}"
    if min_mids_per_label is not None:
        nav_params += f"&min_mids_per_label={min_mids_per_label}"

    prev_idx = label_index - 1 if label_index > 0 else None
    next_idx = label_index + 1 if label_index < (len(props) - 1) else None
    nav_links = []
    if prev_idx is not None:
        nav_links.append(f"""
        <a href="/plot-multi-sample-iter?sample_name={sample_name}&label_index={prev_idx}{nav_params}">
          &laquo; Prev Label
        </a>""")
    if next_idx is not None:
        nav_links.append(f"""
        <a href="/plot-multi-sample-iter?sample_name={sample_name}&label_index={next_idx}{nav_params}">
          Next Label &raquo;
        </a>""")
    nav_html = " ".join(nav_links)

    # Build final HTML
        # The JavaScript for merges, sub-mid, etc.
    js_script = """
        <script>
        function toggleDeleteReasonInput() {
            var sel = document.getElementById("deleteReasonSelect");
            var extra = document.getElementById("deleteReasonInput");
            if(sel.value === "child" || sel.value === "other") {
                extra.style.display = "inline-block";
            } else {
                extra.style.display = "none";
            }
        }

        function getDeleteReason() {
            var sel = document.getElementById("deleteReasonSelect").value;
            var extra = document.getElementById("deleteReasonInput").value;
            if(sel === "child") {
                return extra ? "Child " + extra + " nm apart" : "Child";
            } else if(sel === "other") {
                return extra ? extra : "Other";
            } else {
                return sel;
            }
        }

        function deleteLid(sampleName, targetLabel, lid) {
            if(!confirm(`Delete LID=${lid}?`)) return;
            const payload = {
                sample_name: sampleName,
                label: targetLabel,
                lid_to_delete: lid,
                reason: getDeleteReason()
            };
            fetch('/delete-lid', {
                method:'POST',
                headers:{'Content-Type':'application/json'},
                body:JSON.stringify(payload)
            })
            .then(r=>r.json())
            .then(d=>{ alert('Delete:'+JSON.stringify(d)); location.reload(); })
            .catch(e=>alert('Error:'+e));
        }

        // ADDED: load-label-data button
        function loadLabelData(sampleName, labelVal) {
            const payload = {
            sample_name: sampleName,
            label: labelVal
            };
            fetch('/load-label-data', {
            method: 'POST',
            headers: { 'Content-Type':'application/json' },
            body: JSON.stringify(payload)
            })
            .then(r => r.json())
            .then(d => {
                alert('Label data loaded: ' + JSON.stringify(d));
                // Optionally reload if you like:
                // location.reload();
            })
            .catch(e => alert('Error loading label data: ' + e));
        }

        function onMergeReasonChange() {
            const selVal = document.getElementById("mergeReasonSelect").value;
            const xInput = document.getElementById("mergeCommentInput");
            const otherInput = document.getElementById("mergeCommentOther");
            if(selVal === "from_x_splits") {
                xInput.style.display = "inline-block";
                otherInput.style.display = "none";
            } else if(selVal === "other") {
                xInput.style.display = "none";
                otherInput.style.display = "inline-block";
            } else {
                xInput.style.display = "none";
                otherInput.style.display = "none";
            }
        }
        function onAssignReasonChange() {
            const selVal = document.getElementById("assignCommentSelect").value;
            const otherInput = document.getElementById("assignCommentOther");
            if(selVal === "other") {
                otherInput.style.display = "inline-block";
            } else {
                otherInput.style.display = "none";
            }
        }
        function openLinePeaksView(sampleName, targetLabel, lid, isMod) {
            const url = `/line-peaks-view?sample_name=${sampleName}&target_label=${targetLabel}&specific_lid=${lid}&use_modified_data=${isMod}`;
            window.open(url, '_blank');
        }
        function assignSubMid(sampleName, targetLabel, lid, inputId) {
            const val = document.getElementById(inputId).value;
            if(!val) {
                alert('Provide sub-mid');
                return;
            }
            // gather all lids that share this sub-mid in the DOM
            const inputs = document.querySelectorAll(`input[id^="submid_${sampleName}_${targetLabel}_"]`);
            let lidsToAssign = [];
            inputs.forEach(inp => {
                if(inp.value === val) {
                    const parts = inp.id.split("_");
                    const lidVal = parseInt(parts[parts.length - 1], 10);
                    if(!isNaN(lidVal)) lidsToAssign.push(lidVal);
                }
            });
            let reasonSel = document.getElementById("assignCommentSelect");
            let reasonVal = reasonSel ? reasonSel.value : "";
            if(reasonVal === "other") {
                reasonVal = document.getElementById("assignCommentOther").value || "other";
            }
            const payload = {
                sample_name: sampleName,
                target_label: targetLabel,
                lids: lidsToAssign,
                new_sub_mid: val,
                comment: reasonVal
            };
            fetch('/assign-submid', {
                method:'POST',
                headers:{'Content-Type':'application/json'},
                body: JSON.stringify(payload)
            })
            .then(r=>r.json())
            .then(d=>{ alert('Assigned SubMid: ' + JSON.stringify(d)); location.reload(); })
            .catch(e=> alert('Error:' + e));
        }
        function undoSubMid(sampleName, targetLabel, lid) {
            const payload = { sample_name: sampleName, target_label: targetLabel, lid };
            fetch('/undo-submid', {
                method:'POST',
                headers:{'Content-Type':'application/json'},
                body: JSON.stringify(payload)
            })
            .then(r=>r.json())
            .then(d=>{ alert('Undo SubMid: ' + JSON.stringify(d)); location.reload(); })
            .catch(e=> alert('Error:' + e));
        }
        function mergeLids(sampleName, targetLabel, currentLid) {
            const subMidVal = document.getElementById(`submid_${sampleName}_${targetLabel}_${currentLid}`).value;
            if(!subMidVal) {
                alert("No sub-mid typed. Can't merge.");
                return;
            }
            // gather lids with this sub-mid
            const inputs = document.querySelectorAll(`input[id^="submid_${sampleName}_${targetLabel}_"]`);
            let lidsArray = [];
            inputs.forEach(inp => {
                if(inp.value === subMidVal) {
                    const parts = inp.id.split("_");
                    const lidVal = parseInt(parts[parts.length - 1], 10);
                    if(!isNaN(lidVal)) lidsArray.push(lidVal);
                }
            });
            if(lidsArray.length < 2) {
                alert("Need at least 2 lids with same sub-mid to merge. Found " + lidsArray.length);
                return;
            }
            let selVal = document.getElementById("mergeReasonSelect").value;
            let reasonText = "";
            if(selVal === "from_x_splits") {
                const xInput = document.getElementById("mergeCommentInput");
                reasonText = "from " + (xInput.value || "X") + " splits";
            } else if(selVal === "other") {
                reasonText = document.getElementById("mergeCommentOther").value || "other";
            } else {
                reasonText = selVal || "unspecified";
            }
            const payload = {
                sample_name: sampleName,
                target_label: targetLabel,
                lids: lidsArray,
                merge_comment: reasonText
            };
            fetch('/merge-lids', {
                method:'POST',
                headers:{'Content-Type':'application/json'},
                body: JSON.stringify(payload)
            })
            .then(r=>r.json())
            .then(d=>{ alert('Merge:'+JSON.stringify(d)); location.reload(); })
            .catch(e=> alert('Merge Error:'+ e));
        }
        function applyLabelChanges(sampleName) {
        if (!confirm("Apply changes for all labels in sample " + sampleName + "?")) return;
        fetch("/apply-label-changes", {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sample_name: sampleName
            })
        })
        .then(r => r.json())
        .then(d => {
            alert(d.message);
            location.reload();
        })
        .catch(e => alert('Error:' + e));
        }
        function discardLabelChanges(sampleName, label) {
            if(!confirm("Discard changes?")) return;
            fetch("/discard-label-changes", {
                method:'POST',
                headers:{'Content-Type':'application/json'},
                body: JSON.stringify({sample_name: sampleName, label})
            })
            .then(r=>r.json())
            .then(d=>{ alert(d.message); location.reload(); })
            .catch(e=> alert('Error:'+ e));
        }
        </script>
        """
    


    nav_html = " ".join(nav_links)
    html_out = f"""
    <html>
        <head>
        <title>{sample_name} Label={label_num} {dataset_type}</title>
        <style>
            body {{ font-family: sans-serif; margin:20px; }}
            a {{ font-weight: bold; color:#0066cc; text-decoration: none; margin-right:20px; }}
            a:hover {{ text-decoration: underline; }}
        </style>
        </head>
        <body>
        <h2>{sample_name}, Label={label_num} ({dataset_type} data)</h2>
        <div>{nav_html}</div>

        <!-- Merge reason UI -->
        <div style="margin:8px 0;">
            <label><b>Merge Reason:</b></label>
            <select id="mergeReasonSelect" onchange="onMergeReasonChange()">
            <option value="">--Select--</option>
            <option value="from_x_splits">from x splits</option>
            <option value="other">other</option>
            </select>
            <input type="number" id="mergeCommentInput" style="display:none; margin-left:8px; width:80px;" placeholder="Enter # splits" />
            <input type="text" id="mergeCommentOther" style="display:none; margin-left:8px; width:120px;" placeholder="Enter reason" />
        </div>

        <!-- Assign reason UI -->
        <div style="margin:8px 0;">
            <label><b>Assign Reason:</b></label>
            <select id="assignCommentSelect" onchange="onAssignReasonChange()">
            <option value="">--Select--</option>
            <option value="double mode">double mode</option>
            <option value="more than one mid">more than one mid</option>
            <option value="misassignment">misassignment</option>
            <option value="other">other</option>
            </select>
            <input type="text" id="assignCommentOther" style="display:none; margin-left:8px; width:120px;" placeholder="Enter reason" />
        </div>

        <!-- Add Delete reason UI -->
        <div style="margin:8px 0;">
        <label><b>Delete Reason:</b></label>
        <select id="deleteReasonSelect" onchange="toggleDeleteReasonInput()">
            <option value="">--Select Reason--</option>
            <option value="child">Child</option>
            <option value="noise">Noise</option>
            <option value="other">Other</option>
        </select>
        <input type="text" id="deleteReasonInput" style="display:none; margin-left:8px; width:120px;" placeholder="Enter reason" />
        </div>

        <!-- ADDED: a "Load In-Memory" button so user can call /load-label-data. -->
        <div style="margin: 10px 0;">
        <button onclick="loadLabelData('{sample_name}', {label_num})">Load In-Memory</button>
        <button onclick="applyLabelChanges('{sample_name}', {label_num})">Apply Changes</button>
        <button onclick="discardLabelChanges('{sample_name}', {label_num})">Discard Changes</button>
        </div>

        <div style="margin-top:20px;">
            {big_html_block}
        </div>
        
        <div style="margin-top: 20px;">
        <h3>Label {label_num} Spectra</h3>
        <img src="/plots/{rel_cache_folder}/{spectra_filename}" style="max-width:100%; height:auto;" />
        </div>

        <div>{nav_html}</div>
        {js_script}
        </body>
    </html>
    """

    # save the HTML so next time we can return instantly
    # with open(html_cache_file, "w", encoding="utf-8") as f:
    #     f.write(html_out)

    # return HTMLResponse(html_out)
    # april 25
        # save the HTML so next time we can return instantly
    with open(html_cache_file, "w", encoding="utf-8") as f:
        f.write(html_out)

    # → force the browser to re-download on every reload
    response = HTMLResponse(html_out)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"]        = "no-cache"
    response.headers["Expires"]       = "0"
    return response



    # html_out = f"""
    # <html>
    #     <head>
    #       <title>{sample_name} Label={label_num} {dataset_type}</title>
    #       <style>
    #         body {{ font-family: sans-serif; margin:20px; }}
    #       </style>
    #     </head>
    #     <body>
    #     <h2>{sample_name}, Label={label_num} ({dataset_type} data)</h2>
    #     <div>{nav_html}</div>
    #     <div style="margin-top:20px;">{big_html_block}</div>
    #     <div style="margin-top: 20px;">
    #       <h3>Label {label_num} Spectra</h3>
    #       <img src="/plots/cc_matching_plots/{spectra_filename}" style="max-width:100%; height:auto;" />
    #     </div>
    #     <div>{nav_html}</div>
    #     {js_script}
    #     </body>
    # </html>
    # """

    # # ----------------------------------------------------------------
    # # 3) Write the newly generated HTML to disk => next time is instant
    # # ----------------------------------------------------------------
    # with open(html_cache_file, "w", encoding="utf-8") as f:
    #     f.write(html_out)

    # return HTMLResponse(html_out)



####### above is original

# ----------------------------------------------------------------------------
#  MAIN ENDPOINT with Desired MID Filter + Extra Filtering by MID Count,
#  and Single-Label / Multi-Label / MID-Only Modes
# ----------------------------------------------------------------------------
@app.get("/plot-multi-sample-iter", response_class=HTMLResponse)
def plot_multi_sample_iter(
    sample_name: str,
    use_modified_data: bool = True,
    label_index: Optional[int] = None,
    min_mids_per_label: Optional[float] = None,
    max_samples: int = 8,
    save_changes: bool = False,
    desired_mid: Optional[int] = None,
    min_mid_count: Optional[int] = None    # <-- New parameter for filtering by MID count
):
    """
    Renders an HTML page that compares the same label across multiple samples.
    1) If label_index is set => single-label mode.
    2) If min_mids_per_label is set => multi-label filtering.
    3) If desired_mid is set (and neither label_index nor min_mids_per_label):
       => find all labels that contain that MID.
    4) New: If min_mid_count is set, only include labels where at least one MID
       appears at least that many times.
    """
    # 1) Choose base data
    if use_modified_data:
        base_data = CONNECTED_COMPONENT_DATA_MODIFIED
        label_matches = ALL_LABEL_MATCHES_MODIFIED
        dataset_type = "modified"
    else:
        base_data = CONNECTED_COMPONENT_DATA_ORIG
        label_matches = ALL_LABEL_MATCHES_ORIG
        dataset_type = "original"

    if use_modified_data and save_changes:
        # e.g. accumulate_changes_for_sample(sample_name)
        pass

    if sample_name not in base_data:
        return HTMLResponse(
            f"<h3>Sample '{sample_name}' not found in {dataset_type} data.</h3>",
            status_code=404
        )

    props = base_data[sample_name].get("props", [])
    num_labels = len(props)
    if num_labels == 0:
        return HTMLResponse(
            f"<h3>No labels in sample '{sample_name}' ({dataset_type} data).</h3>",
            status_code=404
        )

    # ------------------------------------------------
    # A) Filter PROPS by desired_mid if specified
    # ------------------------------------------------
    if desired_mid is not None:
        filtered_indices = []
        for i, comp_ in enumerate(props):
            lns_df = comp_.get("lns", pd.DataFrame())
            if not lns_df.empty and "mid" in lns_df.columns:
                if desired_mid in lns_df["mid"].values:
                    filtered_indices.append(i)
        if not filtered_indices:
            return HTMLResponse(
                f"<h3>No labels in sample '{sample_name}' contain mid={desired_mid}.</h3>",
                status_code=404
            )
        props = [props[i] for i in filtered_indices]
        num_labels = len(props)
        if label_index is not None and label_index >= num_labels:
            return HTMLResponse(
                f"<h3>Label index={label_index} out of range after mid={desired_mid} filter. Only {num_labels} left.</h3>",
                status_code=400
            )

    # ------------------------------------------------
    # NEW B) Filter PROPS by min_mid_count if specified
    #     (Keep only labels where at least one mid appears 
    #      >= min_mid_count times)
    # ------------------------------------------------
    if min_mid_count is not None:
        filtered_indices = []
        for i, comp_ in enumerate(props):
            lns_df = comp_.get("lns", pd.DataFrame())
            if lns_df.empty or "mid" not in lns_df.columns:
                continue
            # Count occurrences per MID
            mid_counts = lns_df["mid"].value_counts()
            if (mid_counts == min_mid_count).any():
                filtered_indices.append(i)
        if not filtered_indices:
            return HTMLResponse(
                f"<h3>No labels in sample '{sample_name}' have any MID with count == {min_mid_count}.</h3>",
                status_code=404
            )
        # Reassign props to filtered labels only.
        props = [props[i] for i in filtered_indices]
        num_labels = len(props)
        # If label_index was provided but is out-of-range for the filtered list, error out.
        if label_index is not None and label_index >= num_labels:
            return HTMLResponse(
                f"<h3>Label index={label_index} out of range after min_mid_count filter. Only {num_labels} left.</h3>",
                status_code=400
            )

    # If no label_index is provided but a filter (min_mid_count) is applied—and no other mode parameters—
    # generate an HTML pick-list page.
    if label_index is None and min_mid_count is not None and desired_mid is None and min_mids_per_label is None:
        # Compute total number of filtered labels.
        total_count = len(props)
        # Compute the "Single MID Count": count of labels where the unique MID count equals 1.
        single_mid_count = 0
        for comp in props:
            lns_df = comp.get("lns", pd.DataFrame())
            if "mid" in lns_df.columns:
                unique_mids = lns_df["mid"].unique()
                if len(unique_mids) == 1:
                    single_mid_count += 1

        header_html = (
            f"<h2>{sample_name}: MID Count == {min_mid_count} LIDs. "
            f"(Total Count={total_count}, Single MID Count={single_mid_count})</h2>"
        )
        
        # Build pick-list of links.
        lines = []
        for i, comp in enumerate(props):
            lbl = comp["label"]
            # Optionally, you can compute additional MID count info for each label.
            lns_df = comp.get("lns", pd.DataFrame())
            if "mid" in lns_df.columns:
                # For example, compute the unique MID count.
                num_mids = len(lns_df["mid"].unique())
            else:
                num_mids = 0
            lines.append(
                f"<li>Label {lbl} (index={i}), MIDs={num_mids} "
                f"<a href='/plot-multi-sample-iter?sample_name={sample_name}&label_index={i}"
                f"&min_mid_count={min_mid_count}&use_modified_data={str(use_modified_data).lower()}"
                f"&max_samples={max_samples}'>Open Single-Label View</a></li>"
            )
        pick_list_html = header_html + "<ul>" + "".join(lines) + "</ul>"
        return HTMLResponse(pick_list_html)


    # ------------------------------------------------
    # 1) Always apply min_mids_per_label filter first
    # ------------------------------------------------
    if min_mids_per_label is not None:
        # helper to count unique MIDs in props[idx]
        def count_mids(idx: int) -> int:
            comp_ = props[idx]
            lns_df = comp_.get("lns", pd.DataFrame())
            if "mid" not in lns_df.columns:
                return 0
            return len(lns_df["mid"].unique())

        # compute counts for every label
        all_counts = [count_mids(i) for i in range(len(props))]

        # pick matching indices
        if min_mids_per_label >= 2:
            matching = [i for i, n in enumerate(all_counts) if n >= min_mids_per_label]
            comparison_symbol = ">="
        else:
            matching = [i for i, n in enumerate(all_counts) if n <  min_mids_per_label]
            comparison_symbol = "<"

        if not matching:
            return HTMLResponse(
                f"<h3>No labels in '{sample_name}' match {comparison_symbol} {min_mids_per_label} MIDs.</h3>",
                status_code=404
            )

        # reassign props → filtered subset
        props = [props[i] for i in matching]
        num_labels = len(props)

        # if user asked for a specific label_index, ensure it's still valid
        if label_index is not None and (label_index < 0 or label_index >= num_labels):
            return HTMLResponse(
                f"<h3>Label index={label_index} out of range after filter. Only {num_labels} labels remain.</h3>",
                status_code=400
            )

    # ------------------------------------------------
    # 2) Pick‑list mode: show filtered labels only
    # ------------------------------------------------
    if label_index is None and min_mids_per_label is not None:
        # rebuild counts on the filtered props
        def count_mids_filtered(idx: int) -> int:
            comp_ = props[idx]
            lns_df = comp_.get("lns", pd.DataFrame())
            if "mid" not in lns_df.columns:
                return 0
            return len(lns_df["mid"].unique())

        label_mid_counts = [(i, count_mids_filtered(i)) for i in range(num_labels)]

        # build HTML list
        lines = []
        for idx, nmids in label_mid_counts:
            lbl_num = props[idx]["label"]
            lines.append(
                f"<li>Label {lbl_num} (index={idx}), MIDs={nmids} "
                f"<a href='/plot-multi-sample-iter?sample_name={sample_name}"
                f"&label_index={idx}"
                f"&use_modified_data={str(use_modified_data).lower()}"
                f"&max_samples={max_samples}"
                f"&min_mids_per_label={min_mids_per_label}'>"
                "Open Single-Label View</a></li>"
            )

        html_ = f"""
        <html>
        <head>
            <title>{sample_name} {comparison_symbol} {min_mids_per_label} MIDs ({dataset_type})</title>
        </head>
        <body>
            <h2>{sample_name}: Labels with {comparison_symbol} {min_mids_per_label} MIDs. (Count={num_labels})</h2>
            <ul>{''.join(lines)}</ul>
        </body>
        </html>
        """
        return HTMLResponse(html_)

    # ------------------------------------------------
    # C) Single-Label Mode (label_index provided, no min_mids_per_label)
    # ------------------------------------------------
    if label_index is not None:
        filter_key = None
        if min_mids_per_label is not None:
            filter_key = f"minmids_{min_mids_per_label}"
        if min_mid_count is not None:
            filter_key = f"minmidcount_{min_mid_count}"
        if desired_mid is not None:
            filter_key = f"mid_{desired_mid}"
        # april 25
        # # figure out the *real* label value (not the index)
        # actual_label = props[label_index]["label"]
        # # merge any in-memory splits/merges for that label into the data dict
        # base_data = merge_label_memory_for_plot(sample_name, actual_label, base_data)
        # 2) the *real* label value (not just its index in props)
        actual_label = props[label_index]["label"]

        # 3) pull every split / merge that lives in LABEL_MEMORY
        base_data = merge_label_memory_for_plot(sample_name, actual_label, base_data)

        # 4) blow away any cached PNGs/JSONs for that label
        clear_label_cache(sample_name, actual_label)
        # pass the filter_key into the cache function
        return cached_plot_single_label_mode(
            sample_name=sample_name,
            label_index=label_index,
            props=props,
            base_data=base_data,
            label_matches=label_matches,
            dataset_type=dataset_type,
            max_samples=max_samples,
            use_modified_data=use_modified_data,
            save_changes=save_changes,
            min_mid_count=min_mid_count,
            desired_mid=desired_mid,
            min_mids_per_label=min_mids_per_label,
            filter_key=filter_key              # ← new!
    )
    # ------------------------------------------------
    # E) MID-Only Mode (desired_mid provided and no label_index/min_mids_per_label)
    # ------------------------------------------------
    if desired_mid is not None and label_index is None and min_mids_per_label is None:
        if not props:
            return HTMLResponse(
                f"<h3>No labels in sample '{sample_name}' contain mid={desired_mid}.</h3>",
                status_code=404
            )
        if len(props) == 1:
            # actual_label = props[0]["label"]
            # base_data     = merge_label_memory_for_plot(sample_name, actual_label, base_data)
            # 2) the *real* label value (not just its index in props)
            actual_label = props[0]["label"]

            # 3) pull every split / merge that lives in LABEL_MEMORY
            base_data = merge_label_memory_for_plot(sample_name, actual_label, base_data)

            # 4) blow away any cached PNGs/JSONs for that label
            clear_label_cache(sample_name, actual_label)
            return cached_plot_single_label_mode(
                sample_name=sample_name,
                label_index=0,
                props=props,
                base_data=base_data,
                label_matches=label_matches,
                dataset_type=dataset_type,
                max_samples=max_samples,
                use_modified_data=use_modified_data,
                save_changes=save_changes
            )
        else:
            lines = []
            for i, comp_ in enumerate(props):
                lbl_num = comp_["label"]
                lines.append(f"""
                <li>
                  Label={lbl_num}, index={i}
                  <a href="/plot-multi-sample-iter?sample_name={sample_name}&label_index={i}&use_modified_data={str(use_modified_data).lower()}&max_samples={max_samples}">
                    View Single-Label
                  </a>
                </li>
                """)
            html_ = f"""
            <html>
             <head>
               <title>{sample_name} - mid={desired_mid}</title>
             </head>
             <body>
               <h2>Sample '{sample_name}' - multiple labels contain mid={desired_mid}</h2>
               <ul>
                 {''.join(lines)}
               </ul>
             </body>
            </html>
            """
            return HTMLResponse(html_)

    # ------------------------------------------------
    # F) Fallback if no valid mode is specified
    # ------------------------------------------------
    return HTMLResponse(
        "<h3>No label_index, min_mids_per_label, or desired_mid specified. Nothing to plot.</h3>",
        status_code=400
    )


