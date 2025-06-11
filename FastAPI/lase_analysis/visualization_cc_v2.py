# visualization_cc_v2.py
import os
import math
import json
import random

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # For headless environments
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import DBSCAN

def normalize_sample_id(filename: str) -> str:
    """
    Strip off any trailing .map.lase, .lase or .map so that:
      "foo.map.lase"  → "foo"
      "foo.map"       → "foo"
      "foo.lase"      → "foo"
      otherwise      → unchanged
    """
    s = filename.strip()
    for ext in (".map.lase", ".lase", ".map"):
        if s.endswith(ext):
            return s[:-len(ext)]
    return s

#may2
def get_cc_spectra(cc_dict, storage):
    """
    Fetch raw spectra for one connected component.
    cc_dict MUST contain 'sample_id' and 'group_name' (step‑2 guarantees this).
    """
    key = f"{cc_dict['sample_id']}_{cc_dict['group_name']}"   # canonical key
    try:
        spts = storage[key]["spts"]
    except KeyError:
        raise KeyError(
            f"Spectra cache missing for {key}. "
            f"Available keys: {list(storage)[:5]} ..."
        )

    return [spts[i] for i in cc_dict["ispt_values"] if 0 <= i < len(spts)]
##############################################################################
# 0) Helpers
##############################################################################

def is_neighboring(bbox1, bbox2, padding=10):
    """
    Check if two bounding boxes (min_row, min_col, max_row, max_col)
    are "neighboring" or close within a given padding.
    """
    (r1min, c1min, r1max, c1max) = bbox1
    (r2min, c2min, r2max, c2max) = bbox2
    # Expand each by padding
    r1minP, r1maxP = r1min - padding, r1max + padding
    c1minP, c1maxP = c1min - padding, c1max + padding
    r2minP, r2maxP = r2min - padding, r2max + padding
    c2minP, c2maxP = c2min - padding, c2max + padding

    horizontal_overlap = not (c1maxP < c2minP or c2maxP < c1minP)
    vertical_overlap   = not (r1maxP < r2minP or r2maxP < r1minP)
    return horizontal_overlap and vertical_overlap


def preprocess_spectrum(spectrum: np.ndarray) -> np.ndarray:
    """Simple spectral normalization."""
    if np.std(spectrum) == 0:
        return spectrum - np.mean(spectrum)
    return (spectrum - np.mean(spectrum)) / np.std(spectrum)


##############################################################################
# 1) Delete / Split / Modify LIDs
##############################################################################

def delete_lid(
    connected_component_data: Dict[str, Any],
    sample_name: str,
    target_label: int,
    lid_to_delete: int
):
    """
    Removes lid_to_delete from sample_name & target_label
    """
    sample_dict = connected_component_data.get(sample_name, {})
    pks_df = sample_dict.get('pks', pd.DataFrame())
    lns_df = sample_dict.get('lns', pd.DataFrame())
    props  = sample_dict.get('props', [])

    comp = next((c for c in props if c['label'] == target_label), None)
    if comp is None:
        print(f"[delete_lid] Label={target_label} not found in sample={sample_name}.")
        return

    if 'lns' not in comp or not isinstance(comp['lns'], pd.DataFrame):
        print("[delete_lid] comp['lns'] missing or not a DataFrame.")
        return

    if lid_to_delete not in comp['lns'].index:
        print(f"[delete_lid] LID={lid_to_delete} not in label={target_label}.")
        return

    # Remove from comp['lns']
    comp['lns'].drop(lid_to_delete, inplace=True, errors='ignore')

    # Remove from lns_df
    if lid_to_delete in lns_df.index:
        lns_df.drop(lid_to_delete, inplace=True, errors='ignore')

    # Remove from pks_df
    if 'lid' in pks_df.columns:
        idx_remove = pks_df.index[pks_df['lid'] == lid_to_delete]
        pks_df.drop(idx_remove, inplace=True, errors='ignore')

    print(f"[delete_lid] Deleted LID={lid_to_delete} from sample={sample_name}, label={target_label}.")


def _get_next_unique_lid(lns_df: pd.DataFrame) -> int:
    """
    Generate a new lid that isn't used yet in lns_df.index
    """
    if len(lns_df.index) == 0:
        return 1
    lids_existing = lns_df.index
    lid_new = max(lids_existing) + 1
    while lid_new in lids_existing:
        lid_new += 1
    return lid_new


def split_lid_dbscan(
    connected_component_data: Dict[str, Any],
    sample_name: str,
    target_label: int,
    old_lid: int,
    eps: float = 3.0,
    min_samples: int = 1
) -> List[int]:
    """
    Split LID by clustering pks_df['wl','fwhm'] with DBSCAN
    """
    sample_dict = connected_component_data.get(sample_name, {})
    pks_df = sample_dict.get('pks', pd.DataFrame())
    lns_df = sample_dict.get('lns', pd.DataFrame())
    props  = sample_dict.get('props', [])

    comp = next((c for c in props if c['label'] == target_label), None)
    if comp is None:
        print(f"[split_lid_dbscan] label={target_label} not found.")
        return []

    if 'lns' not in comp or old_lid not in comp['lns'].index:
        print(f"[split_lid_dbscan] LID={old_lid} not in label={target_label}.")
        return []

    peaks_lid = pks_df[pks_df.get('lid', pd.Series(dtype=int)) == old_lid]
    if peaks_lid.empty:
        print(f"[split_lid_dbscan] LID={old_lid} has no peaks.")
        return []

    if not {'wl', 'fwhm'}.issubset(peaks_lid.columns):
        print("[split_lid_dbscan] Missing wl/fwhm in pks_df.")
        return []

    X = peaks_lid[['wl', 'fwhm']].values
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    unique_lbls = set(labels) - {-1}
    n_clusters = len(unique_lbls)
    if n_clusters <= 1:
        print(f"[split_lid_dbscan] {n_clusters} cluster(s). No multi-split.")
        return []

    new_lids = []
    for cluster_id in sorted(unique_lbls):
        cluster_idx = peaks_lid.index[labels == cluster_id]
        if len(cluster_idx) == 0:
            continue

        lid_new = _get_next_unique_lid(lns_df)
        new_lids.append(lid_new)

        pks_df.loc[cluster_idx, 'lid'] = lid_new

        old_mid = lns_df.at[old_lid, 'mid'] if (old_lid in lns_df.index and 'mid' in lns_df.columns) else 1
        mean_wl = pks_df.loc[cluster_idx, 'wl'].mean()
        lns_df.loc[lid_new, 'mid'] = old_mid
        lns_df.loc[lid_new, 'wl']  = mean_wl

        comp['lns'].loc[lid_new] = {'mid': old_mid, 'wl': mean_wl}

    # Remove or update old lid
    remain_idx = pks_df.index[pks_df['lid'] == old_lid]
    if len(remain_idx) > 0:
        old_wl = pks_df.loc[remain_idx, 'wl'].mean()
        if old_lid in lns_df.index:
            lns_df.at[old_lid, 'wl'] = old_wl
        if old_lid in comp['lns'].index:
            comp['lns'].at[old_lid, 'wl'] = old_wl
    else:
        # remove entirely
        if old_lid in lns_df.index:
            lns_df.drop(old_lid, inplace=True)
        if old_lid in comp['lns'].index:
            comp['lns'].drop(old_lid, inplace=True)

    print(f"[split_lid_dbscan] Split old_lid={old_lid} => {n_clusters} clusters. New lids={new_lids}")
    return new_lids


def modify_mid(
    connected_component_data: Dict[str, Any],
    sample_name: str,
    target_label: int,
    old_mid: int,
    lids_to_move: List[int],
    action: str = "split",
    reason: Optional[str] = None
) -> Optional[int]:
    """
    Move or delete a subset of LIDs from an existing MID in a label.
    """
    sample_dict = connected_component_data.get(sample_name, {})
    pks_df = sample_dict.get('pks', pd.DataFrame())
    lns_df = sample_dict.get('lns', pd.DataFrame())
    props  = sample_dict.get('props', [])

    comp = next((c for c in props if c['label'] == target_label), None)
    if comp is None:
        print(f"[modify_mid] label={target_label} not found.")
        return None
    if 'lns' not in comp or not isinstance(comp['lns'], pd.DataFrame):
        print("[modify_mid] comp['lns'] missing or not DataFrame.")
        return None

    lids_of_old_mid = lns_df.index[lns_df.get('mid', pd.Series(dtype=int)) == old_mid]
    lids_in_comp = comp['lns'].index.intersection(lids_of_old_mid)
    if lids_in_comp.empty:
        print(f"[modify_mid] old_mid={old_mid} not in label={target_label}.")
        return None

    valid_lids_to_move = set(lids_to_move).intersection(lids_in_comp)
    if not valid_lids_to_move:
        print("[modify_mid] no valid lids to move.")
        return None

    if 'reason' not in lns_df.columns:
        lns_df['reason'] = None

    for L in valid_lids_to_move:
        if L in lns_df.index:
            lns_df.at[L, 'reason'] = reason

    action = action.lower()
    if action == "delete":
        # Remove from pks, lns, comp['lns']
        idx_remove = pks_df.index[pks_df['lid'].isin(valid_lids_to_move)]
        pks_df.drop(idx_remove, inplace=True, errors='ignore')
        for L in valid_lids_to_move:
            if L in lns_df.index:
                lns_df.drop(L, inplace=True, errors='ignore')
            if L in comp['lns'].index:
                comp['lns'].drop(L, inplace=True, errors='ignore')
        print(f"[modify_mid] Deleted LIDs={valid_lids_to_move}. old_mid={old_mid}")
        return None

    elif action == "split":
        existing_mids = lns_df['mid'].unique()
        if len(existing_mids) == 0:
            new_mid = 1
        else:
            new_mid = int(existing_mids.max()) + 1
            while new_mid in existing_mids:
                new_mid += 1

        # Move lids => new_mid
        lns_df.loc[valid_lids_to_move, 'mid'] = new_mid
        if 'mid' in comp['lns'].columns:
            comp['lns'].loc[valid_lids_to_move, 'mid'] = new_mid

        print(f"[modify_mid] Split LIDs={valid_lids_to_move} from MID={old_mid} => new_mid={new_mid}")
        return new_mid

    else:
        print(f"[modify_mid] Unrecognized action={action}")
        return None


##############################################################################
# 2) Plotting Utility
##############################################################################

def plot_component_with_neighbors(
    ax,
    labeled_image,
    image,
    ref_component,
    ref_components,
    cmp_component,
    title,
    padding_size=10
):
    """
    Basic bounding-box visualization of ref_component vs cmp_component,
    highlighting neighbors.
    """
    if image is None or labeled_image is None:
        ax.set_title(f"No image data.\n{title}")
        ax.axis('off')
        return

    min_row, min_col, max_row, max_col = ref_component.get('bbox', (0,0,0,0))
    cropped_image = image[
        max(0, min_row - padding_size): min(image.shape[0], max_row + padding_size),
        max(0, min_col - padding_size): min(image.shape[1], max_col + padding_size)
    ]
    ax.imshow(cropped_image, cmap='hot', interpolation='none')

    # local bounding box
    r0 = min_row - max(0, min_row - padding_size)
    c0 = min_col - max(0, min_col - padding_size)
    rr = max_row - min_row
    cc = max_col - min_col

    rect_main = patches.Rectangle((c0, r0), cc, rr, linewidth=2, edgecolor='blue', facecolor='none')
    ax.add_patch(rect_main)

    # compare component bounding box
    min_row_cmp, min_col_cmp, max_row_cmp, max_col_cmp = cmp_component.get('bbox', (0,0,0,0))
    r0_cmp = min_row_cmp - max(0, min_row - padding_size)
    c0_cmp = min_col_cmp - max(0, min_col - padding_size)
    rr_cmp = max_row_cmp - min_row_cmp
    cc_cmp = max_col_cmp - min_col_cmp
    rect_cmp = patches.Rectangle((c0_cmp, r0_cmp), cc_cmp, rr_cmp, linewidth=2, edgecolor='green', facecolor='none')
    ax.add_patch(rect_cmp)

    # highlight neighbors
    for neighbor in ref_components:
        if neighbor['label'] == ref_component['label']:
            continue
        nbbox = neighbor.get('bbox', (0,0,0,0))
        if is_neighboring(nbbox, cmp_component.get('bbox', (0,0,0,0))):
            nr0, nc0, nr1, nc1 = nbbox
            ro = nr0 - max(0, min_row - padding_size)
            co = nc0 - max(0, min_col - padding_size)
            rr_n = nr1 - nr0
            cc_n = nc1 - nc0
            rect_neighbor = patches.Rectangle(
                (co, ro), cc_n, rr_n, linewidth=1, edgecolor='yellow', linestyle='--', facecolor='none'
            )
            ax.add_patch(rect_neighbor)

    ax.set_title(title)
    ax.axis('off')


def plot_correlation_scores(ax, correlations, best_corr_idx, title):
    ax.plot(correlations, color='purple', marker='o')
    if 0 <= best_corr_idx < len(correlations):
        best_val = correlations[best_corr_idx]
        ax.axhline(y=best_val, color='red', linestyle='--', label=f"Best={best_val:.3f}")
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel('Comparison Index')
    ax.set_ylabel('Correlation Value')


def generate_mid_lid_title(sample_data: Dict[str, Any], ref_component: Dict[str, Any]) -> str:
    """
    Build a short label for the 'MID' content of the ref_component
    """
    lns_df = sample_data.get('lns', pd.DataFrame())
    c_lns = ref_component.get('lns', pd.DataFrame())
    if c_lns.empty or lns_df.empty:
        return f"Label {ref_component.get('label', -1)} (no LIDs)"

    lids = c_lns.index.unique()
    mids = lns_df.loc[lids, 'mid'].unique() if 'mid' in lns_df.columns else []

    lines = []
    for m in mids:
        lids_this_mid = lns_df.index[lns_df['mid'] == m]
        lines.append(f"MID={m} (LIDs: {len(lids_this_mid)})")
    return "\n".join(lines)


def plot_multiplets_and_mids(
    ax: plt.Axes,
    sample_data: Dict[str, Any],
    matched_component: Dict[str, Any],
    title: str,
    color_map: Dict[int, str],
    title_fontsize: int = 10
):
    """
    Plot lines & their MIDs in 1D (wavelength axis).
    """
    pks_df = sample_data.get('pks', pd.DataFrame())
    lns_df = sample_data.get('lns', pd.DataFrame())
    labeled_image = sample_data.get('labeled_image')
    label_id = matched_component.get('label', None)

    if label_id is None or labeled_image is None or lns_df.empty or pks_df.empty:
        ax.set_title(f"No data for label {label_id}")
        ax.axis('off')
        return

    comp_mask = (labeled_image == label_id)
    coords = np.argwhere(comp_mask)
    unique_lids = set()
    for (i, j) in coords:
        local = pks_df[(pks_df['i'] == i) & (pks_df['j'] == j)]
        unique_lids.update(local.get('lid', []).unique())

    # group by mid
    subset_lns = lns_df.loc[lns_df.index.intersection(unique_lids)]
    if subset_lns.empty:
        ax.set_title(f"No lns for label {label_id}")
        ax.axis('off')
        return

    mids_here = subset_lns['mid'].unique()
    mid_counts = {m: (subset_lns['mid'] == m).sum() for m in mids_here}
    sorted_mids = sorted(mids_here, key=lambda M: mid_counts[M], reverse=False)

    ax.set_xlabel('Wavelength')
    ax.set_yticks([])
    ax.set_title(title, fontsize=title_fontsize)

    current_y = 1.0
    for m in sorted_mids:
        lids_for_m = subset_lns.index[subset_lns['mid'] == m]
        color_ = color_map.get(m, '#000000')
        wls_ = lns_df.loc[lids_for_m, 'wl'].dropna()
        if wls_.empty:
            continue
        mn_wl, mx_wl = wls_.min() - 5, wls_.max() + 5

        # bounding rectangle
        ax.add_patch(
            patches.Rectangle(
                (mn_wl, current_y - 0.2),
                (mx_wl - mn_wl),
                0.6,
                linewidth=2,
                edgecolor=color_,
                facecolor='none'
            )
        )

        # lines
        for LID_ in lids_for_m:
            wl_val  = lns_df.at[LID_, 'wl'] if (LID_ in lns_df.index and 'wl' in lns_df.columns) else 0
            dwl_val = lns_df.at[LID_, 'dwl'] if (LID_ in lns_df.index and 'dwl' in lns_df.columns) else 0.5
            ax.plot([wl_val, wl_val], [current_y - 0.1, current_y + 0.1], color=color_, lw=2)
            ax.fill_between([wl_val - dwl_val, wl_val + dwl_val],
                            (current_y - 0.08), (current_y + 0.08),
                            color=color_, alpha=0.3)
        current_y += 1.0

    # attempt x-limits
    if 'wl' in pks_df.columns:
        ax.set_xlim(pks_df['wl'].min(), pks_df['wl'].max())


def plot_spatial_distribution(
    ax: plt.Axes,
    sample_data: Dict[str, Any],
    matched_component: Dict[str, Any],
    title: str,
    color_map: Dict[int, str]
):
    """
    Plot (i,j) positions of lines in matched_component, color-coded by MID
    """
    pks_df = sample_data.get('pks', pd.DataFrame())
    lns_df = sample_data.get('lns', pd.DataFrame())
    labeled_image = sample_data.get('labeled_image')
    label_id = matched_component.get('label', None)

    if label_id is None or labeled_image is None or pks_df.empty or lns_df.empty:
        ax.set_title("No Spatial Data")
        ax.axis('off')
        return

    mask_ = (labeled_image == label_id)
    coords = np.argwhere(mask_)
    unique_lids = set()
    for (i, j) in coords:
        local = pks_df[(pks_df['i'] == i) & (pks_df['j'] == j)]
        unique_lids.update(local.get('lid', []).unique())

    for lid_ in unique_lids:
        cpks = pks_df[pks_df['lid'] == lid_]
        if lid_ in lns_df.index and 'mid' in lns_df.columns:
            mid_ = lns_df.at[lid_, 'mid']
        else:
            mid_ = -999
        c_ = color_map.get(mid_, '#808080')
        sizes_ = np.sqrt(cpks.get('a', 1.0)) * 20
        ax.scatter(cpks['j'], cpks['i'], s=sizes_, c=c_)

    ax.axis('off')
    ax.set_title(title)


##############################################################################
# 3) Single-CC Plot => "Option 1" in your UI
##############################################################################

def plot_same_cc_across_samples(
    connected_component_data: Dict[str, Any],
    filtered_data_storage: Dict[str, Any],  # NEW parameter
    save_directory: str,
    all_matrices: Dict[Tuple[str, str], Any],
    all_subcorrelations: Dict[Tuple[str, str], Dict[int, Any]],
    start_sample: Optional[str] = None,
    target_label: Optional[int] = None
):
    """
    For the "Iterate Matched CC (Custom) with Default Sample/Label"
    - We'll produce NO clickable areas, so we avoid 404s by writing an empty JSON.
    """
    sample_names = sorted(connected_component_data.keys())
    if not sample_names:
        print("[plot_same_cc_across_samples] no samples in data.")
        return

    if start_sample is None:
        start_sample = sample_names[0]
    start_data = connected_component_data.get(start_sample, {})
    start_props = start_data.get('props', [])
    if target_label is None:
        if not start_props:
            print(f"[plot_same_cc_across_samples] no props in {start_sample}")
            return
        target_label = start_props[0]['label']

    try:
        start_idx = sample_names.index(start_sample)
    except ValueError:
        print(f"[plot_same_cc_across_samples] start_sample={start_sample} not found in keys.")
        return

    n_rows = len(sample_names) - start_idx - 1
    if n_rows <= 0:
        print("[plot_same_cc_across_samples] not enough samples after start_sample to plot matches.")
        return

    os.makedirs(save_directory, exist_ok=True)
    fig, axes = plt.subplots(n_rows, 5, figsize=(30, 6 * n_rows),
                             gridspec_kw={'width_ratios':[1,2,1,1,1]},
                             squeeze=False)

    ref_name = start_sample
    ref_label = target_label

    for row_i in range(n_rows):
        cmp_i = start_idx + row_i + 1
        cmp_name = sample_names[cmp_i]

        ref_dict = connected_component_data.get(ref_name, {})
        cmp_dict = connected_component_data.get(cmp_name, {})
        ref_comps = ref_dict.get('props', [])
        cmp_comps = cmp_dict.get('props', [])
        subcorr_key = (ref_name, cmp_name)
        score_mat = all_matrices.get(subcorr_key, np.zeros((len(ref_comps), len(cmp_comps))))
        subcorr = all_subcorrelations.get(subcorr_key, {})

        # find ref component with ref_label
        ref_idx = next((i for i, c in enumerate(ref_comps) if c['label'] == ref_label), None)
        if ref_idx is None:
            print(f"[plot_same_cc_across_samples] label={ref_label} not found in {ref_name}")
            continue
        ref_comp = ref_comps[ref_idx]
        correlations = subcorr.get(ref_idx, {}).get('correlations', np.zeros(len(cmp_comps)))
        if correlations.size == 0:
            print(f"No correlations from {ref_name} label={ref_label} => {cmp_name}")
            continue
        best_idx = int(np.argmax(correlations))
        if best_idx >= len(cmp_comps):
            print(f"best_idx={best_idx} out of range in {cmp_name}.")
            continue

        cmp_comp = cmp_comps[best_idx]
        best_val = correlations[best_idx]

        # ref_profile = np.mean(ref_comp.get('raw_spectra', np.zeros((1,1))), axis=0)
        # cmp_profile = np.mean(cmp_comp.get('raw_spectra', np.zeros((1,1))), axis=0)
        ref_spectra = get_cc_spectra(ref_comp, filtered_data_storage)
        cmp_spectra = get_cc_spectra(cmp_comp, filtered_data_storage)
        ref_profile = np.mean(ref_spectra, axis=0) if ref_spectra else np.zeros((1,1))
        cmp_profile = np.mean(cmp_spectra, axis=0) if cmp_spectra else np.zeros((1,1))
        wax = ref_dict.get('wax', np.arange(len(ref_profile)))

        color_map = generate_color_map(ref_dict, ref_comp)

        # col0 => CC viz
        ax_cc = axes[row_i, 0]
        plot_component_with_neighbors(
            ax=ax_cc,
            labeled_image=ref_dict.get('labeled_image'),
            image=ref_dict.get('image'),
            ref_component=ref_comp,
            ref_components=ref_comps,
            cmp_component=cmp_comp,
            title=f"{ref_name} label {ref_label}",
            padding_size=10
        )

        # col1 => spectral comparison
        ax_spec = axes[row_i, 1]
        ax_spec.plot(wax, ref_profile, label=f"{ref_name} L{ref_label}")
        ax_spec.plot(wax, cmp_profile, label=f"{cmp_name} L{cmp_comp['label']}")
        ax_spec.set_title(f"{ref_name} vs {cmp_name}\nCorr={best_val:.4f}")
        ax_spec.legend()

        # col2 => correlation scores
        ax_corr = axes[row_i, 2]
        plot_correlation_scores(ax_corr, correlations, best_idx, "Correlation Scores")

        # col3 => MID details
        ax_mids = axes[row_i, 3]
        mid_title = generate_mid_lid_title(ref_dict, ref_comp)
        plot_multiplets_and_mids(ax_mids, ref_dict, ref_comp, mid_title, color_map)

        # col4 => spatial distribution
        ax_spatial = axes[row_i, 4]
        plot_spatial_distribution(ax_spatial, ref_dict, ref_comp, "Spatial Plot", color_map)

        # update ref
        ref_name = cmp_name
        ref_label = cmp_comp['label']

    plt.tight_layout()
    fig_filename = f"cc_starting_with_sample_{start_sample}_label_{target_label}.png"
    fig_save_path = os.path.join(save_directory, fig_filename)
    plt.savefig(fig_save_path, dpi=150)
    plt.close(fig)

    # Write an EMPTY clickable-areas file => to avoid 404
    empty_json_path = fig_save_path + ".json"
    try:
        with open(empty_json_path, 'w') as f:
            json.dump({"clickableAreas": []}, f, indent=2)  # Changed key to "clickableAreas"
    except Exception as e:
        print("[plot_same_cc_across_samples] Error writing empty JSON:", e)

    print(f"[plot_same_cc_across_samples] Saved figure to {fig_save_path} (empty clickable areas)")

##############################################################################
# 4) get_fullimg2 (fixed for integer indexing)
##############################################################################

def get_fullimg2(
    pks: pd.DataFrame,
    amax: float,
    global_bounds: Tuple[int,int,int,int],
    wllims: List[int] = [1200, 1500]
) -> Optional[np.ndarray]:
    """
    Convert (i,j,wl,a) -> RGBA image. Avoid float indexing errors by casting int.
    """
    if pks.empty or amax == 0:
        return None

    i_min, i_max, j_min, j_max = global_bounds
    height = i_max - i_min + 1
    width  = j_max - j_min + 1
    img = np.zeros((height, width, 4), dtype=np.float32)

    range_wl = wllims[1] - wllims[0]
    cmap_ = plt.get_cmap('nipy_spectral', range_wl)

    # Precompute arrays
    i_coords = (pks['i'].values - i_min).astype(int)
    j_coords = (pks['j'].values - j_min).astype(int)
    intensities = pks['a'].values / amax
    wl_array = pks['wl'].values

    for idx in range(len(pks)):
        i_idx = i_coords[idx]
        j_idx = j_coords[idx]
        if i_idx < 0 or j_idx < 0:
            continue
        if i_idx >= height or j_idx >= width:
            continue

        wl_i = int(wl_array[idx] - wllims[0])
        wl_i = max(0, min(wl_i, range_wl - 1))
        alpha_ = intensities[idx]
        rgba = cmap_(wl_i)

        img[i_idx, j_idx, :3] = rgba[:3]
        img[i_idx, j_idx, 3]  = alpha_

    return img


# April 12 :)

bbox_cache = {}  # Simple in-memory cache for bounding-box computations

def get_bbox_and_mask(
    sample: str,
    label: int,
    connected_component_data: dict
) -> Optional[Tuple[int,int,int,int, np.ndarray]]:
    """
    Return (i_min, i_max, j_min, j_max, mask) for the bounding box 
    around all peaks in this (sample, label), caching results 
    to avoid recomputing.

    mask is a 2D boolean array where mask[ii, jj] = True if 
    that pixel belongs to the component's peaks.
    """
    global bbox_cache
    cache_key = (sample, label)
    if cache_key in bbox_cache:
        return bbox_cache[cache_key]

    sdata = connected_component_data.get(sample, {})
    pks_df = sdata.get('pks', pd.DataFrame())
    props = sdata.get('props', [])
    comp = next((c for c in props if c['label'] == label), None)
    if not comp or 'lns' not in comp or comp['lns'].empty:
        return None

    lns_df = comp['lns']
    lids_ = lns_df.index.unique()
    if lids_.empty:
        return None

    # Subset pks in one go
    pks_sub = pks_df[pks_df['lid'].isin(lids_)]
    if pks_sub.empty:
        return None

    coords = pks_sub[['i','j']].values.astype(int)
    i_min, i_max = coords[:,0].min(), coords[:,0].max()
    j_min, j_max = coords[:,1].min(), coords[:,1].max()

    height = i_max - i_min + 1
    width  = j_max - j_min + 1
    if height <= 0 or width <= 0:
        return None

    mask = np.zeros((height, width), dtype=bool)

    # Vectorized mask assignment
    ii = coords[:, 0] - i_min
    jj = coords[:, 1] - j_min
    mask[ii, jj] = True

    result = (i_min, i_max, j_min, j_max, mask)
    bbox_cache[cache_key] = result
    return result


def get_mid_info_sorted(
    sample: str,
    label: int,
    connected_component_data: dict
) -> List[Tuple[int, List[int], float]]:
    """
    Return a list of (mid, [lid indices sorted by wl], min_wl_for_mid),
    sorted by each mid's min wl.

    Vectorized approach with groupby("mid") instead of manual looping.
    """
    sdata = connected_component_data.get(sample, {})
    props = sdata.get('props', [])
    comp = next((xx for xx in props if xx['label'] == label), None)
    if not comp:
        return []

    lns_df = comp.get('lns', pd.DataFrame())
    if lns_df.empty or 'mid' not in lns_df.columns:
        return []

    # Group by mid in one pass
    grouped = lns_df.groupby('mid', sort=False)
    rows_out = []
    for midval, subdf in grouped:
        # Sort the LIDs by ascending 'wl'
        subdf_sorted = subdf.sort_values('wl')
        lids_sorted = list(subdf_sorted.index)  # LID is the DataFrame index
        if len(lids_sorted) == 0:
            continue
        min_wl = subdf_sorted['wl'].iloc[0]
        rows_out.append((midval, lids_sorted, float(min_wl)))

    # Sort all mids by their min_wl
    rows_out.sort(key=lambda x: x[2])
    return rows_out


def get_fullimg2(
    pks: pd.DataFrame,
    amax: float,
    global_bounds: Tuple[int,int,int,int],
    wllims: List[float] = [1200, 1500]
) -> Optional[np.ndarray]:
    """
    Build an RGBA overlay for these peaks without a Python for-loop.

    pks columns: [i,j,a,wl]
    amax is the max amplitude used to normalize alpha channel
    global_bounds = (i_min, i_max, j_min, j_max)
    """
    if pks.empty or amax <= 0:
        return None

    i_min, i_max, j_min, j_max = global_bounds
    height = i_max - i_min + 1
    width  = j_max - j_min + 1
    if height <= 0 or width <= 0:
        return None

    wl_min, wl_max = wllims
    range_wl = wl_max - wl_min

    # We build a segmented colormap with 'range_wl' discrete bins
    # If range_wl is large & fractional, we'll cast it to int; 
    # or you might prefer a simpler approach with a continuous colormap.
    n_bins = int(range_wl) if range_wl > 1 else 1
    cmap_  = plt.get_cmap('nipy_spectral', n_bins)

    # Coordinates
    i_coords = (pks['i'].values - i_min).astype(int)
    j_coords = (pks['j'].values - j_min).astype(int)

    # Clip out-of-bounds
    valid_mask = (
        (i_coords >= 0) & (i_coords < height) &
        (j_coords >= 0) & (j_coords < width)
    )
    i_coords = i_coords[valid_mask]
    j_coords = j_coords[valid_mask]

    # Normalize amplitude => alpha
    alpha_vals = (pks['a'].values[valid_mask] / amax).astype(np.float32)

    # Convert wl -> indices or floats in [0..1]
    # We'll do discrete bins: (wl - wl_min) -> [0..n_bins)
    wl_vals = pks['wl'].values[valid_mask]
    wl_idx  = (wl_vals - wl_min)
    wl_idx  = np.clip(np.floor(wl_idx), 0, n_bins-1).astype(int)

    # Create empty RGBA image
    img = np.zeros((height, width, 4), dtype=np.float32)

    # Map each row to a color
    # We'll pass normalized fraction: wl_idx/(n_bins-1) for continuous
    # or direct indexing for discrete. We'll do direct indexing:
    color_array = cmap_((wl_idx / max(n_bins - 1, 1)).astype(float))
    # color_array has shape (N,4)

    # Assign
    img[i_coords, j_coords, :3] = color_array[:, :3]
    img[i_coords, j_coords, 3]  = alpha_vals

    return img


def fetch_sub_mid(
    sample_: str,
    label_: int,
    lid_: int,
    connected_component_data: dict
) -> Optional[int]:
    """
    Return sub_mid if present in the 'lns' DataFrame row 
    for the given LID, else None.
    """
    sdata = connected_component_data.get(sample_, {})
    props = sdata.get('props', [])
    comp_ = next((c for c in props if c['label'] == label_), None)
    if not comp_ or 'lns' not in comp_:
        return None

    lns_df_ = comp_['lns']
    if 'sub_mid' not in lns_df_.columns or lid_ not in lns_df_.index:
        return None

    val_ = lns_df_.at[lid_, 'sub_mid']
    if pd.isna(val_):
        return None
    return int(val_)


def plot_multi_sample_comparison(
    ref_sample: str,
    ref_label: int,
    sample_label_matches: list,
    connected_component_data: dict,
    filtered_data_storage: Dict[str, Any],
    plot_output_folder: str,
    plot_filename: str = "multi_sample_comparison.png",
    samples_per_row: int = 4,
    max_sample_rows: int = 4,
    columns_per_sample: int = 4,
    save_individual_subplots: bool = False,
    target_label: Optional[int] = None
):
    """
    *This optimized version only uses the single-LID "Option B" logic.*
    Because you said you never use the single-big-figure approach, 
    we skip that code entirely and go directly to per-LID PNG creation.

    Returns a list of clickable_areas, each item like:
      {
        "lid": <int>,
        "sample": <str>,
        "label": <int>,
        "mid": <int>,
        "sub_mid": <int or None>,
        "image_path": <str or None>,
        "mid_wl": <float>,
        "lid_wl": <float>
      }
    """
    import matplotlib
    matplotlib.rcParams.update({'figure.max_open_warning': 0})
    import matplotlib.pyplot as plt

    os.makedirs(plot_output_folder, exist_ok=True)

    # 1) Collect the list of (sample, label) we'll plot
    all_samples = [(ref_sample, ref_label)] + [
        (s, l) for (s, l, _) in sample_label_matches
    ]
    if max_sample_rows is not None:
        limit_ = samples_per_row * max_sample_rows
        all_samples = all_samples[:limit_]
    if not all_samples:
        print("[plot_multi_sample_comparison] No samples to plot.")
        return []

    clickable_areas = []

    # 2) For each sample, get bounding box + mid->lids sorting
    for (samp, lbl) in all_samples:
        mids_list = get_mid_info_sorted(samp, lbl, connected_component_data)
        bbox_ = get_bbox_and_mask(samp, lbl, connected_component_data)
        if not mids_list or not bbox_:
            # no bounding => skip
            continue

        iSmin, iSmax, jSmin, jSmax, s_mask = bbox_
        # For quick reference
        sdict = connected_component_data.get(samp, {})
        pks_  = sdict.get('pks', pd.DataFrame())
        lns_  = sdict.get('lns', pd.DataFrame())

        # 3) For each (midVal, lids_list, fw), build an image per LID
        for (mVal, lid_array, fw) in mids_list:
            # lid_array is already sorted by wl
            for lid_ in lid_array:
                cpks = pks_[pks_['lid'] == lid_]
                if cpks.empty:
                    # record "no image" in clickable_areas
                    clickable_areas.append({
                        "lid": int(lid_),
                        "sample": samp,
                        "label": int(lbl),
                        "mid": int(mVal),
                        "sub_mid": fetch_sub_mid(samp, lbl, lid_, connected_component_data),
                        "image_path": None,
                        "mid_wl": float(fw),
                        "lid_wl": -999
                    })
                    continue

                central_wl = (
                    lns_.at[lid_, 'wl']
                    if (lid_ in lns_.index and 'wl' in lns_.columns)
                    else np.nan
                )
                wl_ = float(cpks['wl'].iloc[0]) if 'wl' in cpks.columns else -999

                # Build figure
                fig_lid, ax_lid = plt.subplots(figsize=(5, 5))
                amax_ = cpks['a'].max() if 'a' in cpks.columns else 1e-9
                # vectorized overlay
                img_ = get_fullimg2(cpks, amax_, (iSmin, iSmax, jSmin, jSmax))

                # Show the mask faintly behind it
                ax_lid.imshow(
                    s_mask,
                    cmap='viridis', alpha=0.15,
                    extent=[jSmin, jSmax, iSmin, iSmax],
                    origin='lower'
                )
                if img_ is not None:
                    ax_lid.imshow(
                        img_,
                        extent=[jSmin, jSmax, iSmin, iSmax],
                        origin='lower',
                        aspect='equal'
                    )
                    # Also add a scatter
                    ax_lid.scatter(
                        cpks['j'], cpks['i'],
                        s=10, color='#808080'
                    )

                ax_lid.set_title(
                    f"MID={mVal}, LID={lid_}, \nwl={central_wl:.1f}",
                    fontsize=24
                )
                ax_lid.axis('off')

                # Save
                out_filename = f"{samp}_label_{lbl}_mid_{mVal}_lid_{lid_}.png"
                out_path = Path(plot_output_folder) / out_filename
                plt.subplots_adjust(
                    left=0.0, right=1.0, top=0.9, bottom=0.00
                )
                plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0)
                plt.close(fig_lid)

                clickable_areas.append({
                    "lid": int(lid_),
                    "sample": samp,
                    "label": int(lbl),
                    "mid": int(mVal),
                    "sub_mid": fetch_sub_mid(samp, lbl, lid_, connected_component_data),
                    "image_path": str(out_path),
                    "mid_wl": float(fw),
                    "lid_wl": float(central_wl) if not pd.isna(central_wl) else -999
                })

    # Finally, write a JSON with all clickable area info
    big_json_path = Path(plot_output_folder) / "individual_lids_info.json"
    out_json = {"clickableAreas": clickable_areas}
    try:
        with open(big_json_path, 'w') as f:
            json.dump(out_json, f, indent=2)
        print(f"[plot_multi_sample_comparison] Wrote clickable areas to {big_json_path}")
    except Exception as e:
        print("[plot_multi_sample_comparison] Error writing JSON:", e)

    return clickable_areas

import plotly.graph_objects as go
import numpy as np

def create_interactive_full_spectrum_plot(
    wax: np.ndarray,
    raw_spectra: list,
    wllims: list,
    central_wl: float,
    lid: int,
    color_map: dict,
    mid: str,
) -> str:
    """
    Create a Plotly figure covering the entire wavelength range in 'wax'
    with all raw_spectra (unfiltered), allowing user zoom/pan.
    Also highlight the ±4 nm region around central_wl (if valid).

    Returns:
      HTML snippet (string) containing the interactive Plotly figure.
    """
    fig = go.Figure()

    # 1) Build a list of line traces (one for each raw spectrum)
    if raw_spectra and len(wax) > 0:
        traces = []
        line_color = color_map.get(mid, '#000000')
        for spectrum in raw_spectra:
            traces.append(go.Scatter(
                x=wax,
                y=spectrum,
                mode='lines',
                line=dict(color=line_color),
                name=f'LID {lid}'
            ))
        fig.add_traces(traces)

    else:
        # If no raw spectra exist
        fig.add_annotation(
            text="No raw spectra found.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )

    # 2) Possibly add highlight rectangle for ±4 nm around central_wl
    shapes = []
    if not np.isnan(central_wl):
        rect_left = central_wl - 4
        rect_right = central_wl + 4
        shapes.append(
            dict(
                type="rect",
                xref="x", yref="paper",
                x0=rect_left,
                x1=rect_right,
                y0=0,  # bottom of plot area
                y1=1,  # top of plot area
                fillcolor="yellow",
                opacity=0.2,
                line_width=0
            )
        )

    # 3) Update layout in a single call
    fig.update_layout(
        title=f"Interactive Full Spectrum (LID={lid}, WL={central_wl:.2f} nm)",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Intensity",
        template="plotly_white",
        dragmode="pan",
        xaxis=dict(
            range=[wllims[0], wllims[1]],
            rangeslider=dict(visible=True),
            type="linear"
        ),
        shapes=shapes
    )

    # Return HTML for embedding. (include_plotlyjs='cdn' -> loads Plotly from CDN)
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from typing import Optional
from pathlib import Path

def visualize_line_peaks(
    connected_component_data: dict,
    filtered_data_storage: dict,  # e.g. an app.state cache
    ref_name: str = '000.map',
    target_label: int = 19,
    wllims: list = [1200, 1500],
    plot_fwhm: bool = True,
    specific_lid: Optional[int] = None,
    output_folder: Optional[Path] = None
) -> str:
    """
    1) Creates the main figure(s) in Matplotlib, focusing on ±4 nm around each line (lid).
    2) If exactly one LID is requested, also builds a Plotly figure (interactive) over the entire 'wax' range,
       returning that figure as an HTML snippet to embed in the final page.
       
    Returns:
      interactive_html (str): If specific_lid is provided and spectra exist, returns the Plotly HTML snippet;
                              otherwise "".
    """
    # Quick check for missing data
    try:
        sample_data = connected_component_data[ref_name]
        pks_df = sample_data['pks']
        lns_df = sample_data['lns']
        wax    = sample_data['wax']
        props  = sample_data['props']
    except KeyError:
        # If the user gave a bad ref_name or if pks/lns is missing
        return ""

    # 1) Find the target component
    target_comp = next((c for c in props if c['label'] == target_label), None)
    if not target_comp or 'lns' not in target_comp:
        return ""

    # 2) Gather the relevant LIDs
    all_lids_in_component = target_comp['lns'].index.unique()
    if specific_lid is not None:
        if specific_lid not in all_lids_in_component:
            return ""
        selected_lids = [specific_lid]
    else:
        selected_lids = all_lids_in_component

    if len(selected_lids) == 0:
        return ""

    # 3) Sort selected LIDs by mean wavelength (just as before)
    def mean_wl_for_lid(lid):
        sub_pks = pks_df[pks_df['lid'] == lid]
        if sub_pks.empty:
            return 999999
        return sub_pks['wl'].mean()
    sorted_lids = sorted(selected_lids, key=mean_wl_for_lid)

    # 4) Build color map for MIDs present in these LIDs
    mid_series = lns_df.loc[sorted_lids, 'mid']
    unique_mids = mid_series.unique()
    color_map = {
        mid: f'#{random.randint(0, 0xFFFFFF):06x}' for mid in unique_mids
    }

    # 5) Compute bounding region for the entire CC (for the spatial overlay)
    #    + build a boolean mask so we can show the outline faintly
    all_coords = pks_df[pks_df['lid'].isin(all_lids_in_component)][['i','j']].dropna().astype(int).values
    if all_coords.size == 0:
        return ""

    i_min, i_max = all_coords[:,0].min(), all_coords[:,0].max()
    j_min, j_max = all_coords[:,1].min(), all_coords[:,1].max()
    global_bounds = (i_min, i_max, j_min, j_max)

    # Combined mask
    height = (i_max - i_min + 1)
    width  = (j_max - j_min + 1)
    combined_mask = np.zeros((height, width), dtype=bool)
    shift_i = all_coords[:,0] - i_min
    shift_j = all_coords[:,1] - j_min
    combined_mask[shift_i, shift_j] = True

    # 6) Helper to make an RGBA overlay for peaks in a single LID
    def make_rgba_overlay(pks: pd.DataFrame, amax: float, bounds: tuple):
        if pks.empty or amax <= 0:
            return None
        (imin, imax, jmin, jmax) = bounds
        h = imax - imin + 1
        w = jmax - jmin + 1
        if h <= 0 or w <= 0:
            return None

        wl_min, wl_max = wllims
        wl_range = wl_max - wl_min
        # We'll do discrete indexing. If wllims is large, we clamp it. 
        n_bins = int(wl_range) if wl_range >= 1 else 1
        cmap_obj = plt.get_cmap('nipy_spectral', n_bins)

        i_coords = (pks['i'].values - imin).astype(int)
        j_coords = (pks['j'].values - jmin).astype(int)
        alpha_vals = (pks['a'].values / amax).clip(0, None)
        wl_vals = pks['wl'].values
        # Clip to [0..(n_bins-1)]
        wl_idx = np.clip(np.floor(wl_vals - wl_min), 0, n_bins - 1).astype(int)

        # mask out-of-bounds
        in_range = (
            (i_coords >= 0) & (i_coords < h) &
            (j_coords >= 0) & (j_coords < w)
        )
        i_coords = i_coords[in_range]
        j_coords = j_coords[in_range]
        alpha_vals = alpha_vals[in_range]
        wl_idx = wl_idx[in_range]

        # Build RGBA
        img_ = np.zeros((h, w, 4), dtype=np.float32)
        colors_ = cmap_obj(wl_idx / max(n_bins - 1, 1))
        # colors_ has shape (N, 4)

        img_[i_coords, j_coords, :3] = colors_[:, :3]
        img_[i_coords, j_coords, 3]  = alpha_vals.astype(np.float32)
        return img_

    # 7) Create blockwise Matplotlib figures – each block up to 16 LIDs wide
    max_cols = 16
    lid_blocks = [sorted_lids[i : i+max_cols] for i in range(0, len(sorted_lids), max_cols)]
    interactive_html = ""

    import math

    # For each block, build a multi-subplot figure:
    for block_index, lids_in_block in enumerate(lid_blocks):
        n_block_lids = len(lids_in_block)
        # 2 rows if no FWHM, else 3 rows
        n_rows = 3 if plot_fwhm else 2
        fig, axs = plt.subplots(
            n_rows, n_block_lids,
            figsize=(5*n_block_lids, 5*n_rows)
        )
        if n_block_lids == 1:
            # axs -> (n_rows, ) shape
            axs = axs.reshape(n_rows, 1)

        for col_i, lid_ in enumerate(lids_in_block):
            # Grab relevant peaks for this LID
            cpks = pks_df.loc[pks_df['lid'] == lid_]
            if cpks.empty:
                continue

            # Determine central wl => define ±4 nm region
            central_wl = lns_df.at[lid_, 'wl'] if lid_ in lns_df.index else np.nan
            if pd.notna(central_wl):
                x_range = [central_wl - 4, central_wl + 4]
                central_wl_str = f"{central_wl:.2f}"
            else:
                x_range = wllims[:]
                central_wl_str = "N/A"

            mid_val = lns_df.loc[lid_, 'mid'] if lid_ in lns_df.index else "Unknown"
            # row 0: Filtered Spectrum
            ax_spec = axs[0, col_i]
            raw_spectra = get_cc_spectra(target_comp, filtered_data_storage)
            if raw_spectra and len(wax) > 0:
                # Plot each raw spectrum in the ±4 nm region
                for spec_ in raw_spectra:
                    in_filt = (wax >= x_range[0]) & (wax <= x_range[1])
                    ax_spec.plot(
                        wax[in_filt],
                        spec_[in_filt],
                        color=color_map.get(mid_val, '#000000'),
                        label=f"LID {lid_}"
                    )
            else:
                ax_spec.text(
                    0.5, 0.5, "No raw spectra found.",
                    transform=ax_spec.transAxes,
                    ha='center', va='center', fontsize=12
                )

            # Count peaks in [central_wl-4.. central_wl+4]
            peak_count = cpks[(cpks['wl']>=x_range[0])&(cpks['wl']<=x_range[1])].shape[0]
            ax_spec.set_title(
                f"LID={lid_} ({peak_count} pks)\nWL={central_wl_str} nm",
                fontsize=14
            )
            ax_spec.tick_params(axis='both', which='major', labelsize=12)

            # row 1: Spatial overlay
            ax_spatial = axs[1, col_i]
            amax_ = cpks['a'].max() if not cpks.empty else 0
            overlay_img = make_rgba_overlay(cpks, amax_, global_bounds)
            # Show the combined mask faintly
            ax_spatial.imshow(
                combined_mask,
                cmap='viridis', alpha=0.2,
                extent=[j_min-0.5, j_max+0.5, i_min-0.5, i_max+0.5],
                origin='lower'
            )
            if overlay_img is not None:
                ax_spatial.imshow(
                    overlay_img,
                    extent=[j_min-0.5, j_max+0.5, i_min-0.5, i_max+0.5],
                    origin='lower',
                    aspect='equal'
                )
                # Possibly scatter peak locations in a contrasting color
                sc_color = color_map.get(mid_val, '#888888')
                ax_spatial.scatter(
                    cpks['j'].values, cpks['i'].values,
                    c=sc_color, edgecolors='white', s=40
                )

            ax_spatial.set_title(
                f"MID={mid_val}, LID={lid_}\nWL={central_wl_str} nm",
                fontsize=14
            )
            ax_spatial.tick_params(axis='both', which='major', labelsize=12)

            # row 2: FWHM vs WL (only if plot_fwhm=True)
            if plot_fwhm:
                ax_fwhm = axs[2, col_i]
                # filter cpks to [x_range]
                csub = cpks[(cpks['wl']>=x_range[0])&(cpks['wl']<=x_range[1])]
                if not csub.empty and 'fwhm' in csub.columns:
                    ax_fwhm.scatter(csub['wl'], csub['fwhm'],
                                    color=color_map.get(mid_val, '#000000'))
                else:
                    ax_fwhm.text(
                        0.5, 0.5, "No FWHM data.",
                        transform=ax_fwhm.transAxes,
                        ha='center', va='center', fontsize=12
                    )
                ax_fwhm.set_title(f"FWHM vs WL (LID={lid_})", fontsize=12)
                ax_fwhm.tick_params(axis='both', which='major', labelsize=10)

        # End col loop
        plt.tight_layout()

        # Save figure if needed
        if output_folder is not None:
            if specific_lid is not None and len(selected_lids) == 1:
                # Single LID => name includes LID
                fname = f"line_peaks_{ref_name}_label_{target_label}_lid_{specific_lid}.png"
            else:
                # Multiple LIDs => no LID in name
                fname = f"line_peaks_{ref_name}_label_{target_label}_block_{block_index}.png"

            out_path = output_folder / fname
            plt.savefig(out_path, dpi=150)
        plt.close(fig)

    # 8) If exactly one LID => build the interactive Plotly figure for the full spectrum
    if specific_lid is not None and len(selected_lids) == 1:
        # We do an entire wax-based figure
        # pick that one LID's central wl & mid
        single_lid = selected_lids[0]
        cpks_sub = pks_df[pks_df['lid'] == single_lid]
        if cpks_sub.empty:
            return ""

        central_wl = (
            lns_df.at[single_lid, 'wl']
            if (single_lid in lns_df.index and 'wl' in lns_df.columns)
            else np.nan
        )
        mid_ = (
            lns_df.loc[single_lid, 'mid']
            if (single_lid in lns_df.index and 'mid' in lns_df.columns)
            else "Unknown MID"
        )
        # raw_spectra
        raw_spectra = get_cc_spectra(target_comp, filtered_data_storage)
        # from . import create_interactive_full_spectrum_plot  # or wherever you placed it

        interactive_html = create_interactive_full_spectrum_plot(
            wax=wax,
            raw_spectra=raw_spectra,
            wllims=wllims,
            central_wl=central_wl,
            lid=single_lid,
            color_map=color_map,
            mid=mid_
        )
        # Optional static fallback
        if output_folder:
            pass  # e.g. fig.write_image(...)

    return interactive_html




def generate_color_map(sample_data: Dict[str, Any], matched_component: Dict[str, Any]) -> Dict[int, str]:
    """
    Generate random color for each MID in matched_component. 
    """
    pks_df = sample_data.get('pks', pd.DataFrame())
    lns_df = sample_data.get('lns', pd.DataFrame())
    labeled_image = sample_data.get('labeled_image')
    label_id = matched_component.get('label', None)
    if label_id is None or labeled_image is None or pks_df.empty or lns_df.empty:
        return {}

    coords = np.argwhere(labeled_image==label_id)
    unique_lids = set()
    for (i, j) in coords:
        local = pks_df[(pks_df['i']==i)&(pks_df['j']==j)]
        unique_lids.update(local['lid'].unique())

    subset_lns = lns_df.loc[lns_df.index.intersection(unique_lids)]
    if 'mid' not in subset_lns.columns:
        return {}
    mids = subset_lns['mid'].unique()
    color_map = {}
    for m in mids:
        color_map[m] = f"#{random.randint(0,0xFFFFFF):06x}"
    return color_map


def plot_label_spectra(data, sample_name, label_num, filtered_data_storage, output_folder):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd  # Ensure pandas is imported

    # Get the data for this label
    sample_data = data[sample_name]
    props = sample_data.get("props", [])
    target_comp = next((c for c in props if c.get("label") == label_num), None)
    if not target_comp:
        return None

    # Get wavelength array
    wax = sample_data.get("wax")
    if wax is None:
        return None

    # Get lns dataframe for available LIDs
    lns_df = target_comp.get("lns", pd.DataFrame())
    if lns_df.empty:
        return None

    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))  # 10 distinct colors

    # Extract raw_spectra as a list from the dictionary target_comp.
    # raw_spectra_list = target_comp.get("raw_spectra", [])
    raw_spectra_list = get_cc_spectra(target_comp, filtered_data_storage)

    
    # Plot raw spectrum for each spectrum in the list
    for i, spectrum in enumerate(raw_spectra_list):
        # Attempt to get corresponding LID from lns_df index; if not available, use the loop index
        try:
            lid = lns_df.index[i]
        except IndexError:
            lid = i
        plt.plot(wax, spectrum,
                 color=colors[i % len(colors)],
                 alpha=0.7, linewidth=1,
                 label=f"LID={lid}")

    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Raw Intensity')
    plt.title(f'Raw Spectra for Sample {sample_name}, Label {label_num}')
    plt.grid(True, alpha=0.3)

    # Only show legend if we have 10 or fewer spectra to avoid cluttering
    if len(raw_spectra_list) <= 10:
        plt.legend(fontsize='small', loc='upper right')

    plt.tight_layout()

    # Save the plot
    filename = f"spectra_{sample_name}_label_{label_num}.png"
    plt.savefig(str(output_folder / filename),
                bbox_inches='tight', dpi=150)
    plt.close()

    return filename

