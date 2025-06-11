import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import label
from skimage.measure import regionprops_table
from joblib import Parallel, delayed
import logging
# for matching make "do_matching=True"
# Suppose these utility functions come from lase_analysis.utilities
from lase_analysis.utilities import (
    preprocess_spectrum,
    is_neighboring,
    calculate_spectral_correlation
)

logger = logging.getLogger(__name__)


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


def get_adjacent_tiles(tile_rc, tile_size=200):
    """
    For tile-based bounding-box pruning, return the 3x3 block of neighboring tiles.
    """
    r, c = tile_rc
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            neighbors.append((r + dr, c + dc))
    return neighbors
#may 2
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


class DataProcessor:
    def __init__(self, output_folder, padding=10, tile_size=200):
        """
        Parameters
        ----------
        output_folder : str or Path
            Directory for saving pickles, CSVs, etc.
        padding : int
            Extra bounding-box distance for 'neighboring' CC checks.
        tile_size : int
            Tile size for spatial pruning of bounding-box comparisons.
        """
        self.output_folder = Path(output_folder)
        self.padding = padding
        self.tile_size = tile_size

        # Data structures
        self.connected_component_data = {}
        self.all_matrices = {}
        self.all_subcorrelations = {}
        self.all_spatial_info = {}
        self.all_label_matches = {}

        # Must store filtered_data_storage so we can fetch raw spectra 
        # during match_consecutive_samples.
        self.filtered_data_storage = {}

        self.output_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized DataProcessor with output folder: {self.output_folder}")

    def analyze_data(self, lase_data_with_paths, analysis_params):
        """
        Example placeholder for your find_peaks, find_lines, refine steps.
        """
        results = {}
        for name_part, lase_data in lase_data_with_paths:
            logger.info(f'Processing File: {name_part}')
            for grp in lase_data.info:
            # from itertools import islice
            # for grp in islice(lase_data.info, 2):   
                lma = lase_data.get_analysis(grp, analysis=None)
                lma.find_peaks(**analysis_params['popt'])
                lma.find_lines(**analysis_params['lopt'])
                lma.refine(**analysis_params['ropt'])
                lma.save_analysis('base', overwrite=True)
            results[name_part] = lase_data.get_data(gname='grp_0', analysis='base')
        print("info type:", type(lase_data.info))    
        logger.info("Completed data analysis.")
        return results

    def analyze_connected_components(self, pks_df, spts, area_indices,
                                     lns_df=None, mlt_df=None, wax=None, name_part=None): #spts,
        """
        Build a 2D intensity image from pks_df, label connected components, 
        store only 'ispt_values'. We do NOT store raw_spectra to keep the 
        output pickle small. We'll fetch them on the fly from filtered_data_storage.
        """
        # 1) Build intensity image
        max_i = int(pks_df['i'].max()) + 1
        max_j = int(pks_df['j'].max()) + 1
        image = np.zeros((max_i, max_j), dtype=float)

        i_coords = pks_df['i'].to_numpy().astype(int)
        j_coords = pks_df['j'].to_numpy().astype(int)
        ph_values = pks_df['ph'].to_numpy()
        np.add.at(image, (i_coords, j_coords), ph_values)

        labeled_image, num_features = label(image > 0, structure=np.ones((3, 3), dtype=int))
        logger.info(f"[{name_part}] Number of components detected: {num_features}")

        # Each peak row gets cc_label
        cc_labels_for_peaks = labeled_image[i_coords, j_coords]
        pks_df['cc_label'] = cc_labels_for_peaks

        # Use regionprops_table for property extraction
        props_table = regionprops_table(
            labeled_image,
            intensity_image=image,
            properties=[
                'label',
                'area',
                'centroid',
                'bbox',
                'perimeter',
                'major_axis_length',
                'minor_axis_length'
            ]
        )
        props_df = pd.DataFrame(props_table)
        grouped = pks_df.groupby('cc_label')
        components_info = []

        # -------------- split "<base>_grp_X" into the two parts --------------
        #may2
        base, grp = name_part.rsplit('_grp_', 1)   # <-- robust even if more “_” in base
        grp = f'grp_{grp}'

        for _, row in props_df.iterrows():
            cc_label = int(row['label'])
            if cc_label == 0:  # background
                continue
            if cc_label not in grouped.groups:
                continue
            group_df = grouped.get_group(cc_label)
            if group_df.empty:
                continue

            ispt_values = group_df['ispt'].values
            component_lids = group_df['lid'].unique()

            if lns_df is not None and not lns_df.empty:
                component_lns = lns_df.loc[lns_df.index.intersection(component_lids)]
            else:
                component_lns = pd.DataFrame()

            component_mids = component_lns['mid'].unique() if not component_lns.empty else []
            if mlt_df is not None and not mlt_df.empty:
                component_mlt = mlt_df.loc[mlt_df.index.intersection(component_mids)]
            else:
                component_mlt = pd.DataFrame()

            comp_dict = {
                # "sample_id" : base,   # without the _grp suffix
                "group_name": grp,    # 'grp_0', 'grp_1', ...
                'label': cc_label,
                'area': row['area'],
                'centroid': (row['centroid-0'], row['centroid-1']),
                'bbox': (row['bbox-0'], row['bbox-1'], row['bbox-2'], row['bbox-3']),
                'perimeter': row['perimeter'],
                'major_axis_length': row['major_axis_length'],
                'minor_axis_length': row['minor_axis_length'],

                'ispt_values': ispt_values,
                'sample_id': normalize_sample_id(base),  # <--- crucial
                'lns': component_lns,
                'mlt': component_mlt,
                'wax': wax
            }
            components_info.append(comp_dict)

        return labeled_image, components_info, image
    

    def process_all_samples(self, filtered_data_storage, save_path, filter_area=None):
        """
        For each sample in filtered_data_storage, run analyze_connected_components.
        Then pickle the results in 'connected_component_data'.
        """
        connected_component_data = {}

        for name_part, data in filtered_data_storage.items():
            pks_df = data['pks']
            spts = data['spts']

            lns_df = data.get('lns', pd.DataFrame())
            mlt_df = data.get('mlt', pd.DataFrame())
            # from pandas import DataFrame
            # raw_lns = data.get('lns')
            # lns_df   = raw_lns if isinstance(raw_lns, DataFrame) else DataFrame(raw_lns or [])

            # raw_mlt = data.get('mlt')
            # mlt_df   = raw_mlt if isinstance(raw_mlt, DataFrame) else DataFrame(raw_mlt or [])
            crds = data['crds']
            wax = data['wax']

            if wax is None:
                logger.error(f"Wavelength axis (wax) missing for sample {name_part}. Skipping.")
                continue

            if filter_area is not None:
                area_indices = np.where(crds[:, 3] == filter_area)[0]
            else:
                area_indices = np.arange(crds.shape[0])

            if pks_df is not None and not pks_df.empty:
                labeled_image, components_info, image = self.analyze_connected_components(
                    pks_df, spts.copy(), area_indices, lns_df, mlt_df, wax, name_part=name_part
                ) #spts.copy(), 
                connected_component_data[name_part] = {
                    'props': components_info,
                    'pks': pks_df,
                    'lns': lns_df,
                    'mlt': mlt_df,
                    'wax': wax,
                    'labeled_image': labeled_image,
                    'image': image
                }

        with open(save_path, 'wb') as file:
            pickle.dump(connected_component_data, file)
        logger.info(f"Saved connected_component_data to {save_path}")

        self.connected_component_data = connected_component_data
        return connected_component_data

    def match_consecutive_samples(self):
        """
        For each consecutive sample pair, match connected components 
        by fetching raw spectra from self.filtered_data_storage on the fly.
        Creates correlation matrices and subcorrelation info for each pair.
        """
        sample_names = sorted(self.connected_component_data.keys())
        logger.info("Starting matching of consecutive samples.")

        for i in range(len(sample_names) - 1):
            ref_name = sample_names[i]
            cmp_name = sample_names[i + 1]

            ref_components = self.connected_component_data[ref_name]['props']
            cmp_components = self.connected_component_data[cmp_name]['props']
            N_ref = len(ref_components)
            N_cmp = len(cmp_components)

            logger.info(f"Processing reference sample: {ref_name} vs {cmp_name} "
                        f"({N_ref} ref CCs, {N_cmp} cmp CCs)")

            cmp_boxes = [c['bbox'] for c in cmp_components]

            # Build tile dictionary for bounding-box adjacency
            tile_cmp_dict = {}
            for cmp_idx, box in enumerate(cmp_boxes):
                minr, minc, _, _ = box
                tile_r = minr // self.tile_size
                tile_c = minc // self.tile_size
                tile_cmp_dict.setdefault((tile_r, tile_c), []).append(cmp_idx)

            score_matrix = np.zeros((N_ref, N_cmp), dtype=float)
            subcorrelation_storage = {}
            spatial_info_storage = {}

            def process_ref_cc(ridx):
                ref_comp = ref_components[ridx]
                ref_bbox = ref_comp['bbox']
                spatial_info = {
                    'label': ref_comp['label'],
                    'bbox': ref_bbox
                }

                # Fetch & preprocess reference spectra
                ref_spectra = get_cc_spectra(ref_comp, self.filtered_data_storage)
                if len(ref_spectra) == 0:
                    return ridx, None, 0.0, np.zeros(N_cmp), spatial_info
                ref_mean = np.mean(ref_spectra, axis=0)
                ref_profile = preprocess_spectrum(ref_mean)

                # Determine candidate tiles
                minr, minc, _, _ = ref_bbox
                ref_tile_r = minr // self.tile_size
                ref_tile_c = minc // self.tile_size

                candidate_indices = []
                from lase_analysis.utilities import calculate_spectral_correlation

                for neigh_tile in get_adjacent_tiles((ref_tile_r, ref_tile_c), self.tile_size):
                    if neigh_tile in tile_cmp_dict:
                        candidate_indices.extend(tile_cmp_dict[neigh_tile])

                if not candidate_indices:
                    return ridx, None, 0.0, np.zeros(N_cmp), spatial_info

                subcorr_vector = np.zeros(N_cmp)
                for cmp_idx in candidate_indices:
                    cbox = cmp_boxes[cmp_idx]
                    if is_neighboring(ref_bbox, cbox, self.padding):
                        cmp_comp = cmp_components[cmp_idx]
                        cmp_spectra = get_cc_spectra(cmp_comp, self.filtered_data_storage)
                        if len(cmp_spectra) == 0:
                            corr_val = 0.0
                        else:
                            cmp_mean = np.mean(cmp_spectra, axis=0)
                            cmp_profile = preprocess_spectrum(cmp_mean)
                            corr_val = calculate_spectral_correlation(ref_profile, cmp_profile)
                        subcorr_vector[cmp_idx] = corr_val

                best_idx = np.argmax(subcorr_vector)
                best_val = subcorr_vector[best_idx]
                if best_val <= 0:
                    return ridx, None, 0.0, subcorr_vector, spatial_info
                else:
                    best_label = cmp_components[best_idx]['label']
                    return ridx, best_label, best_val, subcorr_vector, spatial_info

            from joblib import Parallel, delayed
            results = Parallel(n_jobs=-1, backend='threading')(
                delayed(process_ref_cc)(r) for r in range(N_ref)
            )

            for (ridx, best_label, best_corr, subcorr_vec, sp_info) in results:
                score_matrix[ridx, :] = subcorr_vec
                spatial_info_storage[ridx] = sp_info
                if best_label is not None and best_corr > 0:
                    subcorrelation_storage[ridx] = {
                        'best_match_label': best_label,
                        'best_correlation': best_corr,
                        'correlations': subcorr_vec
                    }
                else:
                    logger.warning(f"No valid match found for reference CC "
                                   f"{ref_components[ridx]['label']} in {ref_name} vs {cmp_name}")

            self.all_matrices[(ref_name, cmp_name)] = score_matrix
            self.all_subcorrelations[(ref_name, cmp_name)] = subcorrelation_storage
            self.all_spatial_info[(ref_name, cmp_name)] = spatial_info_storage

            out_csv = self.output_folder / f"{ref_name}_to_{cmp_name}_corr_matrix_pad{self.padding}.csv"
            pd.DataFrame(score_matrix).to_csv(out_csv, index=False)
            logger.info(f"Saved correlation matrix to {out_csv}")

        logger.info("Completed matching of consecutive samples.")

    def find_best_matching_cc(self, ref_sample, ref_label, next_sample):
        key = (ref_sample, next_sample)
        if key not in self.all_subcorrelations:
            logger.warning(f"No subcorrelations found between {ref_sample} and {next_sample}.")
            return None, 0.0
        subcorr = self.all_subcorrelations[key]
        ref_components = self.connected_component_data[ref_sample]['props']
        ref_idx = next((i for i, c in enumerate(ref_components) if c['label'] == ref_label), None)
        if ref_idx is None:
            logger.warning(f"Label {ref_label} not found in sample {ref_sample}.")
            return None, 0.0
        match_info = subcorr.get(ref_idx, {})
        best_label = match_info.get('best_match_label', None)
        best_corr = match_info.get('best_correlation', 0.0)
        return (best_label, best_corr) if best_label else (None, 0.0)

    def find_consecutive_matches(self, start_sample, start_label, next_samples):
        current_sample = start_sample
        current_label = start_label
        matches = []
        for nsample in next_samples:
            matched_label, match_corr = self.find_best_matching_cc(current_sample, current_label, nsample)
            if matched_label is not None:
                matches.append((nsample, matched_label, match_corr))
                current_sample = nsample
                current_label = matched_label
            else:
                break
        return matches

    def generate_all_label_matches(self):
        logger.info("Generating all_label_matches...")
        sample_names = sorted(self.connected_component_data.keys())
        for i, sample in enumerate(sample_names):
            props = self.connected_component_data[sample]['props']
            for comp in props:
                lbl = comp['label']
                sub_next_samples = sample_names[i + 1:]
                matches = self.find_consecutive_matches(sample, lbl, sub_next_samples)
                self.all_label_matches[(sample, lbl)] = matches
        logger.info("Completed generating all_label_matches.")

    def save_all_data(self):
        """
        Save all relevant data structures as separate pickle files.
        """
        with open(self.output_folder / 'connected_component_data.pkl', 'wb') as f:
            pickle.dump(self.connected_component_data, f)
        with open(self.output_folder / 'all_matrices.pkl', 'wb') as f:
            pickle.dump(self.all_matrices, f)
        with open(self.output_folder / 'all_subcorrelations.pkl', 'wb') as f:
            pickle.dump(self.all_subcorrelations, f)
        with open(self.output_folder / 'all_spatial_info.pkl', 'wb') as f:
            pickle.dump(self.all_spatial_info, f)
        with open(self.output_folder / 'all_label_matches.pkl', 'wb') as f:
            pickle.dump(self.all_label_matches, f)
        logger.info(f"Saved all data structures to {self.output_folder}")

    def load_all_data(self):
        """
        Load the pickled data structures. 
        If they don't exist yet, you'll see warnings. That's normal.
        """
        files_map = {
            'connected_component_data': 'connected_component_data.pkl',
            'all_matrices': 'all_matrices.pkl',
            'all_subcorrelations': 'all_subcorrelations.pkl',
            'all_spatial_info': 'all_spatial_info.pkl',
            'all_label_matches': 'all_label_matches.pkl'
        }
        for key, filename in files_map.items():
            path = self.output_folder / filename
            if not path.exists():
                logger.warning(f"File {filename} not found in {self.output_folder}.")
                continue
            try:
                with open(path, 'rb') as f:
                    setattr(self, key, pickle.load(f))
                logger.info(f"Loaded '{key}' from '{path}'.")
            except Exception as e:
                logger.error(f"Error loading '{filename}' from '{self.output_folder}': {e}")
        logger.info("Completed loading all data.")
# April 28 has changed, do_matching=True has added
    def process_and_save(self, filtered_data_storage, do_matching=True):
        """
        High-level workflow:
          1) Assign filtered_data_storage so we can fetch raw spectra
          2) process_all_samples => connected components
          3) match_consecutive_samples => correlation
          4) generate_all_label_matches
          5) save_all_data => writes .pkl files
        """
        self.filtered_data_storage = filtered_data_storage

        # Step 1: Analyze CC
        self.connected_component_data = self.process_all_samples(
            filtered_data_storage,
            self.output_folder / 'connected_component_data.pkl'
        )
        if do_matching:
            # Step 2: Match across consecutive samples
            self.match_consecutive_samples()

            # Step 3: Generate label matches
            self.generate_all_label_matches()
        else:
            # Step 4: Save pickles
            self.save_all_data()
            logger.info("Processing and saving workflow completed.")

    def get_data(self):
        """
        Return all data structures in a dict.
        """
        return {
            'connected_component_data': self.connected_component_data,
            'all_matrices': self.all_matrices,
            'all_subcorrelations': self.all_subcorrelations,
            'all_spatial_info': self.all_spatial_info,
            'all_label_matches': self.all_label_matches
        }
