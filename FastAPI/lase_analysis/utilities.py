from scipy.ndimage import median_filter
from scipy.stats import pearsonr
import numpy as np


def preprocess_spectrum(spectrum):
    background_removed = spectrum - median_filter(spectrum, size=101)
    return (background_removed - np.mean(background_removed)) / np.std(background_removed)

def calculate_spectral_correlation(ref_profile, cmp_profile):
    if len(ref_profile) > 0 and len(cmp_profile) > 0:
        corr, _ = pearsonr(ref_profile, cmp_profile)
        return corr
    return 0

# Updated neighboring check for bounding boxes with padding
def is_neighboring(bbox1, bbox2, padding=10):
    min_row_1, min_col_1, max_row_1, max_col_1 = bbox1
    min_row_2, min_col_2, max_row_2, max_col_2 = bbox2
    return (
        min_row_2 < max_row_1 + padding and
        max_row_2 > min_row_1 - padding and
        min_col_2 < max_col_1 + padding and
        max_col_2 > min_col_1 - padding
    )