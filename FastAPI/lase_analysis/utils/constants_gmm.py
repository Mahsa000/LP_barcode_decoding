# import numpy as np
# from scipy.constants import h,c,e


# K_nm2meV = 1e12*h*c/e


# class LIDS:
#   NON = np.uint64(((2**32-1)<<32) + 0)
#   FLT = np.uint64(((2**32-1)<<32) + 1)
#   UNC = np.uint64(((2**32-1)<<32) + 2)
#   DRP = np.uint64(((2**32-1)<<32) + 3)
#   INC = np.uint64(((2**32-1)<<32) + 4) # incomplete splitting
#   BADS = np.array([NON,FLT,UNC,INC], dtype=np.uint64)


# class MIDS:
#   NON = np.uint64(((2**32-1)<<32) + 100)
#   BADS = np.array([NON], dtype=np.uint64)


# class FIDS:
#   NON = np.uint64(((2**32-1)<<32) + 10000)
#   UNM = np.uint64(((2**32-1)<<32) + 10001)
#   BADS = np.array([NON,UNM], dtype=np.uint64)


# class COLS:
#   FIT = ['a','wl','fwhm','ph','ispt']

#   # FLUO DATA
#   FFLU_SAVE = {'t': np.float64, 'tl': np.float64, 'tr': np.float64}
#   FFLU = {'t': np.float64, 'tl': np.float64, 'tr': np.float64}

#   # LASE MAP DATA
#   MPKS_SAVE = {'i': np.int32, 'j': np.int32, 'k': np.int32, 'a': np.float32, 'wl': np.float32, 'fwhm': np.float32,
#                'ph': np.float32, 'ispt': np.int32, 'lid': np.uint64}
#   MLNS_SAVE = {'i': np.float32, 'j': np.float32, 'k': np.float32, 'a': np.float32,
#                'wl': np.float32, 'dwl': np.float32, 'E': np.float32, 'dE': np.float32, 'ph': np.float32,
#                'n': np.int32, 'peri': np.int32, 'mid': np.uint64}
#   MMLT_SAVE = {'i': np.float32, 'j': np.float32, 'k': np.float32, 'n': np.int32}

#   MPKS = {'i': np.int32, 'j': np.int32, 'k': np.int32, 'a': np.float64, 'wl': np.float64, 'fwhm': np.float64,
#          'E': np.float64, 'fwhmE': np.float64, 'ph': np.float64, 'ispt': np.int32, 'lid': np.uint64}
#   MLNS = {'i': np.float64, 'j': np.float64, 'k': np.float64, 'a': np.float64,
#          'wl': np.float64, 'dwl': np.float64, 'E': np.float64, 'dE': np.float64, 'ph': np.float64,
#          'n': np.int32, 'peri': np.int32, 'mid': np.uint64}
#   MMLT = {'i': np.float64, 'j': np.float64, 'k': np.float64, 'n': np.int32}

#   # LASE FLOW DATA
#   FPKS_SAVE = {'t': np.float64, 'a': np.float32, 'wl': np.float32, 'fwhm': np.float32,
#                'ph': np.float32, 'ispt': np.int32, 'lid': np.uint64}
#   FLNS_SAVE = {'t': np.float64, 'dt': np.float64, 'a': np.float32, 'wl': np.float32, 'dwl': np.float32, 'E': np.float32, 'dE': np.float32,
#                'ph': np.float32, 'n': np.int32, 'mid': np.uint64}
#   FMLT_SAVE = {'t': np.float64, 'dt': np.float64, 'n': np.int32}
              
#   FPKS = {'t': np.float64, 'a': np.float64, 'wl': np.float64, 'fwhm': np.float64,
#          'E': np.float64, 'fwhmE': np.float64, 'ph': np.float64, 'ispt': np.int32, 'lid': np.uint64}
#   FLNS = {'t': np.float64, 'dt': np.float64, 'a': np.float64, 'wl': np.float64, 'dwl': np.float64,'E': np.float64, 'dE': np.float64,
#           'ph': np.float64, 'n': np.int32, 'mid': np.uint64}
#   FMLT = {'t': np.float64, 'dt': np.float64, 'n': np.int32}


# class SDTP:
#   MLNS = [(k,v) for k,v in COLS.MLNS.items()]
#   MMLT = [(k,v) for k,v in COLS.MMLT.items()]

#   FLNS = [(k,v) for k,v in COLS.FLNS.items()]
#   FMLT = [(k,v) for k,v in COLS.FMLT.items()]

import numpy as np
from scipy.constants import h, c, e

# conversion constant: nanometers → meV
K_nm2meV = 1e12 * h * c / e


class LIDS:
    NON = np.uint64(((2**32 - 1) << 32) + 0)
    FLT = np.uint64(((2**32 - 1) << 32) + 1)
    UNC = np.uint64(((2**32 - 1) << 32) + 2)
    DRP = np.uint64(((2**32 - 1) << 32) + 3)
    INC = np.uint64(((2**32 - 1) << 32) + 4)  # incomplete splitting
    BADS = np.array([NON, FLT, UNC, INC], dtype=np.uint64)


class MIDS:
    NON = np.uint64(((2**32 - 1) << 32) + 100)
    BADS = np.array([NON], dtype=np.uint64)


class FIDS:
    NON = np.uint64(((2**32 - 1) << 32) + 10000)
    UNM = np.uint64(((2**32 - 1) << 32) + 10001)
    BADS = np.array([NON, UNM], dtype=np.uint64)


class COLS:
    FIT = ['a', 'wl', 'fwhm', 'ph', 'ispt']

    # Fluorescence data
    FFLU_SAVE = {'t': np.float64, 'tl': np.float64, 'tr': np.float64}
    FFLU      = {'t': np.float64, 'tl': np.float64, 'tr': np.float64}

    # Map peaks
    MPKS_SAVE = {
        'i': np.int32, 'j': np.int32, 'k': np.int32,
        'a': np.float32, 'wl': np.float32, 'fwhm': np.float32,
        'ph': np.float32, 'ispt': np.int32, 'lid': np.uint64
    }
    MLNS_SAVE = {
        'i': np.float32, 'j': np.float32, 'k': np.float32,
        'a': np.float32, 'wl': np.float32, 'dwl': np.float32,
        'E': np.float32, 'dE': np.float32, 'ph': np.float32,
        'n': np.int32,   'peri': np.float32,  # ← changed to float32
        'mid': np.uint64
    }
    MMLT_SAVE = {'i': np.float32, 'j': np.float32, 'k': np.float32, 'n': np.int32}

    # runtime columns
    MPKS = {
        'i': np.int32, 'j': np.int32, 'k': np.int32,
        'a': np.float64, 'wl': np.float64, 'fwhm': np.float64,
        'E': np.float64, 'fwhmE': np.float64, 'ph': np.float64,
        'ispt': np.int32, 'lid': np.uint64
    }
    MLNS = {
        'i': np.float64, 'j': np.float64, 'k': np.float64,
        'a': np.float64, 'wl': np.float64, 'dwl': np.float32,  # ← float32
        'E': np.float64, 'dE': np.float32,                     # ← float32
        'ph': np.float64, 'n': np.int32, 'peri': np.float32,   # ← float32
        'mid': np.uint64
    }
    MMLT = {'i': np.float64, 'j': np.float64, 'k': np.float64, 'n': np.int32}

    # Flow (not shown)...
    FPKS_SAVE = {
        't': np.float64, 'a': np.float32, 'wl': np.float32,
        'fwhm': np.float32, 'ph': np.float32,
        'ispt': np.int32, 'lid': np.uint64
    }
    FLNS_SAVE = {
        't': np.float64, 'dt': np.float64, 'a': np.float32,
        'wl': np.float32, 'dwl': np.float32, 'E': np.float32,
        'dE': np.float32, 'ph': np.float32, 'n': np.int32,
        'mid': np.uint64
    }
    FMLT_SAVE = {'t': np.float64, 'dt': np.float64, 'n': np.int32}

    FPKS = {
        't': np.float64, 'a': np.float64, 'wl': np.float64,
        'fwhm': np.float64, 'E': np.float64, 'fwhmE': np.float64,
        'ph': np.float64, 'ispt': np.int32, 'lid': np.uint64
    }
    FLNS = {
        't': np.float64, 'dt': np.float64, 'a': np.float64,
        'wl': np.float64, 'dwl': np.float64, 'E': np.float64,
        'dE': np.float64, 'ph': np.float64, 'n': np.int32,
        'mid': np.uint64
    }
    FMLT = {'t': np.float64, 'dt': np.float64, 'n': np.int32}


class SDTP:
    # Structured dtypes for fast recarray conversion
    MLNS = [(k, v) for k, v in COLS.MLNS.items()]
    MMLT = [(k, v) for k, v in COLS.MMLT.items()]

    FLNS = [(k, v) for k, v in COLS.FLNS.items()]
    FMLT = [(k, v) for k, v in COLS.FMLT.items()]
