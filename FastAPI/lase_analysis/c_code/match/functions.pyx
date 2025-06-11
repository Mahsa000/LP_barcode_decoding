cimport cython

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

from scipy.sparse import csr_matrix
from time import sleep

CUINT8 = np.uint8
CINT32 = np.int32
CINT64 = np.int64
CUINT64 = np.uint64
CFLOAT = np.float64
ctypedef np.uint8_t CUINT8_t
ctypedef np.int32_t CINT32_t
ctypedef np.int64_t CINT64_t
ctypedef np.uint64_t CUINT64_t
ctypedef np.float64_t CFLOAT_t

cdef CFLOAT_t EPS = 1e-12
cdef CINT32_t MAX_SQUARE_SIZE = 13



# ----------------------------- SUPPORT FUNCTIONS ------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CFLOAT_t bits_count(CUINT64_t num):
  cdef CFLOAT_t sum_and = 0.
  while num:
    sum_and += 1.
    num &= num-1

  return sum_and

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef CFLOAT_t f_lorentz(CFLOAT_t x, CFLOAT_t gam):
#   return 1./(x*x/(0.25*gam*gam)+1.)

# --------------------------------- PARTITION ----------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def jaccard64_cross_01(CUINT64_t[:] binA0, CUINT64_t[:] binA1, CUINT64_t[:] binB0, CUINT64_t[:] binB1,
                       CINT32_t nA, CINT32_t i0, CFLOAT_t density, CFLOAT_t min_simil):

  cdef CINT32_t lenA0 = binA0.size
  cdef CINT32_t lenA1 = binA1.size
  assert lenA0 == lenA1
  cdef CINT32_t lenB0 = binB0.size
  cdef CINT32_t lenB1 = binB1.size
  assert lenB0 == lenB1
  cdef CUINT64_t nMax = CUINT64(np.ceil(density*lenA0*lenB0))

  indptrAB = np.zeros(nA+1, dtype=CUINT64)
  cdef CUINT64_t[:] _indptrAB_ = indptrAB
  indicesAB = np.full(nMax, -1, dtype=CINT32)
  cdef CINT32_t[:] _indicesAB_ = indicesAB
  dataAB = np.full_like(indicesAB, np.nan, dtype=CUINT8)
  cdef CUINT8_t[:] _dataAB_ = dataAB

  wA0 = _weights64(binA0)
  cdef CFLOAT_t[:] _wA0_ = wA0
  wB0 = _weights64(binB0)
  cdef CFLOAT_t[:] _wB0_ = wB0
  wA1 = _weights64(binA1)
  cdef CFLOAT_t[:] _wA1_ = wA1
  wB1 = _weights64(binB1)
  cdef CFLOAT_t[:] _wB1_ = wB1

  cdef CUINT64_t iseqAB = 0
  cdef CFLOAT_t sum0, sum1, temp
  cdef CINT32_t i, j, j0 = 0

  for i in range(lenA0):
    for j in range(lenB0):
      sum0 = bits_count(binA0[i]&binB0[j])
      sum1 = bits_count(binA1[i]&binB1[j])

      temp = max(sum0/(_wA0_[i]+_wB0_[j]-sum0), sum1/(_wA1_[i]+_wB1_[j]-sum1))
      if temp >= min_simil:
        _dataAB_[iseqAB] = <CUINT8_t>(255.*temp)
        _indicesAB_[iseqAB] = j
        iseqAB += 1
        if iseqAB >= nMax: return None

    _indptrAB_[i0+i+1] = iseqAB

  _indptrAB_[i0+i+1:] = iseqAB

  return csr_matrix((dataAB[:iseqAB], indicesAB[:iseqAB], indptrAB), (nA, lenB0))

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def jaccard64_self_01(CUINT64_t[:] binA0, CUINT64_t[:] binA1, CUINT64_t[:] binB0, CUINT64_t[:] binB1,
                      CINT32_t nA, CINT32_t i0, CFLOAT_t density, CFLOAT_t min_simil):

  cdef CINT32_t lenA0 = binA0.size
  cdef CINT32_t lenA1 = binA1.size
  assert lenA0 == lenA1
  cdef CINT32_t lenB0 = binB0.size
  cdef CINT32_t lenB1 = binB1.size
  assert lenB0 == lenB1
  cdef CUINT64_t nMax = CUINT64(np.ceil(density*lenA0*lenB0))

  indptrAB = np.zeros(nA+1, dtype=CUINT64)
  cdef CUINT64_t[:] _indptrAB_ = indptrAB
  indicesAB = np.full(nMax, -1, dtype=CINT32)
  cdef CINT32_t[:] _indicesAB_ = indicesAB
  dataAB = np.full_like(indicesAB, np.nan, dtype=CUINT8)
  cdef CUINT8_t[:] _dataAB_ = dataAB

  wA0 = _weights64(binA0)
  cdef CFLOAT_t[:] _wA0_ = wA0
  wB0 = _weights64(binB0)
  cdef CFLOAT_t[:] _wB0_ = wB0
  wA1 = _weights64(binA1)
  cdef CFLOAT_t[:] _wA1_ = wA1
  wB1 = _weights64(binB1)
  cdef CFLOAT_t[:] _wB1_ = wB1

  cdef CUINT64_t iseqAB = 0
  cdef CFLOAT_t sum0, sum1, temp
  cdef CINT32_t i, j, j0 = 0

  for i in range(lenA0):
    for j in range(i0+i+1,lenB0):
      sum0 = bits_count(binA0[i]&binB0[j])
      sum1 = bits_count(binA1[i]&binB1[j])

      temp = max(sum0/(_wA0_[i]+_wB0_[j]-sum0), sum1/(_wA1_[i]+_wB1_[j]-sum1))
      if temp >= min_simil:
        _dataAB_[iseqAB] = <CUINT8_t>(255.*temp)
        _indicesAB_[iseqAB] = j
        iseqAB += 1
        if iseqAB >= nMax: return None

    _indptrAB_[i0+i+1] = iseqAB

  _indptrAB_[i0+i+1:] = iseqAB

  return csr_matrix((dataAB[:iseqAB], indicesAB[:iseqAB], indptrAB), (nA, lenB0))

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def jaccard64_cross_all(CUINT64_t[:] binA0, CUINT64_t[:] binA1, CUINT64_t[:] binB0, CUINT64_t[:] binB1,
                        CINT32_t nA, CINT32_t i0, CFLOAT_t density, CFLOAT_t min_simil_01, CFLOAT_t min_simil_00):

  cdef CINT32_t lenA0 = binA0.size
  cdef CINT32_t lenA1 = binA1.size
  assert lenA0 == lenA1
  cdef CINT32_t lenB0 = binB0.size
  cdef CINT32_t lenB1 = binB1.size
  assert lenB0 == lenB1
  cdef CUINT64_t nMax = CUINT64(np.ceil(density*lenA0*lenB0))

  indptrAB = np.zeros(nA+1, dtype=CUINT64)
  cdef CUINT64_t[:] _indptrAB_ = indptrAB
  indptrA = np.zeros(nA+1, dtype=CUINT64)
  cdef CUINT64_t[:] _indptrA_ = indptrA
  indptrB = np.zeros(nA+1, dtype=CUINT64)
  cdef CUINT64_t[:] _indptrB_ = indptrB
  indicesAB = np.full(nMax, -1, dtype=CINT32)
  cdef CINT32_t[:] _indicesAB_ = indicesAB
  indicesA = np.full(nMax, -1, dtype=CINT32)
  cdef CINT32_t[:] _indicesA_ = indicesA
  indicesB = np.full(nMax, -1, dtype=CINT32)
  cdef CINT32_t[:] _indicesB_ = indicesB
  dataAB = np.full_like(indicesAB, np.nan, dtype=CUINT8)
  cdef CUINT8_t[:] _dataAB_ = dataAB
  dataA = np.full_like(indicesA, np.nan, dtype=CUINT8)
  cdef CUINT8_t[:] _dataA_ = dataA
  dataB = np.full_like(indicesB, np.nan, dtype=CUINT8)
  cdef CUINT8_t[:] _dataB_ = dataB

  wA0 = _weights64(binA0)
  cdef CFLOAT_t[:] _wA0_ = wA0
  wB0 = _weights64(binB0)
  cdef CFLOAT_t[:] _wB0_ = wB0
  wA1 = _weights64(binA1)
  cdef CFLOAT_t[:] _wA1_ = wA1
  wB1 = _weights64(binB1)
  cdef CFLOAT_t[:] _wB1_ = wB1

  cdef CUINT64_t iseqAB = 0, iseqA = 0, iseqB = 0
  cdef CFLOAT_t sum0, sum1, temp
  cdef CINT32_t i, j, j0 = 0

  for i in range(lenA0):
    for j in range(lenB0):
      sum0 = bits_count(binA0[i]&binB0[j])
      sum1 = bits_count(binA1[i]&binB1[j])

      temp = max(sum0/(_wA0_[i]+_wB0_[j]-sum0), sum1/(_wA1_[i]+_wB1_[j]-sum1))
      if temp >= min_simil_01:
        _dataAB_[iseqAB] = <CUINT8_t>(255.*temp)
        _indicesAB_[iseqAB] = j
        iseqAB += 1
        if iseqAB >= nMax: return None

      temp = max(sum0/(_wA0_[i]), sum1/(_wA1_[i]))
      if temp >= min_simil_00:
        _dataA_[iseqA] = <CUINT8_t>(255.*temp)
        _indicesA_[iseqA] = j
        iseqA += 1
        if iseqA >= nMax: return None

      temp = max(sum0/(_wB0_[j]), sum1/(_wB1_[j]))
      if temp >= min_simil_00:
        _dataB_[iseqB] = <CUINT8_t>(255.*temp)
        _indicesB_[iseqB] = j
        iseqB += 1
        if iseqB >= nMax: return None

    _indptrAB_[i0+i+1] = iseqAB
    _indptrA_[i0+i+1] = iseqA
    _indptrB_[i0+i+1] = iseqB

  _indptrAB_[i0+i+1:] = iseqAB
  _indptrA_[i0+i+1:] = iseqA
  _indptrB_[i0+i+1:] = iseqB

  return csr_matrix((dataAB[:iseqAB], indicesAB[:iseqAB], indptrAB), (nA, lenB0)),\
         csr_matrix((dataA[:iseqA], indicesA[:iseqA], indptrA), (nA, lenB0)),\
         csr_matrix((dataB[:iseqB], indicesB[:iseqB], indptrB), (nA, lenB0))

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def jaccard64_self_all(CUINT64_t[:] binA0, CUINT64_t[:] binA1, CUINT64_t[:] binB0, CUINT64_t[:] binB1,
                       CINT32_t nA, CINT32_t i0, CFLOAT_t density, CFLOAT_t min_simil_01, CFLOAT_t min_simil_00):

  cdef CINT32_t lenA0 = binA0.size
  cdef CINT32_t lenA1 = binA1.size
  assert lenA0 == lenA1
  cdef CINT32_t lenB0 = binB0.size
  cdef CINT32_t lenB1 = binB1.size
  assert lenB0 == lenB1
  cdef CUINT64_t nMax = CUINT64(np.ceil(density*lenA0*lenB0))

  indptrAB = np.zeros(nA+1, dtype=CUINT64)
  cdef CUINT64_t[:] _indptrAB_ = indptrAB
  indptrA = np.zeros(nA+1, dtype=CUINT64)
  cdef CUINT64_t[:] _indptrA_ = indptrA
  indptrB = np.zeros(nA+1, dtype=CUINT64)
  cdef CUINT64_t[:] _indptrB_ = indptrB
  indicesAB = np.full(nMax, -1, dtype=CINT32)
  cdef CINT32_t[:] _indicesAB_ = indicesAB
  indicesA = np.full(nMax, -1, dtype=CINT32)
  cdef CINT32_t[:] _indicesA_ = indicesA
  indicesB = np.full(nMax, -1, dtype=CINT32)
  cdef CINT32_t[:] _indicesB_ = indicesB
  dataAB = np.full_like(indicesAB, np.nan, dtype=CUINT8)
  cdef CUINT8_t[:] _dataAB_ = dataAB
  dataA = np.full_like(indicesA, np.nan, dtype=CUINT8)
  cdef CUINT8_t[:] _dataA_ = dataA
  dataB = np.full_like(indicesB, np.nan, dtype=CUINT8)
  cdef CUINT8_t[:] _dataB_ = dataB

  wA0 = _weights64(binA0)
  cdef CFLOAT_t[:] _wA0_ = wA0
  wB0 = _weights64(binB0)
  cdef CFLOAT_t[:] _wB0_ = wB0
  wA1 = _weights64(binA1)
  cdef CFLOAT_t[:] _wA1_ = wA1
  wB1 = _weights64(binB1)
  cdef CFLOAT_t[:] _wB1_ = wB1

  cdef CUINT64_t iseqAB = 0, iseqA = 0, iseqB = 0
  cdef CFLOAT_t sum0, sum1, temp
  cdef CINT32_t i, j, j0 = 0

  for i in range(lenA0):
    for j in range(i0+i+1,lenB0):
      sum0 = bits_count(binA0[i]&binB0[j])
      sum1 = bits_count(binA1[i]&binB1[j])

      temp = max(sum0/(_wA0_[i]+_wB0_[j]-sum0), sum1/(_wA1_[i]+_wB1_[j]-sum1))
      if temp >= min_simil_01:
        _dataAB_[iseqAB] = <CUINT8_t>(255.*temp)
        _indicesAB_[iseqAB] = j
        iseqAB += 1
        if iseqAB >= nMax: return None

      temp = max(sum0/(_wA0_[i]), sum1/(_wA1_[i]))
      if temp >= min_simil_00:
        _dataA_[iseqA] = <CUINT8_t>(255.*temp)
        _indicesA_[iseqA] = j
        iseqA += 1
        if iseqA >= nMax: return None

      temp = max(sum0/(_wB0_[j]), sum1/(_wB1_[j]))
      if temp >= min_simil_00:
        _dataB_[iseqB] = <CUINT8_t>(255.*temp)
        _indicesB_[iseqB] = j
        iseqB += 1
        if iseqB >= nMax: return None
    _indptrAB_[i0+i+1] = iseqAB
    _indptrA_[i0+i+1] = iseqA
    _indptrB_[i0+i+1] = iseqB

  _indptrAB_[i0+i+1:] = iseqAB
  _indptrA_[i0+i+1:] = iseqA
  _indptrB_[i0+i+1:] = iseqB

  return csr_matrix((dataAB[:iseqAB], indicesAB[:iseqAB], indptrAB), (nA, lenB0)),\
         csr_matrix((dataA[:iseqA], indicesA[:iseqA], indptrA), (nA, lenB0)),\
         csr_matrix((dataB[:iseqB], indicesB[:iseqB], indptrB), (nA, lenB0))

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def _weights64(CUINT64_t[:] arr):
  cdef Py_ssize_t N = arr.size

  weights = np.zeros(N, dtype=CFLOAT)
  cdef CFLOAT_t[:] _weights_ = weights

  cdef Py_ssize_t i

  for i in range(N):
    _weights_[i] = bits_count(arr[i])

  return weights

# ------------------------------ ALIGNING/SCORING ------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def align_single(EsA, EsB, AsA=None, AsB=None,
                 CFLOAT_t max_dE=1., CFLOAT_t thr_A=1., CFLOAT_t wgt_small=1., pmode='01'):
  cdef CINT32_t nA=np.int32(len(EsA)), nB=np.int32(len(EsB))

  if AsA is None: AsA = np.ones_like(EsA)
  if AsB is None: AsB = np.ones_like(EsB)
  assert nA == len(AsA)
  assert nB == len(AsB)

  cdef CFLOAT_t[:] _EsA_ = EsA
  cdef CFLOAT_t[:] _EsB_ = EsB
  cdef CFLOAT_t[:] _AsA_ = AsA
  cdef CFLOAT_t[:] _AsB_ = AsB

  matchedA = np.full(nA, -1, dtype=CINT32)
  cdef CINT32_t[:] _matchedA_ = matchedA
  matchedB = np.full(nB, -1, dtype=CINT32)
  cdef CINT32_t[:] _matchedB_ = matchedB

  wmat = np.zeros((MAX_SQUARE_SIZE, MAX_SQUARE_SIZE), dtype=CFLOAT)
  cdef CFLOAT_t[:,:] _wmat_ = wmat

  cdef CINT32_t ret
  cdef CFLOAT_t ascr, dscr
  cdef CUINT8_t _pmode_
  if   pmode == '00': _pmode_ = 0
  elif pmode == '11': _pmode_ = 1
  elif pmode == '01': _pmode_ = 2

  ret = _align_lines(_EsA_, _AsA_, 0, nA, _EsB_, _AsB_, 0, nB,
                     _matchedA_, _matchedB_,  _wmat_, &dscr, &ascr,
                     max_dE, thr_A, wgt_small, _pmode_)
  if ret < 0: raise RuntimeError(f'Error {ret}!!!')

  return matchedA, matchedB, ascr, dscr

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def align_single2(EsA, EsB, AsA=None, AsB=None,
                  CFLOAT_t max_dE=1., CFLOAT_t Lfwhm=0.2, CFLOAT_t thr_A0=0., CFLOAT_t thr_A1=1., pmode='01'):
  cdef CINT32_t nA=np.int32(len(EsA)), nB=np.int32(len(EsB))

  if AsA is None: AsA = np.ones_like(EsA)
  if AsB is None: AsB = np.ones_like(EsB)
  assert nA == len(AsA)
  assert nB == len(AsB)

  cdef CFLOAT_t[:] _EsA_ = EsA
  cdef CFLOAT_t[:] _EsB_ = EsB
  cdef CFLOAT_t[:] _AsA_ = AsA
  cdef CFLOAT_t[:] _AsB_ = AsB

  matchedA = np.full(nA, -1, dtype=CINT32)
  cdef CINT32_t[:] _matchedA_ = matchedA
  matchedB = np.full(nB, -1, dtype=CINT32)
  cdef CINT32_t[:] _matchedB_ = matchedB

  wmat = np.zeros((MAX_SQUARE_SIZE, MAX_SQUARE_SIZE), dtype=CFLOAT)
  cdef CFLOAT_t[:,:] _wmat_ = wmat

  cdef CINT32_t ret
  cdef CFLOAT_t scr, dig, KL=(0.5*Lfwhm)**2
  cdef CUINT8_t _pmode_
  if   pmode == '00': _pmode_ = 0
  elif pmode == '11': _pmode_ = 1
  elif pmode == '01': _pmode_ = 2

  ret = _align_lines2(_EsA_, _AsA_, 0, nA, _EsB_, _AsB_, 0, nB,
                      _matchedA_, _matchedB_,  _wmat_, &scr, &dig,
                      max_dE, KL, thr_A0, thr_A1, _pmode_)
  if ret < 0: raise RuntimeError(f'Error {ret}!!!')

  return matchedA, matchedB, scr, dig

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def align_single3(EsA, EsB, AsA=None, AsB=None,
                  CFLOAT_t max_dE=1., CFLOAT_t thr_A0=0., CFLOAT_t thr_A1=1., pmode='01'):
  cdef CINT32_t nA=np.int32(len(EsA)), nB=np.int32(len(EsB))

  if AsA is None: AsA = np.ones_like(EsA)
  if AsB is None: AsB = np.ones_like(EsB)
  assert nA == len(AsA)
  assert nB == len(AsB)

  cdef CFLOAT_t[:] _EsA_ = EsA
  cdef CFLOAT_t[:] _EsB_ = EsB
  cdef CFLOAT_t[:] _AsA_ = AsA
  cdef CFLOAT_t[:] _AsB_ = AsB

  matchedA = np.full(nA, -1, dtype=CINT32)
  cdef CINT32_t[:] _matchedA_ = matchedA
  matchedB = np.full(nB, -1, dtype=CINT32)
  cdef CINT32_t[:] _matchedB_ = matchedB

  cdef CINT32_t ret
  cdef CFLOAT_t scr, dig
  cdef CUINT8_t _pmode_
  if   pmode == '00': _pmode_ = 0
  elif pmode == '11': _pmode_ = 1
  elif pmode == '01': _pmode_ = 2

  ret = _align_lines3(_EsA_, _AsA_, 0, nA, _EsB_, _AsB_, 0, nB,
                      _matchedA_, _matchedB_, &scr, &dig,
                      max_dE, thr_A0, thr_A1, _pmode_)
  if ret < 0: raise RuntimeError(f'Error {ret}!!!')

  return matchedA, matchedB, scr, dig

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def score_lines(dataA, dataB, simil, CUINT8_t pmode, opt):
  cdef CINT32_t lenA=np.int32(len(dataA)), lenB=np.int32(len(dataB))
  assert lenA == simil.shape[0]
  assert lenB == simil.shape[1]

  cdef CFLOAT_t[:] _EsA_ = dataA.Es
  cdef CFLOAT_t[:] _AsA_ = dataA.As
  cdef CINT32_t[:] _nsA_ = dataA.ns
  cumA = np.insert(np.cumsum(dataA.ns,dtype=np.int32),0,0)
  cdef CINT32_t[:] _cumA_ = cumA

  cdef CFLOAT_t[:] _EsB_ = dataB.Es
  cdef CFLOAT_t[:] _AsB_ = dataB.As
  cdef CINT32_t[:] _nsB_ = dataB.ns
  cumB = np.insert(np.cumsum(dataB.ns,dtype=np.int32),0,0)
  cdef CINT32_t[:] _cumB_ = cumB

  # cdef CFLOAT_t[:] _simil_ = simil.data
  cdef CUINT64_t[:] _indptr_ = simil.indptr.astype(CUINT64)
  cdef CINT32_t[:] _indices_ = simil.indices

  dscr_data = np.zeros_like(simil.data, dtype=CFLOAT)
  cdef CFLOAT_t[:] _dscr_data_ = dscr_data
  ascr_data = np.zeros_like(simil.data, dtype=CFLOAT)
  cdef CFLOAT_t[:] _ascr_data_ = ascr_data

  matchedA = np.full(opt.nmax, -1, dtype=CINT32)
  cdef CINT32_t[:] _matchedA_ = matchedA
  matchedB = np.full(opt.nmax, -1, dtype=CINT32)
  cdef CINT32_t[:] _matchedB_ = matchedB

  wmat = np.zeros((MAX_SQUARE_SIZE, MAX_SQUARE_SIZE), dtype=CFLOAT)
  cdef CFLOAT_t[:,:] _wmat_ = wmat

  cdef CFLOAT_t min_dig = opt.min_dig
  cdef CFLOAT_t max_dE = opt.max_dE
  # cdef CFLOAT_t thr_A0 = opt.thr_A0
  # cdef CFLOAT_t thr_A1 = opt.thr_A1
  cdef CFLOAT_t thr_amp = opt.th_amp
  cdef CFLOAT_t wgt_small = opt.wgt_small

  cdef CUINT64_t ptrB
  cdef CINT32_t ret, jA, jB, idat, cA0, cB0
  cdef CINT32_t iA, iB, nA, nB, iB0, iA1, iB1, n_row, n_col, nsq
  cdef CFLOAT_t curr, right, bottom, diag, scr_tmp, scr_tot, wM, wT, ascr, dscr

  idat = -1
  for jA in range(lenA):
    cA0 = _cumA_[jA]
    for ptrB in range(_indptr_[jA],_indptr_[jA+1]):
      idat += 1
      jB = _indices_[ptrB]
      cB0 = _cumB_[jB]
      # ----------------
      nA = _nsA_[jA]
      nB = _nsB_[jB]
      iA = -1
      iB0 = -1
      scr_tot = 0.
      # --- Find aligned lines ---
      while True: # cycling through iA
        iA += 1
        # If there are no more rows left, finish
        if iA >= nA: break
        _matchedA_[iA] = -1

        iB = iB0
        while True: # cycling through iB
          iB += 1
          if iB >= nB: break
          # Calculate current and neighbors positions
          curr = _EsB_[cB0+iB]-_EsA_[cA0+iA]
          if curr > max_dE: break # Past possible matches
          if curr < -max_dE: continue # Before possible matches
          curr = max_dE-abs(curr)

          if iB<nB-1: right = max(0., max_dE-abs(_EsB_[cB0+iB+1]-_EsA_[cA0+iA]))
          else:       right = 0.
          if iA<nA-1: bottom = max(0., max_dE-abs(_EsB_[cB0+iB]-_EsA_[cA0+iA+1]))
          else:       bottom = 0.

          # If current position has lower direct neighbors, it's the best match
          if (curr>right) and (curr>bottom):
            _matchedA_[iA] = iB
            scr_tot += curr/max_dE
            iB0 = iB
            break
          # If current position has a higher direct neighbor...
          else:
            if (iA<nA-1) and (iB<nB-1): diag = max(0., max_dE-abs(_EsB_[cB0+iB+1]-_EsA_[cA0+iA+1]))
            else:                       diag = 0.
            # If diagonal neighbor is 0, move to the highest direct neighbor
            # (in this case direct neighbors cannot be equal by construction)
            if diag == 0.:
              # If the highest neighbor is vertical, go to next row (same column)
              if bottom>right:
                iB0 = iB-1
                break
              # If the highest neighbor is horizontal, go to next column (same row)
              else:
                continue
            # If diagonal neighbor is >0, find the minimum square
            else:
              iA1 = iA+1
              iB1 = iB+1
              while True:
                # if (jA+nsq_row >= nA) or (jB+nsq_col >= nB): break
                if (iA1 < nA-1) and (abs(_EsB_[cB0+iB1]-_EsA_[cA0+iA1+1]) < max_dE):
                  iA1 += 1
                elif (iB1 < nB-1) and (abs(_EsB_[cB0+iB1+1]-_EsA_[cA0+iA1]) < max_dE):
                  iB1 += 1
                else:
                  break

              n_row = iA1-iA+1
              n_col = iB1-iB+1
              nsq = max(n_row, n_col)

              _calc_submatrix(_EsA_, _EsB_, cA0+iA, cB0+iB, n_row, n_col, _wmat_, max_dE)
              if nsq > MAX_SQUARE_SIZE:
                return -1*nsq

              ret = _match_square(_wmat_, n_row, n_col, _matchedA_, iA, iB, &scr_tmp)
              if ret < 0: return ret

              scr_tot += scr_tmp/max_dE
              iB0 = iB1
              iA = iA1
              break

      # --- Calculate scores ---
      wT = 0.
      wM = 0.
      for iB in range(nB): _matchedB_[iB] = -1
      for iA in range(nA):
        if _matchedA_[iA]>=0:
          wT += 2.
          wM += 2.
          _matchedB_[_matchedA_[iA]] = iA
        else:
          if pmode%2==0:
            if _AsA_[cA0+iA] < thr_amp: wT += wgt_small
            else:                       wT += 1.
            # wT += max(0., (min(thr_A1, _AsA_[cA0+iA])-thr_A0)/(thr_A1-thr_A0))
      if pmode>0:
        for iB in range(nB):
          if _matchedB_[iB] < 0:
            if _AsB_[cB0+iB] < thr_amp: wT += wgt_small
            else:                       wT += 1.
            # wT += max(0., (min(thr_A1, _AsB_[cB0+iB])-thr_A0)/(thr_A1-thr_A0))

      dscr = wM/wT
      if dscr < min_dig: continue
      if wM > 0.: ascr = max(EPS, 2.*scr_tot/wM)
      else:       ascr = 0.

      _dscr_data_[idat] = dscr
      _ascr_data_[idat] = ascr

  mat_ascr = csr_matrix((ascr_data, simil.indices.copy(), simil.indptr.copy()), (lenA,lenB))
  mat_ascr.eliminate_zeros()
  mat_dscr = csr_matrix((dscr_data, simil.indices.copy(), simil.indptr.copy()), (lenA,lenB))
  mat_dscr.eliminate_zeros()

  return mat_ascr, mat_dscr

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def score_lines2(dataA, dataB, simil, CUINT8_t pmode, opt):
  cdef CINT32_t lenA=np.int32(len(dataA)), lenB=np.int32(len(dataB))
  assert lenA == simil.shape[0]
  assert lenB == simil.shape[1]

  cdef CFLOAT_t[:] _EsA_ = dataA.Es
  cdef CFLOAT_t[:] _AsA_ = dataA.As
  cdef CINT32_t[:] _nsA_ = dataA.ns
  cumA = np.insert(np.cumsum(dataA.ns,dtype=np.int32),0,0)
  cdef CINT32_t[:] _cumA_ = cumA

  cdef CFLOAT_t[:] _EsB_ = dataB.Es
  cdef CFLOAT_t[:] _AsB_ = dataB.As
  cdef CINT32_t[:] _nsB_ = dataB.ns
  cumB = np.insert(np.cumsum(dataB.ns,dtype=np.int32),0,0)
  cdef CINT32_t[:] _cumB_ = cumB

  # cdef CFLOAT_t[:] _simil_ = simil.data
  cdef CUINT64_t[:] _indptr_ = simil.indptr.astype(CUINT64)
  cdef CINT32_t[:] _indices_ = simil.indices

  scr_data = np.zeros_like(simil.data, dtype=CFLOAT)
  cdef CFLOAT_t[:] _scr_data_ = scr_data

  matchedA = np.full(opt.nmax, -1, dtype=CINT32)
  cdef CINT32_t[:] _matchedA_ = matchedA
  matchedB = np.full(opt.nmax, -1, dtype=CINT32)
  cdef CINT32_t[:] _matchedB_ = matchedB

  wmat = np.zeros((MAX_SQUARE_SIZE, MAX_SQUARE_SIZE), dtype=CFLOAT)
  cdef CFLOAT_t[:,:] _wmat_ = wmat

  cdef CFLOAT_t min_dig = opt.min_dig
  cdef CFLOAT_t max_dE = opt.max_dE
  cdef CFLOAT_t thr_A0 = opt.thr_A0
  cdef CFLOAT_t thr_A1 = opt.thr_A1
  cdef CFLOAT_t KL = (0.5*opt.Lfwhm)**2

  cdef CUINT64_t ptrB
  cdef CINT32_t ret, jA, jB, idat, cA0, cB0, iA, iB, nA, nB, iB0, iA1, iB1, n_row, n_col, nsq
  cdef CFLOAT_t curr, prev, right, bottom, diag, scr_tmp, scr_tot, wM, wT

  idat = -1
  for jA in range(lenA):
    cA0 = _cumA_[jA]
    for ptrB in range(_indptr_[jA],_indptr_[jA+1]):
      idat += 1
      jB = _indices_[ptrB]
      cB0 = _cumB_[jB]
      # ----------------
      nA = _nsA_[jA]
      nB = _nsB_[jB]
      iA = -1
      iB0 = -1
      scr_tot = 0.
      # --- Find aligned lines ---
      while True: # cycling through iA
        iA += 1
        # If there are no more rows left, finish
        if iA >= nA: break
        _matchedA_[iA] = -1

        iB = iB0
        while True: # cycling through iB
          iB += 1
          if iB >= nB: break
          # Calculate current and neighbors positions
          curr = (_EsB_[cB0+iB]-_EsA_[cA0+iA])/max_dE
          if curr > 1.:  break # Past possible matches
          if curr < -1.: continue # Before possible matches
          # curr = max_dE-abs(curr)
          curr = 1.-abs(curr)/max_dE

          if iB<nB-1: right = 1.-abs(_EsB_[cB0+iB+1]-_EsA_[cA0+iA])/max_dE
          else:       right = 0.
          if iA<nA-1: bottom = 1.-abs(_EsB_[cB0+iB]-_EsA_[cA0+iA+1])/max_dE
          else:       bottom = 0.

          # If current position has lower direct neighbors, it's the best match
          if (curr>right) and (curr>bottom):
            _matchedA_[iA] = iB
            scr_tot += curr
            iB0 = iB
            break
          # If current position has a higher direct neighbor...
          else:
            if (iA<nA-1) and (iB<nB-1): diag = max(0., 1.-abs(_EsB_[cB0+iB+1]-_EsA_[cA0+iA+1])/max_dE)
            else:                       diag = 0.
            # If diagonal neighbor is 0, move to the highest direct neighbor
            # (in this case direct neighbors cannot be equal by construction)
            if diag == 0.:
              # If the highest neighbor is vertical, go to next row (same column)
              if bottom>right:
                iB0 = iB-1
                break
              # If the highest neighbor is horizontal, go to next column (same row)
              else:
                continue
            # If diagonal neighbor is >0, find the minimum square
            else:
              iA1 = iA+1
              iB1 = iB+1
              while True:
                if (iA1 < nA-1) and (abs(_EsB_[cB0+iB1]-_EsA_[cA0+iA1+1]) < max_dE):
                  iA1 += 1
                elif (iB1 < nB-1) and (abs(_EsB_[cB0+iB1+1]-_EsA_[cA0+iA1]) < max_dE):
                  iB1 += 1
                else:
                  break

              n_row = iA1-iA+1
              n_col = iB1-iB+1
              nsq = max(n_row, n_col)

              _calc_submatrix2(_EsA_, _EsB_, cA0+iA, cB0+iB, n_row, n_col, _wmat_, max_dE, KL)
              if nsq > MAX_SQUARE_SIZE: return -1*nsq

              ret = _match_square(_wmat_, n_row, n_col, _matchedA_, iA, iB, &scr_tmp)
              if ret < 0: return ret

              scr_tot += scr_tmp
              iB0 = iB1
              iA = iA1
              break

      # --- Calculate scores ---
      wT = 0.
      wM = 0.
      for iB in range(nB): _matchedB_[iB] = -1
      for iA in range(nA):
        if _matchedA_[iA]>=0:
          wT += 2.
          wM += 2.
          _matchedB_[_matchedA_[iA]] = iA
        else:
          if pmode%2==0: wT += max(0., (min(thr_A1, _AsA_[cA0+iA])-thr_A0)/(thr_A1-thr_A0))
      if pmode>0:
        for iB in range(nB):
          if _matchedB_[iB] < 0:
            wT += max(0., (min(thr_A1, _AsB_[cB0+iB])-thr_A0)/(thr_A1-thr_A0))

      if wM/wT < min_dig: continue
      _scr_data_[idat] = 2.*scr_tot/wT

  mat_scr = csr_matrix((scr_data, simil.indices.copy(), simil.indptr.copy()), (lenA,lenB))
  mat_scr.eliminate_zeros()

  return mat_scr

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def score_lines3(dataA, dataB, simil, CUINT8_t pmode, opt):
  cdef CINT32_t lenA=np.int32(len(dataA)), lenB=np.int32(len(dataB))
  assert lenA == simil.shape[0]
  assert lenB == simil.shape[1]

  cdef CFLOAT_t[:] _EsA_ = dataA.Es
  cdef CFLOAT_t[:] _AsA_ = dataA.As
  cdef CINT32_t[:] _nsA_ = dataA.ns
  cumA = np.insert(np.cumsum(dataA.ns,dtype=np.int32),0,0)
  cdef CINT32_t[:] _cumA_ = cumA

  cdef CFLOAT_t[:] _EsB_ = dataB.Es
  cdef CFLOAT_t[:] _AsB_ = dataB.As
  cdef CINT32_t[:] _nsB_ = dataB.ns
  cumB = np.insert(np.cumsum(dataB.ns,dtype=np.int32),0,0)
  cdef CINT32_t[:] _cumB_ = cumB

  # cdef CUINT8_t[:] _simil_ = simil.data
  cdef CUINT64_t[:] _indptr_ = simil.indptr.astype(CUINT64)
  cdef CINT32_t[:] _indices_ = simil.indices

  scr_data = np.zeros_like(simil.data, dtype=CFLOAT)
  cdef CFLOAT_t[:] _scr_data_ = scr_data

  matchedA = np.full(opt.nmax, -1, dtype=CINT32)
  cdef CINT32_t[:] _matchedA_ = matchedA
  matchedB = np.full(opt.nmax, -1, dtype=CINT32)
  cdef CINT32_t[:] _matchedB_ = matchedB

  # wmat = np.zeros((MAX_SQUARE_SIZE, MAX_SQUARE_SIZE), dtype=CFLOAT)
  # cdef CFLOAT_t[:,:] _wmat_ = wmat

  cdef CFLOAT_t min_dig = opt.min_dig
  cdef CFLOAT_t max_dE = opt.max_dE
  cdef CFLOAT_t thr_A0 = opt.thr_A0
  cdef CFLOAT_t thr_A1 = opt.thr_A1

  cdef CUINT64_t ptrB
  cdef CINT32_t ret, jA, jB, idat, cA0, cB0, iA, iB, nA, nB, iB0
  cdef CFLOAT_t curr, right, bottom, diag, scr_tot, wM, wT

  idat = -1
  for jA in range(lenA):
    cA0 = _cumA_[jA]
    for ptrB in range(_indptr_[jA],_indptr_[jA+1]):
      idat += 1
      jB = _indices_[ptrB]
      cB0 = _cumB_[jB]
      # ----------------
      nA = _nsA_[jA]
      nB = _nsB_[jB]
      for iA in range(nA): _matchedA_[iA] = -1
      for iB in range(nB): _matchedB_[iB] = -1

      iA = iB0 = 0
      scr_tot = 0.
      doA = doB = True
      # --- Find aligned lines ---
      while True:
        if (iA >= nA) or (iB0 >= nB): break
        for iB in range(iB0,nB):
          curr = (_EsB_[cB0+iB]-_EsA_[cA0+iA])/max_dE
          iB0 = iB
          if curr>-1.: break
          doB = True

        if curr>=1.:
          iA += 1
          doA = True
          continue

        curr = 1.-abs(curr)

        if iB<nB-1: right = max(0.,1.-abs(_EsB_[cB0+iB+1]-_EsA_[cA0+iA])/max_dE)
        else:       right = 0.
        if iA<nA-1: bottom = max(0.,1.-abs(_EsB_[cB0+iB]-_EsA_[cA0+iA+1])/max_dE)
        else:       bottom = 0.

        if doA and (curr>right):
          _matchedA_[iA] = iB
          scr_tot += curr
          doA = False

        if doB and (curr>bottom):
          _matchedB_[iB] = iA
          scr_tot += curr
          doB = False

        if (bottom>0.) and (bottom>right):
          iA += 1
          doA = True
          continue

        if (right>0.) and (right>bottom):
          iB0 += 1
          doB = True
          continue

        if (bottom>0.) and (bottom==right):
          if (iA<nA-1)&(iB<nB-1): diag = max(0.,1.-abs(_EsB_[cB0+iB+1]-_EsA_[cA0+iA+1])/max_dE)
          else:                   diag = 0.
          if (bottom>diag):
            # print(f'{jA},{jB} --> B: {bottom:.3f}, R: {right:.3f}')
            return -1

        iA += 1
        iB0 += 1
        doA = doB = True
        continue

      # --- Calculate scores ---
      wT = 0.
      wM = 0.
      for iA in range(nA):
        if _matchedA_[iA]>=0:
          wT += 1.
          wM += 1.
        else:
          if pmode%2==0: wT += max(0., (min(thr_A1, _AsA_[cA0+iA])-thr_A0)/(thr_A1-thr_A0))

      for iB in range(nB):
        if _matchedB_[iB]>=0:
          wT += 1.
          wM += 1.
        else:
          if pmode>0: wT += max(0., (min(thr_A1, _AsB_[cB0+iB])-thr_A0)/(thr_A1-thr_A0))

      if wM/wT < min_dig: continue
      _scr_data_[idat] = scr_tot/wT

  mat_scr = csr_matrix((scr_data, simil.indices.copy(), simil.indptr.copy()), (lenA,lenB))
  mat_scr.eliminate_zeros()

  return mat_scr

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def score_lines_prl(CFLOAT_t[:] _EsA_, CFLOAT_t[:] _AsA_, CINT32_t[:] _nsA_, CINT32_t pA0, CINT32_t lenA, CINT32_t totA,
                    CFLOAT_t[:] _EsB_, CFLOAT_t[:] _AsB_, CINT32_t[:] _nsB_, CINT32_t lenB,
                    CUINT8_t[:] _sim_data_, CUINT64_t[:] _sim_indptr_, CINT32_t[:] _sim_indices_, CINT32_t nsim,
                    CINT32_t nmax, CFLOAT_t min_simil, CFLOAT_t min_dig, CFLOAT_t max_dE,
                    CFLOAT_t thr_A, CFLOAT_t wgt_small, CUINT8_t pmode):

  indptr = np.zeros(totA+1, dtype=CUINT64)
  cdef CUINT64_t[:] _indptr_ = indptr
  indices = np.full(nsim, -1, dtype=CINT32)
  cdef CINT32_t[:] _indices_ = indices
  data = np.full(nsim, np.nan, dtype=CFLOAT)
  cdef CFLOAT_t[:] _data_ = data
  dscr_data = np.zeros(nsim, dtype=CFLOAT)
  cdef CFLOAT_t[:] _dscr_data_ = dscr_data
  ascr_data = np.zeros(nsim, dtype=CFLOAT)
  cdef CFLOAT_t[:] _ascr_data_ = ascr_data

  matchedA = np.full(nmax, -1, dtype=CINT32)
  cdef CINT32_t[:] _matchedA_ = matchedA
  matchedB = np.full(nmax, -1, dtype=CINT32)
  cdef CINT32_t[:] _matchedB_ = matchedB

  wmat = np.zeros((MAX_SQUARE_SIZE, MAX_SQUARE_SIZE), dtype=CFLOAT)
  cdef CFLOAT_t[:,:] _wmat_ = wmat

  cdef CUINT8_t min_similINT = CUINT8(255*min_simil)
  cdef CUINT64_t ptrB
  cdef CINT32_t ii, ret, jA, jB, idat, cA0, cB0
  cdef CINT32_t iA, iB, nA, nB, iB0, iA1, iB1, n_row, n_col, nsq
  cdef CFLOAT_t n1SS, n1LS, n1LL, nA0S, nA0L, nB0S, nB0L, nM1, nMw, nTw
  cdef CFLOAT_t curr, right, bottom, diag, scr, cs, scr_tmp, scr_tot, ascr, dscr
  cdef CINT32_t *_pathi_
  cdef CINT32_t *_pathj_

  cumA = np.zeros(lenA, dtype=CINT32)
  cdef CINT32_t[:] _cumA_ = cumA
  for ii in range(1,lenA): cumA[ii] = cumA[ii-1]+_nsA_[ii-1]

  cumB = np.zeros(lenB, dtype=CINT32)
  cdef CINT32_t[:] _cumB_ = cumB
  for ii in range(1,lenB): cumB[ii] = cumB[ii-1]+_nsB_[ii-1]

  idat = -1 # index of current data position in scoring sparse matrix
  for jA in range(lenA): # cycle through rows of simil matrix (sample A)
    cA0 = _cumA_[jA] # initial position for Es and As vector for current A
    for ptrB in range(_sim_indptr_[jA],_sim_indptr_[jA+1]): # cycle through columns of simil matrix (sample B)
      idat += 1
      jB = _sim_indices_[ptrB] # this is the current B (ptrB is used to index the sparse matrix indices array)
      if _sim_data_[idat] < min_similINT: continue
      cB0 = _cumB_[jB] # initial position for Es and As vector for current B
      # ----------------
      nA = _nsA_[jA] # number of lines in current A
      nB = _nsB_[jB] # number of lines in current B
      iA = -1 # index of current line in A
      iB0 = -1 # index of current *initial* line in B
      scr_tot = 0.
      # --- Find aligned lines ---
      while True: # cycling through A lines
        iA += 1
        # If there are no more rows left, finish
        if iA >= nA: break
        _matchedA_[iA] = -1

        iB = iB0 # index of current line in column
        while True: # cycling through B lines
          iB += 1
          if iB >= nB: break
          # Calculate current and neighbors positions
          curr = _EsB_[cB0+iB]-_EsA_[cA0+iA]
          if curr > max_dE: break # Past possible matches
          if curr < -max_dE: continue # Before possible matches
          curr = max_dE-abs(curr)

          if iB<nB-1: right = max(0., max_dE-abs(_EsB_[cB0+iB+1]-_EsA_[cA0+iA]))
          else:       right = 0.
          if iA<nA-1: bottom = max(0., max_dE-abs(_EsB_[cB0+iB]-_EsA_[cA0+iA+1]))
          else:       bottom = 0.

          # If current position has lower direct neighbors, it's the best match
          if (curr>right) and (curr>bottom):
            _matchedA_[iA] = iB
            scr_tot += curr
            iB0 = iB
            break
          # If current position has a higher direct neighbor...
          else:
            if (iA<nA-1) and (iB<nB-1): diag = max(0., max_dE-abs(_EsB_[cB0+iB+1]-_EsA_[cA0+iA+1]))
            else:                       diag = 0.
            # If diagonal neighbor is 0, move to the highest direct neighbor
            # (in this case direct neighbors cannot be equal by construction)
            if diag == 0.:
              # If the highest neighbor is vertical, go to next row (same column)
              if bottom>right:
                iB0 = iB-1
                break
              # If the highest neighbor is horizontal, go to next column (same row)
              else:
                continue
            # If diagonal neighbor is >0, find the minimum square
            else:
              iA1 = iA+1
              iB1 = iB+1
              while True:
                # if (jA+nsq_row >= nA) or (jB+nsq_col >= nB): break
                if (iA1 < nA-1) and (abs(_EsB_[cB0+iB1]-_EsA_[cA0+iA1+1]) < max_dE):
                  iA1 += 1
                elif (iB1 < nB-1) and (abs(_EsB_[cB0+iB1+1]-_EsA_[cA0+iA1]) < max_dE):
                  iB1 += 1
                else:
                  break

              n_row = iA1-iA+1
              n_col = iB1-iB+1
              nsq = max(n_row, n_col)

              _calc_submatrix(_EsA_, _EsB_, cA0+iA, cB0+iB, n_row, n_col, _wmat_, max_dE)
              if nsq > MAX_SQUARE_SIZE:
                return -1*nsq

              ret = _match_square(_wmat_, n_row, n_col, _matchedA_, iA, iB, &scr_tmp)
              if ret < 0: return ret

              scr_tot += scr_tmp
              iB0 = iB1
              iA = iA1
              break

      # --- Calculate scores ---
      n1SS = n1LS = n1LL = nA0S = nA0L = nB0S = nB0L = 0.
      for iB in range(nB): _matchedB_[iB] = -1
      for iA in range(nA):
        if _matchedA_[iA]>=0:
          _matchedB_[_matchedA_[iA]] = iA
          if   (_AsA_[cA0+iA]<thr_A)&(_AsB_[cB0+_matchedA_[iA]]<thr_A): n1SS += 2.
          elif (_AsA_[cA0+iA]<thr_A)|(_AsB_[cB0+_matchedA_[iA]]<thr_A): n1LS += 2.
          else:                                                         n1LL += 2.
        else:
          if _AsA_[cA0+iA] < thr_A: nA0S += 1.
          else:                     nA0L += 1.
      for iB in range(nB):
        if _matchedB_[iB] < 0:
          if _AsB_[cB0+iB] < thr_A: nB0S += 1.
          else:                     nB0L += 1.

      nM1 = (n1SS+n1LS+n1LL)/2.
      nMw = n1SS*wgt_small + n1LL + 0.5*n1LS*(1.+wgt_small)

      if pmode == 2:
        nTw = nMw+(nA0S+nB0S)*wgt_small+nA0L+nB0L
      elif pmode == 0:
        nMw /= 2.
        nTw = nMw+nA0S*wgt_small+nA0L
      elif pmode == 1:
        nMw /= 2.
        nTw = nMw+nB0S*wgt_small+nB0L

      if nM1 == 0:
        ascr = 0
        dscr = 0.
      else:
        dscr = nMw/nTw
        ascr = max(EPS, scr_tot/(max_dE*nM1))

      if dscr < min_dig: continue
      _dscr_data_[idat] = dscr
      _ascr_data_[idat] = ascr
      _indices_[idat] = jB

    _indptr_[pA0+jA+1] = idat+1

  _indptr_[pA0+jA+1:] = idat+1

  mat_ascr = csr_matrix((ascr_data, indices.copy(), indptr.copy()), (totA,lenB))
  mat_ascr.eliminate_zeros()
  mat_dscr = csr_matrix((dscr_data, indices.copy(), indptr.copy()), (totA,lenB))
  mat_dscr.eliminate_zeros()

  return mat_ascr, mat_dscr

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32_t _align_lines(CFLOAT_t[:] _EsA_, CFLOAT_t[:] _AsA_, CINT32_t cA0, CINT32_t nA,
                           CFLOAT_t[:] _EsB_, CFLOAT_t[:] _AsB_, CINT32_t cB0, CINT32_t nB,
                           CINT32_t[:] _matchedA_, CINT32_t[:] _matchedB_, CFLOAT_t[:,:] _wmat_,
                           CFLOAT_t *dscr, CFLOAT_t *ascr, CFLOAT_t max_dE,
                           CFLOAT_t thr_A, CFLOAT_t wgt_small, CUINT8_t pmode):

  cdef CINT32_t ii, iA, iB, iB0, iA1, iB1, n_row, n_col, nsq
  cdef CFLOAT_t n1SS=0., n1LS=0., n1LL=0., nA0S=0., nA0L=0., nB0S=0., nB0L=0., nM1=0., nMw=0., nTw=0.
  cdef CFLOAT_t curr, right, bottom, diag, scr, cs, scr_tmp, scr_tot=0.
  cdef CINT32_t *_pathi_
  cdef CINT32_t *_pathj_

  iA = -1
  iB0 = -1
  # --- Find aligned lines ---
  while True: # cycling through iA
    iA += 1
    # If there are no more rows left, finish
    if iA >= nA: break
    _matchedA_[iA] = -1

    iB = iB0
    while True: # cycling through iB
      iB += 1
      if iB >= nB: break
      # Calculate current and neighbors positions
      curr = _EsB_[cB0+iB]-_EsA_[cA0+iA]
      if curr > max_dE: break # Past possible matches
      if curr < -max_dE: continue # Before possible matches
      curr = max_dE-abs(curr)

      if iB<nB-1: right = max(0., max_dE-abs(_EsB_[cB0+iB+1]-_EsA_[cA0+iA]))
      else:       right = 0.
      if iA<nA-1: bottom = max(0., max_dE-abs(_EsB_[cB0+iB]-_EsA_[cA0+iA+1]))
      else:       bottom = 0.

      # If current position has lower direct neighbors, it's the best match
      if (curr>right) and (curr>bottom):
        _matchedA_[iA] = iB
        scr_tot += curr
        iB0 = iB
        break
      # If current position has a higher direct neighbor...
      else:
        if (iA<nA-1) and (iB<nB-1): diag = max(0., max_dE-abs(_EsB_[cB0+iB+1]-_EsA_[cA0+iA+1]))
        else:                       diag = 0.
        # If diagonal neighbor is 0, move to the highest direct neighbor
        # (in this case direct neighbors cannot be equal by construction)
        if diag == 0.:
          # If the highest neighbor is vertical, go to next row (same column)
          if bottom>right:
            iB0 = iB-1
            break
          # If the highest neighbor is horizontal, go to next column (same row)
          else:
            continue
        # If diagonal neighbor is >0, find the minimum square
        else:
          iA1 = iA+1
          iB1 = iB+1
          while True:
            # if (jA+nsq_row >= nA) or (jB+nsq_col >= nB): break
            if (iA1 < nA-1) and (abs(_EsB_[cB0+iB1]-_EsA_[cA0+iA1+1]) < max_dE):
              iA1 += 1
            elif (iB1 < nB-1) and (abs(_EsB_[cB0+iB1+1]-_EsA_[cA0+iA1]) < max_dE):
              iB1 += 1
            else:
              break

          n_row = iA1-iA+1
          n_col = iB1-iB+1
          nsq = max(n_row, n_col)

          _calc_submatrix(_EsA_, _EsB_, cA0+iA, cB0+iB, n_row, n_col, _wmat_, max_dE)
          if nsq > MAX_SQUARE_SIZE:
            return -1*nsq

          ret = _match_square(_wmat_, n_row, n_col, _matchedA_, iA, iB, &scr_tmp)
          if ret < 0: return ret

          scr_tot += scr_tmp
          iB0 = iB1
          iA = iA1
          break

  # --- Calculate scores ---
  for iB in range(nB): _matchedB_[iB] = -1
  for iA in range(nA):
    if _matchedA_[iA]>=0:
      _matchedB_[_matchedA_[iA]] = iA
      if   (_AsA_[cA0+iA]<thr_A)&(_AsB_[cB0+_matchedA_[iA]]<thr_A): n1SS += 2.
      elif (_AsA_[cA0+iA]<thr_A)|(_AsB_[cB0+_matchedA_[iA]]<thr_A): n1LS += 2.
      else:                                                         n1LL += 2.
    else:
      if _AsA_[cA0+iA] < thr_A: nA0S += 1.
      else:                     nA0L += 1.
  for iB in range(nB):
    if _matchedB_[iB] < 0:
      if _AsB_[cB0+iB] < thr_A: nB0S += 1.
      else:                     nB0L += 1.

  nM1 = (n1SS+n1LS+n1LL)/2.
  nMw = n1SS*wgt_small + n1LL + 0.5*n1LS*(1.+wgt_small)

  if pmode == 2:
    nTw = nMw+(nA0S+nB0S)*wgt_small+nA0L+nB0L
  elif pmode == 0:
    nMw /= 2.
    nTw = nMw+nA0S*wgt_small+nA0L
  elif pmode == 1:
    nMw /= 2.
    nTw = nMw+nB0S*wgt_small+nB0L

  if nM1 == 0:
    ascr[0] = 0
    dscr[0] = 0.
  else:
    dscr[0] = nMw/nTw
    ascr[0] = max(EPS, scr_tot/(max_dE*nM1))
  return 0

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32_t _align_lines2(CFLOAT_t[:] _EsA_, CFLOAT_t[:] _AsA_, CINT32_t cA0, CINT32_t nA,
                            CFLOAT_t[:] _EsB_, CFLOAT_t[:] _AsB_, CINT32_t cB0, CINT32_t nB,
                            CINT32_t[:] _matchedA_, CINT32_t[:] _matchedB_, CFLOAT_t[:,:] _wmat_,
                            CFLOAT_t *scr, CFLOAT_t *dig, CFLOAT_t max_dE, CFLOAT_t KL,
                            CFLOAT_t thr_A0, CFLOAT_t thr_A1, CUINT8_t pmode):

  cdef CINT32_t ii, iA, iB, iB0, iA1, iB1, n_row, n_col, nsq
  cdef CFLOAT_t curr, de, right, bottom, diag, scr_tmp, scr_tot=0., wT, wM

  iA = -1
  iB0 = -1
  # --- Find aligned lines ---
  while True: # cycling through iA
    iA += 1
    # If there are no more rows left, finish
    if iA >= nA: break
    _matchedA_[iA] = -1

    iB = iB0
    while True: # cycling through iB
      iB += 1
      if iB >= nB: break
      # Calculate current and neighbors positions
      curr = (_EsB_[cB0+iB]-_EsA_[cA0+iA])/max_dE
      if curr >= 1.: break # Past possible matches
      if curr <= -1.: continue # Before possible matches
      curr = 1.-abs(curr)

      if iB<nB-1: right = 1.-abs(_EsB_[cB0+iB+1]-_EsA_[cA0+iA])/max_dE
      else:       right = 0.
      if iA<nA-1: bottom = 1.-abs(_EsB_[cB0+iB]-_EsA_[cA0+iA+1])/max_dE
      else:       bottom = 0.

      # If current position has lower direct neighbors, it's the best match
      if (curr>right) and (curr>bottom):
        _matchedA_[iA] = iB
        scr_tot += curr
        iB0 = iB
        break
      # If current position has a higher direct neighbor...
      else:
        if (iA<nA-1) and (iB<nB-1): diag = max(0., 1.-abs(_EsB_[cB0+iB+1]-_EsA_[cA0+iA+1]))
        else:                       diag = 0.
        # If diagonal neighbor is 0, move to the highest direct neighbor
        # (in this case direct neighbors cannot be equal by construction)
        if diag == 0.:
          # If the highest neighbor is vertical, go to next row (same column)
          if bottom>right:
            iB0 = iB-1
            break
          # If the highest neighbor is horizontal, go to next column (same row)
          else:
            continue
        # If diagonal neighbor is >0, find the minimum square
        else:
          iA1 = iA+1
          iB1 = iB+1
          while True:
            # if (jA+nsq_row >= nA) or (jB+nsq_col >= nB): break
            if (iA1 < nA-1) and (abs(_EsB_[cB0+iB1]-_EsA_[cA0+iA1+1]) < max_dE):
              iA1 += 1
            elif (iB1 < nB-1) and (abs(_EsB_[cB0+iB1+1]-_EsA_[cA0+iA1]) < max_dE):
              iB1 += 1
            else:
              break

          n_row = iA1-iA+1
          n_col = iB1-iB+1
          nsq = max(n_row, n_col)

          _calc_submatrix2(_EsA_, _EsB_, cA0+iA, cB0+iB, n_row, n_col, _wmat_, max_dE, KL)
          if nsq > MAX_SQUARE_SIZE:
            return -1*nsq

          ret = _match_square(_wmat_, n_row, n_col, _matchedA_, iA, iB, &scr_tmp)
          if ret < 0: return ret

          scr_tot += scr_tmp
          iB0 = iB1
          iA = iA1
          break

  # --- Calculate scores ---
  wT = 0.
  wM = 0.
  for iB in range(nB): _matchedB_[iB] = -1
  for iA in range(nA):
    if _matchedA_[iA]>=0:
      wT += 2.
      wM += 2.
      _matchedB_[_matchedA_[iA]] = iA
    else:
      if pmode%2==0: wT += max(0., (min(thr_A1, _AsA_[cA0+iA])-thr_A0)/(thr_A1-thr_A0))
  if pmode>0:
    for iB in range(nB):
      if _matchedB_[iB] < 0:
        wT += max(0., (min(thr_A1, _AsB_[cB0+iB])-thr_A0)/(thr_A1-thr_A0))

  scr[0] = 2.*scr_tot/wT
  dig[0] = wM/wT

  return 0

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32_t _align_lines3(CFLOAT_t[:] _EsA_, CFLOAT_t[:] _AsA_, CINT32_t cA0, CINT32_t nA,
                            CFLOAT_t[:] _EsB_, CFLOAT_t[:] _AsB_, CINT32_t cB0, CINT32_t nB,
                            CINT32_t[:] _matchedA_, CINT32_t[:] _matchedB_,
                            CFLOAT_t *scr, CFLOAT_t *dig, CFLOAT_t max_dE,
                            CFLOAT_t thr_A0, CFLOAT_t thr_A1, CUINT8_t pmode):

  cdef CINT32_t ii, iA, iB, iB0
  cdef CFLOAT_t curr, right, bottom, diag, scr_tot, wT, wM

  for iA in range(nA): _matchedA_[iA] = -1
  for iB in range(nB): _matchedB_[iB] = -1

  doA = doB = True
  iA = iB0 = 0
  scr_tot = 0.
  while True:
    if (iA >= nA) or (iB0 >= nB): break
    for iB in range(iB0,nB):
      curr = (_EsB_[cB0+iB]-_EsA_[cA0+iA])/max_dE
      iB0 = iB
      if curr>-1.: break
      doB = True

    if curr>=1.:
      iA += 1
      doA = True
      continue

    curr = 1.-abs(curr)

    if iB<nB-1: right = max(0.,1.-abs(_EsB_[cB0+iB+1]-_EsA_[cA0+iA])/max_dE)
    else:       right = 0.
    if iA<nA-1: bottom = max(0.,1.-abs(_EsB_[cB0+iB]-_EsA_[cA0+iA+1])/max_dE)
    else:       bottom = 0.

    if doA and (curr>right):
      _matchedA_[iA] = iB
      scr_tot += curr
      doA = False

    if doB and (curr>bottom):
      _matchedB_[iB] = iA
      scr_tot += curr
      doB = False

    if (bottom>0.) and (bottom>right):
      iA += 1
      doA = True
      continue

    if (right>0.) and (right>bottom):
      iB0 += 1
      doB = True
      continue

    if (bottom>0.) and (bottom==right):
      if (iA<nA-1)&(iB<nB-1): diag = max(0.,1.-abs(_EsB_[cB0+iB+1]-_EsA_[cA0+iA+1])/max_dE)
      else:                   diag = 0.
      if (bottom>diag):
        # print(f'{jA},{jB} --> B: {bottom:.3f}, R: {right:.3f}')
        return -1

    iA += 1
    iB0 += 1
    doA = doB = True
    continue

  # --- Calculate scores ---
  wT = 0.
  wM = 0.
  for iA in range(nA):
    if _matchedA_[iA]>=0:
      wT += 1.
      wM += 1.
    else:
      if pmode%2==0: wT += max(0., (min(thr_A1, _AsA_[cA0+iA])-thr_A0)/(thr_A1-thr_A0))

  for iB in range(nB):
    if _matchedB_[iB]>=0:
      wT += 1.
      wM += 1.
    else:
      if pmode>0: wT += max(0., (min(thr_A1, _AsB_[cB0+iB])-thr_A0)/(thr_A1-thr_A0))

  scr[0] = scr_tot/wT
  dig[0] = wM/wT

  return 0

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32_t _match_square(CFLOAT_t[:,:] _mat_, CINT32_t nrow, CINT32_t ncol,
                            CINT32_t[:] _matchedA_, CINT32_t iA0, CINT32_t iB0,
                            CFLOAT_t* scr_tmp):

  cdef CINT32_t ret, ii, n_match, nlvl=max(nrow,ncol)
  cdef CINT32_t *pass_rows = <CINT32_t *> malloc(nlvl*sizeof(CINT32_t))
  cdef CINT32_t *pass_cols = <CINT32_t *> malloc(nlvl*sizeof(CINT32_t))

  ret = _rec_walk(_mat_, nrow, ncol, nlvl, 0, 0, pass_rows, pass_cols, scr_tmp, &n_match)
  if ret < 0:
    free(pass_rows)
    free(pass_cols)
    return ret

  for ii in range(n_match):
    _matchedA_[iA0+pass_rows[n_match-ii-1]] = iB0+pass_cols[n_match-ii-1]

  free(pass_rows)
  free(pass_cols)
  return 0

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32_t _rec_walk(CFLOAT_t[:,:] _mat_, CINT32_t nrow, CINT32_t ncol, CINT32_t nlvl,
                          CINT32_t row0, CINT32_t col0, CINT32_t* buff_rows, CINT32_t* buff_cols,
                          CFLOAT_t *buff_scr, CINT32_t *buff_n):

  cdef CINT32_t ret, ii, irow, icol, pass_n
  cdef CFLOAT_t max_scr=0., pass_scr
  cdef CINT32_t *pass_rows = <CINT32_t *> malloc(nlvl*sizeof(CINT32_t))
  cdef CINT32_t *pass_cols = <CINT32_t *> malloc(nlvl*sizeof(CINT32_t))

  for irow in range(row0,nrow):
    for icol in range(col0,ncol):
      if (irow>row0) and (icol>col0): break
      if _mat_[irow,icol] == 0.: continue

      # ----- End of the matrix -----
      if (irow == nrow-1) or (icol == ncol-1):
        if _mat_[irow,icol] <= max_scr: continue
        buff_rows[0] = irow
        buff_cols[0] = icol
        max_scr = _mat_[irow,icol]
        buff_n[0] = 1
        buff_scr[0] = max_scr
      # ----- Middle of the matrix -----
      else:
        ret = _rec_walk(_mat_, nrow, ncol, nlvl, irow+1, icol+1, pass_rows, pass_cols, &pass_scr, &pass_n)
        if ret < 0:
          free(pass_rows)
          free(pass_cols)
          return ret

        if _mat_[irow,icol]+pass_scr <= max_scr: continue

        max_scr = _mat_[irow,icol]+pass_scr
        for ii in range(pass_n):
          buff_rows[ii] = pass_rows[ii]
          buff_cols[ii] = pass_cols[ii]
        buff_rows[ii+1] = irow
        buff_cols[ii+1] = icol
        buff_n[0] = pass_n+1
        buff_scr[0] = max_scr

  free(pass_rows)
  free(pass_cols)

  if max_scr == 0.: return -999
  return 0

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32_t _calc_submatrix(CFLOAT_t[:] _EsA_, CFLOAT_t[:] _EsB_, CINT32_t iA0, CINT32_t iB0,
                              CINT32_t nrow, CINT32_t ncol, CFLOAT_t[:,:] _mat_, CFLOAT_t max_dE):

  cdef CINT32_t irow, icol

  for irow in range(nrow):
    for icol in range(ncol):
      _mat_[irow,icol] = max(0., max_dE-abs(_EsB_[iB0+icol]-_EsA_[iA0+irow]))

  return 0

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32_t _calc_submatrix2(CFLOAT_t[:] _EsA_, CFLOAT_t[:] _EsB_, CINT32_t iA0, CINT32_t iB0,
                              CINT32_t nrow, CINT32_t ncol, CFLOAT_t[:,:] _mat_, CFLOAT_t max_dE, CFLOAT_t KL):

  cdef CINT32_t irow, icol
  cdef CFLOAT_t de

  for irow in range(nrow):
    for icol in range(ncol):
      de = abs(_EsB_[iB0+icol]-_EsA_[iA0+irow])
      if de >= max_dE: _mat_[irow,icol] = 0.
      else:            _mat_[irow,icol] = 1.-de/max_dE

  return 0

# -------------------------------- COMBINATIONS --------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def score_combinations(CFLOAT_t[:] _Es0_, CFLOAT_t[:] _As0_, CINT32_t[:] _ns0_,
                       CFLOAT_t[:] _Es1_, CFLOAT_t[:] _As1_, CINT32_t[:] _ns1_, ijks1,
                       scrtot, CINT32_t nmax, Emax, CFLOAT_t max_dE, CFLOAT_t thr_A, CFLOAT_t wgt_small,
                       CINT32_t comb_n, CINT32_t min_n, CFLOAT_t min_scr, CFLOAT_t min_imp, CFLOAT_t max_dr):

  cdef CINT32_t n0 = scrtot.shape[0]
  cdef CINT32_t n1 = scrtot.shape[1]

  if max_dr==0.:
    ijks1 = np.zeros((1,3),dtype=CFLOAT)
  else:
    assert ijks1.shape[0] == n1
    assert ijks1.shape[1] == 3

  cdef CFLOAT_t[:,:] _ijks1_ = ijks1

  cdef CINT32_t[:] _indices_ = scrtot.indices
  cdef CINT32_t[:] _indptr_ = scrtot.indptr

  csum0 = np.insert(np.cumsum(_ns0_),0,0).astype(CINT32)
  csum1 = np.insert(np.cumsum(_ns1_),0,0).astype(CINT32)
  cdef CINT32_t[:] _csum0_ = csum0
  cdef CINT32_t[:] _csum1_ = csum1

  idxs1 = np.zeros(nmax, dtype=CINT32)
  cdef CINT32_t[:] _idxs1_ = idxs1
  pns1 = np.zeros(nmax, dtype=CINT32)
  cdef CINT32_t[:] _pns1_ = pns1

  mtcA = np.zeros(Emax, dtype=CINT32)
  mtcB = np.zeros(Emax, dtype=CINT32)
  cdef CINT32_t[:] _mtcA_ = mtcA
  cdef CINT32_t[:] _mtcB_ = mtcB

  cdef CINT32_t irow, icol, ii, jj, kk, cnt2, cnt3, cnt4, do_sort
  cdef CFLOAT_t dr

  comb2_idx = np.zeros((nmax*(nmax-1)//2,2), dtype=CINT32)
  comb2_scr = np.zeros((nmax*(nmax-1)//2,3), dtype=CFLOAT)
  cdef CINT32_t[:,:] _comb2_idx_ = comb2_idx
  cdef CFLOAT_t[:,:] _comb2_scr_ = comb2_scr

  comb3_idx = np.zeros((nmax*nmax*(nmax-1)//2,3), dtype=CINT32)
  comb3_scr = np.zeros((nmax*nmax*(nmax-1)//2,3), dtype=CFLOAT)
  cdef CINT32_t[:,:] _comb3_idx_ = comb3_idx
  cdef CFLOAT_t[:,:] _comb3_scr_ = comb3_scr

  comb4_idx = np.zeros((nmax*nmax*(nmax-1)//2,4), dtype=CINT32)
  comb4_scr = np.zeros((nmax*nmax*(nmax-1)//2,3), dtype=CFLOAT)
  cdef CINT32_t[:,:] _comb4_idx_ = comb4_idx
  cdef CFLOAT_t[:,:] _comb4_scr_ = comb4_scr

  wmat = np.zeros((MAX_SQUARE_SIZE, MAX_SQUARE_SIZE), dtype=CFLOAT)
  cdef CFLOAT_t[:,:] _wmat_ = wmat

  ret = {}
  for irow in range(n0):
    if _ns0_[irow]<min_n: continue

    jj = 0
    do_sort = 0
    for ii in range(_indptr_[irow],_indptr_[irow+1]):
      icol = _indices_[ii]
      if _ns0_[irow] <= _ns1_[icol]: continue
      _idxs1_[jj] = icol
      _pns1_[jj] = _ns1_[icol]
      jj += 1
      if jj>=nmax:
        do_sort = 1
        break

    if do_sort:
      _row = scrtot.getrow(irow)
      isort = np.argsort(_row.data)[::-1]
      cols = _row.indices[isort]
      jj = 0
      for cc in cols:
        if _ns0_[irow] <= _ns1_[cc]: continue

        _idxs1_[jj] = cc
        _pns1_[jj] = _ns1_[cc]
        jj += 1
        if jj>=nmax: break

    if jj < 2: continue
    _score_combinations(_Es0_, _As0_, _csum0_[irow], _ns0_[irow],
                        _Es1_, _As1_, _ijks1_, _csum1_, _idxs1_, _pns1_, jj,
                        _mtcA_, _mtcB_, _comb2_idx_, _comb2_scr_, _comb3_idx_, _comb3_scr_, _comb4_idx_, _comb4_scr_,
                        &cnt2, &cnt3, &cnt4, _wmat_, max_dE, thr_A, wgt_small, comb_n, min_imp, min_scr, max_dr)

    tmp = []
    for ii in range(cnt2):
      tmp.append(((_comb2_idx_[ii,0],_comb2_idx_[ii,1]),_comb2_scr_[ii,2]))
    for ii in range(cnt3):
      tmp.append(((_comb3_idx_[ii,0],_comb3_idx_[ii,1],_comb3_idx_[ii,2]),_comb3_scr_[ii,2]))
    for ii in range(cnt4):
      tmp.append(((_comb4_idx_[ii,0],_comb4_idx_[ii,1],_comb4_idx_[ii,2],_comb4_idx_[ii,3]),_comb4_scr_[ii,2]))

    if len(tmp)>0: ret[irow] = tmp

  return ret

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32_t _score_combinations(CFLOAT_t[:] Es0, CFLOAT_t[:] As0, CINT32_t i0, CINT32_t n0,
                                  CFLOAT_t[:] Es1, CFLOAT_t[:] As1, CFLOAT_t[:,:] ijks1, CINT32_t[:] csum1, CINT32_t[:] i1s, CINT32_t[:] n1s, CINT32_t len1,
                                  CINT32_t[:] mtcA, CINT32_t[:] mtcB, CINT32_t[:,:] comb2_idx, CFLOAT_t[:,:] comb2_scr,
                                  CINT32_t[:,:] comb3_idx, CFLOAT_t[:,:] comb3_scr, CINT32_t[:,:] comb4_idx, CFLOAT_t[:,:] comb4_scr,
                                  CINT32_t *cnt2, CINT32_t *cnt3, CINT32_t *cnt4, CFLOAT_t[:,:] wmat, CFLOAT_t max_dE, CFLOAT_t thr_A, CFLOAT_t wgt_small,
                                  CINT32_t comb_n, CFLOAT_t min_imp, CFLOAT_t min_scr, CFLOAT_t max_dr):

  cdef CFLOAT_t dig, ana, tot, dEi, dEj, wM, wT, dr
  cdef CINT32_t ii, jj, kk, n1max=0

  for ii in range(len1):
    if n1s[ii]>n1max: n1max=n1s[ii]

  cdef CFLOAT_t *dEs1 = <CFLOAT_t *> malloc(n0*len1*sizeof(CFLOAT_t))
  cdef CFLOAT_t *dEs2 = <CFLOAT_t *> malloc(n0*(len1*(len1-1)//2)*sizeof(CFLOAT_t))
  cdef CFLOAT_t *dEs3 = <CFLOAT_t *> malloc(n0*(len1*len1*(len1-1)//2)*sizeof(CFLOAT_t))
  cdef CFLOAT_t *digs1 = <CFLOAT_t *> malloc(len1*sizeof(CFLOAT_t))
  cdef CFLOAT_t *digs2 = <CFLOAT_t *> malloc((len1*(len1-1)//2)*sizeof(CFLOAT_t))
  cdef CFLOAT_t *digs3 = <CFLOAT_t *> malloc((len1*len1*(len1-1)//2)*sizeof(CFLOAT_t))
  cdef CFLOAT_t *uw1s = <CFLOAT_t *> malloc(len1*sizeof(CFLOAT_t))
  cdef CFLOAT_t *uw2s = <CFLOAT_t *> malloc((len1*(len1-1)//2)*sizeof(CFLOAT_t))
  cdef CFLOAT_t *uw3s = <CFLOAT_t *> malloc((len1*len1*(len1-1)//2)*sizeof(CFLOAT_t))
  cdef CFLOAT_t *ijks2 = <CFLOAT_t *> malloc(3*(len1*(len1-1)//2)*sizeof(CFLOAT_t))
  cdef CFLOAT_t *ijks3 = <CFLOAT_t *> malloc(3*(len1*len1*(len1-1)//2)*sizeof(CFLOAT_t))
  cdef CFLOAT_t *w0s = <CFLOAT_t *> malloc(n0*sizeof(CFLOAT_t))
  cdef CFLOAT_t *comb1_scr = <CFLOAT_t *> malloc(len1*sizeof(CFLOAT_t))

  for ii in range(n0):
    if As0[ii] > thr_A: w0s[ii] = 1.
    else:               w0s[ii] = wgt_small

  # Get alignments
  for ii in range(len1):
    _align_lines(Es0, As0, i0, n0, Es1, As1, csum1[i1s[ii]], n1s[ii], mtcA, mtcB,
                 wmat, &dig, &ana, max_dE, thr_A, wgt_small, 0)
    digs1[ii] = dig
    comb1_scr[ii] = 0.5*(ana+dig)

    for jj in range(n0):
      if mtcA[jj]>=0: dEs1[ii*n0+jj] = abs(Es0[i0+jj]-Es1[csum1[i1s[ii]]+mtcA[jj]])
      else:           dEs1[ii*n0+jj] = max_dE

    uw1s[ii] = 0.

    for jj in range(n1s[ii]):
      if mtcB[jj] == -1:
        if As1[csum1[i1s[ii]]+jj] > thr_A: uw1s[ii] += 1.
        else:                              uw1s[ii] += wgt_small

  # Find 2-combinations
  cnt2[0] = 0
  if comb_n >= 2:
    for ii in range(len1):
      for jj in range(ii+1,len1):
        if max_dr>0.:
          dr = ((ijks1[i1s[ii],0]-ijks1[i1s[jj],0])**2+\
                (ijks1[i1s[ii],1]-ijks1[i1s[jj],1])**2+\
                (ijks1[i1s[ii],2]-ijks1[i1s[jj],2])**2)**0.5
          if dr>max_dr: continue

        wM = 0.
        wT = 0.
        ana = 0.
        for kk in range(n0):
          dEi = dEs1[ii*n0+kk]
          dEj = dEs1[jj*n0+kk]
          if dEi < dEj:
            dEs2[cnt2[0]*n0+kk] = dEi
            ana += dEi/max_dE
            wM += 2.
            wT += 2.
          elif (dEj < dEi) or (dEj < max_dE):
            dEs2[cnt2[0]*n0+kk] = dEj
            ana += dEj/max_dE
            wM += 2.
            wT += 2.
          else:
            dEs2[cnt2[0]*n0+kk] = max_dE
            wT += w0s[kk]

        ana = 1.-(2.*ana/wM)
        dig = wM/(wT+uw1s[ii]+uw1s[jj])
        tot = 0.5*(ana+dig)

        if (dig>digs1[ii]+.05) and (dig>digs1[jj]+.05) and \
           (tot>comb1_scr[ii]+min_imp) and (tot>comb1_scr[jj]+min_imp) and (tot>min_scr):
          comb2_idx[cnt2[0],0] = i1s[ii]
          comb2_idx[cnt2[0],1] = i1s[jj]
          comb2_scr[cnt2[0],0] = ana
          comb2_scr[cnt2[0],1] = dig
          comb2_scr[cnt2[0],2] = tot
          digs2[cnt2[0]] = dig
          uw2s[cnt2[0]] = uw1s[ii]+uw1s[jj]
          if max_dr>0:
            ijks2[3*cnt2[0]+0] = 0.5*(ijks1[i1s[ii],0]+ijks1[i1s[jj],0])
            ijks2[3*cnt2[0]+1] = 0.5*(ijks1[i1s[ii],1]+ijks1[i1s[jj],1])
            ijks2[3*cnt2[0]+2] = 0.5*(ijks1[i1s[ii],2]+ijks1[i1s[jj],2])
          cnt2[0] += 1

  # Find 3-combinations
  cnt3[0] = 0
  if comb_n >= 3:
    for ii in range(cnt2[0]):
      for jj in range(len1):
        if max_dr>0.:
          dr = ((ijks2[3*ii+0]-ijks1[i1s[jj],0])**2+\
                (ijks2[3*ii+1]-ijks1[i1s[jj],1])**2+\
                (ijks2[3*ii+2]-ijks1[i1s[jj],2])**2)**0.5
          if dr>max_dr: continue

        wM = 0.
        wT = 0.
        ana = 0.
        for kk in range(n0):
          dEi = dEs2[ii*n0+kk]
          dEj = dEs1[jj*n0+kk]
          if dEi < dEj:
            dEs3[cnt3[0]*n0+kk] = dEi
            ana += dEi/max_dE
            wM += 2.
            wT += 2.
          elif (dEj < dEi) or (dEj < max_dE):
            dEs3[cnt3[0]*n0+kk] = dEj
            ana += dEj/max_dE
            wM += 2.
            wT += 2.
          else:
            dEs3[cnt3[0]*n0+kk] = max_dE
            wT += w0s[kk]

        ana = 1.-(2.*ana/wM)
        dig = wM/(wT+uw2s[ii]+uw1s[jj])
        tot = 0.5*(ana+dig)

        if (dig>digs2[ii]+.05) and (dig>digs1[jj]+.05) and \
           (tot>comb2_scr[ii,2]+min_imp) and (tot>comb1_scr[jj]+min_imp) and (tot>min_scr):
          comb3_idx[cnt3[0],0] = comb2_idx[ii,0]
          comb3_idx[cnt3[0],1] = comb2_idx[ii,1]
          comb3_idx[cnt3[0],2] = i1s[jj]
          comb3_scr[cnt3[0],0] = ana
          comb3_scr[cnt3[0],1] = dig
          comb3_scr[cnt3[0],2] = tot
          digs3[cnt3[0]] = dig
          uw3s[cnt3[0]] = uw2s[ii]+uw1s[jj]
          if max_dr>0:
            ijks3[3*cnt3[0]+0] = 0.5*(ijks2[3*ii+0]+ijks1[i1s[jj],0])
            ijks3[3*cnt3[0]+1] = 0.5*(ijks2[3*ii+1]+ijks1[i1s[jj],1])
            ijks3[3*cnt3[0]+2] = 0.5*(ijks2[3*ii+2]+ijks1[i1s[jj],2])
          cnt3[0] += 1

  # Find 4-combinations
  cnt4[0] = 0
  if comb_n >= 4:
    for ii in range(cnt3[0]):
      for jj in range(len1):
        if max_dr>0.:
          dr = ((ijks3[3*ii+0]-ijks1[i1s[jj],0])**2+\
                (ijks3[3*ii+1]-ijks1[i1s[jj],1])**2+\
                (ijks3[3*ii+2]-ijks1[i1s[jj],2])**2)**0.5
          if dr>max_dr: continue

        wM = 0.
        wT = 0.
        ana = 0.
        for kk in range(n0):
          dEi = dEs3[ii*n0+kk]
          dEj = dEs1[jj*n0+kk]
          if dEi < dEj:
            ana += dEi/max_dE
            wM += 2.
            wT += 2.
          elif (dEj < dEi) or (dEj < max_dE):
            ana += dEj/max_dE
            wM += 2.
            wT += 2.
          else:
            wT += w0s[kk]

        ana = 1.-(2.*ana/wM)
        dig = wM/(wT+uw3s[ii]+uw1s[jj])
        tot = 0.5*(ana+dig)

        if (dig>digs3[ii]+.05) and (dig>digs1[jj]+.05) and \
           (tot>comb3_scr[ii,2]+min_imp) and (tot>comb1_scr[jj]+min_imp) and (tot>min_scr):
          comb4_idx[cnt4[0],0] = comb3_idx[ii,0]
          comb4_idx[cnt4[0],1] = comb3_idx[ii,1]
          comb4_idx[cnt4[0],2] = comb3_idx[ii,2]
          comb4_idx[cnt4[0],3] = i1s[jj]
          comb4_scr[cnt4[0],0] = ana
          comb4_scr[cnt4[0],1] = dig
          comb4_scr[cnt4[0],2] = tot
          cnt4[0] += 1

  # free(mtcs1)
  free(dEs1)
  free(dEs2)
  free(dEs3)
  free(digs1)
  free(digs2)
  free(digs3)
  free(uw1s)
  free(uw2s)
  free(uw3s)
  free(ijks2)
  free(ijks3)
  free(w0s)
  free(comb1_scr)

  return 0

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def score_combinations2(CFLOAT_t[:] _Es0_, CFLOAT_t[:] _As0_, CINT32_t[:] _ns0_,
                        CFLOAT_t[:] _Es1_, CFLOAT_t[:] _As1_, CINT32_t[:] _ns1_, ijks1,
                        scrtot, CINT32_t nmax, Emax, CFLOAT_t max_dE, CFLOAT_t Lfwhm, CFLOAT_t thr_A0, CFLOAT_t thr_A1,
                        CINT32_t comb_n, CINT32_t min_n, CFLOAT_t min_scr, CFLOAT_t min_dig, CFLOAT_t min_imp, CFLOAT_t max_dr):

  cdef CINT32_t n0 = scrtot.shape[0]
  cdef CINT32_t n1 = scrtot.shape[1]

  if max_dr==0.:
    ijks1 = np.zeros((1,3),dtype=CFLOAT)
  else:
    assert ijks1.shape[0] == n1
    assert ijks1.shape[1] == 3

  cdef CFLOAT_t[:,:] _ijks1_ = ijks1

  cdef CINT32_t[:] _indices_ = scrtot.indices
  cdef CINT32_t[:] _indptr_ = scrtot.indptr

  csum0 = np.insert(np.cumsum(_ns0_),0,0).astype(CINT32)
  csum1 = np.insert(np.cumsum(_ns1_),0,0).astype(CINT32)
  cdef CINT32_t[:] _csum0_ = csum0
  cdef CINT32_t[:] _csum1_ = csum1

  idxs1 = np.zeros(nmax, dtype=CINT32)
  cdef CINT32_t[:] _idxs1_ = idxs1
  pns1 = np.zeros(nmax, dtype=CINT32)
  cdef CINT32_t[:] _pns1_ = pns1

  mtcA = np.zeros(Emax, dtype=CINT32)
  mtcB = np.zeros(Emax, dtype=CINT32)
  cdef CINT32_t[:] _mtcA_ = mtcA
  cdef CINT32_t[:] _mtcB_ = mtcB

  cdef CINT32_t irow, icol, ii, jj, kk, cnt2, cnt3, cnt4, do_sort
  cdef CFLOAT_t dr

  comb2_idx = np.zeros((nmax*(nmax-1)//2,2), dtype=CINT32)
  comb2_scr = np.zeros((nmax*(nmax-1)//2,2), dtype=CFLOAT)
  cdef CINT32_t[:,:] _comb2_idx_ = comb2_idx
  cdef CFLOAT_t[:,:] _comb2_scr_ = comb2_scr

  comb3_idx = np.zeros((nmax*nmax*(nmax-1)//2,3), dtype=CINT32)
  comb3_scr = np.zeros((nmax*nmax*(nmax-1)//2,2), dtype=CFLOAT)
  cdef CINT32_t[:,:] _comb3_idx_ = comb3_idx
  cdef CFLOAT_t[:,:] _comb3_scr_ = comb3_scr

  comb4_idx = np.zeros((nmax*nmax*(nmax-1)//2,4), dtype=CINT32)
  comb4_scr = np.zeros((nmax*nmax*(nmax-1)//2,2), dtype=CFLOAT)
  cdef CINT32_t[:,:] _comb4_idx_ = comb4_idx
  cdef CFLOAT_t[:,:] _comb4_scr_ = comb4_scr

  wmat = np.zeros((MAX_SQUARE_SIZE, MAX_SQUARE_SIZE), dtype=CFLOAT)
  cdef CFLOAT_t[:,:] _wmat_ = wmat

  cdef CFLOAT_t KL = (0.5*Lfwhm)**2

  ret = {}
  for irow in range(n0):
    if _ns0_[irow]<min_n: continue

    jj = 0
    do_sort = 0
    for ii in range(_indptr_[irow],_indptr_[irow+1]):
      icol = _indices_[ii]
      if _ns0_[irow] <= _ns1_[icol]: continue
      _idxs1_[jj] = icol
      _pns1_[jj] = _ns1_[icol]
      jj += 1
      if jj>=nmax:
        do_sort = 1
        break

    if do_sort:
      _row = scrtot.getrow(irow)
      isort = np.argsort(_row.data)[::-1]
      cols = _row.indices[isort]
      jj = 0
      for cc in cols:
        if _ns0_[irow] <= _ns1_[cc]: continue

        _idxs1_[jj] = cc
        _pns1_[jj] = _ns1_[cc]
        jj += 1
        if jj>=nmax: break

    if jj < 2: continue
    _score_combinations2(_Es0_, _As0_, _csum0_[irow], _ns0_[irow],
                         _Es1_, _As1_, _ijks1_, _csum1_, _idxs1_, _pns1_, jj,
                         _mtcA_, _mtcB_, _comb2_idx_, _comb2_scr_, _comb3_idx_, _comb3_scr_, _comb4_idx_, _comb4_scr_,
                         &cnt2, &cnt3, &cnt4, _wmat_, max_dE, KL, thr_A0, thr_A1, comb_n, min_imp, min_scr, max_dr)

    tmp = []
    for ii in range(cnt2):
      if _comb2_scr_[ii,1] < min_dig: continue
      tmp.append(((_comb2_idx_[ii,0],_comb2_idx_[ii,1]),_comb2_scr_[ii,0]))
    for ii in range(cnt3):
      if _comb3_scr_[ii,1] < min_dig: continue
      tmp.append(((_comb3_idx_[ii,0],_comb3_idx_[ii,1],_comb3_idx_[ii,2]),_comb3_scr_[ii,0]))
    for ii in range(cnt4):
      if _comb4_scr_[ii,1] < min_dig: continue
      tmp.append(((_comb4_idx_[ii,0],_comb4_idx_[ii,1],_comb4_idx_[ii,2],_comb4_idx_[ii,3]),_comb4_scr_[ii,0]))

    if len(tmp)>0: ret[irow] = tmp

  return ret

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32_t _score_combinations2(CFLOAT_t[:] Es0, CFLOAT_t[:] As0, CINT32_t i0, CINT32_t n0,
                                  CFLOAT_t[:] Es1, CFLOAT_t[:] As1, CFLOAT_t[:,:] ijks1, CINT32_t[:] csum1, CINT32_t[:] i1s, CINT32_t[:] n1s, CINT32_t len1,
                                  CINT32_t[:] mtcA, CINT32_t[:] mtcB, CINT32_t[:,:] comb2_idx, CFLOAT_t[:,:] comb2_scr,
                                  CINT32_t[:,:] comb3_idx, CFLOAT_t[:,:] comb3_scr, CINT32_t[:,:] comb4_idx, CFLOAT_t[:,:] comb4_scr,
                                  CINT32_t *cnt2, CINT32_t *cnt3, CINT32_t *cnt4, CFLOAT_t[:,:] wmat, CFLOAT_t max_dE, CFLOAT_t KL, CFLOAT_t thr_A0, CFLOAT_t thr_A1,
                                  CINT32_t comb_n, CFLOAT_t min_imp, CFLOAT_t min_scr, CFLOAT_t max_dr):

  cdef CFLOAT_t dig, scr, dEi, dEj, wM, wT, dr
  cdef CINT32_t ii, jj, kk, n1max=0

  for ii in range(len1):
    if n1s[ii]>n1max: n1max=n1s[ii]

  cdef CFLOAT_t *dEs1 = <CFLOAT_t *> malloc(n0*len1*sizeof(CFLOAT_t))
  cdef CFLOAT_t *dEs2 = <CFLOAT_t *> malloc(n0*(len1*(len1-1)//2)*sizeof(CFLOAT_t))
  cdef CFLOAT_t *dEs3 = <CFLOAT_t *> malloc(n0*(len1*len1*(len1-1)//2)*sizeof(CFLOAT_t))
  cdef CFLOAT_t *digs1 = <CFLOAT_t *> malloc(len1*sizeof(CFLOAT_t))
  cdef CFLOAT_t *uw1s = <CFLOAT_t *> malloc(len1*sizeof(CFLOAT_t))
  cdef CFLOAT_t *uw2s = <CFLOAT_t *> malloc((len1*(len1-1)//2)*sizeof(CFLOAT_t))
  cdef CFLOAT_t *uw3s = <CFLOAT_t *> malloc((len1*len1*(len1-1)//2)*sizeof(CFLOAT_t))
  cdef CFLOAT_t *ijks2 = <CFLOAT_t *> malloc(3*(len1*(len1-1)//2)*sizeof(CFLOAT_t))
  cdef CFLOAT_t *ijks3 = <CFLOAT_t *> malloc(3*(len1*len1*(len1-1)//2)*sizeof(CFLOAT_t))
  cdef CFLOAT_t *w0s = <CFLOAT_t *> malloc(n0*sizeof(CFLOAT_t))
  cdef CFLOAT_t *comb1_scr = <CFLOAT_t *> malloc(len1*sizeof(CFLOAT_t))

  for ii in range(n0):
    w0s[ii] = max(0.,(min(thr_A1,As0[ii])-thr_A0)/(thr_A1-thr_A0))

  # Get alignments
  for ii in range(len1):
    _align_lines2(Es0, As0, i0, n0, Es1, As1, csum1[i1s[ii]], n1s[ii], mtcA, mtcB,
                  wmat, &scr, &dig, max_dE, KL, thr_A0, thr_A1, 0)
    digs1[ii] = dig
    comb1_scr[ii] = scr

    for jj in range(n0):
      if mtcA[jj]>=0: dEs1[ii*n0+jj] = abs(Es0[i0+jj]-Es1[csum1[i1s[ii]]+mtcA[jj]])
      else:           dEs1[ii*n0+jj] = max_dE

    uw1s[ii] = 0.
    for jj in range(n1s[ii]):
      if mtcB[jj] == -1:
        uw1s[ii] += max(0.,(min(thr_A1,As1[csum1[i1s[ii]]+jj])-thr_A0)/(thr_A1-thr_A0))

  # Find 2-combinations
  cnt2[0] = 0
  if comb_n >= 2:
    for ii in range(len1):
      for jj in range(ii+1,len1):
        if max_dr>0.:
          dr = ((ijks1[i1s[ii],0]-ijks1[i1s[jj],0])**2+\
                (ijks1[i1s[ii],1]-ijks1[i1s[jj],1])**2+\
                (ijks1[i1s[ii],2]-ijks1[i1s[jj],2])**2)**0.5
          if dr>max_dr: continue

        wM = 0.
        wT = uw1s[ii]+uw1s[jj]
        scr = 0.
        for kk in range(n0):
          dEi = dEs1[ii*n0+kk]
          dEj = dEs1[jj*n0+kk]
          if dEi < dEj:
            dEs2[cnt2[0]*n0+kk] = dEi
            scr += 1.-dEi/max_dE
            wM += 2.
            wT += 2.
          elif (dEj < dEi) or (dEj < max_dE):
            dEs2[cnt2[0]*n0+kk] = dEj
            scr += 1.-dEj/max_dE
            wM += 2.
            wT += 2.
          else:
            dEs2[cnt2[0]*n0+kk] = max_dE
            wT += w0s[kk]

        scr = 2.*scr/wT
        dig = wM/wT

        if (dig>digs1[ii]+.05) and (dig>digs1[jj]+.05) and \
           (scr>comb1_scr[ii]+min_imp) and (scr>comb1_scr[jj]+min_imp) and (scr>min_scr):
          comb2_idx[cnt2[0],0] = i1s[ii]
          comb2_idx[cnt2[0],1] = i1s[jj]
          comb2_scr[cnt2[0],0] = scr
          comb2_scr[cnt2[0],1] = dig
          uw2s[cnt2[0]] = uw1s[ii]+uw1s[jj]
          if max_dr>0:
            ijks2[3*cnt2[0]+0] = 0.5*(ijks1[i1s[ii],0]+ijks1[i1s[jj],0])
            ijks2[3*cnt2[0]+1] = 0.5*(ijks1[i1s[ii],1]+ijks1[i1s[jj],1])
            ijks2[3*cnt2[0]+2] = 0.5*(ijks1[i1s[ii],2]+ijks1[i1s[jj],2])
          cnt2[0] += 1

  # Find 3-combinations
  cnt3[0] = 0
  if comb_n >= 3:
    for ii in range(cnt2[0]):
      for jj in range(len1):
        if max_dr>0.:
          dr = ((ijks2[3*ii+0]-ijks1[i1s[jj],0])**2+\
                (ijks2[3*ii+1]-ijks1[i1s[jj],1])**2+\
                (ijks2[3*ii+2]-ijks1[i1s[jj],2])**2)**0.5
          if dr>max_dr: continue

        wM = 0.
        wT = uw2s[ii]+uw1s[jj]
        scr = 0.
        for kk in range(n0):
          dEi = dEs2[ii*n0+kk]
          dEj = dEs1[jj*n0+kk]
          if dEi < dEj:
            dEs3[cnt3[0]*n0+kk] = dEi
            scr += 1.-dEi/max_dE
            wM += 2.
            wT += 2.
          elif (dEj < dEi) or (dEj < max_dE):
            dEs3[cnt3[0]*n0+kk] = dEj
            scr += 1.-dEj/max_dE
            wM += 2.
            wT += 2.
          else:
            dEs3[cnt3[0]*n0+kk] = max_dE
            wT += w0s[kk]

        scr = 2.*scr/wT
        dig = wM/wT

        if (dig>comb2_scr[ii,1]+.05) and (dig>digs1[jj]+.05) and \
           (scr>comb2_scr[ii,0]+min_imp) and (scr>comb1_scr[jj]+min_imp) and (scr>min_scr):
          comb3_idx[cnt3[0],0] = comb2_idx[ii,0]
          comb3_idx[cnt3[0],1] = comb2_idx[ii,1]
          comb3_idx[cnt3[0],2] = i1s[jj]
          comb3_scr[cnt3[0],0] = scr
          comb3_scr[cnt3[0],1] = dig
          uw3s[cnt3[0]] = uw2s[ii]+uw1s[jj]
          if max_dr>0:
            ijks3[3*cnt3[0]+0] = 0.5*(ijks2[3*ii+0]+ijks1[i1s[jj],0])
            ijks3[3*cnt3[0]+1] = 0.5*(ijks2[3*ii+1]+ijks1[i1s[jj],1])
            ijks3[3*cnt3[0]+2] = 0.5*(ijks2[3*ii+2]+ijks1[i1s[jj],2])
          cnt3[0] += 1

  # Find 4-combinations
  cnt4[0] = 0
  if comb_n >= 4:
    for ii in range(cnt3[0]):
      for jj in range(len1):
        if max_dr>0.:
          dr = ((ijks3[3*ii+0]-ijks1[i1s[jj],0])**2+\
                (ijks3[3*ii+1]-ijks1[i1s[jj],1])**2+\
                (ijks3[3*ii+2]-ijks1[i1s[jj],2])**2)**0.5
          if dr>max_dr: continue

        wM = 0.
        wT = uw3s[ii]+uw1s[jj]
        scr = 0.
        for kk in range(n0):
          dEi = dEs3[ii*n0+kk]
          dEj = dEs1[jj*n0+kk]
          if dEi < dEj:
            scr += 1.-dEi/max_dE
            wM += 2.
            wT += 2.
          elif (dEj < dEi) or (dEj < max_dE):
            scr += 1.-dEj/max_dE
            wM += 2.
            wT += 2.
          else:
            wT += w0s[kk]

        scr = 2.*scr/wT
        dig = wM/wT

        if (dig>comb3_scr[ii,1]+.05) and (dig>digs1[jj]+.05) and \
           (scr>comb3_scr[ii,0]+min_imp) and (scr>comb1_scr[jj]+min_imp) and (scr>min_scr):
          comb4_idx[cnt4[0],0] = comb3_idx[ii,0]
          comb4_idx[cnt4[0],1] = comb3_idx[ii,1]
          comb4_idx[cnt4[0],2] = comb3_idx[ii,2]
          comb4_idx[cnt4[0],3] = i1s[jj]
          comb4_scr[cnt4[0],0] = scr
          comb4_scr[cnt4[0],1] = dig
          cnt4[0] += 1

  # free(mtcs1)
  free(dEs1)
  free(dEs2)
  free(dEs3)
  free(digs1)
  free(uw1s)
  free(uw2s)
  free(uw3s)
  free(ijks2)
  free(ijks3)
  free(w0s)
  free(comb1_scr)

  return 0

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def score_combinations3(CFLOAT_t[:] _Es0_, CFLOAT_t[:] _As0_, CINT32_t[:] _ns0_,
                        CFLOAT_t[:] _Es1_, CFLOAT_t[:] _As1_, CINT32_t[:] _ns1_, ijks1,
                        scrtot, CINT32_t nmax, Emax, CFLOAT_t max_dE, CFLOAT_t thr_A0, CFLOAT_t thr_A1,
                        CINT32_t comb_n, CINT32_t min_n, CFLOAT_t min_scr, CFLOAT_t min_dig, CFLOAT_t min_imp, CFLOAT_t max_dr):

  cdef CINT32_t n0 = scrtot.shape[0]
  cdef CINT32_t n1 = scrtot.shape[1]

  if max_dr==0.:
    ijks1 = np.zeros((1,3),dtype=CFLOAT)
  else:
    assert ijks1.shape[0] == n1
    assert ijks1.shape[1] == 3

  cdef CFLOAT_t[:,:] _ijks1_ = ijks1

  cdef CINT32_t[:] _indices_ = scrtot.indices
  cdef CINT32_t[:] _indptr_ = scrtot.indptr

  csum0 = np.insert(np.cumsum(_ns0_),0,0).astype(CINT32)
  csum1 = np.insert(np.cumsum(_ns1_),0,0).astype(CINT32)
  cdef CINT32_t[:] _csum0_ = csum0
  cdef CINT32_t[:] _csum1_ = csum1

  idxs1 = np.zeros(nmax, dtype=CINT32)
  cdef CINT32_t[:] _idxs1_ = idxs1
  pns1 = np.zeros(nmax, dtype=CINT32)
  cdef CINT32_t[:] _pns1_ = pns1

  mtcA = np.zeros(Emax, dtype=CINT32)
  mtcB = np.zeros(Emax, dtype=CINT32)
  cdef CINT32_t[:] _mtcA_ = mtcA
  cdef CINT32_t[:] _mtcB_ = mtcB

  cdef CINT32_t irow, icol, ii, jj, kk, cnt2, cnt3, cnt4, do_sort
  cdef CFLOAT_t dr

  comb2_idx = np.zeros((nmax*(nmax-1)//2,2), dtype=CINT32)
  comb2_scr = np.zeros((nmax*(nmax-1)//2,2), dtype=CFLOAT)
  cdef CINT32_t[:,:] _comb2_idx_ = comb2_idx
  cdef CFLOAT_t[:,:] _comb2_scr_ = comb2_scr

  comb3_idx = np.zeros((nmax*nmax*(nmax-1)//2,3), dtype=CINT32)
  comb3_scr = np.zeros((nmax*nmax*(nmax-1)//2,2), dtype=CFLOAT)
  cdef CINT32_t[:,:] _comb3_idx_ = comb3_idx
  cdef CFLOAT_t[:,:] _comb3_scr_ = comb3_scr

  comb4_idx = np.zeros((nmax*nmax*(nmax-1)//2,4), dtype=CINT32)
  comb4_scr = np.zeros((nmax*nmax*(nmax-1)//2,2), dtype=CFLOAT)
  cdef CINT32_t[:,:] _comb4_idx_ = comb4_idx
  cdef CFLOAT_t[:,:] _comb4_scr_ = comb4_scr

  ret = {}
  for irow in range(n0):
    if _ns0_[irow]<min_n: continue

    jj = 0
    do_sort = 0
    for ii in range(_indptr_[irow],_indptr_[irow+1]):
      icol = _indices_[ii]
      if _ns0_[irow] <= _ns1_[icol]: continue
      _idxs1_[jj] = icol
      _pns1_[jj] = _ns1_[icol]
      jj += 1
      if jj>=nmax:
        do_sort = 1
        break

    if do_sort:
      _row = scrtot.getrow(irow)
      isort = np.argsort(_row.data)[::-1]
      cols = _row.indices[isort]
      jj = 0
      for cc in cols:
        if _ns0_[irow] <= _ns1_[cc]: continue

        _idxs1_[jj] = cc
        _pns1_[jj] = _ns1_[cc]
        jj += 1
        if jj>=nmax: break

    if jj < 2: continue
    _score_combinations3(_Es0_, _As0_, _csum0_[irow], _ns0_[irow],
                         _Es1_, _As1_, _ijks1_, _csum1_, _idxs1_, _pns1_, jj,
                         _mtcA_, _mtcB_, _comb2_idx_, _comb2_scr_, _comb3_idx_, _comb3_scr_, _comb4_idx_, _comb4_scr_,
                         &cnt2, &cnt3, &cnt4, max_dE, thr_A0, thr_A1, comb_n, min_imp, min_scr, max_dr)

    tmp = []
    for ii in range(cnt2):
      if _comb2_scr_[ii,1] < min_dig: continue
      tmp.append(((_comb2_idx_[ii,0],_comb2_idx_[ii,1]),_comb2_scr_[ii,0]))
    for ii in range(cnt3):
      if _comb3_scr_[ii,1] < min_dig: continue
      tmp.append(((_comb3_idx_[ii,0],_comb3_idx_[ii,1],_comb3_idx_[ii,2]),_comb3_scr_[ii,0]))
    for ii in range(cnt4):
      if _comb4_scr_[ii,1] < min_dig: continue
      tmp.append(((_comb4_idx_[ii,0],_comb4_idx_[ii,1],_comb4_idx_[ii,2],_comb4_idx_[ii,3]),_comb4_scr_[ii,0]))

    if len(tmp)>0: ret[irow] = tmp

  return ret

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32_t _score_combinations3(CFLOAT_t[:] Es0, CFLOAT_t[:] As0, CINT32_t i0, CINT32_t n0,
                                  CFLOAT_t[:] Es1, CFLOAT_t[:] As1, CFLOAT_t[:,:] ijks1, CINT32_t[:] csum1, CINT32_t[:] i1s, CINT32_t[:] n1s, CINT32_t len1,
                                  CINT32_t[:] mtcA, CINT32_t[:] mtcB, CINT32_t[:,:] comb2_idx, CFLOAT_t[:,:] comb2_scr,
                                  CINT32_t[:,:] comb3_idx, CFLOAT_t[:,:] comb3_scr, CINT32_t[:,:] comb4_idx, CFLOAT_t[:,:] comb4_scr,
                                  CINT32_t *cnt2, CINT32_t *cnt3, CINT32_t *cnt4, CFLOAT_t max_dE, CFLOAT_t thr_A0, CFLOAT_t thr_A1,
                                  CINT32_t comb_n, CFLOAT_t min_imp, CFLOAT_t min_scr, CFLOAT_t max_dr):

  cdef CFLOAT_t dig, scr, dEi, dEj, wM, wT, dr
  cdef CINT32_t ii, jj, kk, n1max=0

  for ii in range(len1):
    if n1s[ii]>n1max: n1max=n1s[ii]

  cdef CFLOAT_t *dEs1 = <CFLOAT_t *> malloc(n0*len1*sizeof(CFLOAT_t))
  cdef CFLOAT_t *dEs2 = <CFLOAT_t *> malloc(n0*(len1*(len1-1)//2)*sizeof(CFLOAT_t))
  cdef CFLOAT_t *dEs3 = <CFLOAT_t *> malloc(n0*(len1*len1*(len1-1)//2)*sizeof(CFLOAT_t))
  cdef CFLOAT_t *digs1 = <CFLOAT_t *> malloc(len1*sizeof(CFLOAT_t))
  cdef CFLOAT_t *uw1s = <CFLOAT_t *> malloc(len1*sizeof(CFLOAT_t))
  cdef CFLOAT_t *uw2s = <CFLOAT_t *> malloc((len1*(len1-1)//2)*sizeof(CFLOAT_t))
  cdef CFLOAT_t *uw3s = <CFLOAT_t *> malloc((len1*len1*(len1-1)//2)*sizeof(CFLOAT_t))
  cdef CFLOAT_t *ijks2 = <CFLOAT_t *> malloc(3*(len1*(len1-1)//2)*sizeof(CFLOAT_t))
  cdef CFLOAT_t *ijks3 = <CFLOAT_t *> malloc(3*(len1*len1*(len1-1)//2)*sizeof(CFLOAT_t))
  cdef CFLOAT_t *w0s = <CFLOAT_t *> malloc(n0*sizeof(CFLOAT_t))
  cdef CFLOAT_t *comb1_scr = <CFLOAT_t *> malloc(len1*sizeof(CFLOAT_t))

  for ii in range(n0):
    w0s[ii] = max(0.,(min(thr_A1,As0[ii])-thr_A0)/(thr_A1-thr_A0))

  # Get alignments
  for ii in range(len1):
    _align_lines3(Es0, As0, i0, n0, Es1, As1, csum1[i1s[ii]], n1s[ii], mtcA, mtcB,
                  &scr, &dig, max_dE, thr_A0, thr_A1, 0)
    digs1[ii] = dig
    comb1_scr[ii] = scr

    for jj in range(n0):
      if mtcA[jj]>=0: dEs1[ii*n0+jj] = abs(Es0[i0+jj]-Es1[csum1[i1s[ii]]+mtcA[jj]])
      else:           dEs1[ii*n0+jj] = max_dE

    uw1s[ii] = 0.
    for jj in range(n1s[ii]):
      if mtcB[jj] == -1:
        uw1s[ii] += max(0.,(min(thr_A1,As1[csum1[i1s[ii]]+jj])-thr_A0)/(thr_A1-thr_A0))

  # Find 2-combinations
  cnt2[0] = 0
  if comb_n >= 2:
    for ii in range(len1):
      for jj in range(ii+1,len1):
        if max_dr>0.:
          dr = ((ijks1[i1s[ii],0]-ijks1[i1s[jj],0])**2+\
                (ijks1[i1s[ii],1]-ijks1[i1s[jj],1])**2+\
                (ijks1[i1s[ii],2]-ijks1[i1s[jj],2])**2)**0.5
          if dr>max_dr: continue

        wM = 0.
        wT = uw1s[ii]+uw1s[jj]
        scr = 0.
        for kk in range(n0):
          dEi = dEs1[ii*n0+kk]
          dEj = dEs1[jj*n0+kk]
          if dEi < dEj:
            dEs2[cnt2[0]*n0+kk] = dEi
            scr += 1.-dEi/max_dE
            wM += 2.
            wT += 2.
          elif (dEj < dEi) or (dEj < max_dE):
            dEs2[cnt2[0]*n0+kk] = dEj
            scr += 1.-dEj/max_dE
            wM += 2.
            wT += 2.
          else:
            dEs2[cnt2[0]*n0+kk] = max_dE
            wT += w0s[kk]

        scr = 2.*scr/wT
        dig = wM/wT

        if (dig>digs1[ii]+.05) and (dig>digs1[jj]+.05) and \
           (scr>comb1_scr[ii]+min_imp) and (scr>comb1_scr[jj]+min_imp) and (scr>min_scr):
          comb2_idx[cnt2[0],0] = i1s[ii]
          comb2_idx[cnt2[0],1] = i1s[jj]
          comb2_scr[cnt2[0],0] = scr
          comb2_scr[cnt2[0],1] = dig
          uw2s[cnt2[0]] = uw1s[ii]+uw1s[jj]
          if max_dr>0:
            ijks2[3*cnt2[0]+0] = 0.5*(ijks1[i1s[ii],0]+ijks1[i1s[jj],0])
            ijks2[3*cnt2[0]+1] = 0.5*(ijks1[i1s[ii],1]+ijks1[i1s[jj],1])
            ijks2[3*cnt2[0]+2] = 0.5*(ijks1[i1s[ii],2]+ijks1[i1s[jj],2])
          cnt2[0] += 1

  # Find 3-combinations
  cnt3[0] = 0
  if comb_n >= 3:
    for ii in range(cnt2[0]):
      for jj in range(len1):
        if max_dr>0.:
          dr = ((ijks2[3*ii+0]-ijks1[i1s[jj],0])**2+\
                (ijks2[3*ii+1]-ijks1[i1s[jj],1])**2+\
                (ijks2[3*ii+2]-ijks1[i1s[jj],2])**2)**0.5
          if dr>max_dr: continue

        wM = 0.
        wT = uw2s[ii]+uw1s[jj]
        scr = 0.
        for kk in range(n0):
          dEi = dEs2[ii*n0+kk]
          dEj = dEs1[jj*n0+kk]
          if dEi < dEj:
            dEs3[cnt3[0]*n0+kk] = dEi
            scr += 1.-dEi/max_dE
            wM += 2.
            wT += 2.
          elif (dEj < dEi) or (dEj < max_dE):
            dEs3[cnt3[0]*n0+kk] = dEj
            scr += 1.-dEj/max_dE
            wM += 2.
            wT += 2.
          else:
            dEs3[cnt3[0]*n0+kk] = max_dE
            wT += w0s[kk]

        scr = 2.*scr/wT
        dig = wM/wT

        if (dig>comb2_scr[ii,1]+.05) and (dig>digs1[jj]+.05) and \
           (scr>comb2_scr[ii,0]+min_imp) and (scr>comb1_scr[jj]+min_imp) and (scr>min_scr):
          comb3_idx[cnt3[0],0] = comb2_idx[ii,0]
          comb3_idx[cnt3[0],1] = comb2_idx[ii,1]
          comb3_idx[cnt3[0],2] = i1s[jj]
          comb3_scr[cnt3[0],0] = scr
          comb3_scr[cnt3[0],1] = dig
          uw3s[cnt3[0]] = uw2s[ii]+uw1s[jj]
          if max_dr>0:
            ijks3[3*cnt3[0]+0] = 0.5*(ijks2[3*ii+0]+ijks1[i1s[jj],0])
            ijks3[3*cnt3[0]+1] = 0.5*(ijks2[3*ii+1]+ijks1[i1s[jj],1])
            ijks3[3*cnt3[0]+2] = 0.5*(ijks2[3*ii+2]+ijks1[i1s[jj],2])
          cnt3[0] += 1

  # Find 4-combinations
  cnt4[0] = 0
  if comb_n >= 4:
    for ii in range(cnt3[0]):
      for jj in range(len1):
        if max_dr>0.:
          dr = ((ijks3[3*ii+0]-ijks1[i1s[jj],0])**2+\
                (ijks3[3*ii+1]-ijks1[i1s[jj],1])**2+\
                (ijks3[3*ii+2]-ijks1[i1s[jj],2])**2)**0.5
          if dr>max_dr: continue

        wM = 0.
        wT = uw3s[ii]+uw1s[jj]
        scr = 0.
        for kk in range(n0):
          dEi = dEs3[ii*n0+kk]
          dEj = dEs1[jj*n0+kk]
          if dEi < dEj:
            scr += 1.-dEi/max_dE
            wM += 2.
            wT += 2.
          elif (dEj < dEi) or (dEj < max_dE):
            scr += 1.-dEj/max_dE
            wM += 2.
            wT += 2.
          else:
            wT += w0s[kk]

        scr = 2.*scr/wT
        dig = wM/wT

        if (dig>comb3_scr[ii,1]+.05) and (dig>digs1[jj]+.05) and \
           (scr>comb3_scr[ii,0]+min_imp) and (scr>comb1_scr[jj]+min_imp) and (scr>min_scr):
          comb4_idx[cnt4[0],0] = comb3_idx[ii,0]
          comb4_idx[cnt4[0],1] = comb3_idx[ii,1]
          comb4_idx[cnt4[0],2] = comb3_idx[ii,2]
          comb4_idx[cnt4[0],3] = i1s[jj]
          comb4_scr[cnt4[0],0] = scr
          comb4_scr[cnt4[0],1] = dig
          cnt4[0] += 1

  # free(mtcs1)
  free(dEs1)
  free(dEs2)
  free(dEs3)
  free(digs1)
  free(uw1s)
  free(uw2s)
  free(uw3s)
  free(ijks2)
  free(ijks3)
  free(w0s)
  free(comb1_scr)

  return 0

# ---------------------------------- MATCHING ----------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def match(score, CFLOAT_t min_score):
  cdef CINT32_t N = score.shape[0]

  cdef CFLOAT_t[:] _score_row_ = score.data
  cdef CINT32_t[:] _indices_row_ = score.indices
  cdef CINT32_t[:] _indptr_row_ = score.indptr

  score_c = score.tocsc()
  cdef CFLOAT_t[:] _score_col_ = score_c.data
  cdef CINT32_t[:] _indices_col_ = score_c.indices
  cdef CINT32_t[:] _indptr_col_ = score_c.indptr

  cdef CINT32_t iA, iB, iBA, iAB, j, j0, j1, jmax
  cdef CFLOAT_t cmax, cval

  matches = np.full(N, -1, dtype=CINT32)
  cdef CINT32_t[:] _matches_ = matches

  for iA in range(N):
    j0 = _indptr_row_[iA]
    j1 = _indptr_row_[iA+1]
    if j0 == j1: continue

    iBA = 0
    cmax = -1.
    for j in range(j0, j1):
      cval = _score_row_[j]
      if cval >= max(cmax, min_score):
        cmax = cval
        iBA = _indices_row_[j]
        jmax = j

    if cmax == -1.: continue

    j0 = _indptr_col_[iBA]
    j1 = _indptr_col_[iBA+1]
    if j0 == j1: continue

    iAB = 0
    cmax = -1.
    for j in range(j0,j1):
      cval = _score_col_[j]
      if cval >= max(cmax, min_score):
        cmax = cval
        iAB = _indices_col_[j]

    if cmax == -1.: continue

    if iAB == iA:
      _matches_[iA] = iBA

  return matches

# ------------------------------ FIND DUPLICATES -------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def find_duplicates(Es, ns, CINT32_t n0, CFLOAT_t max_dE):

  if ns.sum() != len(Es): raise ValueError("Lentgh of 'Es' must be equal to the sum of 'ns'!")

  cum_all = np.insert(np.cumsum(ns[:-1],dtype=np.int32),0,0)
  cdef CINT32_t[:] _cum_all_ = cum_all
  cum_sng = np.zeros_like(cum_all, dtype=np.int32)
  cdef CINT32_t[:] _cum_sng_ = cum_sng
  idx0 = np.zeros_like(cum_all, dtype=np.int32)
  cdef CINT32_t[:] _idx0_ = idx0

  doub = np.zeros(len(ns), dtype=np.uint8)
  cdef CUINT8_t[:] _doub_ = doub

  cdef CINT32_t[:] _ns_ = ns
  cdef CFLOAT_t[:] _Es_ = Es

  cdef CINT32_t nn, ii, jj, n_cum=len(cum_all), n_max=ns.max()

  for nn in range(n0,n_max):
    jj = 0
    for ii in range(n_cum):
      if _ns_[ii]==nn:
        _cum_sng_[jj] = _cum_all_[ii]
        _idx0_[jj] = ii
        jj += 1

    _find_duplicates(_Es_, _idx0_, _cum_sng_, jj, nn, max_dE, _doub_)

  return doub.astype(bool)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32_t _find_duplicates(CFLOAT_t[:] Es, CINT32_t[:] idx0, CINT32_t[:] cum, CINT32_t nidx,
                               CINT32_t M, CFLOAT_t max_dE, CUINT8_t[:] doub):

  cdef CINT32_t ii, jj, kk, cnt

  cnt = 0
  for ii in range(nidx):
    for jj in range(ii+1,nidx):
      for kk in range(M):
        if abs(Es[cum[ii]+kk]-Es[cum[jj]+kk]) > max_dE: break
      else:
        doub[idx0[ii]] = 1
        doub[idx0[jj]] = 1

  return 0
