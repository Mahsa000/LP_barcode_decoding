cimport cython

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

from scipy.sparse import csr_matrix


CUINT8 = np.uint8
CUINT16 = np.uint16
CINT32 = np.int32
CINT64 = np.int64
CUINT64 = np.uint64
CFLOAT = np.float64
ctypedef np.uint8_t CUINT8_t
ctypedef np.uint16_t CUINT16_t
ctypedef np.int32_t CINT32_t
ctypedef np.int64_t CINT64_t
ctypedef np.uint64_t CUINT64_t
ctypedef np.float64_t CFLOAT_t

cdef CFLOAT_t EPS = 1e-12

# ----------------------------- ARGSORT FUNCTIONS ------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

cdef extern from "stdlib.h":
  ctypedef void const_void "const void"
  void qsort(void *base, int nmemb, int size, int(*compar)(const_void *, const_void *)) nogil

cdef struct IndexedElement:
  CINT32_t index
  CFLOAT_t value

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _compare(const_void *a, const_void *b):
  cdef CFLOAT_t v = (<IndexedElement*> a).value-(<IndexedElement*> b).value
  if v < 0: return -1
  if v >= 0: return 1

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int  argsort(CFLOAT_t[:] data, CINT32_t[:] idxs, CINT32_t n):
  cdef CINT32_t ii
  cdef IndexedElement *order_struct = <IndexedElement *> malloc(n * sizeof(IndexedElement))
  
  for ii in range(n):
    order_struct[ii].index = ii
    order_struct[ii].value = data[ii]
      
  qsort(<void *> order_struct, n, sizeof(IndexedElement), _compare)
  
  for ii in range(n): idxs[ii] = order_struct[ii].index

  free(order_struct)

  return 0

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
# -------
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
# -------
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CUINT16_t ns2fcl(CINT32_t n0, CINT32_t n1, CINT32_t nM):
  return ((<CUINT16_t> min(min(n0,n1),64)-1)<<10) +\
         ((<CUINT16_t> min(max(n0,n1),64)-1)<<4) +\
         (<CUINT16_t> min(n1+n0-nM, 15))
# -------

# ------------------------------ ALIGNING/SCORING ------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def score_cross_symm(CFLOAT_t[:] Es0, CFLOAT_t[:] As0, CINT32_t[:] ns0, CUINT64_t[:] codes0A, CUINT64_t[:] codes0B, CINT32_t i0_start, CINT32_t ntot0, 
                     CFLOAT_t[:] Es1, CFLOAT_t[:] As1, CINT32_t[:] ns1, CUINT64_t[:] codes1A, CUINT64_t[:] codes1B,
                     CFLOAT_t density_symm, CFLOAT_t min_simil_symm, CINT32_t mmax, CFLOAT_t max_dE, CFLOAT_t thr_A0, CFLOAT_t thr_A1, CFLOAT_t min_dig_symm, CFLOAT_t min_scr_symm):
  
  cidx0 = np.insert(np.cumsum(ns0,dtype=np.int32),0,0)
  cdef CINT32_t[:] _cidx0_ = cidx0
  cidx1 = np.insert(np.cumsum(ns1,dtype=np.int32),0,0)
  cdef CINT32_t[:] _cidx1_ = cidx1

  cdef CINT32_t len0 = len(ns0)
  assert len(Es0) == sum(ns0)
  assert len(Es0) == len(As0)  
  assert len0 == len(codes0A)
  assert len0 == len(codes0B)

  cdef CINT32_t ntot1 = len(ns1)
  assert len(Es1) == sum(ns1)
  assert len(Es1) == len(As1)
  assert ntot1 == len(codes1A)
  assert ntot1 == len(codes1B)

  cdef CUINT64_t nMax_symm = CUINT64(np.ceil(density_symm*len0*ntot1))

  indptr01 = np.zeros(ntot0+1, dtype=CUINT64)
  cdef CUINT64_t[:] _indptr01_ = indptr01
  indices01 = np.full(nMax_symm, -1, dtype=CINT32)
  cdef CINT32_t[:] _indices01_ = indices01
  data_scr01 = np.full_like(indices01, 0, dtype=CFLOAT)
  cdef CFLOAT_t[:] _data_scr01_ = data_scr01
  data_fcl01 = np.full_like(indices01, 0, dtype=CUINT16)
  cdef CUINT16_t[:] _data_fcl01_ = data_fcl01

  w0A = _weights64(codes0A)
  cdef CFLOAT_t[:] _w0A_ = w0A
  w0B = _weights64(codes0B)
  cdef CFLOAT_t[:] _w0B_ = w0B
  w1A = _weights64(codes1A)
  cdef CFLOAT_t[:] _w1A_ = w1A
  w1B = _weights64(codes1B)
  cdef CFLOAT_t[:] _w1B_ = w1B

  matched0 = np.full(mmax, -1, dtype=CINT32)
  cdef CINT32_t[:] _matched0_ = matched0
  matched1 = np.full(mmax, -1, dtype=CINT32)
  cdef CINT32_t[:] _matched1_ = matched1

  cdef CUINT64_t iseq01=0
  cdef CFLOAT_t sumA, sumB, scr01, scr0in1, scr1in0
  cdef CINT32_t i, j, mmin, ret
  cdef CUINT16_t fcl

  for i in range(len0):
    for j in range(ntot1):
      sumA = bits_count(codes0A[i]&codes1A[j])
      sumB = bits_count(codes0B[i]&codes1B[j])

      if max(sumA/(_w0A_[i]+_w1A_[j]-sumA), sumB/(_w0B_[i]+_w1B_[j]-sumB)) >= min_simil_symm:
        ret = _score_pair(Es0, As0, _cidx0_[i], ns0[i], _matched0_, Es1, As1, _cidx1_[j], ns1[j], _matched1_,
                          max_dE, thr_A0, thr_A1, min_dig_symm, 0., &scr01, &scr0in1, &scr1in0, &fcl)
        if ret == -1: return -1

        if scr01>min_scr_symm:
          _data_scr01_[iseq01] = scr01
          _data_fcl01_[iseq01] = fcl
          _indices01_[iseq01] = j
          iseq01 += 1
          if iseq01 >= nMax_symm: return -2

    _indptr01_[i0_start+i+1] = iseq01

  _indptr01_[i0_start+i+1:] = iseq01

  return csr_matrix((data_fcl01[:iseq01], indices01[:iseq01], indptr01), (ntot0, ntot1)),\
         csr_matrix((data_scr01[:iseq01], indices01[:iseq01], indptr01), (ntot0, ntot1)),\
         None, None
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def score_cross_all(CFLOAT_t[:] Es0, CFLOAT_t[:] As0, CINT32_t[:] ns0, CUINT64_t[:] codes0A, CUINT64_t[:] codes0B, CINT32_t i0_start, CINT32_t ntot0, 
                    CFLOAT_t[:] Es1, CFLOAT_t[:] As1, CINT32_t[:] ns1, CUINT64_t[:] codes1A, CUINT64_t[:] codes1B,
                    CFLOAT_t density_symm, CFLOAT_t density_comb, CFLOAT_t min_simil_symm, CFLOAT_t min_simil_comb, CINT32_t m0, CINT32_t mrat,
                    CINT32_t mmax, CFLOAT_t max_dE, CFLOAT_t thr_A0, CFLOAT_t thr_A1, CFLOAT_t min_dig_symm, CFLOAT_t min_dig_comb, CFLOAT_t min_scr_symm, CFLOAT_t min_scr_comb):

  cidx0 = np.insert(np.cumsum(ns0,dtype=np.int32),0,0)
  cdef CINT32_t[:] _cidx0_ = cidx0
  cidx1 = np.insert(np.cumsum(ns1,dtype=np.int32),0,0)
  cdef CINT32_t[:] _cidx1_ = cidx1

  cdef CINT32_t len0 = len(ns0)
  assert len(Es0) == sum(ns0)
  assert len(Es0) == len(As0)  
  assert len0 == len(codes0A)
  assert len0 == len(codes0B)

  cdef CINT32_t ntot1 = len(ns1)
  assert len(Es1) == sum(ns1)
  assert len(Es1) == len(As1)
  assert ntot1 == len(codes1A)
  assert ntot1 == len(codes1B)

  cdef CUINT64_t nMax_symm = CUINT64(np.ceil(density_symm*len0*ntot1))
  cdef CUINT64_t nMax_comb = CUINT64(np.ceil(density_comb*len0*ntot1))

  indptr01 = np.zeros(ntot0+1, dtype=CUINT64)
  cdef CUINT64_t[:] _indptr01_ = indptr01
  indptr0in1 = np.zeros(ntot0+1, dtype=CUINT64)
  cdef CUINT64_t[:] _indptr0in1_ = indptr0in1
  indptr1in0 = np.zeros(ntot0+1, dtype=CUINT64)
  cdef CUINT64_t[:] _indptr1in0_ = indptr1in0

  indices01 = np.full(nMax_symm, -1, dtype=CINT32)
  cdef CINT32_t[:] _indices01_ = indices01
  indices0in1 = np.full(nMax_comb, -1, dtype=CINT32)
  cdef CINT32_t[:] _indices0in1_ = indices0in1
  indices1in0 = np.full(nMax_comb, -1, dtype=CINT32)
  cdef CINT32_t[:] _indices1in0_ = indices1in0

  data_scr01 = np.full_like(indices01, 0, dtype=CFLOAT)
  cdef CFLOAT_t[:] _data_scr01_ = data_scr01
  data_scr0in1 = np.full_like(indices0in1, 0, dtype=CFLOAT)
  cdef CFLOAT_t[:] _data_scr0in1_ = data_scr0in1
  data_scr1in0 = np.full_like(indices1in0, 0, dtype=CFLOAT)
  cdef CFLOAT_t[:] _data_scr1in0_ = data_scr1in0

  data_fcl01 = np.full_like(indices01, 0, dtype=CUINT16)
  cdef CUINT16_t[:] _data_fcl01_ = data_fcl01

  w0A = _weights64(codes0A)
  cdef CFLOAT_t[:] _w0A_ = w0A
  w0B = _weights64(codes0B)
  cdef CFLOAT_t[:] _w0B_ = w0B
  w1A = _weights64(codes1A)
  cdef CFLOAT_t[:] _w1A_ = w1A
  w1B = _weights64(codes1B)
  cdef CFLOAT_t[:] _w1B_ = w1B

  matched0 = np.full(mmax, -1, dtype=CINT32)
  cdef CINT32_t[:] _matched0_ = matched0
  matched1 = np.full(mmax, -1, dtype=CINT32)
  cdef CINT32_t[:] _matched1_ = matched1

  cdef CUINT64_t iseq01=0, iseq0in1=0, iseq1in0=0
  cdef CFLOAT_t sumA, sumB, scr01, scr0in1, scr1in0
  cdef CINT32_t i, j, mmin, ret
  cdef CUINT8_t do01, do0in1, do1in0
  cdef CUINT16_t fcl

  for i in range(len0):
    for j in range(ntot1):
      sumA = bits_count(codes0A[i]&codes1A[j])
      sumB = bits_count(codes0B[i]&codes1B[j])

      do01 = max(sumA/(_w0A_[i]+_w1A_[j]-sumA), sumB/(_w0B_[i]+_w1B_[j]-sumB)) >= min_simil_symm
      
      mmin = ns1[j]//mrat
      if ns1[j]%mrat>0: mmin += 1
      do0in1 = (ns1[j]>=m0) and (ns0[i]>=mmin) and (max(sumA/_w0A_[i], sumB/_w0B_[i]) >= min_simil_comb)

      mmin = ns0[i]//mrat
      if ns0[i]%mrat>0: mmin += 1
      do1in0 = (ns0[i]>=m0) and (ns1[j]>=mmin) and (max(sumA/_w1A_[j], sumB/_w1B_[j]) >= min_simil_comb)

      if do01 or do0in1 or do1in0:
        ret = _score_pair(Es0, As0, _cidx0_[i], ns0[i], _matched0_, Es1, As1, _cidx1_[j], ns1[j], _matched1_,
                          max_dE, thr_A0, thr_A1, min_dig_symm, min_dig_comb, &scr01, &scr0in1, &scr1in0, &fcl)
        if ret == -1: return -1

        if do01 and scr01>min_scr_symm:
          _data_scr01_[iseq01] = scr01
          _data_fcl01_[iseq01] = fcl
          _indices01_[iseq01] = j
          iseq01 += 1
          if iseq01 >= nMax_symm: return -2
        
        if do0in1 and scr0in1>min_scr_comb:
          _data_scr0in1_[iseq0in1] = scr0in1
          _indices0in1_[iseq0in1] = j
          iseq0in1 += 1
          if iseq0in1 >= nMax_comb: return -3

        if do1in0 and scr1in0>min_scr_comb:
          _data_scr1in0_[iseq1in0] = scr1in0
          _indices1in0_[iseq1in0] = j
          iseq1in0 += 1
          if iseq1in0 >= nMax_comb: return -3

    _indptr01_[i0_start+i+1] = iseq01
    _indptr0in1_[i0_start+i+1] = iseq0in1
    _indptr1in0_[i0_start+i+1] = iseq1in0

  _indptr01_[i0_start+i+1:] = iseq01
  _indptr0in1_[i0_start+i+1:] = iseq0in1
  _indptr1in0_[i0_start+i+1:] = iseq1in0

  return csr_matrix((data_fcl01[:iseq01], indices01[:iseq01], indptr01), (ntot0, ntot1)),\
         csr_matrix((data_scr01[:iseq01], indices01[:iseq01], indptr01), (ntot0, ntot1)),\
         csr_matrix((data_scr0in1[:iseq0in1], indices0in1[:iseq0in1], indptr0in1), (ntot0, ntot1)),\
         csr_matrix((data_scr1in0[:iseq1in0], indices1in0[:iseq1in0], indptr1in0), (ntot0, ntot1))
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def score_cross_ana(CFLOAT_t[:] Es0, CFLOAT_t[:] As0, CINT32_t[:] ns0, CUINT64_t[:] codes0A, CUINT64_t[:] codes0B, CINT32_t i0_start, CINT32_t ntot0, 
                    CFLOAT_t[:] Es1, CFLOAT_t[:] As1, CINT32_t[:] ns1, CUINT64_t[:] codes1A, CUINT64_t[:] codes1B,
                    CINT32_t mmax, CFLOAT_t max_dE, CFLOAT_t density_symm, CFLOAT_t min_simil_symm, CFLOAT_t min_dig_symm, CFLOAT_t min_scr_symm):

  cidx0 = np.insert(np.cumsum(ns0,dtype=np.int32),0,0)
  cdef CINT32_t[:] _cidx0_ = cidx0
  cidx1 = np.insert(np.cumsum(ns1,dtype=np.int32),0,0)
  cdef CINT32_t[:] _cidx1_ = cidx1

  cdef CINT32_t len0 = len(ns0)
  assert len(Es0) == sum(ns0)
  assert len(Es0) == len(As0)  
  assert len0 == len(codes0A)
  assert len0 == len(codes0B)

  cdef CINT32_t ntot1 = len(ns1)
  assert len(Es1) == sum(ns1)
  assert len(Es1) == len(As1)
  assert ntot1 == len(codes1A)
  assert ntot1 == len(codes1B)

  cdef CUINT64_t nMax_symm = CUINT64(np.ceil(density_symm*len0*ntot1))

  indptr01 = np.zeros(ntot0+1, dtype=CUINT64)
  cdef CUINT64_t[:] _indptr01_ = indptr01

  indices01 = np.full(nMax_symm, -1, dtype=CINT32)
  cdef CINT32_t[:] _indices01_ = indices01

  data_scr01 = np.full_like(indices01, 0, dtype=CFLOAT)
  cdef CFLOAT_t[:] _data_scr01_ = data_scr01

  data_fcl01 = np.full_like(indices01, 0, dtype=CUINT16)
  cdef CUINT16_t[:] _data_fcl01_ = data_fcl01

  w0A = _weights64(codes0A)
  cdef CFLOAT_t[:] _w0A_ = w0A
  w0B = _weights64(codes0B)
  cdef CFLOAT_t[:] _w0B_ = w0B
  w1A = _weights64(codes1A)
  cdef CFLOAT_t[:] _w1A_ = w1A
  w1B = _weights64(codes1B)
  cdef CFLOAT_t[:] _w1B_ = w1B

  matched0 = np.full(mmax, -1, dtype=CINT32)
  cdef CINT32_t[:] _matched0_ = matched0
  matched1 = np.full(mmax, -1, dtype=CINT32)
  cdef CINT32_t[:] _matched1_ = matched1

  cdef CUINT64_t iseq01=0
  cdef CFLOAT_t sumA, sumB, scr01, dig01
  cdef CINT32_t i, j, mmin, ret, nM
  cdef CUINT8_t do01
  cdef CUINT16_t fcl

  for i in range(len0):
    for j in range(ntot1):
      sumA = bits_count(codes0A[i]&codes1A[j])
      sumB = bits_count(codes0B[i]&codes1B[j])

      do01 = max(sumA/(_w0A_[i]+_w1A_[j]-sumA), sumB/(_w0B_[i]+_w1B_[j]-sumB)) >= min_simil_symm
      if do01:
        ret = _score_analog(Es0, As0, _cidx0_[i], ns0[i], _matched0_, Es1, As1, _cidx1_[j], ns1[j], _matched1_, max_dE, &scr01, &nM)
        if ret == -1: return -1
        dig01 = <CFLOAT_t>nM / <CFLOAT_t>(ns0[i]+ns1[j])

        if (dig01>min_dig_symm) & (scr01>min_scr_symm):
          _data_scr01_[iseq01] = scr01
          _data_fcl01_[iseq01] = ns2fcl(ns0[i],ns1[j],nM)
          _indices01_[iseq01] = j
          iseq01 += 1
          if iseq01 >= nMax_symm: return -2
        
    _indptr01_[i0_start+i+1] = iseq01

  _indptr01_[i0_start+i+1:] = iseq01

  return csr_matrix((data_fcl01[:iseq01], indices01[:iseq01], indptr01), (ntot0, ntot1)),\
         csr_matrix((data_scr01[:iseq01], indices01[:iseq01], indptr01), (ntot0, ntot1))
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def score_self_symm(CFLOAT_t[:] Es, CFLOAT_t[:] As, CINT32_t[:] ns, CUINT64_t[:] codesA, CUINT64_t[:] codesB, CINT32_t i0_start, CINT32_t i0_stop,
                    CFLOAT_t density_symm, CFLOAT_t min_simil_symm, CINT32_t mmax, CFLOAT_t max_dE, CFLOAT_t thr_A0, CFLOAT_t thr_A1, CFLOAT_t min_dig_symm, CFLOAT_t min_scr_symm):

  cidx = np.insert(np.cumsum(ns,dtype=np.int32),0,0)
  cdef CINT32_t[:] _cidx_ = cidx

  cdef CINT32_t ntot = len(ns)
  assert len(Es) == sum(ns)
  assert len(Es) == len(As)  
  assert ntot == len(codesA)
  assert ntot == len(codesB)

  cdef CUINT64_t nMax_symm = CUINT64(np.ceil(density_symm*(ntot-i0_start)*(i0_stop-i0_start)))

  indptr01 = np.zeros(ntot+1, dtype=CUINT64)
  cdef CUINT64_t[:] _indptr01_ = indptr01
  indices01 = np.full(nMax_symm, -1, dtype=CINT32)
  cdef CINT32_t[:] _indices01_ = indices01
  data_scr01 = np.full_like(indices01, 0, dtype=CFLOAT)
  cdef CFLOAT_t[:] _data_scr01_ = data_scr01
  data_fcl01 = np.full_like(indices01, 0, dtype=CUINT16)
  cdef CUINT16_t[:] _data_fcl01_ = data_fcl01

  wA = _weights64(codesA)
  cdef CFLOAT_t[:] _wA_ = wA
  wB = _weights64(codesB)
  cdef CFLOAT_t[:] _wB_ = wB

  matched0 = np.full(mmax, -1, dtype=CINT32)
  cdef CINT32_t[:] _matched0_ = matched0
  matched1 = np.full(mmax, -1, dtype=CINT32)
  cdef CINT32_t[:] _matched1_ = matched1

  cdef CUINT64_t iseq01=0
  cdef CFLOAT_t sumA, sumB, temp, scr01, scr0in1, scr1in0
  cdef CINT32_t i, j, ret
  cdef CUINT16_t fcl

  for i in range(i0_start, i0_stop):
    for j in range(i+1,ntot):
      sumA = bits_count(codesA[i]&codesA[j])
      sumB = bits_count(codesB[i]&codesB[j])

      if max(sumA/(_wA_[i]+_wA_[j]-sumA), sumB/(_wB_[i]+_wB_[j]-sumB)) >= min_simil_symm:
        ret = _score_pair(Es, As, _cidx_[i], ns[i], _matched0_, Es, As, _cidx_[j], ns[j], _matched1_,
                          max_dE, thr_A0, thr_A1, min_dig_symm, 0., &scr01, &scr0in1, &scr1in0, &fcl)
        if ret == -1: return -1

        if scr01>min_scr_symm:
          _data_scr01_[iseq01] = scr01
          _data_fcl01_[iseq01] = fcl
          _indices01_[iseq01] = j
          iseq01 += 1
          if iseq01 >= nMax_symm: return -2

    _indptr01_[i+1] = iseq01

  _indptr01_[i+1:] = iseq01

  return csr_matrix((data_fcl01[:iseq01], indices01[:iseq01], indptr01), (ntot, ntot)),\
         csr_matrix((data_scr01[:iseq01], indices01[:iseq01], indptr01), (ntot, ntot)),\
         None, None
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def score_self_all(CFLOAT_t[:] Es, CFLOAT_t[:] As, CINT32_t[:] ns, CUINT64_t[:] codesA, CUINT64_t[:] codesB, CINT32_t i0_start, CINT32_t i0_stop,
                   CFLOAT_t density_symm, CFLOAT_t density_comb, CFLOAT_t min_simil_symm, CFLOAT_t min_simil_comb, CINT32_t m0, CINT32_t mrat,
                   CINT32_t mmax, CFLOAT_t max_dE, CFLOAT_t thr_A0, CFLOAT_t thr_A1, CFLOAT_t min_dig_symm, CFLOAT_t min_dig_comb, CFLOAT_t min_scr_symm, CFLOAT_t min_scr_comb):

  cidx = np.insert(np.cumsum(ns,dtype=np.int32),0,0)
  cdef CINT32_t[:] _cidx_ = cidx

  cdef CINT32_t ntot = len(ns)
  assert len(Es) == sum(ns)
  assert len(Es) == len(As)  
  assert ntot == len(codesA)
  assert ntot == len(codesB)

  cdef CUINT64_t nMax_symm = CUINT64(np.ceil(density_symm*(ntot-i0_start)*(i0_stop-i0_start)))
  cdef CUINT64_t nMax_comb = CUINT64(np.ceil(density_comb*(ntot-i0_start)*(i0_stop-i0_start)))

  indptr01 = np.zeros(ntot+1, dtype=CUINT64)
  cdef CUINT64_t[:] _indptr01_ = indptr01
  indptr0in1 = np.zeros(ntot+1, dtype=CUINT64)
  cdef CUINT64_t[:] _indptr0in1_ = indptr0in1
  indptr1in0 = np.zeros(ntot+1, dtype=CUINT64)
  cdef CUINT64_t[:] _indptr1in0_ = indptr1in0
  
  indices01 = np.full(nMax_symm, -1, dtype=CINT32)
  cdef CINT32_t[:] _indices01_ = indices01
  indices0in1 = np.full(nMax_comb, -1, dtype=CINT32)
  cdef CINT32_t[:] _indices0in1_ = indices0in1
  indices1in0 = np.full(nMax_comb, -1, dtype=CINT32)
  cdef CINT32_t[:] _indices1in0_ = indices1in0

  data_scr01 = np.full_like(indices01, 0, dtype=CFLOAT)
  cdef CFLOAT_t[:] _data_scr01_ = data_scr01
  data_scr0in1 = np.full_like(indices0in1, 0, dtype=CFLOAT)
  cdef CFLOAT_t[:] _data_scr0in1_ = data_scr0in1
  data_scr1in0 = np.full_like(indices1in0, 0, dtype=CFLOAT)
  cdef CFLOAT_t[:] _data_scr1in0_ = data_scr1in0

  data_fcl01 = np.full_like(indices01, 0, dtype=CUINT16)
  cdef CUINT16_t[:] _data_fcl01_ = data_fcl01

  wA = _weights64(codesA)
  cdef CFLOAT_t[:] _wA_ = wA
  wB = _weights64(codesB)
  cdef CFLOAT_t[:] _wB_ = wB

  matched0 = np.full(mmax, -1, dtype=CINT32)
  cdef CINT32_t[:] _matched0_ = matched0
  matched1 = np.full(mmax, -1, dtype=CINT32)
  cdef CINT32_t[:] _matched1_ = matched1

  cdef CUINT64_t iseq01 = 0, iseq0in1 = 0, iseq1in0 = 0
  cdef CFLOAT_t sumA, sumB, temp, scr01, scr0in1, scr1in0
  cdef CINT32_t i, j, mmin, ret
  cdef CUINT8_t do01, do0in1, do1in0
  cdef CUINT16_t fcl

  for i in range(i0_start, i0_stop):
    for j in range(i+1,ntot):
      sumA = bits_count(codesA[i]&codesA[j])
      sumB = bits_count(codesB[i]&codesB[j])

      do01 = max(sumA/(_wA_[i]+_wA_[j]-sumA), sumB/(_wB_[i]+_wB_[j]-sumB)) >= min_simil_symm

      mmin = ns[j]//mrat
      if ns[j]%mrat>0: mmin += 1
      do0in1 = (ns[j]>=m0) and (ns[i]>=mmin) and (max(sumA/(_wA_[i]), sumB/(_wB_[i]))>=min_simil_comb)

      mmin = ns[i]//mrat
      if ns[i]%mrat>0: mmin += 1
      do1in0 = (ns[i]>=m0) and (ns[j]>=mmin) and (max(sumA/_wA_[j], sumB/_wB_[j]) >= min_simil_comb)

      if do01 or do0in1 or do1in0:
        ret = _score_pair(Es, As, _cidx_[i], ns[i], _matched0_, Es, As, _cidx_[j], ns[j], _matched1_,
                          max_dE, thr_A0, thr_A1, min_dig_symm, min_dig_comb, &scr01, &scr0in1, &scr1in0, &fcl)
        if ret == -1: return -1

        if do01 and scr01>min_scr_symm:
          _data_scr01_[iseq01] = scr01
          _data_fcl01_[iseq01] = fcl
          _indices01_[iseq01] = j
          iseq01 += 1
          if iseq01 >= nMax_symm: return -2
        
        if do0in1 and scr0in1>min_scr_comb:
          _data_scr0in1_[iseq0in1] = scr0in1
          _indices0in1_[iseq0in1] = j
          iseq0in1 += 1
          if iseq0in1 >= nMax_comb: return -3

        if do1in0 and scr1in0>min_scr_comb:
          _data_scr1in0_[iseq1in0] = scr1in0
          _indices1in0_[iseq1in0] = j
          iseq1in0 += 1
          if iseq1in0 >= nMax_comb: return -3

    _indptr01_[i+1] = iseq01
    _indptr0in1_[i+1] = iseq0in1
    _indptr1in0_[i+1] = iseq1in0

  _indptr01_[i+1:] = iseq01
  _indptr0in1_[i+1:] = iseq0in1
  _indptr1in0_[i+1:] = iseq1in0

  return csr_matrix((data_fcl01[:iseq01], indices01[:iseq01], indptr01), (ntot, ntot)),\
         csr_matrix((data_scr01[:iseq01], indices01[:iseq01], indptr01), (ntot, ntot)),\
         csr_matrix((data_scr0in1[:iseq0in1], indices0in1[:iseq0in1], indptr0in1), (ntot, ntot)),\
         csr_matrix((data_scr1in0[:iseq1in0], indices1in0[:iseq1in0], indptr1in0), (ntot, ntot))
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def score_self_ana(CFLOAT_t[:] Es, CFLOAT_t[:] As, CINT32_t[:] ns, CUINT64_t[:] codesA, CUINT64_t[:] codesB, CINT32_t i0_start, CINT32_t i0_stop,
                   CINT32_t mmax, CFLOAT_t max_dE, CFLOAT_t density_symm, CFLOAT_t min_simil_symm, CFLOAT_t min_dig_symm, CFLOAT_t min_scr_symm):

  cidx = np.insert(np.cumsum(ns,dtype=np.int32),0,0)
  cdef CINT32_t[:] _cidx_ = cidx

  cdef CINT32_t ntot = len(ns)
  assert len(Es) == sum(ns)
  assert len(Es) == len(As)  
  assert ntot == len(codesA)
  assert ntot == len(codesB)

  cdef CUINT64_t nMax_symm = CUINT64(np.ceil(density_symm*(ntot-i0_start)*(i0_stop-i0_start)))

  indptr01 = np.zeros(ntot+1, dtype=CUINT64)
  cdef CUINT64_t[:] _indptr01_ = indptr01
  indices01 = np.full(nMax_symm, -1, dtype=CINT32)
  cdef CINT32_t[:] _indices01_ = indices01
  data_scr01 = np.full_like(indices01, 0, dtype=CFLOAT)
  cdef CFLOAT_t[:] _data_scr01_ = data_scr01
  data_fcl01 = np.full_like(indices01, 0, dtype=CUINT16)
  cdef CUINT16_t[:] _data_fcl01_ = data_fcl01

  wA = _weights64(codesA)
  cdef CFLOAT_t[:] _wA_ = wA
  wB = _weights64(codesB)
  cdef CFLOAT_t[:] _wB_ = wB

  matched0 = np.full(mmax, -1, dtype=CINT32)
  cdef CINT32_t[:] _matched0_ = matched0
  matched1 = np.full(mmax, -1, dtype=CINT32)
  cdef CINT32_t[:] _matched1_ = matched1

  cdef CUINT64_t iseq01=0
  cdef CFLOAT_t sumA, sumB, scr01, dig01
  cdef CINT32_t i, j, ret, nM

  for i in range(i0_start, i0_stop):
    for j in range(i+1,ntot):
      sumA = bits_count(codesA[i]&codesA[j])
      sumB = bits_count(codesB[i]&codesB[j])

      if max(sumA/(_wA_[i]+_wA_[j]-sumA), sumB/(_wB_[i]+_wB_[j]-sumB)) >= min_simil_symm:
        ret = _score_analog(Es, As, _cidx_[i], ns[i], _matched0_, Es, As, _cidx_[j], ns[j], _matched1_, max_dE, &scr01, &nM)
        if ret == -1: return -1

        dig01 = <CFLOAT_t>nM / <CFLOAT_t>(ns[i]+ns[j])

        if (dig01>min_dig_symm) & (scr01>min_scr_symm):
          _data_scr01_[iseq01] = scr01
          _data_fcl01_[iseq01] = ns2fcl(ns[i],ns[j],nM)
          _indices01_[iseq01] = j
          iseq01 += 1
          if iseq01 >= nMax_symm: return -2

    _indptr01_[i+1] = iseq01

  _indptr01_[i+1:] = iseq01

  return csr_matrix((data_fcl01[:iseq01], indices01[:iseq01], indptr01), (ntot, ntot)),\
         csr_matrix((data_scr01[:iseq01], indices01[:iseq01], indptr01), (ntot, ntot))
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def score_pair(CFLOAT_t[:] Es0, CFLOAT_t[:] As0, CINT32_t cidx0, CINT32_t n0,
               CFLOAT_t[:] Es1, CFLOAT_t[:] As1, CINT32_t cidx1, CINT32_t n1,
               CFLOAT_t max_dE, CFLOAT_t thr_A0, CFLOAT_t thr_A1, CFLOAT_t min_dig_symm, CFLOAT_t min_dig_comb):
  
  matched0 = np.full(n0, -1, dtype=CINT32)
  cdef CINT32_t[:] _matched0_ = matched0
  matched1 = np.full(n1, -1, dtype=CINT32)
  cdef CINT32_t[:] _matched1_ = matched1

  cdef CFLOAT_t scr01, scr0in1, scr1in0
  cdef CUINT16_t fcl

  ret = _score_pair(Es0, As0, cidx0, n0, _matched0_, Es1, As1, cidx1, n1, _matched1_,
                    max_dE, thr_A0, thr_A1, min_dig_symm, min_dig_comb, &scr01, &scr0in1, &scr1in0, &fcl)

  return (matched0, matched1), (scr01,scr0in1,scr1in0,fcl)
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32_t _score_pair(CFLOAT_t[:] _Es0_, CFLOAT_t[:] _As0_, CINT32_t cidx0, CINT32_t n0, CINT32_t[:] _matched0_,
                          CFLOAT_t[:] _Es1_, CFLOAT_t[:] _As1_, CINT32_t cidx1, CINT32_t n1, CINT32_t[:] _matched1_,
                          CFLOAT_t max_dE, CFLOAT_t thr_A0, CFLOAT_t thr_A1, CFLOAT_t min_dig_symm, CFLOAT_t min_dig_comb,
                          CFLOAT_t* scr01, CFLOAT_t* scr0in1, CFLOAT_t* scr1in0, CUINT16_t* fcl):
  
  cdef CINT32_t i0=0, i1=0, j1
  cdef CFLOAT_t curr, right, bottom, score=0., wM=0., wT01=0., wT0in1=0., wT1in0=0.
  cdef CUINT8_t do0=True, do1=True

  for i0 in range(n0): _matched0_[i0] = -1
  for i1 in range(n1): _matched1_[i1] = -1

  i0 = i1 = 0
  while True:
    if (i0>=n0) or (i1>=n1): break
    for j1 in range(i1,n1):
      curr = (_Es1_[cidx1+j1]-_Es0_[cidx0+i0])/max_dE
      i1 = j1
      if curr>-1.: break # until E1 is far below current E0, cycle through Es1
      do1 = True
    
    # If current E0 is far above current E1, go to next E0
    if curr>=1.:
      i0 += 1
      do0 = True
      continue

    curr = 1.-abs(curr)

    # Check scores of adjacent matches
    if j1<n1-1: right = max(0.,1.-abs(_Es1_[cidx1+j1+1]-_Es0_[cidx0+i0])/max_dE)
    else:       right = 0.
    if i0<n0-1: bottom = max(0.,1.-abs(_Es1_[cidx1+j1]-_Es0_[cidx0+i0+1])/max_dE)
    else:       bottom = 0.

    if do0 and (curr>right):
      _matched0_[i0] = j1
      score += curr
      do0 = False

    if do1 and (curr>bottom):
      _matched1_[j1] = i0
      score += curr
      do1 = False

    if (bottom>0.) and (bottom>right):
      i0 += 1
      do0 = True
      continue

    if (right>0.) and (right>bottom):
      i1 += 1
      do1 = True
      continue

    if (bottom>0.) and (bottom==right): return -1

    i0 += 1
    i1 += 1
    do0 = do1 = True

  # --- Calculate scores ---
  for i0 in range(n0):
    if _matched0_[i0]>=0:
      wM += 1.
      wT01 += 1.
      wT0in1 += 1.
      wT1in0 += 1.
    else:
      curr = max(0.,(min(thr_A1, _As0_[cidx0+i0])-thr_A0)/(thr_A1-thr_A0))
      wT01 += curr
      wT0in1 += curr

  for i1 in range(n1):
    if _matched1_[i1]>=0:
      wM += 1.
      wT01 += 1.
      wT0in1 += 1.
      wT1in0 += 1.
    else:
      curr = max(0.,(min(thr_A1, _As1_[cidx1+i1])-thr_A0)/(thr_A1-thr_A0))
      wT01 += curr
      wT1in0 += curr
  
  fcl[0] = ((<CUINT16_t> min(min(n0,n1),64)-1)<<10) +\
           ((<CUINT16_t> min(max(n0,n1),64)-1)<<4) +\
           (<CUINT16_t> min(n1 + n0 - <CINT32_t> wM, 15))

  if wM/wT01 >= min_dig_symm: scr01[0] = score/wT01
  else:                       scr01[0] = 0.

  if wM/wT0in1 >= min_dig_comb: scr0in1[0] = score/wT0in1
  else:                         scr0in1[0] = 0.

  if wM/wT1in0 >= min_dig_comb: scr1in0[0] = score/wT1in0
  else:                         scr1in0[0] = 0.

  return 0
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32_t _score_analog(CFLOAT_t[:] _Es0_, CFLOAT_t[:] _As0_, CINT32_t cidx0, CINT32_t n0, CINT32_t[:] _matched0_,
                            CFLOAT_t[:] _Es1_, CFLOAT_t[:] _As1_, CINT32_t cidx1, CINT32_t n1, CINT32_t[:] _matched1_,
                            CFLOAT_t max_dE, CFLOAT_t* scr01, CINT32_t* nM):
  
  cdef CINT32_t i0=0, i1=0, j1
  cdef CFLOAT_t curr, right, bottom
  cdef CUINT8_t do0=True, do1=True

  for i0 in range(n0): _matched0_[i0] = -1
  for i1 in range(n1): _matched1_[i1] = -1

  i0 = i1 = 0
  scr01[0] = 0.
  nM[0] = 0
  while True:
    if (i0>=n0) or (i1>=n1): break
    for j1 in range(i1,n1):
      curr = (_Es1_[cidx1+j1]-_Es0_[cidx0+i0])/max_dE
      i1 = j1
      if curr>-1.: break # until E1 is far below current E0, cycle through Es1
      do1 = True
    
    # If current E0 is far above current E1, go to next E0
    if curr>=1.:
      i0 += 1
      do0 = True
      continue

    curr = 1.-abs(curr)

    # Check scores of adjacent matches
    if j1<n1-1: right = max(0.,1.-abs(_Es1_[cidx1+j1+1]-_Es0_[cidx0+i0])/max_dE)
    else:       right = 0.
    if i0<n0-1: bottom = max(0.,1.-abs(_Es1_[cidx1+j1]-_Es0_[cidx0+i0+1])/max_dE)
    else:       bottom = 0.

    if do0 and (curr>right):
      _matched0_[i0] = j1
      scr01[0] += curr
      nM[0] += 1
      do0 = False

    if do1 and (curr>bottom):
      _matched1_[j1] = i0
      scr01[0] += curr
      nM[0] += 1
      do1 = False

    if (bottom>0.) and (bottom>right):
      i0 += 1
      do0 = True
      continue

    if (right>0.) and (right>bottom):
      i1 += 1
      do1 = True
      continue

    if (bottom>0.) and (bottom==right): return -1

    i0 += 1
    i1 += 1
    do0 = do1 = True

  scr01[0] /= <CFLOAT_t>nM[0]

  return 0
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def align_single(EsA, EsB, AsA=None, AsB=None,
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
  else:               raise ValueError(f'Invalid value for pmode: {pmode}')

  ret = _align_lines(_EsA_, _AsA_, 0, nA, _EsB_, _AsB_, 0, nB,
                     _matchedA_, _matchedB_, &scr, &dig,
                     max_dE, thr_A0, thr_A1, _pmode_)
  if ret < 0: raise RuntimeError(f'Error {ret}!!!')

  return matchedA, matchedB, scr, dig
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32_t _align_lines(CFLOAT_t[:] _EsA_, CFLOAT_t[:] _AsA_, CINT32_t cA0, CINT32_t nA,
                           CFLOAT_t[:] _EsB_, CFLOAT_t[:] _AsB_, CINT32_t cB0, CINT32_t nB,
                           CINT32_t[:] _matchedA_, CINT32_t[:] _matchedB_,
                           CFLOAT_t *scr, CFLOAT_t *dig, CFLOAT_t max_dE,
                           CFLOAT_t thr_A0, CFLOAT_t thr_A1, CUINT8_t pmode):

  cdef CINT32_t ii, iA, iB, iB0
  cdef CFLOAT_t curr, right, bottom, scr_tot, wT, wM

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

    if (bottom>0.) and (bottom==right): return -1

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
# -------

# -------------------------------- COMBINATIONS --------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def score_combinations(CFLOAT_t[:] _Es0_, CFLOAT_t[:] _As0_, CINT32_t[:] _ns0_,
#                        CFLOAT_t[:] _Es1_, CFLOAT_t[:] _As1_, CINT32_t[:] _ns1_, ijks1,
#                        scrtot, CINT32_t nmax, Emax, CFLOAT_t max_dE, CFLOAT_t thr_A0, CFLOAT_t thr_A1,
#                        CINT32_t comb_n, CINT32_t min_n, CFLOAT_t min_scr, CFLOAT_t min_dig, CFLOAT_t min_imp, CFLOAT_t min_tmp, CFLOAT_t max_dr):

#   cdef CINT32_t n0 = scrtot.shape[0]
#   cdef CINT32_t n1 = scrtot.shape[1]

#   if max_dr==0.:
#     ijks1 = np.zeros((1,3),dtype=CFLOAT)
#   else:
#     assert ijks1.shape[0] == n1
#     assert ijks1.shape[1] == 3

#   cdef CFLOAT_t[:,:] _ijks1_ = ijks1

#   cdef CINT32_t[:] _indices_ = scrtot.indices
#   cdef CINT32_t[:] _indptr_ = scrtot.indptr

#   csum0 = np.insert(np.cumsum(_ns0_),0,0).astype(CINT32)
#   csum1 = np.insert(np.cumsum(_ns1_),0,0).astype(CINT32)
#   cdef CINT32_t[:] _csum0_ = csum0
#   cdef CINT32_t[:] _csum1_ = csum1

#   idxs1 = np.zeros(nmax, dtype=CINT32)
#   cdef CINT32_t[:] _idxs1_ = idxs1
#   pns1 = np.zeros(nmax, dtype=CINT32)
#   cdef CINT32_t[:] _pns1_ = pns1

#   mtcA = np.zeros(Emax, dtype=CINT32)
#   mtcB = np.zeros(Emax, dtype=CINT32)
#   cdef CINT32_t[:] _mtcA_ = mtcA
#   cdef CINT32_t[:] _mtcB_ = mtcB

#   cdef CINT32_t irow, icol, ii, jj, cnt2, cnt3, cnt4, do_sort
#   cdef CFLOAT_t dr

#   comb2_idx = np.zeros((nmax*(nmax-1)//2,2), dtype=CINT32)
#   comb2_scr = np.zeros((nmax*(nmax-1)//2,2), dtype=CFLOAT)
#   cdef CINT32_t[:,:] _comb2_idx_ = comb2_idx
#   cdef CFLOAT_t[:,:] _comb2_scr_ = comb2_scr

#   comb3_idx = np.zeros((nmax*nmax*(nmax-1)//2,3), dtype=CINT32)
#   comb3_scr = np.zeros((nmax*nmax*(nmax-1)//2,2), dtype=CFLOAT)
#   cdef CINT32_t[:,:] _comb3_idx_ = comb3_idx
#   cdef CFLOAT_t[:,:] _comb3_scr_ = comb3_scr

#   comb4_idx = np.zeros((nmax*nmax*(nmax-1)//2,4), dtype=CINT32)
#   comb4_scr = np.zeros((nmax*nmax*(nmax-1)//2,2), dtype=CFLOAT)
#   cdef CINT32_t[:,:] _comb4_idx_ = comb4_idx
#   cdef CFLOAT_t[:,:] _comb4_scr_ = comb4_scr

#   ret = {}
#   for irow in range(n0):
#     if _ns0_[irow]<min_n: continue

#     jj = 0
#     do_sort = 0
#     for ii in range(_indptr_[irow],_indptr_[irow+1]):
#       icol = _indices_[ii]
#       if _ns0_[irow] <= _ns1_[icol]: continue
#       _idxs1_[jj] = icol
#       _pns1_[jj] = _ns1_[icol]
#       jj += 1
#       if jj>=nmax:
#         do_sort = 1
#         break

#     if do_sort:
#       _row = scrtot.getrow(irow)
#       isort = np.argsort(_row.data)[::-1]
#       cols = _row.indices[isort]
#       jj = 0
#       for cc in cols:
#         if _ns0_[irow] <= _ns1_[cc]: continue

#         _idxs1_[jj] = cc
#         _pns1_[jj] = _ns1_[cc]
#         jj += 1
#         if jj>=nmax: break

#     if jj < 2: continue
#     _score_combinations(_Es0_, _As0_, _csum0_[irow], _ns0_[irow],
#                         _Es1_, _As1_, _ijks1_, _csum1_, _idxs1_, _pns1_, jj,
#                         _mtcA_, _mtcB_, _comb2_idx_, _comb2_scr_, _comb3_idx_, _comb3_scr_, _comb4_idx_, _comb4_scr_,
#                         &cnt2, &cnt3, &cnt4, max_dE, thr_A0, thr_A1, comb_n, min_imp, min_tmp, max_dr)

#     tmp = []  
#     for ii in range(cnt2):
#       if (_comb2_scr_[ii,0] < min_scr) or (_comb2_scr_[ii,1] < min_dig): continue
#       tmp.append(((_comb2_idx_[ii,0],_comb2_idx_[ii,1]),_comb2_scr_[ii,0]))
#     for ii in range(cnt3):
#       if (_comb3_scr_[ii,0] < min_scr) or (_comb3_scr_[ii,1] < min_dig): continue
#       tmp.append(((_comb3_idx_[ii,0],_comb3_idx_[ii,1],_comb3_idx_[ii,2]),_comb3_scr_[ii,0]))
#     for ii in range(cnt4):
#       if (_comb4_scr_[ii,0] < min_scr) or (_comb4_scr_[ii,1] < min_dig): continue
#       tmp.append(((_comb4_idx_[ii,0],_comb4_idx_[ii,1],_comb4_idx_[ii,2],_comb4_idx_[ii,3]),_comb4_scr_[ii,0]))

#     if len(tmp)>0: ret[irow] = tmp

#   return ret
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def score_combinations(CFLOAT_t[:] _Es0_, CFLOAT_t[:] _As0_, CINT32_t[:] _ns0_,
                       CFLOAT_t[:] _Es1_, CFLOAT_t[:] _As1_, CINT32_t[:] _ns1_, ijks1,
                       scrtot, CINT32_t nmax, Emax, CFLOAT_t max_dE, CFLOAT_t thr_A0, CFLOAT_t thr_A1,
                       CFLOAT_t density, CINT32_t comb_n, CINT32_t min_n, CFLOAT_t min_scr, CFLOAT_t min_dig, CFLOAT_t min_imp, CFLOAT_t min_tmp, CFLOAT_t max_dr, CUINT8_t best):

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
  cdef CFLOAT_t[:] _data_ = scrtot.data

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

  cdef CUINT16_t _n0, _n1, _nc, _nl, _nm
  cdef CINT32_t nsort, n_tmp2, n_tmp3, n_tmp4, n_ret2, n_ret3, n_ret4, ic2=0, ic3=0, ic4=0
  cdef CINT32_t irow, icol, ii0, ic, ii, jj, cnt2, cnt3, cnt4, idx
  cdef CFLOAT_t dr

  n_tmp2 = nmax*(nmax-1)//2
  comb2_idx = np.zeros((n_tmp2,2), dtype=CINT32)
  comb2_scr = np.zeros((n_tmp2,4), dtype=CFLOAT)
  cdef CINT32_t[:,:] _comb2_idx_ = comb2_idx
  cdef CFLOAT_t[:,:] _comb2_scr_ = comb2_scr

  n_ret2 = n0*<CINT32_t>density
  ret2_idx = np.zeros((n_ret2,3), dtype=CINT32)
  ret2_scr = np.zeros(n_ret2, dtype=CFLOAT)
  ret2_fcl = np.zeros(n_ret2, dtype=CUINT16)
  cdef CINT32_t[:,:] _ret2_idx_ = ret2_idx
  cdef CFLOAT_t[:] _ret2_scr_ = ret2_scr
  cdef CUINT16_t[:] _ret2_fcl_ = ret2_fcl

  n_tmp3 = nmax*nmax*(nmax-1)//2 if comb_n>=3 else 1
  comb3_idx = np.zeros((n_tmp3,4), dtype=CINT32)
  comb3_scr = np.zeros((n_tmp3,4), dtype=CFLOAT)
  cdef CINT32_t[:,:] _comb3_idx_ = comb3_idx
  cdef CFLOAT_t[:,:] _comb3_scr_ = comb3_scr

  n_ret3 = n0*<CINT32_t>density if comb_n>=3 else 1
  ret3_idx = np.zeros((n_ret3,5), dtype=CINT32)
  ret3_scr = np.zeros(n_ret3, dtype=CFLOAT)
  ret3_fcl = np.zeros(n_ret3, dtype=CUINT16)
  cdef CINT32_t[:,:] _ret3_idx_ = ret3_idx
  cdef CFLOAT_t[:] _ret3_scr_ = ret3_scr
  cdef CUINT16_t[:] _ret3_fcl_ = ret3_fcl

  n_tmp4 = nmax*nmax*(nmax-1)//2 if comb_n>=4 else 1
  comb4_idx = np.zeros((n_tmp4,4), dtype=CINT32)
  comb4_scr = np.zeros((n_tmp4,4), dtype=CFLOAT)
  cdef CINT32_t[:,:] _comb4_idx_ = comb4_idx
  cdef CFLOAT_t[:,:] _comb4_scr_ = comb4_scr

  n_ret4 = n0*<CINT32_t>density if comb_n>=4 else 1
  ret4_idx = np.zeros((n_ret4,4), dtype=CINT32)
  ret4_scr = np.zeros(n_ret4, dtype=CFLOAT)
  ret4_fcl = np.zeros(n_ret4, dtype=CUINT16)
  cdef CINT32_t[:,:] _ret4_idx_ = ret4_idx
  cdef CFLOAT_t[:] _ret4_scr_ = ret4_scr
  cdef CUINT16_t[:] _ret4_fcl_ = ret4_fcl

  used = np.zeros(n1, dtype=CUINT8)
  cdef CUINT8_t[:] _used_ = used

  val_sort = np.zeros(n1, dtype=CFLOAT)
  cdef CFLOAT_t[:] _val_sort_ = val_sort

  idx_sort = np.zeros(nmax*nmax*(nmax-1)//2, dtype=CINT32)
  cdef CINT32_t[:] _idx_sort_ = idx_sort

  for irow in range(n0):
    if _ns0_[irow]<min_n: continue

    ii0 = _indptr_[irow]
    nsort = _indptr_[irow+1]-ii0
    if nsort < 2: continue

    if nsort <= nmax:
      for jj,ii in enumerate(range(_indptr_[irow],_indptr_[irow+1])):
        _idxs1_[jj] = _indices_[ii]
        _pns1_[jj] = _ns1_[_indices_[ii]]

    else:
      for ii in range(nsort):
        _val_sort_[ii] = _data_[ii0+ii]
        _idx_sort_[ii] = ii
      argsort(_val_sort_, _idx_sort_, nsort)

      for ii in range(nmax):
        ic = _idx_sort_[nsort-ii-1]
        _idxs1_[ii] = _indices_[ii0+ic]
        _pns1_[ii] = _ns1_[_indices_[ii0+ic]]
      nsort = nmax

    # for ii in range(_indptr_[irow],_indptr_[irow+1]):
    #   icol = _indices_[ii]
    #   if _ns0_[irow] <= _ns1_[icol]: continue
    #   _idxs1_[jj] = icol
    #   _pns1_[jj] = _ns1_[icol]
    #   jj += 1
    #   if jj>=nmax:
    #     do_sort = 1
    #     break

    # if do_sort:
    #   _row = scrtot.getrow(irow)
    #   isort = np.argsort(_row.data)[::-1]
    #   cols = _row.indices[isort]
    #   jj = 0
    #   for cc in cols:
    #     if _ns0_[irow] <= _ns1_[cc]: continue

    #     _idxs1_[jj] = cc
    #     _pns1_[jj] = _ns1_[cc]
    #     jj += 1
    #     if jj>=nmax: break

    _score_combinations(_Es0_, _As0_, _csum0_[irow], _ns0_[irow],
                        _Es1_, _As1_, _ijks1_, _csum1_, _idxs1_, _pns1_, nsort,
                        _mtcA_, _mtcB_, _comb2_idx_, _comb2_scr_, _comb3_idx_, _comb3_scr_, _comb4_idx_, _comb4_scr_,
                        &cnt2, &cnt3, &cnt4, max_dE, thr_A0, thr_A1, comb_n, min_imp, min_tmp, max_dr)
    _nl = _ns0_[irow]
    for ii in range(n1): _used_[ii] = 0

    for ii in range(cnt2): _idx_sort_[ii] = ii
    argsort(_comb2_scr_[:,0], _idx_sort_, cnt2)
    for ii in range(cnt2):
      idx = _idx_sort_[cnt2-ii-1]
      if _comb2_scr_[idx,0] < min_scr: break
      if _comb2_scr_[idx,1] < min_dig: continue
      if best and (_used_[_comb2_idx_[idx,0]] or _used_[_comb2_idx_[idx,1]]): continue

      _nc = <CUINT16_t>_comb2_scr_[idx,2]
      _n0 = min(_nc, _nl)
      _n1 = max(_nc, _nl)
      _nm = <CUINT16_t>_comb2_scr_[idx,3]

      _ret2_idx_[ic2,0] = irow
      _ret2_idx_[ic2,1] = _comb2_idx_[idx,0]
      _ret2_idx_[ic2,2] = _comb2_idx_[idx,1]
      _ret2_scr_[ic2] = _comb2_scr_[idx,0]
      _ret2_fcl_[ic2] = (min(_n0-1,63)<<10) + (min(_n1-1,63)<<4) + min(_nm,15)
      ic2 += 1
      if ic2 >= n_ret2: return -1

      _used_[_comb2_idx_[idx,0]] = 1
      _used_[_comb2_idx_[idx,1]] = 1

    if comb_n >= 3:
      for ii in range(cnt3): _idx_sort_[ii] = ii
      argsort(_comb3_scr_[:,0], _idx_sort_, cnt3)
      for ii in range(cnt3):
        idx = _idx_sort_[cnt3-ii-1]
        if _comb3_scr_[idx,0] < min_scr: break
        if _comb3_scr_[idx,1] < min_dig: continue
        if best and (_used_[_comb3_idx_[idx,0]] or _used_[_comb3_idx_[idx,1]] or _used_[_comb3_idx_[idx,2]]): continue

        _nc = <CUINT16_t>_comb3_scr_[idx,2]
        _n0 = min(_nc, _nl)
        _n1 = max(_nc, _nl)
        _nm = <CUINT16_t>_comb3_scr_[idx,3]

        _ret3_idx_[ic3,0] = irow
        _ret3_idx_[ic3,1] = _comb3_idx_[idx,0]
        _ret3_idx_[ic3,2] = _comb3_idx_[idx,1]
        _ret3_idx_[ic3,3] = _comb3_idx_[idx,2]
        _ret3_scr_[ic3] = _comb3_scr_[idx,0]
        _ret3_fcl_[ic3] = (min(_n0-1,63)<<10) + (min(_n1-1,63)<<4) + min(_nm,15)
        ic3 += 1
        if ic3 >= n_ret2: return -1

        _used_[_comb3_idx_[idx,0]] = 1
        _used_[_comb3_idx_[idx,1]] = 1
        _used_[_comb3_idx_[idx,2]] = 1

    if comb_n >= 4:
      for ii in range(cnt4): _idx_sort_[ii] = ii
      argsort(_comb4_scr_[:,0], _idx_sort_, cnt4)
      for ii in range(cnt4):
        idx = _idx_sort_[cnt4-ii-1]
        if _comb4_scr_[idx,0] < min_scr: break
        if _comb4_scr_[idx,1] < min_dig: continue
        if best and (_used_[_comb4_idx_[idx,0]] or _used_[_comb4_idx_[idx,1]] or _used_[_comb4_idx_[idx,2]] or _used_[_comb4_idx_[idx,3]]): continue

        _nc = <CUINT16_t>_comb4_scr_[idx,2]
        _n0 = min(_nc, _nl)
        _n1 = max(_nc, _nl)
        _nm = <CUINT16_t>_comb4_scr_[idx,3]

        _ret4_idx_[ic4,0] = irow
        _ret4_idx_[ic4,1] = _comb4_idx_[idx,0]
        _ret4_idx_[ic4,2] = _comb4_idx_[idx,1]
        _ret4_idx_[ic4,3] = _comb4_idx_[idx,2]
        _ret4_idx_[ic4,4] = _comb4_idx_[idx,3]
        _ret4_scr_[ic4] = _comb3_scr_[idx,0]
        _ret4_fcl_[ic4] = (min(_n0-1,63)<<10) + (min(_n1-1,63)<<4) + min(_nm,15)
        ic4 += 1
        if ic4 >= n_ret2: return -1

        _used_[_comb4_idx_[idx,0]] = 1
        _used_[_comb4_idx_[idx,1]] = 1
        _used_[_comb4_idx_[idx,2]] = 1
        _used_[_comb4_idx_[idx,3]] = 1

  return {2: {'matches': ret2_idx[:ic2], 'score': ret2_scr[:ic2], 'fcl': ret2_fcl[:ic2]},
          3: {'matches': ret3_idx[:ic3], 'score': ret3_scr[:ic3], 'fcl': ret3_fcl[:ic3]} if comb_n>=3 else None,
          4: {'matches': ret4_idx[:ic4], 'score': ret4_scr[:ic4], 'fcl': ret4_fcl[:ic4]} if comb_n>=4 else None}
# -------


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def dist_combinations(CFLOAT_t[:] _Es0_, CFLOAT_t[:] _As0_, CINT32_t[:] _ns0_,
                      CFLOAT_t[:] _Es1_, CFLOAT_t[:] _As1_, CINT32_t[:] _ns1_, ijks1,
                      scrtot, CINT32_t nmax, Emax, CFLOAT_t max_dE, CFLOAT_t thr_A0, CFLOAT_t thr_A1,
                      CINT32_t comb_n, CINT32_t min_n, CFLOAT_t min_scr, CFLOAT_t min_dig, CFLOAT_t min_imp, CFLOAT_t min_tmp, CFLOAT_t max_dr):

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

  cdef CINT32_t irow, icol, ii, jj, cnt2, cnt3, cnt4, do_sort, icomb=0
  cdef CUINT16_t _nl, _nc, _n0, _n1, _nm
  cdef CFLOAT_t dr

  comb2_idx = np.zeros((nmax*(nmax-1)//2,2), dtype=CINT32)
  comb2_scr = np.zeros((nmax*(nmax-1)//2,4), dtype=CFLOAT)
  cdef CINT32_t[:,:] _comb2_idx_ = comb2_idx
  cdef CFLOAT_t[:,:] _comb2_scr_ = comb2_scr

  comb3_idx = np.zeros((nmax*nmax*(nmax-1)//2,3), dtype=CINT32)
  comb3_scr = np.zeros((nmax*nmax*(nmax-1)//2,4), dtype=CFLOAT)
  cdef CINT32_t[:,:] _comb3_idx_ = comb3_idx
  cdef CFLOAT_t[:,:] _comb3_scr_ = comb3_scr

  comb4_idx = np.zeros((nmax*nmax*(nmax-1)//2,4), dtype=CINT32)
  comb4_scr = np.zeros((nmax*nmax*(nmax-1)//2,4), dtype=CFLOAT)
  cdef CINT32_t[:,:] _comb4_idx_ = comb4_idx
  cdef CFLOAT_t[:,:] _comb4_scr_ = comb4_scr

  cmax = n0*(comb2_scr.shape[0])
  # if comb_n>=3: cmax += comb3_scr.shape[0]
  # if comb_n>=4: cmax += comb4_scr.shape[0]

  scrs = np.zeros(cmax, dtype=CFLOAT)
  cdef CFLOAT_t[:] _scrs_ = scrs
  fcls = np.zeros(cmax, dtype=CUINT16)
  cdef CUINT16_t[:] _fcls_ = fcls

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
                        &cnt2, &cnt3, &cnt4, max_dE, thr_A0, thr_A1, comb_n, min_imp, min_tmp, max_dr)

    _nl = <CUINT16_t>_ns0_[irow]
    for ii in range(cnt2):
      if (_comb2_scr_[ii,0] < min_scr) or (_comb2_scr_[ii,1] < min_dig): continue
      _scrs_[icomb] = _comb2_scr_[ii,0]

      _nc = <CUINT16_t>_comb2_scr_[ii,2]
      _n0 = min(_nc, _nl)
      _n1 = max(_nc, _nl)
      _nm = <CUINT16_t>_comb2_scr_[ii,3]
      _fcls_[icomb] = (min(_n0-1,63)<<10) + (min(_n1-1,63)<<4) + min(_nm,15)
      
      icomb += 1
      if icomb>=cmax: return None
    
    for ii in range(cnt3):
      if (_comb3_scr_[ii,0] < min_scr) or (_comb3_scr_[ii,1] < min_dig): continue

      _scrs_[icomb] = _comb3_scr_[ii,0]

      _nc = <CUINT16_t>_comb3_scr_[ii,2]
      _n0 = min(_nc, _nl)
      _n1 = max(_nc, _nl)
      _nm = <CUINT16_t>_comb3_scr_[ii,3]
      _fcls_[icomb] = (min(_n0-1,63)<<10) + (min(_n1-1,63)<<4) + min(_nm,15)
      
      icomb += 1
      if icomb>=cmax: return None


    for ii in range(cnt4):
      if (_comb4_scr_[ii,0] < min_scr) or (_comb4_scr_[ii,1] < min_dig): continue

      _scrs_[icomb] = _comb4_scr_[ii,0]

      _nc = <CUINT16_t>_comb4_scr_[ii,2]
      _n0 = min(_nc, _nl)
      _n1 = max(_nc, _nl)
      _nm = <CUINT16_t>_comb4_scr_[ii,3]
      _fcls_[icomb] = (min(_n0-1,63)<<10) + (min(_n1-1,63)<<4) + min(_nm,15)
      
      icomb += 1
      if icomb>=cmax: return None

  return scrs[:icomb],fcls[:icomb]
# -------


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32_t _score_combinations(CFLOAT_t[:] Es0, CFLOAT_t[:] As0, CINT32_t i0, CINT32_t n0,
                                  CFLOAT_t[:] Es1, CFLOAT_t[:] As1, CFLOAT_t[:,:] ijks1, CINT32_t[:] csum1, CINT32_t[:] i1s, CINT32_t[:] n1s, CINT32_t len1,
                                  CINT32_t[:] mtcA, CINT32_t[:] mtcB, CINT32_t[:,:] comb2_idx, CFLOAT_t[:,:] comb2_scr,
                                  CINT32_t[:,:] comb3_idx, CFLOAT_t[:,:] comb3_scr, CINT32_t[:,:] comb4_idx, CFLOAT_t[:,:] comb4_scr,
                                  CINT32_t *cnt2, CINT32_t *cnt3, CINT32_t *cnt4, CFLOAT_t max_dE, CFLOAT_t thr_A0, CFLOAT_t thr_A1,
                                  CINT32_t comb_n, CFLOAT_t min_imp, CFLOAT_t min_tmp, CFLOAT_t max_dr):

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
    _align_lines(Es0, As0, i0, n0, Es1, As1, csum1[i1s[ii]], n1s[ii], mtcA, mtcB,
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
      for jj in range(len1):
        if i1s[jj] <= i1s[ii]: continue

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
           (scr>comb1_scr[ii]+min_imp) and (scr>comb1_scr[jj]+min_imp) and (scr>min_tmp):
          comb2_idx[cnt2[0],0] = i1s[ii]
          comb2_idx[cnt2[0],1] = i1s[jj]
          comb2_scr[cnt2[0],0] = scr
          comb2_scr[cnt2[0],1] = dig
          comb2_scr[cnt2[0],2] = <CFLOAT_t>(n1s[ii]+n1s[jj])
          comb2_scr[cnt2[0],3] = <CFLOAT_t>(n0) + comb2_scr[cnt2[0],2] - wM
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
        if i1s[jj] <= comb2_idx[ii,1]: continue

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
           (scr>comb2_scr[ii,0]+min_imp) and (scr>comb1_scr[jj]+min_imp) and (scr>min_tmp):
          comb3_idx[cnt3[0],0] = comb2_idx[ii,0]
          comb3_idx[cnt3[0],1] = comb2_idx[ii,1]
          comb3_idx[cnt3[0],2] = i1s[jj]
          comb3_scr[cnt3[0],0] = scr
          comb3_scr[cnt3[0],1] = dig
          comb3_scr[cnt3[0],2] = comb2_scr[ii,2] + <CFLOAT_t>(n1s[jj])
          comb3_scr[cnt3[0],3] = <CFLOAT_t>(n0) + comb3_scr[cnt3[0],2] - wM
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
        if i1s[jj] <= comb3_idx[ii,2]: continue

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
           (scr>comb3_scr[ii,0]+min_imp) and (scr>comb1_scr[jj]+min_imp) and (scr>min_tmp):
          comb4_idx[cnt4[0],0] = comb3_idx[ii,0]
          comb4_idx[cnt4[0],1] = comb3_idx[ii,1]
          comb4_idx[cnt4[0],2] = comb3_idx[ii,2]
          comb4_idx[cnt4[0],3] = i1s[jj]
          comb4_scr[cnt4[0],0] = scr
          comb4_scr[cnt4[0],1] = dig
          comb4_scr[cnt4[0],2] = comb3_scr[ii,2] + <CFLOAT_t>(n1s[jj])
          comb4_scr[cnt4[0],3] = <CFLOAT_t>(n0) + comb4_scr[cnt4[0],2] - wM
          cnt4[0] += 1

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
# -------

# # --------------------------------- PARTITION ----------------------------------
# # ------------------------------------------------------------------------------
# # ------------------------------------------------------------------------------

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def jaccard64_cross_symm(CUINT64_t[:] bin0A, CUINT64_t[:] bin0B, CUINT64_t[:] bin1A, CUINT64_t[:] bin1B,
#                          CINT32_t i0_start, CINT32_t ntot0, CFLOAT_t density_symm, CFLOAT_t min_simil_symm):

#   cdef CINT32_t len0 = bin0A.size
#   assert len0 == bin0B.size
#   cdef CINT32_t ntot1 = bin1A.size
#   assert ntot1 == bin1B.size
#   cdef CUINT64_t nMax = CUINT64(np.ceil(density_symm*len0*ntot1))

#   indptr01 = np.zeros(ntot0+1, dtype=CUINT64)
#   cdef CUINT64_t[:] _indptr01_ = indptr01
#   indices01 = np.full(nMax, -1, dtype=CINT32)
#   cdef CINT32_t[:] _indices01_ = indices01
#   data01 = np.full_like(indices01, np.nan, dtype=CUINT8)
#   cdef CUINT8_t[:] _data01_ = data01

#   w0A = _weights64(bin0A)
#   cdef CFLOAT_t[:] _w0A_ = w0A
#   w0B = _weights64(bin0B)
#   cdef CFLOAT_t[:] _w0B_ = w0B
#   w1A = _weights64(bin1A)
#   cdef CFLOAT_t[:] _w1A_ = w1A
#   w1B = _weights64(bin1B)
#   cdef CFLOAT_t[:] _w1B_ = w1B

#   cdef CUINT64_t iseq01 = 0
#   cdef CFLOAT_t sumA, sumB, temp
#   cdef CINT32_t i, j

#   for i in range(len0):
#     for j in range(ntot1):
#       sumA = bits_count(bin0A[i]&bin1A[j])
#       sumB = bits_count(bin0B[i]&bin1B[j])

#       temp = max(sumA/(_w0A_[i]+_w1A_[j]-sumA), sumB/(_w0B_[i]+_w1B_[j]-sumB))
#       if temp >= min_simil_symm:
#         _data01_[iseq01] = <CUINT8_t>(255.*temp)
#         _indices01_[iseq01] = j
#         iseq01 += 1
#         if iseq01 >= nMax: return None

#     _indptr01_[i0_start+i+1] = iseq01

#   _indptr01_[i0_start+i+1:] = iseq01

#   return csr_matrix((data01[:iseq01], indices01[:iseq01], indptr01), (ntot0, ntot1))

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def jaccard64_self_symm(CUINT64_t[:] binA, CUINT64_t[:] binB, CINT32_t i0_start, CINT32_t i0_stop,
#                         CFLOAT_t density_symm, CFLOAT_t min_simil_symm):

#   cdef CINT32_t ntot = binA.size
#   assert ntot == binB.size
#   assert i0_stop > i0_start
#   cdef CUINT64_t nMax = CUINT64(np.ceil(density_symm*(ntot-i0_start)*(i0_stop-i0_start)))

#   indptr01 = np.zeros(ntot+1, dtype=CUINT64)
#   cdef CUINT64_t[:] _indptr01_ = indptr01
#   indices01 = np.full(nMax, -1, dtype=CINT32)
#   cdef CINT32_t[:] _indices01_ = indices01
#   data01 = np.full_like(indices01, np.nan, dtype=CUINT8)
#   cdef CUINT8_t[:] _data01_ = data01

#   wA = _weights64(binA)
#   cdef CFLOAT_t[:] _wA_ = wA
#   wB = _weights64(binB)
#   cdef CFLOAT_t[:] _wB_ = wB

#   cdef CUINT64_t iseq01 = 0
#   cdef CFLOAT_t sumA, sumB, temp
#   cdef CINT32_t i, j

#   for i in range(i0_start,i0_stop):
#     for j in range(i+1,ntot):
#       sumA = bits_count(binA[i]&binA[j])
#       sumB = bits_count(binB[i]&binB[j])

#       temp = max(sumA/(_wA_[i]+_wA_[j]-sumA), sumB/(_wB_[i]+_wB_[j]-sumB))
#       if temp >= min_simil_symm:
#         _data01_[iseq01] = <CUINT8_t>(255.*temp)
#         _indices01_[iseq01] = j
#         iseq01 += 1
#         if iseq01 >= nMax: return None

#     _indptr01_[i+1] = iseq01

#   _indptr01_[i+1:] = iseq01

#   return csr_matrix((data01[:iseq01], indices01[:iseq01], indptr01), (ntot, ntot))

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def jaccard64_cross_all(CUINT64_t[:] bin0A, CUINT64_t[:] bin0B, CUINT64_t[:] bin1A, CUINT64_t[:] bin1B,
#                         CINT32_t[:] ns0, CINT32_t[:] ns1, CINT32_t i0_start, CINT32_t ntot0,
#                         CFLOAT_t density_symm, CFLOAT_t density_comb, CFLOAT_t min_simil_symm, CFLOAT_t min_simil_comb, CINT32_t m0, CINT32_t mrat):

#   cdef CINT32_t len0 = bin0A.size
#   assert len0 == bin0B.size
#   cdef CINT32_t ntot1 = bin1A.size
#   assert ntot1 == bin1B.size
#   cdef CUINT64_t nMax_symm = CUINT64(np.ceil(density_symm*len0*ntot1))
#   cdef CUINT64_t nMax_comb = CUINT64(np.ceil(density_comb*len0*ntot1))

#   indptr01 = np.zeros(ntot0+1, dtype=CUINT64)
#   cdef CUINT64_t[:] _indptr01_ = indptr01
#   indptr0in1 = np.zeros(ntot0+1, dtype=CUINT64)
#   cdef CUINT64_t[:] _indptr0in1_ = indptr0in1
#   indptr1in0 = np.zeros(ntot0+1, dtype=CUINT64)
#   cdef CUINT64_t[:] _indptr1in0_ = indptr1in0

#   indices01 = np.full(nMax_symm, -1, dtype=CINT32)
#   cdef CINT32_t[:] _indices01_ = indices01
#   indices0in1 = np.full(nMax_comb, -1, dtype=CINT32)
#   cdef CINT32_t[:] _indices0in1_ = indices0in1
#   indices1in0 = np.full(nMax_comb, -1, dtype=CINT32)
#   cdef CINT32_t[:] _indices1in0_ = indices1in0

#   data01 = np.full_like(indices01, 0, dtype=CUINT8)
#   cdef CUINT8_t[:] _data01_ = data01
#   data0in1 = np.full_like(indices0in1, 0, dtype=CUINT8)
#   cdef CUINT8_t[:] _data0in1_ = data0in1
#   data1in0 = np.full_like(indices1in0, 0, dtype=CUINT8)
#   cdef CUINT8_t[:] _data1in0_ = data1in0

#   w0A = _weights64(bin0A)
#   cdef CFLOAT_t[:] _w0A_ = w0A
#   w0B = _weights64(bin0B)
#   cdef CFLOAT_t[:] _w0B_ = w0B
#   w1A = _weights64(bin1A)
#   cdef CFLOAT_t[:] _w1A_ = w1A
#   w1B = _weights64(bin1B)
#   cdef CFLOAT_t[:] _w1B_ = w1B

#   cdef CUINT64_t iseq01 = 0, iseq0in1 = 0, iseq1in0 = 0
#   cdef CFLOAT_t sumA, sumB, temp
#   cdef CINT32_t i, j, mmin

#   for i in range(len0):
#     for j in range(ntot1):
#       sumA = bits_count(bin0A[i]&bin1A[j])
#       sumB = bits_count(bin0B[i]&bin1B[j])

#       temp = max(sumA/(_w0A_[i]+_w1A_[j]-sumA), sumB/(_w0B_[i]+_w1B_[j]-sumB))
#       if temp >= min_simil_symm:
#         _data01_[iseq01] = <CUINT8_t>(255.*temp)
#         _indices01_[iseq01] = j
#         iseq01 += 1
#         if iseq01 >= nMax_symm: return None
      
#       mmin = ns1[j]//mrat
#       if ns1[j]%mrat>0: mmin += 1
#       if (ns1[j]>=m0) and (ns0[i]>=mmin):
#         temp = max(sumA/_w0A_[i], sumB/_w0B_[i])
#         if temp >= min_simil_comb:
#           _data0in1_[iseq0in1] = <CUINT8_t>(255.*temp)
#           _indices0in1_[iseq0in1] = j
#           iseq0in1 += 1
#           if iseq0in1 >= nMax_comb: return None

#       mmin = ns0[i]//mrat
#       if ns0[i]%mrat>0: mmin += 1
#       if (ns0[i]>=m0) and (ns1[j]>=mmin):
#         temp = max(sumA/_w1A_[j], sumB/_w1B_[j])
#         if temp >= min_simil_comb:
#           _data1in0_[iseq1in0] = <CUINT8_t>(255.*temp)
#           _indices1in0_[iseq1in0] = j
#           iseq1in0 += 1
#           if iseq1in0 >= nMax_comb: return None

#     _indptr01_[i0_start+i+1] = iseq01
#     _indptr0in1_[i0_start+i+1] = iseq0in1
#     _indptr1in0_[i0_start+i+1] = iseq1in0

#   _indptr01_[i0_start+i+1:] = iseq01
#   _indptr0in1_[i0_start+i+1:] = iseq0in1
#   _indptr1in0_[i0_start+i+1:] = iseq1in0

#   return csr_matrix((data01[:iseq01], indices01[:iseq01], indptr01), (ntot0, ntot1)),\
#          csr_matrix((data0in1[:iseq0in1], indices0in1[:iseq0in1], indptr0in1), (ntot0, ntot1)),\
#          csr_matrix((data1in0[:iseq1in0], indices1in0[:iseq1in0], indptr1in0), (ntot0, ntot1))


# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def jaccard64_self_all(CUINT64_t[:] binA, CUINT64_t[:] binB, CINT32_t[:] ns, CINT32_t i0_start, CINT32_t i0_stop,
#                        CFLOAT_t density_symm, CFLOAT_t density_comb, CFLOAT_t min_simil_symm, CFLOAT_t min_simil_comb, CINT32_t m0, CINT32_t mrat):

#   cdef CINT32_t ntot = binA.size
#   assert ntot == binB.size
#   assert i0_stop > i0_start
#   cdef CUINT64_t nMax_symm = CUINT64(np.ceil(density_symm*(ntot-i0_start)*(i0_stop-i0_start))/2)
#   cdef CUINT64_t nMax_comb = CUINT64(np.ceil(density_comb*(ntot-i0_start)*(i0_stop-i0_start))/2)

#   indptr01 = np.zeros(ntot+1, dtype=CUINT64)
#   cdef CUINT64_t[:] _indptr01_ = indptr01
#   indptr0in1 = np.zeros(ntot+1, dtype=CUINT64)
#   cdef CUINT64_t[:] _indptr0in1_ = indptr0in1
#   indptr1in0 = np.zeros(ntot+1, dtype=CUINT64)
#   cdef CUINT64_t[:] _indptr1in0_ = indptr1in0
  
#   indices01 = np.full(nMax_symm, -1, dtype=CINT32)
#   cdef CINT32_t[:] _indices01_ = indices01
#   indices0in1 = np.full(nMax_comb, -1, dtype=CINT32)
#   cdef CINT32_t[:] _indices0in1_ = indices0in1
#   indices1in0 = np.full(nMax_comb, -1, dtype=CINT32)
#   cdef CINT32_t[:] _indices1in0_ = indices1in0

#   data01 = np.full_like(indices01, 0, dtype=CUINT8)
#   cdef CUINT8_t[:] _data01_ = data01
#   data0in1 = np.full_like(indices0in1, 0, dtype=CUINT8)
#   cdef CUINT8_t[:] _data0in1_ = data0in1
#   data1in0 = np.full_like(indices1in0, 0, dtype=CUINT8)
#   cdef CUINT8_t[:] _data1in0_ = data1in0

#   wA = _weights64(binA)
#   cdef CFLOAT_t[:] _wA_ = wA
#   wB = _weights64(binB)
#   cdef CFLOAT_t[:] _wB_ = wB

#   cdef CUINT64_t iseq01 = 0, iseq0in1 = 0, iseq1in0 = 0
#   cdef CFLOAT_t sumA, sumB, temp
#   cdef CINT32_t i, j, mmin

#   for i in range(i0_start, i0_stop):
#     for j in range(i+1,ntot):
#       sumA = bits_count(binA[i]&binA[j])
#       sumB = bits_count(binB[i]&binB[j])

#       temp = max(sumA/(_wA_[i]+_wA_[j]-sumA), sumB/(_wB_[i]+_wB_[j]-sumB))
#       if temp >= min_simil_symm:
#         _data01_[iseq01] = <CUINT8_t>(255.*temp)
#         _indices01_[iseq01] = j
#         iseq01 += 1
#         if iseq01 >= nMax_symm: return None

#       mmin = ns[j]//mrat
#       if ns[j]%mrat>0: mmin += 1
#       if (ns[j]>=m0) and (ns[i]>=mmin):
#         temp = max(sumA/(_wA_[i]), sumB/(_wB_[i]))
#         if temp >= min_simil_comb:
#           _data0in1_[iseq0in1] = <CUINT8_t>(255.*temp)
#           _indices0in1_[iseq0in1] = j
#           iseq0in1 += 1
#           if iseq0in1 >= nMax_comb: return None

#       mmin = ns[i]//mrat
#       if ns[i]%mrat>0: mmin += 1
#       if (ns[i]>=m0) and (ns[j]>=mmin):
#         temp = max(sumA/(_wA_[j]), sumB/(_wB_[j]))
#         if temp >= min_simil_comb:
#           _data1in0_[iseq1in0] = <CUINT8_t>(255.*temp)
#           _indices1in0_[iseq1in0] = j
#           iseq1in0 += 1
#           if iseq1in0 >= nMax_comb: return None

#     _indptr01_[i+1] = iseq01
#     _indptr0in1_[i+1] = iseq0in1
#     _indptr1in0_[i+1] = iseq1in0

#   _indptr01_[i+1:] = iseq01
#   _indptr0in1_[i+1:] = iseq0in1
#   _indptr1in0_[i+1:] = iseq1in0

#   return csr_matrix((data01[:iseq01], indices01[:iseq01], indptr01), (ntot, ntot)),\
#          csr_matrix((data0in1[:iseq0in1], indices0in1[:iseq0in1], indptr0in1), (ntot, ntot)),\
#          csr_matrix((data1in0[:iseq1in0], indices1in0[:iseq1in0], indptr1in0), (ntot, ntot))

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def score_lines(dataA, dataB, simil, CUINT8_t pmode, opt):
#   cdef CINT32_t lenA=np.int32(len(dataA)), lenB=np.int32(len(dataB))
#   assert lenA == simil.shape[0]
#   assert lenB == simil.shape[1]

#   cdef CFLOAT_t[:] _EsA_ = dataA.Es
#   cdef CFLOAT_t[:] _AsA_ = dataA.As
#   cdef CINT32_t[:] _nsA_ = dataA.ns
#   cumA = np.insert(np.cumsum(dataA.ns,dtype=np.int32),0,0)
#   cdef CINT32_t[:] _cumA_ = cumA

#   cdef CFLOAT_t[:] _EsB_ = dataB.Es
#   cdef CFLOAT_t[:] _AsB_ = dataB.As
#   cdef CINT32_t[:] _nsB_ = dataB.ns
#   cumB = np.insert(np.cumsum(dataB.ns,dtype=np.int32),0,0)
#   cdef CINT32_t[:] _cumB_ = cumB

#   cdef CUINT64_t[:] _indptr_ = simil.indptr.astype(CUINT64)
#   cdef CINT32_t[:] _indices_ = simil.indices

#   scr_data = np.zeros_like(simil.data, dtype=CFLOAT)
#   cdef CFLOAT_t[:] _scr_data_ = scr_data

#   matchedA = np.full(opt.nmax, -1, dtype=CINT32)
#   cdef CINT32_t[:] _matchedA_ = matchedA
#   matchedB = np.full(opt.nmax, -1, dtype=CINT32)
#   cdef CINT32_t[:] _matchedB_ = matchedB

#   cdef CFLOAT_t min_dig = opt.min_dig
#   cdef CFLOAT_t max_dE = opt.max_dE
#   cdef CFLOAT_t thr_A0 = opt.thr_A0
#   cdef CFLOAT_t thr_A1 = opt.thr_A1

#   cdef CUINT64_t ptrB
#   cdef CINT32_t ret, jA, jB, idat, cA0, cB0, iA, iB, nA, nB, iB0
#   cdef CFLOAT_t curr, right, bottom, scr_tot, wM, wT

#   idat = -1
#   for jA in range(lenA):
#     cA0 = _cumA_[jA]
#     for ptrB in range(_indptr_[jA],_indptr_[jA+1]):
#       idat += 1
#       jB = _indices_[ptrB]
#       cB0 = _cumB_[jB]
#       # ----------------
#       nA = _nsA_[jA]
#       nB = _nsB_[jB]
#       for iA in range(nA): _matchedA_[iA] = -1
#       for iB in range(nB): _matchedB_[iB] = -1

#       iA = iB0 = 0
#       scr_tot = 0.
#       doA = doB = True
#       # --- Find aligned lines ---
#       while True:
#         if (iA >= nA) or (iB0 >= nB): break
#         for iB in range(iB0,nB):
#           curr = (_EsB_[cB0+iB]-_EsA_[cA0+iA])/max_dE
#           iB0 = iB
#           if curr>-1.: break
#           doB = True

#         if curr>=1.:
#           iA += 1
#           doA = True
#           continue

#         curr = 1.-abs(curr)

#         if iB<nB-1: right = max(0.,1.-abs(_EsB_[cB0+iB+1]-_EsA_[cA0+iA])/max_dE)
#         else:       right = 0.
#         if iA<nA-1: bottom = max(0.,1.-abs(_EsB_[cB0+iB]-_EsA_[cA0+iA+1])/max_dE)
#         else:       bottom = 0.

#         if doA and (curr>right):
#           _matchedA_[iA] = iB
#           scr_tot += curr
#           doA = False

#         if doB and (curr>bottom):
#           _matchedB_[iB] = iA
#           scr_tot += curr
#           doB = False

#         if (bottom>0.) and (bottom>right):
#           iA += 1
#           doA = True
#           continue

#         if (right>0.) and (right>bottom):
#           iB0 += 1
#           doB = True
#           continue

#         if (bottom>0.) and (bottom==right): return -1

#         iA += 1
#         iB0 += 1
#         doA = doB = True
#         continue

#       # --- Calculate scores ---
#       wT = 0.
#       wM = 0.
#       for iA in range(nA):
#         if _matchedA_[iA]>=0:
#           wT += 1.
#           wM += 1.
#         else:
#           if pmode%2==0: wT += max(0., (min(thr_A1, _AsA_[cA0+iA])-thr_A0)/(thr_A1-thr_A0))

#       for iB in range(nB):
#         if _matchedB_[iB]>=0:
#           wT += 1.
#           wM += 1.
#         else:
#           if pmode>0: wT += max(0., (min(thr_A1, _AsB_[cB0+iB])-thr_A0)/(thr_A1-thr_A0))

#       if wM/wT < min_dig: continue
#       _scr_data_[idat] = scr_tot/wT

#   mat_scr = csr_matrix((scr_data, simil.indices.copy(), simil.indptr.copy()), (lenA,lenB))
#   mat_scr.eliminate_zeros()

#   return mat_scr
# # -------
