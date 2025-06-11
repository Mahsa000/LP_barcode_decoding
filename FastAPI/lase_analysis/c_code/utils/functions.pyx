cimport cython

cimport numpy as np
import numpy as np
import pandas as pd
from libc.stdlib cimport malloc, free

from scipy.sparse import csr_matrix, coo_matrix


EPS = 1e-6

CINT32 = np.int32
CUINT64 = np.uint64
CFLOAT64 = np.float64
ctypedef np.int32_t CINT32_t
ctypedef np.float64_t CFLOAT64_t
ctypedef np.uint64_t CUINT64_t


cdef extern from "stdlib.h":
  ctypedef void const_void "const void"
  void qsort(void *base, int nmemb, int size, int(*compar)(const_void *, const_void *)) nogil
# -------

cdef struct IndexedElement:
    CINT32_t index
    CFLOAT64_t value
# -------

cdef int _compare(const_void *a, const_void *b) noexcept:
    cdef CFLOAT64_t v = (<IndexedElement*> a).value-(<IndexedElement*> b).value
    if v < 0: return -1
    if v >= 0: return 1
# -------

cdef argsort(CFLOAT64_t[:] data, CINT32_t[:] idxs, CINT32_t n):
    cdef CINT32_t ii
    
    # Allocate index tracking array.
    cdef IndexedElement *order_struct = <IndexedElement *> malloc(n * sizeof(IndexedElement))
    
    # Copy data into index tracking array.
    for ii in range(n):
        order_struct[ii].index = ii
        order_struct[ii].value = data[ii]
        
    # Sort index tracking array.
    qsort(<void *> order_struct, n, sizeof(IndexedElement), _compare)
    
    # Copy indices from index tracking array to output array.
    for ii in range(n):
        idxs[ii] = order_struct[ii].index
        
    # Free index tracking array.
    free(order_struct)
# -------

cdef CFLOAT64_t time_overlap(CFLOAT64_t tA0, CFLOAT64_t tA1, CFLOAT64_t tB0, CFLOAT64_t tB1):
  return max(0.,(min(tA1,tB1)-max(tA0,tB0)))/min(tA1-tA0,tB1-tB0)
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def ideal_scoring(CFLOAT64_t[:,:] Es, CFLOAT64_t density, CFLOAT64_t max_dE):
  cdef CINT32_t m=Es.shape[1], n=Es.shape[0]

  cdef CUINT64_t nMax = CUINT64(np.ceil(density*n*n))

  indptr = np.zeros(n+1, dtype=CUINT64)
  cdef CUINT64_t[:] _indptr_ = indptr
  indices = np.full(nMax, -1, dtype=CINT32)
  cdef CINT32_t[:] _indices_ = indices
  data_scr = np.full_like(indices, 0, dtype=CFLOAT64)
  cdef CFLOAT64_t[:] _data_scr_ = data_scr

  cdef CUINT64_t iseq=0
  cdef CFLOAT64_t dE, scr
  cdef CINT32_t i, j, k

  for i in range(n):
    for j in range(i+1,n):
      scr = 0.
      for k in range(m):
        dE = abs(Es[i,k]-Es[j,k])
        if dE>max_dE: break
        scr += dE/(m*max_dE)
      else:
        _data_scr_[iseq] = scr
        _indices_[iseq] = j
        iseq += 1
        if iseq >= nMax: return -2

    _indptr_[i+1] = iseq
  _indptr_[i+1:] = iseq

  return csr_matrix((data_scr[:iseq], indices[:iseq], indptr), (n,n))
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def pks2limgs(CINT32_t[:,:] crds_view, CINT32_t[:] crd0_view, CINT32_t[:] crd1_view,
              CFLOAT64_t[:] Es_view, CFLOAT64_t[:] As_view):

  assert len(Es_view) == crds_view.shape[0]
  assert len(Es_view) == len(As_view)

  cdef CINT32_t k0,k1,j0,j1,i0,i1
  (i0,j0,k0) = crd0_view
  (i1,j1,k1) = crd1_view

  cdef CINT32_t nk=k1-k0+1, nj=j1-j0+1, ni=i1-i0+1
  cdef CINT32_t i, ci, cj, ck, ndat=len(Es_view)

  imgA = np.full((nk,nj,ni), 0., dtype=CFLOAT64)
  cdef CFLOAT64_t[:,:,:] imgA_view = imgA

  imgE = np.full_like(imgA, np.nan, dtype=CFLOAT64)
  cdef CFLOAT64_t[:,:,:] imgE_view = imgE

  pos = np.full(ndat, -1, dtype=CINT32)
  cdef CINT32_t[:] pos_view = pos

  for i in range(ndat):
    ci = crds_view[i,0]-i0
    cj = crds_view[i,1]-j0
    ck = crds_view[i,2]-k0

    if imgA_view[ck,cj,ci]==0.:
      imgA_view[ck,cj,ci] = As_view[i]
      imgE_view[ck,cj,ci] = Es_view[i]
      pos_view[i] = ci+cj*ni+ck*nj*ni

  return imgE, imgA, pos
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def line_ovlp(CFLOAT64_t[:,:] mat, Py_ssize_t max_out):
  assert mat.shape[1] == 2
  cdef Py_ssize_t len0 = mat.shape[0]
  cdef Py_ssize_t max_len = len0*10*max_out

  rows = np.zeros(max_len, dtype=np.intp)
  cdef Py_ssize_t[:] rows_view = rows
  cols = np.zeros(max_len, dtype=np.intp)
  cdef Py_ssize_t[:] cols_view = cols
  vals = np.zeros(max_len, dtype=CFLOAT64)
  cdef CFLOAT64_t[:] vals_view = vals

  cdef Py_ssize_t ir, ic, j, out
  cdef CFLOAT64_t ovlp, eps
  eps = EPS

  j = 0
  for ir in range(len0-1):
    out = 0
    for ic in range(ir+1,len0):
      ovlp = time_overlap(mat[ir,0], mat[ir,1], mat[ic,0], mat[ic,1])
      if ovlp > eps:
        vals_view[j] = ovlp
        vals_view[j+1] = ovlp
        rows_view[j] = ir
        rows_view[j+1] = ic
        cols_view[j] = ic
        cols_view[j+1] = ir
        j += 2
      else:
        out += 1

      if out > max_out: break

  coo = coo_matrix((vals[:j], (rows[:j], cols[:j])), shape=(len0,len0))

  return coo.tocsr()
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def follow_lines(lines, CFLOAT64_t max_dE, CFLOAT64_t max_dt):
  lines = lines.sort_values(by='E', axis=0)

  # ------------------------------- Definitions --------------------------------
  Es = lines.E.values
  cdef CFLOAT64_t[:] _Es_ = Es

  indexes = lines.index.values.astype(np.uint64)
  cdef np.uint64_t[:] _indexes_ = indexes

  npks = lines.n.values.astype(np.int32)
  cdef np.int32_t[:] _npks_ = npks

  ts = lines.t.values
  cdef CFLOAT64_t[:] _ts_ = ts

  hdts = lines.dt.values/2
  cdef CFLOAT64_t[:] _hdts_ = hdts

  cdef Py_ssize_t len0 = len(Es)

  dt_next = np.full(len0, np.inf, dtype=CFLOAT64)
  cdef CFLOAT64_t[:] _dt_next_ = dt_next

  dt_prev = np.full(len0, np.inf, dtype=CFLOAT64)
  cdef CFLOAT64_t[:] _dt_prev_ = dt_prev

  lid_next = np.full(len0, 0, dtype=np.uint64)
  cdef np.uint64_t[:] _lid_next_ = lid_next

  lid_prev = np.full(len0, 0, dtype=np.uint64)
  cdef np.uint64_t[:] _lid_prev_ = lid_prev

  idx_next = np.full(len0, -1, dtype=np.intp)
  cdef Py_ssize_t[:] _idx_next_ = idx_next

  idx_prev = np.full(len0, -1, dtype=np.intp)
  cdef Py_ssize_t[:] _idx_prev_ = idx_prev

  idx_strk = np.full(len0, -1, dtype=np.intp)
  cdef Py_ssize_t[:] _idx_strk_ = idx_strk

  lns_next = np.full(len0, 0, dtype=np.int32)
  cdef np.int32_t[:] _lns_next_ = lns_next

  pks_next = np.full(len0, 0, dtype=np.int32)
  cdef np.int32_t[:] _pks_next_ = pks_next

  blk_next = np.full(len0, 0, dtype=CFLOAT64)
  cdef CFLOAT64_t[:] _blk_next_ = blk_next

  strk_1st = np.full(len0, 0, dtype=np.uint8)
  cdef np.uint8_t[:] _strk_1st_ = strk_1st

  cdef Py_ssize_t ii, jj, right, left, idx_plus, idx_minus, idx
  cdef CFLOAT64_t E0, nan, inf, dt_plus, dt_minus, delta
  right = 1
  left = 0

  nan = np.nan
  inf = np.inf

  # ------------------------------ Find neighbors ------------------------------
  for ii in range(len0):
    E0 = _Es_[ii] + max_dE
    for right in range(right,len0):
      if _Es_[right] >= E0: break
    else:
      right = len0

    E0 = _Es_[ii] - max_dE
    for left in range(ii-1,-1,-1):
      if _Es_[left] <= E0: break
    else:
      left = 0

    dt_plus = inf
    idx_plus = -1
    dt_minus = inf
    idx_minus = -1
    for jj in range(left+1,right):
      delta = _ts_[jj]-_ts_[ii]
      if delta > 0:
        delta = delta - _hdts_[jj] - _hdts_[ii]
        if delta < dt_plus:
          idx_plus = jj
          dt_plus = delta
      elif delta < 0:
        delta = -delta - _hdts_[jj] - _hdts_[ii]
        if delta < dt_minus:
          idx_minus = jj
          dt_minus = delta

    if idx_plus >= 0:
      _idx_next_[ii] = idx_plus
      _lid_next_[ii] = _indexes_[idx_plus]
      _dt_next_[ii] = dt_plus

    if idx_minus >= 0:
      _idx_prev_[ii] = idx_minus
      _lid_prev_[ii] = _indexes_[idx_minus]
      _dt_prev_[ii] = dt_minus

  # ------------------------------- Follow lines -------------------------------

  for ii in range(len0):
    if _lns_next_[ii] > 0: continue

    jj = 0
    idx = ii
    while True:
      if _dt_next_[idx] < max_dt:
        _idx_strk_[jj] = idx
      else:
        break

      idx = _idx_next_[idx]
      jj += 1
      if _lns_next_[idx] > 0:
        _strk_1st_[idx] = 0
        break

    if _lns_next_[idx] == 0:
      _lns_next_[idx] = 1
      _pks_next_[idx] = _npks_[idx]
      _blk_next_[idx] = 0.

    for jj in range(jj-1,-1,-1):
      idx = _idx_strk_[jj]
      _lns_next_[idx] = _lns_next_[_idx_next_[idx]] + 1
      _pks_next_[idx] = _pks_next_[_idx_next_[idx]] + _npks_[idx]
      _blk_next_[idx] = _blk_next_[_idx_next_[idx]] + _dt_next_[idx]

    _strk_1st_[ii] = 1

  return pd.DataFrame(index=lines.index, data={
    'lid_next': lid_next, 'dt_next': dt_next,
    'lid_prev': lid_prev, 'dt_prev': dt_prev,
    'lns_next': lns_next, 'pks_next': pks_next,
    'blk_next': blk_next, 'streak_1st': strk_1st.astype(bool)
  }).sort_values('lid')
# -------
