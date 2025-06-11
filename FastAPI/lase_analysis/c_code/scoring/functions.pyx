cimport cython

import numpy as np
cimport numpy as np

from scipy.sparse import csr_array

from libc.stdlib cimport malloc, free
from libc.math cimport floor, round
from warnings import warn

ctypedef np.uint8_t CUINT8
ctypedef np.int8_t CINT8
ctypedef np.uint16_t CUINT16
ctypedef np.int32_t CINT32
ctypedef np.uint32_t CUINT32
ctypedef np.uint64_t CUINT64
ctypedef np.float32_t CFLOAT32
ctypedef np.float64_t CFLOAT64

cdef CFLOAT32 _0 = 0.
cdef CFLOAT32 _1 = 1.
cdef CFLOAT32 _2 = 2.
cdef CFLOAT32 _05 = .5
cdef CFLOAT32 _INF = np.finfo('f4').max
cdef CFLOAT64 _INF_64 = np.finfo('f8').max

cdef CUINT64 _1_64 = 1
cdef CUINT64 _2_64 = 2

cdef extern from "c_argsort.h":
  CINT32 argsort_F32(CFLOAT32* array, CINT32* indices, CINT32 length, CUINT8 reverse, CUINT8 inplace)
  CINT32 argsort_F64(CFLOAT64* array, CINT32* indices, CINT32 length, CUINT8 reverse, CUINT8 inplace)
# ----

cdef struct ItpMatrix:
  CFLOAT32* x
  CFLOAT32* y
  CFLOAT32* Z

  CFLOAT32 dx
  CFLOAT32 dy
  CINT32 nx
  CINT32 ny
# -------

cdef struct PinfoStrct:
  ItpMatrix Mok
  ItpMatrix Mno
  CFLOAT64* Coks # ni x nj x nk
  
  CINT32 ni # mL
  CINT32 nj # mH
  CINT32 nk # nx
# -------

cdef struct LprobStrct:
  CFLOAT64* plr
  CINT32* lnx0
  CINT32* lnx1
  CINT32* isort

  CFLOAT64* pprods
  CINT32* nxs0
  CINT32* nxs1

  CINT32 mL
  CINT32 mH
  CINT32 nx0
  CINT32 nx1
  CINT32 nx
  CINT32 npairs

  CINT32 cnt
  CINT32 lmax
  CINT32 imax
# -------

# ----------------------------- SUPPORT FUNCTIONS ------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CFLOAT32 bits_count(CUINT64 num):
  cdef CFLOAT32 sum_and = 0.
  while num:
    sum_and += _1
    num &= num-1

  return sum_and
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def _nbits64(CUINT64[:] arr):
  cdef CINT32 i,N = <CINT32>arr.size
  
  sums = np.zeros(N, dtype=np.float32)
  cdef CFLOAT32[:] _sums_ = sums

  for i in range(N): _sums_[i] = bits_count(arr[i])

  return sums
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
# error code -1000
cdef CINT32 binom(CINT32 k, CINT32 n):
  if k==0: return 0
  if k==n: return 1
  if k>n:  return -1000

  cdef CINT32 i,val=1
  for i in range(1,k+1): val = val*(n-i+1)//i
  return val
# -------

# --------------------------- INTERPOLATION FUNCTION ---------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def interp1d(CFLOAT32 val, CFLOAT32[:] x, CFLOAT32[:] y):
  cdef CINT32 n = <CINT32>len(x)
  cdef CFLOAT32 out

  _interp1d(val, x, y, n, &out)
  return out
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
# error code: -1100
cdef CINT32 _interp1d(CFLOAT32 xp, CFLOAT32[:] x, CFLOAT32[:] y, CINT32 n, CFLOAT32* out):
  xp = max(min(xp,x[n-1]),x[0])

  cdef CFLOAT32 dx=x[1]-x[0], f
  cdef CINT32 k0,k1
  
  k0 = <CINT32>floor((xp-x[0])/dx)
  if (k0>=0)&(k0<n-1): k1 = k0+1
  elif (k0==n-1):      k1 = k0
  else:                return -1100

  f = (xp-x[k0])/dx
  out[0] = y[k0] + f*(y[k1]-y[k0])
  return 0
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def interp2d(CFLOAT32 xp, CFLOAT32 yp, CFLOAT32[:] x, CFLOAT32[:] y, CFLOAT32[:] Z):
  cdef CFLOAT32 out
  cdef ItpMatrix imat = {'x': &x[0], 'y': &y[0], 'Z': &Z[0], 'dx': <CFLOAT32>np.mean(np.diff(x)), 'dy': <CFLOAT32>np.mean(np.diff(y)),
                         'nx': <CINT32>len(x), 'ny': <CINT32>len(y)}

  _interp2d(xp, yp, &imat, &out)
  return out
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
# error code: -1200
cdef CINT32 _interp2d(CFLOAT32 xp, CFLOAT32 yp, ItpMatrix* imat, CFLOAT32* out): 
  xp = max(min(xp,imat[0].x[imat[0].nx-1]),imat[0].x[0])
  yp = max(min(yp,imat[0].y[imat[0].ny-1]),imat[0].y[0])

  cdef CFLOAT32 fx, fy
  cdef CINT32 kx0, kx1, ky0, ky1, ny=imat[0].ny
  
  kx0 = <CINT32>floor((xp-imat[0].x[0])/imat[0].dx)
  if (kx0>=0)&(kx0<imat[0].nx-1): kx1 = kx0+1
  elif kx0==imat[0].nx-1:         kx1 = kx0
  else:                           return -1201

  ky0 = <CINT32>floor((yp-imat[0].y[0])/imat[0].dy)
  if (ky0>=0)&(ky0<imat[0].ny-1): ky1 = ky0+1
  elif ky0==imat[0].ny-1:         ky1 = ky0
  else:                           return -1202

  fx = min(_1,max(_0,(xp-imat[0].x[kx0])/imat[0].dx))
  fy = min(_1,max(_0,(yp-imat[0].y[ky0])/imat[0].dy))

  out[0] = (_1-fy)*((_1-fx)*imat[0].Z[kx0*ny+ky0]+fx*imat[0].Z[kx1*ny+ky0]) +\
           fy*((_1-fx)*imat[0].Z[kx0*ny+ky1]+fx*imat[0].Z[kx1*ny+ky1])
  return 0 
# -------

# ----------------------------- DIGITIZE FUNCTIONS -----------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def digitize(CFLOAT32[:] Es, CUINT8[:] ns, CFLOAT32[:] bins, CFLOAT32 max_dE):
  cidx = np.insert(np.cumsum(ns,dtype=np.int32),0,0)
  cdef CINT32[:] _cidx_ = cidx

  cdef CINT32 N = <CINT32>len(ns)
  assert len(Es) == sum(ns)
  assert len(bins) == 65

  codes = np.zeros(N, dtype=np.uint64)
  cdef CUINT64[:] _codes_ = codes

  cdef CFLOAT32 ee
  cdef CINT32 i, k0, k
  cdef CUINT64 j

  for i in range(N):
    j = 0
    k0 = _cidx_[i]
    for k in range(ns[i]):
      ee = Es[k0+k]
      if ee<bins[0]-max_dE: continue
      if ee>bins[64]+max_dE: continue
      while (ee>=bins[j]) and (j<64): j+=1
      
      if j==0:
        _codes_[i] |= 1
        continue

      _codes_[i] |= (_1_64<<(j-1))
      if (j>1) and (ee-bins[j-1]<max_dE): _codes_[i] |= _1_64<<(j-2)
      if (j<64) and (bins[j]-ee<max_dE):  _codes_[i] |= _1_64<<j

  return codes
# -------

# ------------------------ SCORE/PROB SUPPORT FUNCTIONS ------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
# Error code -10100
cdef inline CINT32 _match_pair(CFLOAT32[:] Es0, CINT32 cidx0, CUINT8 _n0, CFLOAT32[:] dEs0, CINT32[:] mtc0,
                               CFLOAT32[:] Es1, CINT32 cidx1, CUINT8 _n1, CFLOAT32[:] dEs1, CINT32[:] mtc1):
  
  cdef CINT32 i,j,k, n0=_n0, n1=_n1, nadd, bb
  cdef CFLOAT32 curr, right, bottom
  cdef CUINT8 do0=True, do1=True

  for i in range(n0): mtc0[i] = -1
  for i in range(n1): mtc1[i] = -1

  i=j=0
  while True:
    if (i>=n0) or (j>=n1): break
    curr = Es1[cidx1+j]-Es0[cidx0+i]

    # Check scores of adjacent matches
    if j<n1-1: right = abs(Es1[cidx1+j+1]-Es0[cidx0+i])
    else:      right = _INF
    if i<n0-1: bottom = abs(Es1[cidx1+j]-Es0[cidx0+i+1])
    else:      bottom = _INF
    
    if do0 and (abs(curr)<=right):
      dEs0[i] = curr
      mtc0[i] = j
      do0 = False
    if do1 and (abs(curr)<=bottom):
      dEs1[j] = curr
      mtc1[j] = i
      do1 = False

    if (bottom<_INF) and (bottom<right):
      i += 1
      do0 = True
      continue
    elif (right<_INF) and (right<bottom):
      j += 1
      do1 = True
      continue
    elif (bottom<_INF) and (bottom==right):
      if bottom < abs(Es1[cidx1+j+1]-Es0[cidx0+i+1]): return -10101

    i += 1
    j += 1
    do0=do1=True

  for i in range(n0):
    if (mtc0[i]>=n1) or (mtc0[i]<0): return -10102
  for i in range(n1):
    if (mtc1[i]>=n0) or (mtc1[i]<0): return -10103

  return 0
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32 _match_pair_score(CFLOAT32[:] Es0, CINT32 cidx0, CUINT8 _n0, CFLOAT32[:] dEs0, CINT32[:] mtc0, 
                              CFLOAT32[:] Es1, CINT32 cidx1, CUINT8 _n1, CFLOAT32[:] dEs1, CINT32[:] mtc1,
                              CFLOAT32 max_dE, CFLOAT32* ana, CINT32* cnt, CINT32* npairs, CINT32* nx0, CINT32* nx1):

  cdef CINT32 k, ret

  ret = _match_pair(Es0, cidx0, _n0, dEs0, mtc0, Es1, cidx1, _n1, dEs1, mtc1)
  if ret < 0: return ret

  cdef CUINT8 *done = <CUINT8*>malloc(<CINT32>_n1*sizeof(CUINT8))
  for k in range(<CINT32>_n1): done[k] = 0

  ana[0] = _0
  nx0[0] = 0
  nx1[0] = 0
  npairs[0] = 0
  cnt[0] = 0
  for k in range(<CINT32>_n0):
    if abs(dEs0[k])<max_dE:
      ana[0] += _1-abs(dEs0[k])/max_dE
      if mtc1[mtc0[k]]==k:
        done[mtc0[k]] = <CUINT8>1
        npairs[0] += 1
      cnt[0] += 1
    else:
      nx0[0] += 1

  for k in range(<CINT32>_n1):
    if done[k]: continue
    if abs(dEs1[k])<max_dE:
      ana[0] += _1-abs(dEs1[k])/max_dE
      cnt[0] += 1
    else:
      nx1[0] += 1
  
  if cnt[0]>0: ana[0] /= <CFLOAT32>cnt[0]
  else:        ana[0] = _0

  free(done)
  return 0
# -------

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# error code -22000
cdef CINT32 _match_pair_prob(CFLOAT32[:] Es0, CINT32 cidx0, CUINT8 _n0, CFLOAT32[:] dEs0, CINT32[:] mtc0,
                             CFLOAT32[:] Es1, CINT32 cidx1, CUINT8 _n1, CFLOAT32[:] dEs1, CINT32[:] mtc1,
                             LprobStrct* lprob, PinfoStrct* pinfo, CFLOAT32 max_dE):

  cdef CINT32 k, ret, n0=_n0, n1=_n1

  ret = _match_pair(Es0, cidx0, _n0, dEs0, mtc0, Es1, cidx1, _n1, dEs1, mtc1)
  if ret < 0: return ret

  cdef CUINT8 *done = <CUINT8*>malloc(n1*sizeof(CUINT8))
  for k in range(n1): done[k] = 0

  lprob.nx0 = 0
  lprob.nx1 = 0
  lprob.npairs = 0
  lprob.cnt = 0
  for k in range(n0):
    if abs(dEs0[k])<max_dE:
      ret = _interp_ede((Es0[cidx0+k]+Es1[cidx1+mtc0[k]])/_2, dEs0[k], pinfo, lprob)
      if ret<0:
        free(done)
        return ret

      if mtc1[mtc0[k]]==k:
        done[mtc0[k]] = <CUINT8>1
        lprob.npairs += 1
        lprob.lnx0[lprob.cnt-1] = 1
        lprob.lnx1[lprob.cnt-1] = 1
      else:
        lprob.lnx0[lprob.cnt-1] = 1
        lprob.lnx1[lprob.cnt-1] = 0
    else:
      lprob.nx0 += 1

  for k in range(n1):
    if done[k]: continue

    if abs(dEs1[k])<max_dE:
      ret = _interp_ede((Es0[cidx0+mtc1[k]]+Es1[cidx1+k])/_2, dEs1[k], pinfo, lprob)
      if ret<0:
        free(done)
        return ret

      lprob.lnx0[lprob.cnt-1] = 0
      lprob.lnx1[lprob.cnt-1] = 1
    else:
      lprob.nx1 += 1
  
  if _n0==_n1:
    lprob.mL = <CINT32>_n0
    lprob.mH = lprob.mL
    lprob.nx = min(lprob.nx0,lprob.nx1)
  elif _n0>_n1:
    lprob.mL = <CINT32>_n1
    lprob.mH = <CINT32>_n0
    lprob.nx = lprob.nx1
  else:
    lprob.mL = <CINT32>_n0
    lprob.mH = <CINT32>_n1
    lprob.nx = lprob.nx0

  free(done)
  return 0
# -------

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# error code -23000
cdef inline CINT32 _interp_ede(CFLOAT32 ee, CFLOAT32 dEs, PinfoStrct* pinfo, LprobStrct* lprob):
  if lprob.cnt>=lprob.lmax: return -23000

  cdef CINT32 ret
  cdef CFLOAT32 val

  ret = _interp2d(ee, dEs, &(pinfo[0].Mno), &val)
  if ret<0:  return ret
  if val<0.: return -23001
  lprob.plr[lprob.cnt] = <CFLOAT64>val

  ret = _interp2d(ee, dEs, &(pinfo[0].Mok), &val)
  if ret<0:  return ret
  if val<0.: return -23002

  if val==_0: lprob.plr[lprob.cnt] = _INF_64
  else:       lprob.plr[lprob.cnt] /= <CFLOAT64>val

  lprob.cnt += 1

  return 0
# -------

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# error code -24000
cdef CINT32 _calc_prob(LprobStrct* lprob, PinfoStrct* pinfo, CINT32 nr_max, CFLOAT32* pret, CINT32* nxret):
  cdef CINT32 i,ret, nx, nx_best=0
  cdef CINT32 iok0 = lprob.mL*pinfo.nj*pinfo.nk + lprob.mH*pinfo.nk
  cdef CFLOAT64 pprod = 1., pp, cc, pp_best = 0.

  # cdef CINT32 nx_min = min(lprob.nx0,lprob.nx1)
  # cdef CINT32 nx_max = nx_min + nr_max

  # sort lprob by pno/pok ratios (low ratios lead to high probabilities)
  for i in range(lprob.cnt): lprob.isort[i] = i
  ret = argsort_F64(lprob.plr, lprob.isort, lprob.cnt, 1, 0)
  if ret<0: return ret

  for i in range(nr_max, lprob.cnt):
    if lprob.plr[lprob.isort[i]] == _INF_64:
      pprod == _INF_64
      break
    pprod *= lprob.plr[lprob.isort[i]]

  if pprod != _INF_64:
    for i in range(nr_max+1):
      ret = _lines_combs_rec(lprob,pprod,0,0,i,nr_max)
      if ret<0: return ret
      for j in range(ret):
        pp = lprob.pprods[j]
        nx = min(lprob.nxs0[j],lprob.nxs1[j])
        cc = pinfo.Coks[iok0+nx]

        if (pp == _INF_64) or (cc == _INF_64): continue
        if (pp == 0.) or (cc == 0.):
          pp_best = 1.
          nx_best = nx
          break

        pp = 1./(1.+pp*cc)
        if pp>pp_best:
          pp_best = pp
          nx_best = nx
      
      if pp_best >= 1.: break
  else:
    pp_best = 0.
      
  if (pp_best<0.) or (pp_best>1.): return -24000
  pret[0] = <CFLOAT32>pp_best
  nxret[0] = nx_best

  return 0
# -------

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# error code -2100
cdef CINT32 _lines_combs_rec(LprobStrct* lprob, CFLOAT64 pprod, CINT32 i0, CINT32 j0, CINT32 left, CINT32 nr_max):
  cdef CINT32 i,j,j1

  if left==0:
    if j0>=lprob.imax: return -2100
    lprob.nxs0[j0] = lprob.nx0
    lprob.nxs1[j0] = lprob.nx1
    lprob.pprods[j0] = pprod
    for i in range(i0,nr_max):
      if lprob.plr[lprob.isort[i]] == _INF_64:
        lprob.pprods[j0] = _INF_64
        break
      lprob.pprods[j0] *= lprob.plr[lprob.isort[i]]
    return j0+1
  
  if left==nr_max-i0:
    if j0>=lprob.imax: return -2101
    lprob.nxs0[j0] = lprob.nx0
    lprob.nxs1[j0] = lprob.nx1
    lprob.pprods[j0] = pprod
    for i in range(i0,nr_max):
      lprob.nxs0[j0] += lprob.lnx0[lprob.isort[i]]
      lprob.nxs1[j0] += lprob.lnx1[lprob.isort[i]]
    return j0+1

  if left>nr_max-i0: return -2102

  for i in range(i0,nr_max-left+1):
    # current pair is not matching
    j1 = _lines_combs_rec(lprob,pprod,i+1,j0,left-1,nr_max)
    if j1<0: return j1

    for j in range(j0,j1):
      lprob.nxs0[j] += lprob.lnx0[lprob.isort[i]]
      lprob.nxs1[j] += lprob.lnx1[lprob.isort[i]]

    j0 = j1

    # for next cycles, current pair will be matching
    if lprob.plr[lprob.isort[i]] == _INF_64: break
    pprod *= lprob.plr[lprob.isort[i]]

  return j0
# -------

# ---------------------------------- SCORING -----------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
# Error code: -10000
def score_sing(CFLOAT32[:] Es0, CUINT8[:] ns0, CUINT64[:] codes0, CINT32 i0,
               CFLOAT32[:] Es1, CUINT8[:] ns1, CUINT64[:] codes1, CINT32 j0, CUINT8 pself, CINT32 shape0, CINT32 shape1,
               CFLOAT32 density, CFLOAT32 max_dE, CFLOAT32 min_simil, CFLOAT32 min_dig, CFLOAT32 min_ana):

  cidx0 = np.insert(np.cumsum(ns0,dtype=np.int32),0,0)
  cdef CINT32[:] _cidx0_ = cidx0
  cidx1 = np.insert(np.cumsum(ns1,dtype=np.int32),0,0)
  cdef CINT32[:] _cidx1_ = cidx1

  cdef CINT32 len0 = <CINT32>len(ns0)
  assert len(Es0) == sum(ns0)
  assert len0 == len(codes0)

  cdef CINT32 len1 = <CINT32>len(ns1)
  assert len(Es1) == sum(ns1)
  assert len1 == len(codes1)

  cdef CUINT64 max_data = np.uint64(np.ceil(density*len0*len1))
  cdef CINT32 lmax = max(max(ns0),max(ns1))

  indptr = np.zeros(shape0+1, dtype=np.uint64)
  cdef CUINT64[:] _indptr_ = indptr

  indices = np.full(max_data, -1, dtype=np.int32)
  cdef CINT32[:] _indices_ = indices

  data = np.full_like(indices, 0, dtype=np.uint64)
  cdef CUINT64[:] _data_ = data

  nbit0 = _nbits64(codes0)
  cdef CFLOAT32[:] _nbit0_ = nbit0
  nbit1 = _nbits64(codes1)
  cdef CFLOAT32[:] _nbit1_ = nbit1

  dEs0 = np.zeros(lmax, dtype=np.float32)
  cdef CFLOAT32[:] _dEs0_ = dEs0
  dEs1 = np.zeros(lmax, dtype=np.float32)
  cdef CFLOAT32[:] _dEs1_ = dEs1

  mtc0 = np.full(lmax, -1, dtype=np.int32)
  cdef CINT32[:] _mtc0_ = mtc0
  mtc1 = np.full(lmax, -1, dtype=np.int32)
  cdef CINT32[:] _mtc1_ = mtc1


  cdef CUINT64 iseq=0
  cdef CFLOAT32 dig, ana
  cdef CINT32 i, j, k, ret, cnt, npairs, nx0, nx1, mm, mL, mH, nx

  for i in range(len0):
    for j in range(len1):
      if pself and (j0+j <= i0+i): continue
      if bits_count(codes0[i]&codes1[j])/min(_nbit0_[i],_nbit1_[j]) < min_simil: continue
      ret = _match_pair_score(Es0, _cidx0_[i], ns0[i], _dEs0_, _mtc0_,
                              Es1, _cidx1_[j], ns1[j], _dEs1_, _mtc1_, max_dE, &ana, &cnt, &npairs, &nx0, &nx1)
      if ret<0: return ret
      if ana==_0: continue

      if ns0[i]==ns1[j]:
        mL = ns0[i]
        mH = mL
        nx = min(nx0,nx1)
      elif ns0[i]>ns1[j]:
        mL = ns1[j]
        mH = ns0[i]
        nx = nx1
      else:
        mL = ns0[i]
        mH = ns1[j]
        nx = nx0

      dig = _1 - <CFLOAT32>nx/<CFLOAT32>mL

      if (dig<0.) or (dig>1.): return -10000
      if (ana<0.) or (ana>1.): return -10001
      if (ana<min_ana) or (dig<min_dig): continue

      _data_[iseq] = ((<CUINT64>(65535*ana))<<48)+((<CUINT64>(65535*dig))<<32)+((<CUINT64>mH)<<16)+((<CUINT64>mL)<<8)+(<CUINT64>nx)
      _indices_[iseq] = j0+j
      iseq += 1
      if iseq >= max_data: return -10002
        
    _indptr_[i0+i+1] = iseq

  _indptr_[i0+i+1:] = iseq

  return csr_array((data[:iseq], indices[:iseq], indptr), (shape0, shape1))
# -------

# # -------------------------------- PROBABILITY ---------------------------------
# # ------------------------------------------------------------------------------
# # ------------------------------------------------------------------------------

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# error code -20000
def prob_sing(CFLOAT32[:] Es0, CUINT8[:] ns0, CUINT64[:] codes0, CINT32 i0,
              CFLOAT32[:] Es1, CUINT8[:] ns1, CUINT64[:] codes1, CINT32 j0, CINT32 shape0, CINT32 shape1, CUINT8 pself, CFLOAT32 density,
              CFLOAT32 min_simil, CFLOAT32 min_dig, CFLOAT32 max_dE, CFLOAT32 min_prob, pfit):

  _cidx0 = np.insert(np.cumsum(ns0,dtype=np.int32),0,0)
  cdef CINT32[:] cidx0 = _cidx0
  _cidx1 = np.insert(np.cumsum(ns1,dtype=np.int32),0,0)
  cdef CINT32[:] cidx1 = _cidx1

  cdef CINT32 len0 = <CINT32>len(ns0)
  assert len(Es0) == sum(ns0)
  assert len0 == len(codes0)

  cdef CINT32 len1 = <CINT32>len(ns1)
  assert len(Es1) == sum(ns1)
  assert len1 == len(codes1)

  assert len(pfit.pok_x) == pfit.pok_val.shape[0]
  assert len(pfit.pok_y) == pfit.pok_val.shape[1]
  assert len(pfit.pno_x) == pfit.pno_val.shape[0]
  assert len(pfit.pno_y) == pfit.pno_val.shape[1]

  cdef CUINT64 iseq_max,iseq=0
  cdef CINT32 i,j,ret,nxret, mL,mH,nx_max,nr_max
  cdef CINT32 lmax=<CINT32>max(max(ns0),max(ns1))
  cdef CFLOAT32 pret

  assert lmax < 256

  iseq_max = np.uint64(np.ceil(density*len0*len1))

  _indptr = np.zeros(shape0+1, dtype=np.uint64)
  cdef CUINT64[:] indptr = _indptr
  _indices = np.full(iseq_max, -1, dtype=np.int32)
  cdef CINT32[:] indices = _indices
  _data = np.full_like(indices, 0, dtype=np.uint64)
  cdef CUINT64[:] data = _data

  _nbit0 = _nbits64(codes0)
  cdef CFLOAT32[:] nbit0 = _nbit0
  _nbit1 = _nbits64(codes1)
  cdef CFLOAT32[:] nbit1 = _nbit1

  _dEs0 = np.zeros(lmax, dtype=np.float32)
  cdef CFLOAT32[:] dEs0 = _dEs0
  _dEs1 = np.zeros(lmax, dtype=np.float32)
  cdef CFLOAT32[:] dEs1 = _dEs1

  _mtc0 = np.full(lmax, -1, dtype=np.int32)
  cdef CINT32[:] mtc0 = _mtc0
  _mtc1 = np.full(lmax, -1, dtype=np.int32)
  cdef CINT32[:] mtc1 = _mtc1

  cdef CFLOAT32[:] pok_x = pfit.pok_x
  cdef CFLOAT32[:] pok_y = pfit.pok_y
  cdef CFLOAT32[:] pno_x = pfit.pno_x
  cdef CFLOAT32[:] pno_y = pfit.pno_y
  cdef CFLOAT32[:] pok_val = pfit.pok_val.ravel('C')
  cdef CFLOAT32[:] pno_val = pfit.pno_val.ravel('C')
  
  _Coks = pfit.poks.copy().ravel('C')
  cdef CFLOAT64[:] Coks = _Coks
  for i in range(np.prod(_Coks.shape)):
    if Coks[i] == 0.: Coks[i] = _INF_64
    else:             Coks[i] = 1./Coks[i]-1.
  
  cdef ItpMatrix Mok = {'x': &pok_x[0], 'y': &pok_y[0], 'Z': &pok_val[0], 'dx': <CFLOAT32>np.mean(np.diff(pok_x)), 'dy': <CFLOAT32>np.mean(np.diff(pok_y)),
                        'nx': <CINT32>len(pok_x), 'ny': <CINT32>len(pok_y)}

  cdef ItpMatrix Mno = {'x': &pno_x[0], 'y': &pno_y[0], 'Z': &pno_val[0], 'dx': <CFLOAT32>np.mean(np.diff(pno_x)), 'dy': <CFLOAT32>np.mean(np.diff(pno_y)),
                        'nx': <CINT32>len(pno_x), 'ny': <CINT32>len(pno_y)}

  cdef PinfoStrct pinfo = {'Mok': Mok, 'Mno': Mno, 'Coks': &Coks[0],
                           'ni': <CINT32>(pfit.poks.shape[0]), 'nj': <CINT32>(pfit.poks.shape[1]), 'nk': <CINT32>(pfit.poks.shape[2])}

  _plr = np.zeros(lmax, dtype=np.float64)
  cdef CFLOAT64[:] plr = _plr
  _lnx0 = np.zeros(lmax, dtype=np.int32)
  cdef CINT32[:] lnx0 = _lnx0
  _lnx1 = np.zeros(lmax, dtype=np.int32)
  cdef CINT32[:] lnx1 = _lnx1
  _isort = np.zeros(lmax, dtype=np.int32)
  cdef CINT32[:] isort = _isort

  cdef CINT32 imax = binom(pinfo.nk//2, pinfo.nk)+1
  _pprod = np.zeros(imax, dtype=np.float64)
  cdef CFLOAT64[:] pprod = _pprod
  _nxs0 = np.zeros(imax, dtype=np.int32)
  cdef CINT32[:] nxs0 = _nxs0
  _nxs1 = np.zeros(imax, dtype=np.int32)
  cdef CINT32[:] nxs1 = _nxs1

  cdef LprobStrct lprob = {'plr': &plr[0], 'lnx0': &lnx0[0], 'lnx1': &lnx1[0], 'isort': &isort[0], 'pprods': &pprod[0], 'nxs0': &nxs0[0], 'nxs1': &nxs1[0],
                           'mL': 0, 'mH': 0, 'nx0': 0, 'nx1': 0, 'nx': 0, 'npairs': 0, 'cnt': 0, 'lmax': lmax, 'imax': imax}

  for i in range(len0):
    for j in range(len1):
      if pself and (j0+j <= i0+i): continue
      if bits_count(codes0[i]&codes1[j])/min(nbit0[i],nbit1[j]) < min_simil: continue
      
      ret = _match_pair_prob(Es0, cidx0[i], ns0[i], dEs0, mtc0, Es1, cidx1[j], ns1[j], dEs1, mtc1,
                             &lprob, &pinfo, max_dE)
      if ret<0: return ret

      if (_1-<CFLOAT32>lprob.nx/<CFLOAT32>lprob.mL) < min_dig: continue

      nx_max = min(pinfo.nk-1, <CINT32>floor((_1-min_dig)*<CFLOAT32>lprob.mL))
      nr_max = max(0,nx_max-lprob.nx)

      mL = lprob.mL
      mH = lprob.mH
      # when calc probabilities, limit mL and mH based on the size of poks
      lprob.mL = min(mL,pinfo.ni-1)
      lprob.mH = min(mH,pinfo.nj-1)

      ret = _calc_prob(&lprob, &pinfo, nr_max, &pret, &nxret)
      if ret<0: return ret
      if pret<min_prob: continue

      if iseq >= iseq_max: return -20000
      # when saving data, use the real values of mH and mL
      data[iseq] = ((<CUINT64>(<CUINT32*>&pret)[0])<<32) + ((<CUINT64>mH)<<16) + ((<CUINT64>mL)<<8) + (<CUINT64>nxret)
      indices[iseq] = j0+j
      iseq += 1
        
    indptr[i0+i+1] = iseq

  for j in range(i0+i+2, shape0+1): indptr[j] = iseq

  return csr_array((data[:iseq], indices[:iseq], indptr), (shape0, shape1))
# -------

# ------------------------------ SIMPLE MATCHING -------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def simple_match_symm(poss, key):
  cdef CINT32 nrow = <CINT32>poss.shape[0]-1

  cdef CINT32[:] indptr_r = poss.rindptr
  cdef CINT32[:] indices_r = poss.rindices
  cdef CFLOAT32[:] score_r = poss.vals[key]

  cdef CINT32[:] indptr_c = poss.cindptr
  cdef CINT32[:] indices_c = poss.cindices
  cdef CINT32[:] pidxs_c = poss.cdata

  cdef CINT32 i,j, rr,ccol,crow
  cdef CFLOAT32 curr
  
  match0 = np.full(poss.shape[0], -1, dtype=np.int32)
  cdef CINT32[:] _match0_ = match0
  match1 = np.full(poss.shape[1], -1, dtype=np.int32)
  cdef CINT32[:] _match1_ = match1

  for rr in range(nrow):
    curr = _0
    ccol = -1
    for j in range(indptr_r[rr],indptr_r[rr+1]):
      if score_r[j]>curr:
        curr = score_r[j]
        ccol = indices_r[j]
    if ccol<0: continue

    curr = _0
    crow = -1
    for j in range(indptr_c[ccol],indptr_c[ccol+1]):
      if score_r[pidxs_c[j]]>curr:
        curr = score_r[pidxs_c[j]]
        crow = indices_c[j]
    if crow<0: return -1

    if crow!=rr: continue

    if _match0_[rr]>=0: return -2
    if _match1_[ccol]>=0: return -3

    _match0_[rr] = ccol
    _match1_[ccol] = rr
  
  return match0, match1
# -------

# ------------------------- PYTHON INTEFACE FUNCTIONS --------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def find_dEs(CFLOAT32[:] Es0, CFLOAT32[:] Es1):

  cdef CUINT8 n0=<CUINT8>len(Es0), n1=<CUINT8>len(Es1)

  dEs0 = np.zeros(n0, dtype=np.float32)
  cdef CFLOAT32[:] _dEs0_ = dEs0
  dEs1 = np.zeros(n1, dtype=np.float32)
  cdef CFLOAT32[:] _dEs1_ = dEs1

  mtc0 = np.zeros(n0, dtype=np.int32)
  cdef CINT32[:] _mtc0_ = mtc0
  mtc1 = np.zeros(n1, dtype=np.int32)
  cdef CINT32[:] _mtc1_ = mtc1

  _match_pair(Es0, 0, n0, _dEs0_, _mtc0_, Es1, 0, n1, _dEs1_, _mtc1_)

  return dEs0, dEs1, mtc0, mtc1
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
# Error code -25000
def info_match_lns(CFLOAT32[:] Es0, CFLOAT32[:] As0, CUINT8[:] ns0, CINT32[:] idxs0,
                   CFLOAT32[:] Es1, CFLOAT32[:] As1, CUINT8[:] ns1, CINT32[:] idxs1, CFLOAT32 max_dE, CFLOAT32 density=_1):

  cdef CINT32 N = <CINT32>len(idxs0)
  assert N == len(idxs1)
  assert len(Es0) == sum(ns0)
  assert len(Es1) == sum(ns1)
  cdef CINT32 lmax = <CINT32>max(max(ns0),max(ns1))

  cidx0 = np.insert(np.cumsum(ns0,dtype=np.int32),0,0)
  cdef CINT32[:] _cidx0_ = cidx0
  cidx1 = np.insert(np.cumsum(ns1,dtype=np.int32),0,0)
  cdef CINT32[:] _cidx1_ = cidx1

  cdef CUINT64 nMax = <CUINT64>(2*N*lmax*density)

  out_dEs = np.zeros(nMax, dtype=np.float32)
  cdef CFLOAT32[:] _out_dEs_ = out_dEs
  out_Es = np.zeros(nMax, dtype=np.float32)
  cdef CFLOAT32[:] _out_Es_ = out_Es
  out_As = np.zeros(nMax, dtype=np.float32)
  cdef CFLOAT32[:] _out_As_ = out_As
  out_mtc = np.zeros(nMax, dtype=np.int8)
  cdef CINT8[:] _out_mtc_ = out_mtc
  out_cnt = np.zeros(N, dtype=np.int32)
  cdef CINT32[:] _out_cnt_ = out_cnt

  dEs0 = np.zeros(lmax, dtype=np.float32)
  cdef CFLOAT32[:] _dEs0_ = dEs0
  dEs1 = np.zeros(lmax, dtype=np.float32)
  cdef CFLOAT32[:] _dEs1_ = dEs1

  mtc0 = np.full(lmax, -1, dtype=np.int32)
  cdef CINT32[:] _mtc0_ = mtc0
  mtc1 = np.full(lmax, -1, dtype=np.int32)
  cdef CINT32[:] _mtc1_ = mtc1

  done = np.zeros(lmax, dtype=np.uint8)
  cdef CUINT8[:] _done_ = done

  cdef CINT32 i,k,ret
  cdef CUINT64 iseq=0


  for i in range(N):
    ret=_match_pair(Es0, _cidx0_[idxs0[i]], ns0[idxs0[i]], _dEs0_, _mtc0_, Es1, _cidx1_[idxs1[i]], ns1[idxs1[i]], _dEs1_, _mtc1_)
    if ret<0: return ret
    
    for k in range(lmax): _done_[k] = 0

    for k in range(ns0[idxs0[i]]):
      _out_cnt_[i] += 1
      _out_dEs_[iseq] = _dEs0_[k]
      # Maybe matching lines
      if abs(_dEs0_[k])<max_dE:
        _out_Es_[iseq] = (Es0[_cidx0_[idxs0[i]]+k]+Es1[_cidx1_[idxs1[i]]+_mtc0_[k]])/_2
        _out_As_[iseq] = (As0[_cidx0_[idxs0[i]]+k]+As1[_cidx1_[idxs1[i]]+_mtc0_[k]])/_2
        if _mtc1_[_mtc0_[k]]==k:
          _out_mtc_[iseq] = 2
          _done_[_mtc0_[k]] = 1
        else:
          _out_mtc_[iseq] = 0
      # Non-matching lines
      else:
        _out_Es_[iseq] = Es0[_cidx0_[idxs0[i]]+k]
        _out_As_[iseq] = As0[_cidx0_[idxs0[i]]+k]
        _out_mtc_[iseq] = -1

      iseq += 1
      if iseq>=nMax: return -25001

    for k in range(ns1[idxs1[i]]):
      if _done_[k]: continue
      _out_cnt_[i] += 1
      _out_dEs_[iseq] = _dEs1_[k]
      # Maybe matching lines
      if abs(_dEs1_[k])<max_dE:
        _out_Es_[iseq] = (Es0[_cidx0_[idxs0[i]]+_mtc1_[k]]+Es1[_cidx1_[idxs1[i]]+k])/_2
        _out_As_[iseq] = (As0[_cidx0_[idxs0[i]]+_mtc1_[k]]+As1[_cidx1_[idxs1[i]]+k])/_2
        _out_mtc_[iseq] = 1
      # Non-matching lines
      else:
        _out_Es_[iseq] = Es1[_cidx1_[idxs1[i]]+k]
        _out_As_[iseq] = As1[_cidx1_[idxs1[i]]+k]
        _out_mtc_[iseq] = -1

      iseq += 1
      if iseq>=nMax: return -25002
  
  return out_dEs[:iseq], out_Es[:iseq], out_As[:iseq], out_mtc[:iseq], out_cnt
# -------