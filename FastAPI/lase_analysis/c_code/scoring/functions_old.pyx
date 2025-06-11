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

cdef CUINT64 _1_64 = 1
cdef CUINT64 _2_64 = 2

cdef extern from "c_argsort.h":
  CINT32 argsort_F32(CFLOAT32* array, CINT32* indices, CINT32 length, CUINT8 reverse, CUINT8 inplace)
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
  # CFLOAT32* pok_set
  # CFLOAT32* pset
  # CFLOAT32* pok_set_comb
  # CFLOAT32* pset_comb
  CFLOAT32* pwrgs
  CFLOAT32* poknx
  CINT32 nmax
  CINT32 mmax
# -------

cdef struct LprobStrct:
  CFLOAT32* plm
  CFLOAT32* plx
  CINT32* lnx
  CINT32* isort

  CFLOAT32* pprod
  CINT32* nxs
  CFLOAT32* pok_nx
  
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

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
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

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
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

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
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

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef CINT32 _match_pair_score(CFLOAT32[:] Es0, CINT32 cidx0, CUINT8 _n0, CFLOAT32[:] dEs0, CINT32[:] mtc0, 
                              CFLOAT32[:] Es1, CINT32 cidx1, CUINT8 _n1, CFLOAT32[:] dEs1, CINT32[:] mtc1,
                              CFLOAT32 max_dE, CFLOAT32* ana, CINT32* cnt, CINT32* nm, CINT32* nx0, CINT32* nx1):

  cdef CINT32 k, ret

  ret = _match_pair(Es0, cidx0, _n0, dEs0, mtc0, Es1, cidx1, _n1, dEs1, mtc1)
  if ret < 0: return ret

  cdef CUINT8 *done = <CUINT8*>malloc(<CINT32>_n1*sizeof(CUINT8))
  for k in range(<CINT32>_n1): done[k] = 0

  ana[0] = _0
  nx0[0] = 0
  nx1[0] = 0
  nm[0] = 0
  cnt[0] = 0
  for k in range(<CINT32>_n0):
    if abs(dEs0[k])<max_dE:
      ana[0] += _1-abs(dEs0[k])/max_dE
      if mtc1[mtc0[k]]==k:
        done[mtc0[k]] = <CUINT8>1
        nm[0] += 1
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
                             LprobStrct* lprob, CINT32* nm, CINT32* nx0, CINT32* nx1, PinfoStrct* pinfo, CFLOAT32 max_dE):

  cdef CINT32 k, ret, n0=_n0, n1=_n1

  ret = _match_pair(Es0, cidx0, _n0, dEs0, mtc0, Es1, cidx1, _n1, dEs1, mtc1)
  if ret < 0: return ret

  cdef CUINT8 *done = <CUINT8*>malloc(n1*sizeof(CUINT8))
  for k in range(n1): done[k] = 0

  nx0[0] = 0
  nx1[0] = 0
  nm[0] = 0
  lprob.cnt = 0
  for k in range(n0):
    if abs(dEs0[k])<max_dE:
      ret = _interp_ede((Es0[cidx0+k]+Es1[cidx1+mtc0[k]])/_2, dEs0[k], pinfo, lprob)
      if ret<0:
        free(done)
        return ret

      if mtc1[mtc0[k]]==k:
        done[mtc0[k]] = <CUINT8>1
        nm[0] += 1
        lprob.lnx[lprob.cnt-1] = 2
      else:
        lprob.lnx[lprob.cnt-1] = 1
    else:
      nx0[0] += 1

  for k in range(n1):
    if done[k]: continue

    if abs(dEs1[k])<max_dE:
      ret = _interp_ede((Es0[cidx0+mtc1[k]]+Es1[cidx1+k])/_2, dEs1[k], pinfo, lprob)
      if ret<0:
        free(done)
        return ret

      lprob.lnx[lprob.cnt-1] = 1
    else:
      nx1[0] += 1
  
  free(done)
  return 0
# -------

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# error code -2000
cdef CINT32 _lines_combs(LprobStrct* lprob, CINT32 nx, CINT32 xmax):
  cdef CINT32 i
  cdef CFLOAT32 pprod0
  
  pprod0 = _1
  for i in range(xmax, lprob.cnt): pprod0 *= lprob.plm[lprob.isort[i]]
  
  return _lines_combs_rec(lprob,pprod0,0,0,xmax,nx)

# -------

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# error code -2100
cdef CINT32 _lines_combs_rec(LprobStrct* lprob, CFLOAT32 pprod0, CINT32 i0, CINT32 j0, CINT32 n, CINT32 left):
  cdef CINT32 i

  if left==0:
    if j0>=lprob.imax: return -2100
    lprob.nxs[j0] = 0
    lprob.pprod[j0] = pprod0
    for i in range(i0,n): lprob.pprod[j0] *= lprob.plm[lprob.isort[i]]
    return j0+1
  
  if left==n:
    if j0>=lprob.imax: return -2101
    lprob.pprod[j0] = pprod0
    for i in range(i0,n):
      lprob.pprod[j0] *= lprob.plx[lprob.isort[i]]
      lprob.nxs[j0] += lprob.lnx[lprob.isort[i]]
    return j0+1

  cdef CINT32 j,j1,new=0
  cdef CFLOAT32 tmp=_1

  for i in range(i0,n):
    j1 = _lines_combs_rec(lprob,pprod0,i+1,j0,n,left-1)
    if j1<0: return j1

    for j in range(j0,j1):
      lprob.pprod[j] *= lprob.plx[lprob.isort[i]]*tmp
      lprob.nxs[j] += lprob.lnx[lprob.isort[i]]
    
    tmp *= lprob.plm[lprob.isort[i]]
    j0 = j1

  return j0
# -------

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# error code -23000
cdef inline CINT32 _interp_ede(CFLOAT32 ee, CFLOAT32 dEs, PinfoStrct* pinfo, LprobStrct* lprob):
    if lprob.cnt>=lprob.lmax: return -23000

    cdef CINT32 ret

    ret = _interp2d(ee, dEs, &(pinfo[0].Mok), &lprob.plm[lprob.cnt])
    if ret<0: return ret
    if lprob.plm[lprob.cnt]<0: return -23001

    ret = _interp2d(ee, dEs, &(pinfo[0].Mno), &lprob.plx[lprob.cnt])
    if ret<0: return ret
    if lprob.plx[lprob.cnt]<0: return -23002

    lprob.cnt += 1
    return 0
# -------

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# error code -24000
cdef CINT32 _calc_prob(LprobStrct* lprob, PinfoStrct* pinfo, CINT32 mm, CINT32 nx, CINT32 xmax, CFLOAT32* pret):

  cdef CFLOAT32 pok, pno, num, den
  cdef CINT32 i,j,ret, i0=mm*(pinfo.mmax+1), cmmax=min(pinfo.mmax, nx+xmax*2)

  pno = pinfo.pwrgs[mm]
  for i in range(lprob.cnt): pno *= lprob.plx[i]

  # sort lprob by plm (will only consider the lines with lowest scores as possibly missing)
  for i in range(lprob.cnt): lprob.isort[i] = i
  ret = argsort_F32(lprob.plm, lprob.isort, lprob.cnt, 0, 0)
  if ret<0: return ret

  # k is number of elements of plm that can be missed
  for i in range(nx, cmmax+1): lprob.pok_nx[i] = _0
  for i in range(xmax+1):
    ret = _lines_combs(lprob, i, xmax)
    if ret<0: return ret
    for j in range(ret):
      lprob.pok_nx[min(nx+lprob.nxs[j], cmmax)] += lprob.pprod[j]

  pok = _0
  for i in range(nx,cmmax+1):
    if lprob.pok_nx[i]==_0: continue
    # pok += lprob.pok_nx[i]*pinfo.pset[i0+i]*pinfo.pok_set[i0+i]
    pok += lprob.pok_nx[i]*pinfo.poknx[i0+i]


  den = pok+pno
  if den==_0: pret[0] = _0
  else:       pret[0] = pok/den
  if pret[0]>_1: return -24000

  return 0
# -------

# ---------------------------------- SCORING -----------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# Error code: -10000
def score_sing(CFLOAT32[:] Es0, CUINT8[:] ns0, CUINT64[:] codes0, CINT32 i0,
               CFLOAT32[:] Es1, CUINT8[:] ns1, CUINT64[:] codes1, CINT32 j0, CUINT8 pself, CINT32 shape0, CINT32 shape1, CUINT8 symm, CFLOAT32 density,
               CINT32 lmin, CINT32 nmax, CINT32 mmax, CFLOAT32 max_dE, CFLOAT32 min_simil, CFLOAT32 min_dig, CFLOAT32 min_ana):

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
  cdef CINT32 i, j, k, ret, cnt, nm, nx0, nx1, mm, mL, mH, nx

  for i in range(len0):
    if ns0[i]<lmin: continue
    for j in range(len1):
      if ns1[j]<lmin: continue
      if symm:
        if pself and (j0+j <= i0+i): continue
        if _2*bits_count(codes0[i]&codes1[j])/(_nbit0_[i]+_nbit1_[j]) < min_simil: continue

        ret = _match_pair_score(Es0, _cidx0_[i], ns0[i], _dEs0_, _mtc0_,
                                Es1, _cidx1_[j], ns1[j], _dEs1_, _mtc1_, max_dE, &ana, &cnt, &nm, &nx0, &nx1)
        if ret<0: return ret
        if ana==_0: continue
        dig = <CFLOAT32>(nm+nm)/<CFLOAT32>(nm+nm+nx0+nx1)
        # mm = min(nm+max(nx0,nx1), nmax)
        mL = min(min(ns0[i],ns1[j]), nmax)
        mH = min(max(ns0[i],ns1[j]), nmax)
        nx = min(nx0+nx1, mmax)

      else:
        if pself and (j0+j <= i0+i): continue
        if ns0[i]==ns1[j]:
          if bits_count(codes0[i]&codes1[j])/min(_nbit0_[i],_nbit1_[j]) < min_simil: continue
          ret = _match_pair_score(Es0, _cidx0_[i], ns0[i], _dEs0_, _mtc0_,
                                  Es1, _cidx1_[j], ns1[j], _dEs1_, _mtc1_, max_dE, &ana, &cnt, &nm, &nx0, &nx1)
          if ret<0: return ret
          if ana==_0: continue

          dig = <CFLOAT32>(nm)/<CFLOAT32>(nm+max(nx1,nx0))
          # mm = min(ns0[i], nmax)
          mL = ns0[i]
          mH = mL
          nx = min(max(nx1,nx0), mmax)

        elif ns0[i]>ns1[j]:
          if bits_count(codes0[i]&codes1[j])/_nbit1_[j] < min_simil: continue
          ret = _match_pair_score(Es0, _cidx0_[i], ns0[i], _dEs0_, _mtc0_,
                                  Es1, _cidx1_[j], ns1[j], _dEs1_, _mtc1_, max_dE, &ana, &cnt, &nm, &nx0, &nx1)
          if ret<0: return ret
          if ana==_0: continue

          dig = <CFLOAT32>(nm)/<CFLOAT32>(nm+nx1)
          # mm = min(nm+nx1, nmax)
          mL = ns1[j]
          mH = ns0[i]
          nx = min(nx1, mmax)

        else:
          if bits_count(codes0[i]&codes1[j])/_nbit0_[i] < min_simil: continue
          ret = _match_pair_score(Es0, _cidx0_[i], ns0[i], _dEs0_, _mtc0_,
                                  Es1, _cidx1_[j], ns1[j], _dEs1_, _mtc1_, max_dE, &ana, &cnt, &nm, &nx0, &nx1)
          if ret<0: return ret
          if ana==_0: continue

          dig = <CFLOAT32>(nm)/<CFLOAT32>(nm+nx0)
          # mm = min(nm+nx0, nmax)
          mL = ns0[i]
          mH = ns1[j]
          nx = min(nx0, mmax)

      if (dig<0.) or (dig>1.): return -10000
      if (ana<0.) or (ana>1.): return -10001
      if (ana<min_ana) or (dig<min_dig): continue

      _data_[iseq] = ((<CUINT64>(65535*ana))<<48)+((<CUINT64>(65535*dig))<<32)+((<CUINT64>mH)<<16)+((<CUINT64>mL)<<8)+((<CUINT64>nx))
      _indices_[iseq] = j0+j
      iseq += 1
      if iseq >= max_data: return -10002
        
    _indptr_[i0+i+1] = iseq

  _indptr_[i0+i+1:] = iseq

  return csr_array((data[:iseq], indices[:iseq], indptr), (shape0, shape1))
# -------

# -------------------------------- PROBABILITY ---------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# error code -20000
def prob_sing(CFLOAT32[:] Es0, CUINT8[:] ns0, CUINT64[:] codes0, CINT32 i0,
              CFLOAT32[:] Es1, CUINT8[:] ns1, CUINT64[:] codes1, CINT32 j0, CINT32 shape0, CINT32 shape1, CUINT8 pself, CUINT8 symm, CFLOAT32 density,
              CINT32 lmin, CFLOAT32 min_simil, CFLOAT32 min_dig, CFLOAT32 max_dE, CFLOAT32 ratio_miss, CFLOAT32 min_prob, CFLOAT32 p0lm, pfit):

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

  assert len(pfit.pok_x) == pfit.pok_val.shape[0]
  assert len(pfit.pok_y) == pfit.pok_val.shape[1]
  assert len(pfit.pno_x) == pfit.pno_val.shape[0]
  assert len(pfit.pno_y) == pfit.pno_val.shape[1]
  assert len(pfit.pwrgs) == pfit.poknx.shape[0]
  # assert pfit.pset.shape[0] == pfit.pok_set.shape[0]
  # assert pfit.pset.shape[1] == pfit.pok_set.shape[1]
  # assert pfit.pset_comb.shape[0] == pfit.pok_set_comb.shape[0]
  # assert pfit.pset_comb.shape[1] == pfit.pok_set_comb.shape[1]

  cdef CUINT64 _nn0=0, _nn1=0, nMax, iseq=0
  cdef CINT32 i, j, ret, nx, nx0, nx1, nm, mm, xmax, lmax=<CINT32>max(max(ns0),max(ns1))
  cdef CFLOAT32 pout

  for i in range(len0):
    if ns0[i]>lmin: _nn0 += 1
  for i in range(len1):
    if ns1[i]>lmin: _nn1 += 1
  nMax = np.uint64(np.ceil(density*_nn0*_nn1))

  indptr = np.zeros(shape0+1, dtype=np.uint64)
  cdef CUINT64[:] _indptr_ = indptr
  indices = np.full(nMax, -1, dtype=np.int32)
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

  cdef CFLOAT32[:] pok_x = pfit.pok_x
  cdef CFLOAT32[:] pok_y = pfit.pok_y
  cdef CFLOAT32[:] pno_x = pfit.pno_x
  cdef CFLOAT32[:] pno_y = pfit.pno_y
  cdef CFLOAT32[:] pok_val = pfit.pok_val.ravel('C')
  cdef CFLOAT32[:] pno_val = pfit.pno_val.ravel('C')
  cdef CFLOAT32[:] pwrgs = pfit.pwrgs
  cdef CFLOAT32[:] poknx = pfit.poknx.ravel('C')
  # cdef CFLOAT32[:] pset = pfit.pset.ravel('C')
  # cdef CFLOAT32[:] pok_set = pfit.pok_set.ravel('C')
  # cdef CFLOAT32[:] pset_comb = pfit.pset_comb.ravel('C')
  # cdef CFLOAT32[:] pok_set_comb = pfit.pok_set_comb.ravel('C')
  
  cdef ItpMatrix Mok = {'x': &pok_x[0], 'y': &pok_y[0], 'Z': &pok_val[0], 'dx': <CFLOAT32>np.mean(np.diff(pok_x)), 'dy': <CFLOAT32>np.mean(np.diff(pok_y)),
                        'nx': <CINT32>len(pok_x), 'ny': <CINT32>len(pok_y)}

  cdef ItpMatrix Mno = {'x': &pno_x[0], 'y': &pno_y[0], 'Z': &pno_val[0], 'dx': <CFLOAT32>np.mean(np.diff(pno_x)), 'dy': <CFLOAT32>np.mean(np.diff(pno_y)),
                        'nx': <CINT32>len(pno_x), 'ny': <CINT32>len(pno_y)}

  # cdef PinfoStrct pinfo = {'Mok': Mok, 'Mno': Mno, 'pset': &pset[0], 'pok_set': &pok_set[0], 'pset_comb': &pset[0], 'pok_set_comb': &pok_set[0], 'pwrgs': &pwrgs[0],
  #                          'nmax': <CINT32>(pfit.pset.shape[0]-1), 'mmax': <CINT32>(pfit.pset.shape[1]-1)}
  cdef PinfoStrct pinfo = {'Mok': Mok, 'Mno': Mno, 'poknx': &poknx[0], 'pwrgs': &pwrgs[0], 'nmax': <CINT32>(pfit.poknx.shape[0]-1), 'mmax': <CINT32>(pfit.poknx.shape[1]-1)}
  
  plm = np.zeros(lmax, dtype=np.float32)
  cdef CFLOAT32[:] _plm_ = plm
  plx = np.zeros(lmax, dtype=np.float32)
  cdef CFLOAT32[:] _plx_ = plx
  lnx = np.zeros(lmax, dtype=np.int32)
  cdef CINT32[:] _lnx_ = lnx
  isort = np.zeros(lmax, dtype=np.int32)
  cdef CINT32[:] _isort_ = isort

  cdef CINT32 imax = binom(pinfo.mmax//4, pinfo.mmax//2)+1
  pprod = np.zeros(imax, dtype=np.float32)
  cdef CFLOAT32[:] _pprod_ = pprod
  nxs = np.zeros(imax, dtype=np.int32)
  cdef CINT32[:] _nxs_ = nxs
  pok_nx = np.zeros(pinfo.mmax+1, dtype=np.float32)
  cdef CFLOAT32[:] _pok_nx_ = pok_nx

  cdef LprobStrct lprob = {'plm': &_plm_[0], 'plx': &_plx_[0], 'lnx': &_lnx_[0], 'isort': &_isort_[0], 'pprod': &_pprod_[0], 'nxs': &_nxs_[0], 'pok_nx': &_pok_nx_[0],
                           'cnt': 0, 'lmax': lmax, 'imax': imax}

  for i in range(len0):
    for j in range(len1):
      if symm:
        if pself and (j0+j <= i0+i): continue
        if _2*bits_count(codes0[i]&codes1[j])/(_nbit0_[i]+_nbit1_[j]) < min_simil: continue

        ret = _match_pair_prob(Es0, _cidx0_[i], ns0[i], _dEs0_, _mtc0_, Es1, _cidx1_[j], ns1[j], _dEs1_, _mtc1_,
                               &lprob, &nm, &nx0, &nx1, &pinfo, max_dE)
        if ret<0: return ret
        if (<CFLOAT32>(nm+nm))/<CFLOAT32>(nm+nm+nx0+nx1) < min_dig: continue

        mm = min(max(nm+nx0,nm+nx1), pinfo.nmax)
        nx = min(nx0+nx1, pinfo.mmax)

      else:
        # if pself and (j0+j == i0+i): continue
        # if (ns0[i]<lmin) or (ns1[j]>=ns0[i]-1): continue

        # if bits_count(codes0[i]&codes1[j])/_nbit1_[j] < min_simil: continue

        # ret = _match_pair_prob(Es0, _cidx0_[i], ns0[i], _dEs0_, _mtc0_, Es1, _cidx1_[j], ns1[j], _dEs1_, _mtc1_,
        #                        &lprob, &nm, &nx0, &nx1, &pinfo, max_dE)
        # if ret<0: return ret
        # if (<CFLOAT32>nm)/<CFLOAT32>(nm+nx1) < min_dig: continue

        # mm = min(nm+nx1, pinfo.nmax)
        # nx = min(nx1, pinfo.mmax)

        if pself and (j0+j <= i0+i): continue
        if ns0[i]==ns1[j]:
          if bits_count(codes0[i]&codes1[j])/min(_nbit0_[i],_nbit1_[j]) < min_simil: continue
          ret = _match_pair_prob(Es0, _cidx0_[i], ns0[i], _dEs0_, _mtc0_, Es1, _cidx1_[j], ns1[j], _dEs1_, _mtc1_,
                                &lprob, &nm, &nx0, &nx1, &pinfo, max_dE)
          if ret<0: return ret
          if <CFLOAT32>(nm)/<CFLOAT32>(nm+max(nx1,nx0)) < min_dig: continue

          mm = min(ns0[i], pinfo.nmax)
          nx = min(max(nx1,nx0), pinfo.mmax)

        elif ns0[i]>ns1[j]:
          if bits_count(codes0[i]&codes1[j])/_nbit1_[j] < min_simil: continue
          ret = _match_pair_prob(Es0, _cidx0_[i], ns0[i], _dEs0_, _mtc0_, Es1, _cidx1_[j], ns1[j], _dEs1_, _mtc1_,
                                &lprob, &nm, &nx0, &nx1, &pinfo, max_dE)
          if ret<0: return ret
          if <CFLOAT32>(nm)/<CFLOAT32>(nm+nx1) < min_dig: continue

          mm = min(nm+nx1, pinfo.nmax)
          nx = min(nx1, pinfo.mmax)

        else:
          if bits_count(codes0[i]&codes1[j])/_nbit0_[i] < min_simil: continue
          ret = _match_pair_prob(Es0, _cidx0_[i], ns0[i], _dEs0_, _mtc0_, Es1, _cidx1_[j], ns1[j], _dEs1_, _mtc1_,
                                &lprob, &nm, &nx0, &nx1, &pinfo, max_dE)
          if ret<0: return ret
          if <CFLOAT32>(nm)/<CFLOAT32>(nm+nx0) < min_dig: continue

          mm = min(nm+nx0, pinfo.nmax)
          nx = min(nx0, pinfo.mmax)

      if pinfo.mmax-nx<2: xmax = 0
      else:               xmax = min(max(1,<CINT32>round(<CFLOAT32>lprob.cnt/ratio_miss)), (pinfo.mmax-nx)//2)

      ret = _calc_prob(&lprob, &pinfo, mm, nx, xmax, &pout)
      if ret<0: return ret
      if pout<min_prob: continue

      _data_[iseq] = ((<CUINT64>(<CUINT32*>&pout)[0])<<32) | ((<CUINT64>mm)<<8) | (<CUINT64>nx)
      _indices_[iseq] = j0+j
      iseq += 1
      if iseq >= nMax: return -20000
        
    _indptr_[i0+i+1] = iseq

  _indptr_[i0+i+1:] = iseq

  return csr_array((data[:iseq], indices[:iseq], indptr), (shape0, shape1))
# -------

# # @cython.cdivision(True)
# # @cython.boundscheck(False)
# # @cython.wraparound(False)
# # errror code -21000
# def prob_sing_list(CFLOAT32[:] Es0, CUINT8[:] ns0, CINT32[:] idxs0, CFLOAT32[:] Es1, CUINT8[:] ns1, CINT32[:] idxs1,
#                    CUINT8 symm, CFLOAT32 max_dE, CFLOAT32 ratio_miss, pfit):

#   assert len(Es0) == sum(ns0)
#   assert len(Es1) == sum(ns1)
#   assert len(idxs0) == len(idxs1)

#   assert len(pfit.pok_x) == pfit.pok_val.shape[0]
#   assert len(pfit.pok_y) == pfit.pok_val.shape[1]
#   assert len(pfit.pno_x) == pfit.pno_val.shape[0]
#   assert len(pfit.pno_y) == pfit.pno_val.shape[1]
#   assert len(pfit.pwrgs) == pfit.pset.shape[0]
#   assert pfit.pset.shape[0] == pfit.pok_set.shape[0]
#   assert pfit.pset.shape[1] == pfit.pok_set.shape[1]
#   assert pfit.pset_comb.shape[0] == pfit.pok_set_comb.shape[0]
#   assert pfit.pset_comb.shape[1] == pfit.pok_set_comb.shape[1]

#   cdef CUINT8 lmax = <CUINT8>max(max(ns0),max(ns1))

#   cidx0 = np.insert(np.cumsum(ns0,dtype=np.int32),0,0)
#   cdef CINT32[:] _cidx0_ = cidx0
#   cidx1 = np.insert(np.cumsum(ns1,dtype=np.int32),0,0)
#   cdef CINT32[:] _cidx1_ = cidx1

#   pdata = np.zeros(len(idxs0), dtype=np.float32)
#   cdef CFLOAT32[:] _pdata_ = pdata

#   dEs0 = np.zeros(lmax, dtype=np.float32)
#   cdef CFLOAT32[:] _dEs0_ = dEs0
#   dEs1 = np.zeros(lmax, dtype=np.float32)
#   cdef CFLOAT32[:] _dEs1_ = dEs1

#   mtc0 = np.full(lmax, -1, dtype=np.float32)
#   cdef CINT32[:] _mtc0_ = mtc0
#   mtc1 = np.full(lmax, -1, dtype=np.float32)
#   cdef CINT32[:] _mtc1_ = mtc1

#   cdef CINT32 i, k, ret, nm, nx, xmax, mm, nx0, nx1, N=<CINT32>len(idxs0)
#   cdef CFLOAT32 pok, pno

#   cdef CFLOAT32[:] pok_x = pfit.pok_x
#   cdef CFLOAT32[:] pok_y = pfit.pok_y
#   cdef CFLOAT32[:] pno_x = pfit.pno_x
#   cdef CFLOAT32[:] pno_y = pfit.pno_y
#   cdef CFLOAT32[:] pok_val = pfit.pok_val.ravel('C')
#   cdef CFLOAT32[:] pno_val = pfit.pno_val.ravel('C')
#   cdef CFLOAT32[:] pwrgs = pfit.pwrgs
#   cdef CFLOAT32[:] pset = pfit.pset.ravel('C')
#   cdef CFLOAT32[:] pok_set = pfit.pok_set.ravel('C')
#   cdef CFLOAT32[:] pset_comb = pfit.pset_comb.ravel('C')
#   cdef CFLOAT32[:] pok_set_comb = pfit.pok_set_comb.ravel('C')
  
#   cdef ItpMatrix Mok = {'x': &pok_x[0], 'y': &pok_y[0], 'Z': &pok_val[0], 'dx': <CFLOAT32>np.mean(np.diff(pok_x)), 'dy': <CFLOAT32>np.mean(np.diff(pok_y)),
#                         'nx': <CINT32>len(pok_x), 'ny': <CINT32>len(pok_y)}

#   cdef ItpMatrix Mno = {'x': &pno_x[0], 'y': &pno_y[0], 'Z': &pno_val[0], 'dx': <CFLOAT32>np.mean(np.diff(pno_x)), 'dy': <CFLOAT32>np.mean(np.diff(pno_y)),
#                         'nx': <CINT32>len(pno_x), 'ny': <CINT32>len(pno_y)}

#   cdef PinfoStrct pinfo = {'Mok': Mok, 'Mno': Mno, 'pset': &pset[0], 'pok_set': &pok_set[0], 'pset_comb': &pset[0], 'pok_set_comb': &pok_set[0], 'pwrgs': &pwrgs[0],
#                            'nmax': <CINT32>(pfit.pset.shape[0]-1), 'mmax': <CINT32>(pfit.pset.shape[1]-1)}

#   plm = np.zeros(lmax, dtype=np.float32)
#   cdef CFLOAT32[:] _plm_ = plm
#   plx = np.zeros(lmax, dtype=np.float32)
#   cdef CFLOAT32[:] _plx_ = plx
#   lnx = np.zeros(lmax, dtype=np.int32)
#   cdef CINT32[:] _lnx_ = lnx
#   isort = np.zeros(lmax, dtype=np.int32)
#   cdef CINT32[:] _isort_ = isort

#   cdef CINT32 imax = binom(pinfo.mmax//4, pinfo.mmax//2)+1
#   pprod = np.zeros(imax, dtype=np.float32)
#   cdef CFLOAT32[:] _pprod_ = pprod
#   nxs = np.zeros(imax, dtype=np.int32)
#   cdef CINT32[:] _nxs_ = nxs
#   pok_nx = np.zeros(pinfo.mmax+1, dtype=np.float32)
#   cdef CFLOAT32[:] _pok_nx_ = pok_nx

#   cdef LprobStrct lprob = {'plm': &_plm_[0], 'plx': &_plx_[0], 'lnx': &_lnx_[0], 'isort': &_isort_[0], 'pprod': &_pprod_[0], 'nxs': &_nxs_[0], 'pok_nx': &_pok_nx_[0],
#                           'cnt': 0, 'lmax': lmax, 'imax': imax}

#   for i in range(N):
#     if (not symm) and (ns1[idxs1[i]]>=ns0[idxs0[i]]-1): continue 

#     ret = _match_pair_prob(Es0, _cidx0_[idxs0[i]], ns0[idxs0[i]], _dEs0_, _mtc0_, Es1, _cidx1_[idxs1[i]], ns1[idxs1[i]], _dEs1_, _mtc1_,
#                             &lprob, &nm, &nx0, &nx1, &pinfo, max_dE)
#     if ret<0: return ret

#     if symm:
#       mm = min(max(nm+nx0,nm+nx1), pinfo.nmax)
#       nx = min(nx0+nx1, pinfo.mmax)
#     else:
#       mm = min(nm+nx1, pinfo.nmax)
#       nx = min(nx1, pinfo.mmax)
#     if pinfo.mmax-nx<2: xmax = 0
#     else:               xmax = min(max(1,<CINT32>round(<CFLOAT32>lprob.cnt/ratio_miss)), (pinfo.mmax-nx)//2)

#     ret = _calc_prob(&lprob, &pinfo, mm, nx, xmax, &_pdata_[i])
#     if ret<0: return ret
  
#   return pdata
# # -------

# ------------------------------ SIMPLE MATCHING -------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def simple_match_symm(poss, key):
  cdef CINT32 nrow = <CINT32>poss.shape[0]-1

  cdef CINT32[:] indptr_r = poss.indptr
  cdef CINT32[:] indices_r = poss.indices
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



# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def lines_combs(CFLOAT32[:] lm, CFLOAT32[:] lx, CINT32 m):
#   # All possible products of m elements from lx vs the remaining elements from lm

#   assert len(lm)==len(lx)
#   assert m<=len(lm)

#   cdef CINT32 ret
#   cdef CFLOAT32 out
#   ret = _lines_combs(&lm[0],&lx[0], <CINT32>len(lm), m, &out)
#   if ret<0: return ret
  
#   return out
# # -------


# cdef extern from "stdlib.h":
#   ctypedef void const_void "const void"
#   void qsort(void *base, int nmemb, int size, int(*compar)(const_void *, const_void *)) nogil

# cdef struct IndexedElement:
#   CINT32_t index
#   CFLOAT32_t value

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef int _compare(const_void *a, const_void *b):
#   cdef CFLOAT32_t v = (<IndexedElement*> a).value-(<IndexedElement*> b).value
#   if v < 0: return -1
#   if v >= 0: return 1
# # -------

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef int argsort(CFLOAT32_t[:] data, CINT32_t[:] idxs, CINT32_t n):
#   cdef CINT32_t ii
#   cdef IndexedElement *order_struct = <IndexedElement *> malloc(n * sizeof(IndexedElement))
  
#   for ii in range(n):
#     order_struct[ii].index = ii
#     order_struct[ii].value = data[ii]
      
#   qsort(<void *> order_struct, n, sizeof(IndexedElement), _compare)
  
#   for ii in range(n): idxs[ii] = order_struct[ii].index

#   free(order_struct)

#   return 0
# # -------





# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def score_comb(CFLOAT32_t[:] Es0, CFLOAT32_t[:] Ws0, CUINT8_t[:] ns0, CUINT64_t[:] codes0, CINT32_t i0,
#                CFLOAT32_t[:] Es1, CFLOAT32_t[:] Ws1, CFLOAT32_t[:,:] ijks1, CUINT8_t[:] ns1, CUINT64_t[:] codes1,
#                CUINT8_t pself, CUINT8_t best, CINT32_t pmax, CUINT8_t mmax, CUINT8_t mmin0, CFLOAT32_t ratio, CFLOAT32_t max_dE, CFLOAT32_t min_simil, CFLOAT32_t min_dig, CFLOAT32_t min_ana,
#                CINT32_t comb_n, CFLOAT32_t density, CFLOAT32_t min_imp, CFLOAT32_t min_tmp, CFLOAT32_t max_dr, CFLOAT32_t min_dig_comb, CFLOAT32_t min_scr_comb):
#   # This functions finds combinations of 1s in each 0 ----------

#   cidx0 = np.insert(np.cumsum(ns0,dtype=np.int32),0,0)
#   cdef CINT32_t[:] _cidx0_ = cidx0
#   cidx1 = np.insert(np.cumsum(ns1,dtype=np.int32),0,0)
#   cdef CINT32_t[:] _cidx1_ = cidx1

#   cdef Py_ssize_t len0 = len(ns0)
#   assert len0 < 2**32
#   assert len(Es0) == sum(ns0)
#   assert len(Es0) == len(Ws0)  
#   assert len0 == len(codes0)

#   cdef Py_ssize_t len1 = len(ns1)
#   assert len1 < 2**32
#   assert len(Es1) == sum(ns1)
#   assert len(Es1) == len(Ws1)
#   assert len1 == len(codes1)

#   # --- Variables ---
#   cdef CINT32_t Nlarge=0, i, j, k, idx, ret, pcnt, pc2, pc3, pc4, pr2, pr3, pr4, cnt2=0, cnt3=0, cnt4=0, ic2=0, ic3=0, ic4=0
#   cdef CFLOAT32_t ana, dig, nMA, nMB, wA, wB

#   for i in range(len0):
#     if ns0[i]>=mmin0: Nlarge+=1

#   # --- Arrays for scoring --- 
#   matched0 = np.full(mmax, -1, dtype=CINT32)
#   cdef CINT32_t[:] _matched0_ = matched0
#   matched1 = np.full(mmax, -1, dtype=CINT32)
#   cdef CINT32_t[:] _matched1_ = matched1

#   tots = np.full(len1, 0., dtype=CFLOAT32)
#   cdef CFLOAT32_t[:] _tots_ = tots
#   idxs = np.full(len1, -1, dtype=CINT32)
#   cdef CINT32_t[:] _idxs_ = idxs
#   isort = np.full(len1, -1, dtype=CINT32)
#   cdef CINT32_t[:] _isort_ = isort
#   idxs1 = np.full(pmax, -1, dtype=CINT32)
#   cdef CINT32_t[:] _idxs1_ = idxs1
#   isortC = np.full(pmax*pmax*(pmax-1)//2, -1, dtype=CINT32)
#   cdef CINT32_t[:] _isortC_ = isortC

#   nbit1 = _nbits64(codes1)
#   cdef CFLOAT32_t[:] _nbit1_ = nbit1

#   # --- Arrays for combinations --- 
#   pc2 = <CINT32_t>(pmax*(pmax-1)//2)
#   comb2_idx = np.zeros((pc2,2), dtype=CINT32)
#   comb2_scr = np.zeros((pc2,2), dtype=CFLOAT32)
#   comb2_nms = np.zeros((pc2,3), dtype=CUINT8)
#   cdef CINT32_t[:,:] _comb2_idx_ = comb2_idx
#   cdef CFLOAT32_t[:,:] _comb2_scr_ = comb2_scr
#   cdef CUINT8_t[:,:] _comb2_nms_ = comb2_nms

#   pr2 = pc2*<CINT32_t>max(1.,(density*<CFLOAT32_t>Nlarge))
#   ret2_idx = np.zeros((pr2,3), dtype=CINT32)
#   ret2_scr = np.zeros(pr2, dtype=CUINT64)
#   cdef CINT32_t[:,:] _ret2_idx_ = ret2_idx
#   cdef CUINT64_t[:] _ret2_scr_ = ret2_scr

#   pc3 = <CINT32_t>(pmax*pmax*(pmax-1)//2) if comb_n>=3 else 1
#   comb3_idx = np.zeros((pc3,3), dtype=CINT32)
#   comb3_scr = np.zeros((pc3,2), dtype=CFLOAT32)
#   comb3_nms = np.zeros((pc3,3), dtype=CUINT8)
#   cdef CINT32_t[:,:] _comb3_idx_ = comb3_idx
#   cdef CFLOAT32_t[:,:] _comb3_scr_ = comb3_scr
#   cdef CUINT8_t[:,:] _comb3_nms_ = comb3_nms

#   pr3 = pc3*<CINT32_t>max(1.,(density*<CFLOAT32_t>Nlarge))
#   ret3_idx = np.zeros((pr3,4), dtype=CINT32)
#   ret3_scr = np.zeros(pr3, dtype=CUINT64)
#   cdef CINT32_t[:,:] _ret3_idx_ = ret3_idx
#   cdef CUINT64_t[:] _ret3_scr_ = ret3_scr

#   pc4 = <CINT32_t>(pmax*pmax*(pmax-1)//2) if comb_n>=4 else 1
#   comb4_idx = np.zeros((pc4,4), dtype=CINT32)
#   comb4_scr = np.zeros((pc4,2), dtype=CFLOAT32)
#   comb4_nms = np.zeros((pc4,3), dtype=CUINT8)
#   cdef CINT32_t[:,:] _comb4_idx_ = comb4_idx
#   cdef CFLOAT32_t[:,:] _comb4_scr_ = comb4_scr
#   cdef CUINT8_t[:,:] _comb4_nms_ = comb4_nms

#   pr4 = pc4*<CINT32_t>max(1.,(density*<CFLOAT32_t>Nlarge))
#   ret4_idx = np.zeros((pr4,5), dtype=CINT32)
#   ret4_scr = np.zeros(pr4, dtype=CUINT64)
#   cdef CINT32_t[:,:] _ret4_idx_ = ret4_idx
#   cdef CUINT64_t[:] _ret4_scr_ = ret4_scr

#   used = np.zeros(len1, dtype=CUINT8)
#   cdef CUINT8_t[:] _used_ = used

#   for i in range(len0):
#     if ns0[i]<mmin0: continue
#     # --- Find possible 1 in 0 ---
#     pcnt = 0
#     for j in range(len1):
#       if pself and i==j: continue
#       if (ns1[j] < <CUINT8_t>floor(<CFLOAT32_t>ns0[i]/ratio)) or (ns1[j] > ns0[i]-2): continue

#       if bits_count(codes0[i]&codes1[j])/_nbit1_[j] >= min_simil:
#         ret = _fscore(Es0, Ws0, _cidx0_[i], ns0[i], _matched0_, Es1, Ws1, _cidx1_[j], ns1[j], _matched1_, max_dE, &ana, &wA, &wB, &nMA, &nMB)
#         if ret<0: return ret

#         dig = nMB/wB
#         if (ana>min_ana) and (dig>min_dig):
#           _tots_[pcnt] = ana*dig
#           _idxs_[pcnt] = j
#           pcnt += 1
    
#     if pcnt<2: continue

#     # --- Find and score possible combinations ---
#     argsort_F32(&_tots_[0], &_isort_[0], pcnt)
#     for j in range(min(pmax,pcnt)): _idxs1_[j] = _idxs_[_isort_[pcnt-j-1]]
#     ret = _score_combinations(Es0, Ws0, _cidx0_[i], ns0[i], Es1, Ws1, ijks1, _cidx1_, ns1, _idxs1_, min(pmax,pcnt),
#                               _comb2_idx_, _comb2_scr_, _comb2_nms_, pc2, _comb3_idx_, _comb3_scr_, _comb3_nms_, pc3, _comb4_idx_, _comb4_scr_, _comb4_nms_, pc4,
#                               _matched0_, _matched1_, &cnt2, &cnt3, &cnt4, max_dE, comb_n, min_imp, min_tmp, max_dr)

#     if (ret==-2)or(ret==-3)or(ret==-4): raise RuntimeError(f'Error in "score_combinations": comb{-ret} too small!')

#     for j in range(len1): _used_[j] = 0
#     # --- Return the best combinations (n=2) --- 
#     argsort_F32(&_comb2_scr_[:,0], &_isortC_[0], cnt2)
#     for j in range(cnt2):
#       idx = _isortC_[cnt2-j-1]
#       if _comb2_scr_[idx,0] < min_scr_comb: break
#       if _comb2_scr_[idx,1] < min_dig_comb: continue
#       if best and (_used_[_comb2_idx_[idx,0]] or _used_[_comb2_idx_[idx,1]]): continue
#       _ret2_idx_[ic2,0] = i0+i
#       _ret2_idx_[ic2,1] = _comb2_idx_[idx,0]
#       _ret2_idx_[ic2,2] = _comb2_idx_[idx,1]
#       _ret2_scr_[ic2] = (<CUINT64_t>(65535.*_comb2_scr_[idx,0]/_comb2_scr_[idx,1])<<48)+(<CUINT64_t>(65535.*_comb2_scr_[idx,1])<<32)+((<CUINT64_t>ns0[i])<<24)+\
#                         ((<CUINT64_t>_comb2_nms_[idx,0])<<16)+((<CUINT64_t>_comb2_nms_[idx,1])<<8)+((<CUINT64_t>_comb2_nms_[idx,2]))

#       _used_[_comb2_idx_[idx,0]] = 1
#       _used_[_comb2_idx_[idx,1]] = 1

#       ic2 += 1
#       if ic2 >= pr2: raise RuntimeError('Not enough memory for ret2')

#     # --- Return the best combinations (n=3) --- 
#     if comb_n >= 3:
#       argsort_F32(_comb3_scr_[:,0], _isortC_, cnt3)
#       for j in range(cnt3):
#         idx = _isortC_[cnt3-j-1]
#         if _comb3_scr_[idx,0] < min_scr_comb: break
#         if _comb3_scr_[idx,1] < min_dig_comb: continue
#         if best and (_used_[_comb3_idx_[idx,0]] or _used_[_comb3_idx_[idx,1]] or _used_[_comb3_idx_[idx,2]]): continue

#         _ret3_idx_[ic3,0] = i0+i
#         _ret3_idx_[ic3,1] = _comb3_idx_[idx,0]
#         _ret3_idx_[ic3,2] = _comb3_idx_[idx,1]
#         _ret3_idx_[ic3,3] = _comb3_idx_[idx,2]
#         _ret3_scr_[ic3] = (<CUINT64_t>(65535.*_comb3_scr_[idx,0]/_comb3_scr_[idx,1])<<48)+(<CUINT64_t>(65535.*_comb3_scr_[idx,1])<<32)+((<CUINT64_t>ns0[i])<<24)+\
#                           ((<CUINT64_t>_comb3_nms_[idx,0])<<16)+((<CUINT64_t>_comb3_nms_[idx,1])<<8)+((<CUINT64_t>_comb3_nms_[idx,2]))

#         _used_[_comb3_idx_[idx,0]] = 1
#         _used_[_comb3_idx_[idx,1]] = 1
#         _used_[_comb3_idx_[idx,2]] = 1

#         ic3 += 1
#         if ic3 >= pr3: raise RuntimeError('Not enough memory for ret3')
  
#     # --- Return the best combinations (n=4) --- 
#     if comb_n >= 4:
#       argsort_F32(_comb4_scr_[:,0], _isortC_, cnt4)
#       for j in range(cnt4):
#         idx = _isortC_[cnt4-j-1]
#         if _comb4_scr_[idx,0] < min_scr_comb: break
#         if _comb4_scr_[idx,1] < min_dig_comb: continue
#         if best and (_used_[_comb4_idx_[idx,0]] or _used_[_comb4_idx_[idx,1]] or _used_[_comb4_idx_[idx,2]] or _used_[_comb4_idx_[idx,3]]): continue

#         _ret4_idx_[ic4,0] = i0+i
#         _ret4_idx_[ic4,1] = _comb4_idx_[idx,0]
#         _ret4_idx_[ic4,2] = _comb4_idx_[idx,1]
#         _ret4_idx_[ic4,3] = _comb4_idx_[idx,2]
#         _ret4_idx_[ic4,4] = _comb4_idx_[idx,3]
#         _ret4_scr_[ic4] = (<CUINT64_t>(65535.*_comb4_scr_[idx,0]/_comb4_scr_[idx,1])<<48)+(<CUINT64_t>(65535.*_comb4_scr_[idx,1])<<32)+\
#                           ((<CUINT64_t>_comb4_nms_[idx,0])<<16)+((<CUINT64_t>_comb4_nms_[idx,1])<<8)+((<CUINT64_t>_comb4_nms_[idx,2]))

#         _used_[_comb4_idx_[idx,0]] = 1
#         _used_[_comb4_idx_[idx,1]] = 1
#         _used_[_comb4_idx_[idx,2]] = 1
#         _used_[_comb4_idx_[idx,3]] = 1

#         ic4 += 1
#         if ic4 >= pr4: raise RuntimeError('Not enough memory for ret4')

#   return {2: {'matches': ret2_idx[:ic2], 'score': ret2_scr[:ic2]},
#           3: {'matches': ret3_idx[:ic3], 'score': ret3_scr[:ic3]} if comb_n>=3 else None,
#           4: {'matches': ret4_idx[:ic4], 'score': ret4_scr[:ic4]} if comb_n>=4 else None}
# # -------

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef CINT32_t _score_combinations(CFLOAT32_t[:] Es0, CFLOAT32_t[:] Ws0, CINT32_t i0, CUINT8_t _n0,
#                                   CFLOAT32_t[:] Es1, CFLOAT32_t[:] Ws1, CFLOAT32_t[:,:] ijks1, CINT32_t[:] cidx1, CUINT8_t[:] ns1, CINT32_t[:] idxs1, CINT32_t len1,
#                                   CINT32_t[:,:] comb2_idx, CFLOAT32_t[:,:] comb2_scr, CUINT8_t[:,:] comb2_nms, CINT32_t ncomb2,
#                                   CINT32_t[:,:] comb3_idx, CFLOAT32_t[:,:] comb3_scr, CUINT8_t[:,:] comb3_nms, CINT32_t ncomb3,
#                                   CINT32_t[:,:] comb4_idx, CFLOAT32_t[:,:] comb4_scr, CUINT8_t[:,:] comb4_nms, CINT32_t ncomb4,
#                                   CINT32_t[:] mtcA, CINT32_t[:] mtcB, CINT32_t *cnt2, CINT32_t *cnt3, CINT32_t *cnt4,
#                                   CFLOAT32_t max_dE, CINT32_t comb_n, CFLOAT32_t min_imp, CFLOAT32_t min_tmp, CFLOAT32_t max_dr):

#   # --- Looking for 1 in 0 ----------
#   cdef CFLOAT32_t dig, ana, scr, w0, w1, nM0, nM1, aa, nM, wT, dr
#   cdef CINT32_t ret, i, j, k, n1max=0, n0=_n0
  
#   for i in range(len1):
#     if ns1[i]>n1max: n1max=ns1[i]

#   cdef CFLOAT32_t *cells1 = <CFLOAT32_t *> malloc(n0*len1*sizeof(CFLOAT32_t))
#   cdef CFLOAT32_t *cells2 = <CFLOAT32_t *> malloc(n0*((len1-1)*len1//2)*sizeof(CFLOAT32_t))
#   cdef CFLOAT32_t *cells3 = <CFLOAT32_t *> malloc(n0*((len1-1)*len1*len1//2)*sizeof(CFLOAT32_t))
#   cdef CFLOAT32_t *anas1 = <CFLOAT32_t *> malloc(len1*sizeof(CFLOAT32_t))
#   cdef CFLOAT32_t *anas2 = <CFLOAT32_t *> malloc(((len1-1)*len1//2)*sizeof(CFLOAT32_t))
#   cdef CFLOAT32_t *anas3 = <CFLOAT32_t *> malloc(((len1-1)*len1*len1//2)*sizeof(CFLOAT32_t))
#   cdef CFLOAT32_t *nM1s = <CFLOAT32_t *> malloc(len1*sizeof(CFLOAT32_t))
#   cdef CFLOAT32_t *nM2s = <CFLOAT32_t *> malloc(((len1-1)*len1//2)*sizeof(CFLOAT32_t))
#   cdef CFLOAT32_t *nM3s = <CFLOAT32_t *> malloc(((len1-1)*len1*len1//2)*sizeof(CFLOAT32_t))
#   cdef CFLOAT32_t *w1s = <CFLOAT32_t *> malloc(len1*sizeof(CFLOAT32_t))
#   cdef CFLOAT32_t *w2s = <CFLOAT32_t *> malloc(((len1-1)*len1//2)*sizeof(CFLOAT32_t))
#   cdef CFLOAT32_t *w3s = <CFLOAT32_t *> malloc(((len1-1)*len1*len1//2)*sizeof(CFLOAT32_t))
#   cdef CFLOAT32_t *ijks2 = <CFLOAT32_t *> malloc(3*((len1-1)*len1//2)*sizeof(CFLOAT32_t))
#   cdef CFLOAT32_t *ijks3 = <CFLOAT32_t *> malloc(3*((len1-1)*len1*len1//2)*sizeof(CFLOAT32_t))
#   cdef CFLOAT32_t *scrs1 = <CFLOAT32_t *> malloc(len1*sizeof(CFLOAT32_t))
#   cdef CFLOAT32_t *digs1 = <CFLOAT32_t *> malloc(len1*sizeof(CFLOAT32_t))

#   # Get alignments
#   for i in range(len1):
#     ret = _fscore(Es0, Ws0, i0, _n0, mtcA, Es1, Ws1, cidx1[idxs1[i]], ns1[idxs1[i]], mtcB, max_dE, &ana, &w0, &w1, &nM0, &nM1)

#     digs1[i] = (nM0+nM1)/(w0+w1)
#     scrs1[i] = ana*digs1[i]
#     for k in range(n0):
#       if mtcA[k]>=0: cells1[i*n0+k] = _1-abs(Es0[i0+k]-Es1[cidx1[idxs1[i]]+mtcA[k]])/max_dE
#       else:          cells1[i*n0+k] = 0.

#     w1s[i] = _0
#     nM1s[i] = _0
#     anas1[i] = _0
#     for k in range(ns1[idxs1[i]]):
#       if mtcB[k]>=0:
#         w1s[i] += _1
#         nM1s[i] += _1
#         anas1[i] += cells1[i*n0+mtcB[k]]
#       else:
#         w1s[i] += Ws1[cidx1[idxs1[i]]+k]

#   # Find 2-combinations
#   cnt2[0] = 0
#   if comb_n >= 2:
#     for i in range(len1):
#       for j in range(len1):
#         if idxs1[j] <= idxs1[i]: continue

#         if max_dr>0.:
#           dr = ((ijks1[idxs1[i],0]-ijks1[idxs1[j],0])**_2+\
#                 (ijks1[idxs1[i],1]-ijks1[idxs1[j],1])**_2+\
#                 (ijks1[idxs1[i],2]-ijks1[idxs1[j],2])**_2)**_05
#           if dr>max_dr: continue

#         nM0 = _0
#         wT = w1s[i]+w1s[j]
#         ana = anas1[i]+anas1[j]
#         for k in range(n0):
#           aa = max(cells1[i*n0+k],cells1[j*n0+k])
#           cells2[cnt2[0]*n0+k] = aa
#           if aa>0.:
#             ana += aa
#             nM0 += _1
#             wT += _1
#           else:
#             wT += Ws0[i0+k]
        
#         scr = ana/wT
#         dig = (nM0+nM1s[i]+nM1s[j])/wT
#         if (dig>digs1[i]+.05) and (dig>digs1[j]+.05) and \
#            (scr>scrs1[i]+min_imp) and (scr>scrs1[j]+min_imp) and (scr>min_tmp):
#           comb2_idx[cnt2[0],0] = idxs1[i]
#           comb2_idx[cnt2[0],1] = idxs1[j]

#           comb2_scr[cnt2[0],0] = scr
#           comb2_scr[cnt2[0],1] = dig
          
#           comb2_nms[cnt2[0],0] = <CUINT8_t>nM0
#           comb2_nms[cnt2[0],1] = ns1[idxs1[i]]+ns1[idxs1[j]]
#           comb2_nms[cnt2[0],2] = <CUINT8_t>(nM1s[i]+nM1s[j])

#           if max_dr>0:
#             ijks2[3*cnt2[0]+0] = (ijks1[idxs1[i],0]+ijks1[idxs1[j],0])/_2
#             ijks2[3*cnt2[0]+1] = (ijks1[idxs1[i],1]+ijks1[idxs1[j],1])/_2
#             ijks2[3*cnt2[0]+2] = (ijks1[idxs1[i],2]+ijks1[idxs1[j],2])/_2

#           w2s[cnt2[0]] = w1s[i]+w1s[j]
#           nM2s[cnt2[0]] = nM1s[i]+nM1s[j]
#           anas2[cnt2[0]] = anas1[i]+anas1[j]
#           cnt2[0] += 1
#           if cnt2[0]>ncomb2: return -1

#   # Find 3-combinations
#   cnt3[0] = 0
#   if comb_n >= 3:
#     for i in range(cnt2[0]):
#       for j in range(len1):
#         if idxs1[j] <= comb2_idx[i,1]: continue

#         if max_dr>0.:
#           dr = ((ijks2[3*i+0]-ijks1[idxs1[j],0])**_2+\
#                 (ijks2[3*i+1]-ijks1[idxs1[j],1])**_2+\
#                 (ijks2[3*i+2]-ijks1[idxs1[j],2])**_2)**_05
#           if dr>max_dr: continue

#         nM0 = _0
#         wT = w2s[i]+w1s[j]
#         ana = anas2[i]+anas1[j]
#         for k in range(n0):
#           aa = max(cells2[i*n0+k],cells1[j*n0+k])
#           cells3[cnt3[0]*n0+k] = aa
#           if aa>0.:
#             ana += aa
#             nM0 += _1
#             wT += _1
#           else:
#             wT += Ws0[i0+k]

#         scr = ana/wT
#         dig = (nM0+nM2s[i]+nM1s[j])/wT
#         if (dig>comb2_scr[i,1]+.05) and (dig>digs1[j]+.05) and \
#            (scr>comb2_scr[i,0]+min_imp) and (scr>scrs1[j]+min_imp) and (scr>min_tmp):
#           comb3_idx[cnt3[0],0] = comb2_idx[i,0]
#           comb3_idx[cnt3[0],1] = comb2_idx[i,1]
#           comb3_idx[cnt3[0],2] = idxs1[j]

#           comb3_scr[cnt3[0],0] = scr
#           comb3_scr[cnt3[0],1] = dig

#           comb3_nms[cnt3[0],0] = <CUINT8_t>nM0
#           comb3_nms[cnt3[0],1] = comb2_nms[i,1]+ns1[idxs1[j]]
#           comb3_nms[cnt3[0],2] = <CUINT8_t>(nM2s[i]+nM1s[j])

#           if max_dr>0:
#             ijks3[3*cnt3[0]+0] = (ijks2[3*i+0]+ijks1[idxs1[j],0])/_2
#             ijks3[3*cnt3[0]+1] = (ijks2[3*i+1]+ijks1[idxs1[j],1])/_2
#             ijks3[3*cnt3[0]+2] = (ijks2[3*i+2]+ijks1[idxs1[j],2])/_2

#           w3s[cnt3[0]] = w2s[i]+w1s[j]
#           nM3s[cnt3[0]] = nM2s[i]+nM1s[j]
#           anas3[cnt3[0]] = anas2[i]+anas1[j]
#           cnt3[0] += 1
#           if cnt3[0]>ncomb3: return -1

#   # Find 4-combinations
#   cnt4[0] = 0
#   if comb_n >= 4:
#     for i in range(cnt3[0]):
#       for j in range(len1):
#         if idxs1[j] <= comb3_idx[i,2]: continue

#         if max_dr>0.:
#           dr = ((ijks3[3*i+0]-ijks1[idxs1[j],0])**_2+\
#                 (ijks3[3*i+1]-ijks1[idxs1[j],1])**_2+\
#                 (ijks3[3*i+2]-ijks1[idxs1[j],2])**_2)**_05
#           if dr>max_dr: continue

#         nM0 = _0
#         wT = w3s[i]+w1s[j]
#         ana = anas3[i]+anas1[j]
#         for k in range(n0):
#           aa = max(cells3[i*n0+k],cells1[j*n0+k])
#           if aa>0.:
#             ana += aa
#             nM0 += _1
#             wT += _1
#           else:
#             wT += Ws0[i0+k]

#         scr = ana/wT
#         dig = (nM0+nM3s[i]+nM1s[j])/wT
#         if (dig>comb3_scr[i,1]+.05) and (dig>digs1[j]+.05) and \
#            (scr>comb3_scr[i,0]+min_imp) and (scr>scrs1[j]+min_imp) and (scr>min_tmp):
#           comb4_idx[cnt4[0],0] = comb3_idx[i,0]
#           comb4_idx[cnt4[0],1] = comb3_idx[i,1]
#           comb4_idx[cnt4[0],2] = comb3_idx[i,2]
#           comb4_idx[cnt4[0],3] = idxs1[j]

#           comb4_scr[cnt4[0],0] = scr
#           comb4_scr[cnt4[0],1] = dig

#           comb4_nms[cnt4[0],0] = <CUINT8_t>nM0
#           comb4_nms[cnt4[0],1] = comb3_nms[i,1]+ns1[idxs1[j]]
#           comb4_nms[cnt4[0],2] = <CUINT8_t>(nM3s[i]+nM1s[j])

#           cnt4[0] += 1
#           if cnt4[0]>ncomb4: return -1

#   free(cells1)
#   free(cells2)
#   free(cells3)
#   free(anas1)
#   free(anas2)
#   free(anas3)
#   free(nM1s)
#   free(nM2s)
#   free(nM3s)
#   free(w1s)
#   free(w2s)
#   free(w3s)
#   free(ijks2)
#   free(ijks3)
#   free(digs1)
#   free(scrs1)

#   return 0
# -------



# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef CINT32_t _fscore(CFLOAT32_t[:] Es0, CFLOAT32_t[:] Ws0, CINT32_t cidx0, CUINT8_t _n0, CINT32_t[:] matched0,
#                       CFLOAT32_t[:] Es1, CFLOAT32_t[:] Ws1, CINT32_t cidx1, CUINT8_t _n1, CINT32_t[:] matched1,
#                       CFLOAT32_t max_dE, CFLOAT32_t* ana, CFLOAT32_t* w0, CFLOAT32_t* w1, CFLOAT32_t* nM0, CFLOAT32_t* nM1):
  
#   cdef CINT32_t i0, i1, j1, k, n0=_n0, n1=_n1
#   cdef CFLOAT32_t curr, right, bottom
#   cdef CUINT8_t do0=True, do1=True, mtc0=False, mtc1=False

#   for i0 in range(n0): matched0[i0] = -1
#   for i1 in range(n1): matched1[i1] = -1

#   i0 = i1 = 0
#   ana[0] = _0
#   nM0[0] = _0
#   nM1[0] = _0
#   w0[0] = _0
#   w1[0] = _0

#   while True:
#     if (i0>=n0) or (i1>=n1): break
#     for j1 in range(i1,n1):
#       curr = (Es1[cidx1+j1]-Es0[cidx0+i0])/max_dE
#       i1 = j1
#       if curr>-1.: break # until E1 is far below current E0, cycle through Es1
#       do1 = True
    
#     # If current E0 is far above current E1, go to next E0
#     if curr>=1.:
#       i0 += 1
#       do0 = True
#       continue

#     curr = _1-abs(curr)

#     # Check scores of adjacent matches
#     if j1<n1-1: right = max(_0,_1-abs(Es1[cidx1+j1+1]-Es0[cidx0+i0])/max_dE)
#     else:       right = 0.
#     if i0<n0-1: bottom = max(_0,_1-abs(Es1[cidx1+j1]-Es0[cidx0+i0+1])/max_dE)
#     else:       bottom = 0.
    
#     if do0 and (curr>right):
#       matched0[i0] = j1
#       ana[0] += curr
#       nM0[0] += _1
#       do0 = False

#     if do1 and (curr>bottom):
#       matched1[j1] = i0
#       ana[0] += curr
#       nM1[0] += _1
#       do1 = False

#     if (bottom>0.) and (bottom>right):
#       i0 += 1
#       do0 = True
#       continue

#     if (right>0.) and (right>bottom):
#       i1 += 1
#       do1 = True
#       continue

#     if (bottom>0.) and (bottom==right):
#       if bottom > max(_0,_1-abs(Es1[cidx1+j1+1]-Es0[cidx0+i0+1])/max_dE): return -1

#     i0 += 1
#     i1 += 1
#     do0 = do1 = True

#   for k in range(n0):
#     if matched0[k]>=0: w0[0] += _1
#     else:              w0[0] += Ws0[cidx0+k]
#   for k in range(n1):
#     if matched1[k]>=0: w1[0] += _1
#     else:              w1[0] += Ws1[cidx1+k]

#   if nM0[0]>0: ana[0] = ana[0]/(nM0[0]+nM1[0])
#   else:        ana[0] = _0

#   return 0
# # -------





# # @cython.cdivision(True)
# # @cython.boundscheck(False)
# # @cython.wraparound(False)
# # error code -25000
# def prob_comb(CFLOAT32_t[:] Es0, CUINT8_t[:] ns0, CUINT64_t[:] codes0, CINT32_t i0,
#               CFLOAT32_t[:] Es1, CUINT8_t[:] ns1, CUINT64_t[:] codes1, CFLOAT32_t[:,:] ijks1, CUINT8_t pself, CUINT8_t best,
#               CUINT8_t mmax, CUINT8_t mmin, CINT32_t pmax, CFLOAT32_t ratio, CFLOAT32_t max_dE, CFLOAT32_t pmtc_in, CFLOAT32_t min_simil_in, CFLOAT32_t min_prb_in,
#               CINT32_t comb_n, CFLOAT32_t density, CFLOAT32_t min_prob, CFLOAT32_t ratio_miss, CFLOAT32_t pmtc,
#               CFLOAT32_t[:] bok_e, CFLOAT32_t[:] bok_de, CFLOAT32_t[:,:] pok_ede, CFLOAT32_t[:,:] pok_miss, CFLOAT32_t[:,:] pok_miss_comb, 
#               CFLOAT32_t[:] bno_e, CFLOAT32_t[:] bno_de, CFLOAT32_t[:,:] pno_ede, CFLOAT32_t[:,:] pno_miss, CFLOAT32_t[:,:] pno_miss_comb):
#   # This functions finds combinations of 1s in each 0 ----------

#   cidx0 = np.insert(np.cumsum(ns0,dtype=np.int32),0,0)
#   cdef CINT32_t[:] _cidx0_ = cidx0
#   cidx1 = np.insert(np.cumsum(ns1,dtype=np.int32),0,0)
#   cdef CINT32_t[:] _cidx1_ = cidx1

#   cdef Py_ssize_t len0 = len(ns0)
#   assert len0 < 2**32
#   assert len(Es0) == sum(ns0)
#   assert len0 == len(codes0)

#   cdef Py_ssize_t len1 = len(ns1)
#   assert len1 < 2**32
#   assert len(Es1) == sum(ns1)
#   assert len1 == len(codes1)

#   # --- Variables ---
#   cdef CINT32_t Nlarge=0, i, j, k, idx, ret,
#   cdef CINT32_t cnt, nM, nm0, nm1, nm, nH, max_miss, ny=<CINT32_t>pok_miss.shape[1]
#   cdef CINT32_t pcnt, pc2, pc3, pc4, pr2, pr3, pr4, cnt2=0, cnt3=0, cnt4=0, ic2=0, ic3=0, ic4=0
#   cdef CFLOAT32_t pout, nMA, nMB, wA, wB

#   for i in range(len0):
#     if ns0[i]>=mmin: Nlarge+=1

#   # --- Arrays for scoring ---
#   prbs = np.full(len1, 0., dtype=CFLOAT32)
#   cdef CFLOAT32_t[:] _prbs_ = prbs
#   idxs = np.full(len1, -1, dtype=CINT32)
#   cdef CINT32_t[:] _idxs_ = idxs
#   isort = np.full(len1, -1, dtype=CINT32)
#   cdef CINT32_t[:] _isort_ = isort
#   idxs1 = np.full(pmax, -1, dtype=CINT32)
#   cdef CINT32_t[:] _idxs1_ = idxs1
#   isortC = np.full(pmax*pmax*(pmax-1)//2, -1, dtype=CINT32)
#   cdef CINT32_t[:] _isortC_ = isortC

#   nbit1 = _nbits64(codes1)
#   cdef CFLOAT32_t[:] _nbit1_ = nbit1

#   # --- Prob calculation stuff --- 
#   dEs0 = np.zeros(mmax, dtype=CFLOAT32)
#   cdef CFLOAT32_t[:] _dEs0_ = dEs0
#   dEs1 = np.zeros(mmax, dtype=CFLOAT32)
#   cdef CFLOAT32_t[:] _dEs1_ = dEs1

#   mtc0 = np.full(mmax, -1, dtype=CINT32)
#   cdef CINT32_t[:] _mtc0_ = mtc0
#   mtc1 = np.full(mmax, -1, dtype=CINT32)
#   cdef CINT32_t[:] _mtc1_ = mtc1

#   plm = np.zeros(mmax, dtype=CFLOAT32)
#   cdef CFLOAT32_t[:] _plm_ = plm
#   plx = np.zeros(mmax, dtype=CFLOAT32)
#   cdef CFLOAT32_t[:] _plx_ = plx

#   cdef CFLOAT32_t[:] pok_ede1 = np.asarray(pok_ede).ravel()
#   cdef CFLOAT32_t[:] pno_ede1 = np.asarray(pno_ede).ravel()
#   cdef CFLOAT32_t[:] pok_miss1 = np.asarray(pok_miss).ravel()
#   cdef CFLOAT32_t[:] pno_miss1 = np.asarray(pno_miss).ravel()
#   cdef CFLOAT32_t[:] pok_miss_comb1 = np.asarray(pok_miss_comb).ravel()
#   cdef CFLOAT32_t[:] pno_miss_comb1 = np.asarray(pno_miss_comb).ravel()

#   cdef ItpMatrix Mok = {'x': &bok_e[0], 'y': &bok_de[0], 'Z': &pok_ede1[0],
#                         'dx': <CFLOAT32_t>np.mean(np.diff(bok_e)),
#                         'dy': <CFLOAT32_t>np.mean(np.diff(bok_de)),
#                         'nx': <CINT32_t>len(bok_e),
#                         'ny': <CINT32_t>len(bok_de)}

#   cdef ItpMatrix Mno = {'x': &bno_e[0], 'y': &bno_de[0], 'Z': &pno_ede1[0],
#                         'dx': <CFLOAT32_t>np.mean(np.diff(bno_e)),
#                         'dy': <CFLOAT32_t>np.mean(np.diff(bno_de)),
#                         'nx': <CINT32_t>len(bno_e),
#                         'ny': <CINT32_t>len(bno_de)}

#   cdef PinfoStrct pinfo = {'Mok': Mok, 'Mno': Mno, 'Pok_miss': &pok_miss1[0], 'Pno_miss': &pno_miss1[0], 'Pok_miss_comb': &pok_miss_comb1[0], 'Pno_miss_comb': &pno_miss_comb1[0],
#                            'nmax': <CINT32_t>(pok_miss.shape[0]-1), 'mmax': <CINT32_t>(pok_miss.shape[1]-1)}

#   # --- Arrays for combinations --- 
#   pc2 = <CINT32_t>(pmax*(pmax-1)//2)
#   comb2_idx = np.zeros((pc2,2), dtype=CINT32)
#   comb2_scr = np.zeros(pc2, dtype=CFLOAT32)
#   # comb2_nms = np.zeros((pc2,3), dtype=CUINT8)
#   cdef CINT32_t[:,:] _comb2_idx_ = comb2_idx
#   cdef CFLOAT32_t[:] _comb2_scr_ = comb2_scr
#   # cdef CUINT8_t[:,:] _comb2_nms_ = comb2_nms

#   pr2 = pc2*<CINT32_t>max(1.,(density*<CFLOAT32_t>Nlarge))
#   ret2_idx = np.zeros((pr2,3), dtype=CINT32)
#   ret2_scr = np.zeros(pr2, dtype=CFLOAT32)
#   cdef CINT32_t[:,:] _ret2_idx_ = ret2_idx
#   cdef CFLOAT32_t[:] _ret2_scr_ = ret2_scr

#   pc3 = <CINT32_t>(pmax*pmax*(pmax-1)//2) if comb_n>=3 else 1
#   comb3_idx = np.zeros((pc3,3), dtype=CINT32)
#   comb3_scr = np.zeros(pc3, dtype=CFLOAT32)
#   # comb3_nms = np.zeros((pc3,3), dtype=CUINT8)
#   cdef CINT32_t[:,:] _comb3_idx_ = comb3_idx
#   cdef CFLOAT32_t[:] _comb3_scr_ = comb3_scr
#   # cdef CUINT8_t[:,:] _comb3_nms_ = comb3_nms

#   pr3 = pc3*<CINT32_t>max(1.,(density*<CFLOAT32_t>Nlarge))
#   ret3_idx = np.zeros((pr3,4), dtype=CINT32)
#   ret3_scr = np.zeros(pr3, dtype=CFLOAT32)
#   cdef CINT32_t[:,:] _ret3_idx_ = ret3_idx
#   cdef CFLOAT32_t[:] _ret3_scr_ = ret3_scr

#   pc4 = <CINT32_t>(pmax*pmax*(pmax-1)//2) if comb_n>=4 else 1
#   comb4_idx = np.zeros((pc4,4), dtype=CINT32)
#   comb4_scr = np.zeros(pc4, dtype=CFLOAT32)
#   # comb4_nms = np.zeros((pc4,3), dtype=CUINT8)
#   cdef CINT32_t[:,:] _comb4_idx_ = comb4_idx
#   cdef CFLOAT32_t[:] _comb4_scr_ = comb4_scr
#   # cdef CUINT8_t[:,:] _comb4_nms_ = comb4_nms

#   pr4 = pc4*<CINT32_t>max(1.,(density*<CFLOAT32_t>Nlarge))
#   ret4_idx = np.zeros((pr4,5), dtype=CINT32)
#   ret4_scr = np.zeros(pr4, dtype=CFLOAT32)
#   cdef CINT32_t[:,:] _ret4_idx_ = ret4_idx
#   cdef CFLOAT32_t[:] _ret4_scr_ = ret4_scr

#   # used = np.zeros(len1, dtype=CUINT8)
#   # cdef CUINT8_t[:] _used_ = used

#   for i in range(len0):
#     if ns0[i]<mmin: continue
#     # --- Find possible 1 in 0 ---
#     pcnt = 0
#     for j in range(len1):
#       if pself and i==j: continue
#       if (ns1[j] < <CUINT8_t>floor(<CFLOAT32_t>ns0[i]/ratio)) or (ns1[j] >= ns0[i]-1): continue
#       if bits_count(codes0[i]&codes1[j])/_nbit1_[j] < min_simil_in: continue

#       ret = _match_pair_prob(Es0, _cidx0_[i], ns0[i], Es1, _cidx1_[j], ns1[j], max_dE,
#                              _plm_, _plx_, mmax, &cnt, &nM, &nm0, &nm1, &pinfo, _dEs0_, _dEs1_, _mtc0_, _mtc1_)
#       if ret<0: return ret

#       nH = min(nM+nm1, pinfo.nmax)
#       nm = min(2*nm1, pinfo.mmax)
#       max_miss = max(1,<CINT32_t>round(_2*(<CFLOAT32_t>nH)/ratio_miss))

#       ret = _calc_prob(&_plm_[0], &_plx_[0], &pinfo.Pok_miss[nH*ny], &pinfo.Pno_miss[nH*ny], pmtc_in, nm, cnt, min(max_miss,cnt), pinfo.mmax, &pout)
#       if ret<0: return ret
#       if pout < min_prb_in: continue

#       _prbs_[pcnt] = pout
#       _idxs_[pcnt] = j
#       pcnt += 1
    
#     if pcnt<2: continue

#     # --- Find and score possible combinations ---
#     argsort_F32(&_prbs_[0], &_isort_[0], pcnt)
#     for j in range(min(pmax,pcnt)): _idxs1_[j] = _idxs_[_isort_[pcnt-j-1]]

#     ret = _prob_combinations(Es0, _cidx0_[i], ns0[i], Es1, ijks1, _cidx1_, ns1, _idxs1_, min(pmax,pcnt),
#                              _comb2_idx_, _comb2_scr_, pc2, &cnt2, _comb3_idx_, _comb3_scr_, pc3, &cnt3, _comb4_idx_, _comb4_scr_, pc4, &cnt4,
#                              _dEs0_, _dEs1_, _mtc0_, _mtc1_, _plm_, _plx_, &pinfo, max_dE, mmax, ratio_miss, comb_n, pmtc)
#     if ret<0: return ret

#   #   for j in range(len1): _used_[j] = 0
#     # --- Return the best combinations (n=2) --- 
#     if comb_n >= 2:
#       argsort_F32(&_comb2_scr_[0], &_isortC_[0], cnt2)
#       for j in range(cnt2):
#         idx = _isortC_[cnt2-j-1]
#         if _comb2_scr_[idx] < min_prob: break
#         # if best and (_used_[_comb2_idx_[idx,0]] or _used_[_comb2_idx_[idx,1]]): continue

#         _ret2_idx_[ic2,0] = i0+i
#         _ret2_idx_[ic2,1] = _idxs1_[_comb2_idx_[idx,0]]
#         _ret2_idx_[ic2,2] = _idxs1_[_comb2_idx_[idx,1]]
#         _ret2_scr_[ic2] = _comb2_scr_[idx]

#         # _used_[_comb2_idx_[idx,0]] = 1
#         # _used_[_comb2_idx_[idx,1]] = 1

#         ic2 += 1
#         if ic2 >= pr2: return -25002

#   #   # --- Return the best combinations (n=3) --- 
#     if comb_n >= 3:
#       argsort_F32(&_comb3_scr_[0], &_isortC_[0], cnt3)
#       for j in range(cnt3):
#         idx = _isortC_[cnt3-j-1]
#         if _comb3_scr_[idx] < min_prob: break
#         # if best and (_used_[_comb3_idx_[idx,0]] or _used_[_comb3_idx_[idx,1]] or _used_[_comb3_idx_[idx,2]]): continue

#         _ret3_idx_[ic3,0] = i0+i
#         _ret3_idx_[ic3,1] = _idxs1_[_comb3_idx_[idx,0]]
#         _ret3_idx_[ic3,2] = _idxs1_[_comb3_idx_[idx,1]]
#         _ret3_idx_[ic3,3] = _idxs1_[_comb3_idx_[idx,2]]
#         _ret3_scr_[ic3] = _comb3_scr_[idx]

#         # _used_[_comb3_idx_[idx,0]] = 1
#         # _used_[_comb3_idx_[idx,1]] = 1
#         # _used_[_comb3_idx_[idx,2]] = 1

#         ic3 += 1
#         if ic3 >= pr3: return -25003
  
#   #   # --- Return the best combinations (n=4) --- 
#   #   if comb_n >= 4:
#   #     argsort(_comb4_scr_[:,0], _isortC_, cnt4)
#   #     for j in range(cnt4):
#   #       idx = _isortC_[cnt4-j-1]
#   #       if _comb4_scr_[idx,0] < min_scr_comb: break
#   #       if _comb4_scr_[idx,1] < min_dig_comb: continue
#   #       if best and (_used_[_comb4_idx_[idx,0]] or _used_[_comb4_idx_[idx,1]] or _used_[_comb4_idx_[idx,2]] or _used_[_comb4_idx_[idx,3]]): continue

#   #       _ret4_idx_[ic4,0] = i0+i
#   #       _ret4_idx_[ic4,1] = _comb4_idx_[idx,0]
#   #       _ret4_idx_[ic4,2] = _comb4_idx_[idx,1]
#   #       _ret4_idx_[ic4,3] = _comb4_idx_[idx,2]
#   #       _ret4_idx_[ic4,4] = _comb4_idx_[idx,3]
#   #       _ret4_scr_[ic4] = (<CUINT64_t>(65535.*_comb4_scr_[idx,0]/_comb4_scr_[idx,1])<<48)+(<CUINT64_t>(65535.*_comb4_scr_[idx,1])<<32)+\
#   #                         ((<CUINT64_t>_comb4_nms_[idx,0])<<16)+((<CUINT64_t>_comb4_nms_[idx,1])<<8)+((<CUINT64_t>_comb4_nms_[idx,2]))

#   #       _used_[_comb4_idx_[idx,0]] = 1
#   #       _used_[_comb4_idx_[idx,1]] = 1
#   #       _used_[_comb4_idx_[idx,2]] = 1
#   #       _used_[_comb4_idx_[idx,3]] = 1

#   #       ic4 += 1
#   #       if ic4 >= pr4: raise RuntimeError('Not enough memory for ret4')

#   return {2: {'matches': ret2_idx[:ic2], 'score': ret2_scr[:ic2]} if comb_n>=2 else None,
#           3: {'matches': ret3_idx[:ic3], 'score': ret3_scr[:ic3]} if comb_n>=3 else None,}

#   # return {2: {'matches': ret2_idx[:ic2], 'score': ret2_scr[:ic2]},
#   #         3: {'matches': ret3_idx[:ic3], 'score': ret3_scr[:ic3]} if comb_n>=3 else None,
#   #         4: {'matches': ret4_idx[:ic4], 'score': ret4_scr[:ic4]} if comb_n>=4 else None}
# # -------

# # @cython.cdivision(True)
# # @cython.boundscheck(False)
# # @cython.wraparound(False)
# # error code -26000
# cdef CINT32_t _prob_combinations(CFLOAT32_t[:] Es0, CINT32_t cidx0, CUINT8_t _n0, CFLOAT32_t[:] Es1, CFLOAT32_t[:,:] ijks1, CINT32_t[:] cidx1, CUINT8_t[:] ns1, CINT32_t[:] idxs1, CINT32_t len1,
#                                  CINT32_t[:,:] comb2_idx, CFLOAT32_t[:] comb2_scr, CINT32_t ncomb2, CINT32_t *cnt2,
#                                  CINT32_t[:,:] comb3_idx, CFLOAT32_t[:] comb3_scr, CINT32_t ncomb3, CINT32_t *cnt3,
#                                  CINT32_t[:,:] comb4_idx, CFLOAT32_t[:] comb4_scr, CINT32_t ncomb4, CINT32_t *cnt4,
#                                  CFLOAT32_t[:] dEs0, CFLOAT32_t[:] dEs1, CINT32_t[:] mtc0, CINT32_t[:] mtc1, CFLOAT32_t[:] plm, CFLOAT32_t[:] plx,                                 
#                                  PinfoStrct* pinfo, CFLOAT32_t max_dE, CINT32_t mmax, CFLOAT32_t ratio_miss, CINT32_t comb_n, CFLOAT32_t pmtc):

#   # --- Looking for 1 in 0 ----------
  
#   cdef CINT32_t ret, iA,iB,iC, j, s, k, n0=_n0, nH, nM, nm0, max_cnts=0, ny=pinfo[0].mmax+1
#   cdef CFLOAT32_t pout

#   cdef CFLOAT32_t *plms = <CFLOAT32_t*> malloc(mmax*len1*sizeof(CFLOAT32_t))
#   cdef CFLOAT32_t *plxs = <CFLOAT32_t*> malloc(mmax*len1*sizeof(CFLOAT32_t))
#   cdef CUINT8_t   *mtcs = <CUINT8_t*>   malloc(n0*len1*sizeof(CUINT8_t))
#   cdef CINT32_t   *cnts = <CINT32_t*>   malloc(len1*sizeof(CINT32_t))
#   cdef CINT32_t   *nm1s = <CINT32_t*>   malloc(len1*sizeof(CINT32_t))
#   cdef CINT32_t   *nM1s = <CINT32_t*>   malloc(len1*sizeof(CINT32_t))

#   # --- Get line match probabilities ----------
#   for j in range(len1):
#     ret = _match_pair_prob(Es0, cidx0, _n0, Es1, cidx1[idxs1[j]], ns1[idxs1[j]], max_dE, plm, plx, mmax,
#                            &cnts[j], &nM, &nm0, &nm1s[j], pinfo, dEs0, dEs1, mtc0, mtc1)
#     if ret<0: return ret
#     if cnts[j]>max_cnts: max_cnts=cnts[j]

#     for k in range(cnts[j]):
#       plms[j*n0+k] = plm[k]
#       plxs[j*n0+k] = plx[k]

#     nM1s[j] = 0
#     for k in range(n0):
#       if abs(dEs0[k])<max_dE:
#         mtcs[j*n0+k] = 1
#         nM1s[j] += 1
#       else:
#         mtcs[j*n0+k] = 0
  
#   max_cnts *= comb_n
#   cdef CFLOAT32_t *clms = <CFLOAT32_t*> malloc(max_cnts*len1*sizeof(CFLOAT32_t))
#   cdef CFLOAT32_t *clxs = <CFLOAT32_t*> malloc(max_cnts*len1*sizeof(CFLOAT32_t))

#   # Find 2-combinations
#   cnt2[0] = 0
#   if comb_n >= 2:
#     for iA in range(len1):
#       for j in range(iA+1,len1):
#         ret = _eval_comb2(&plms[iA*n0], &plxs[iA*n0], &mtcs[iA*n0], cnts[iA], nM1s[iA], nm1s[iA],
#                           &plms[j*n0],  &plxs[j*n0],  &mtcs[j*n0],  cnts[j],  nM1s[j],  nm1s[j],
#                           pinfo, n0, clms, clxs, ratio_miss, pmtc, &pout)
#         if ret==0: continue
#         if ret==1:
#           comb2_idx[cnt2[0],0] = iA
#           comb2_idx[cnt2[0],1] = j
#           comb2_scr[cnt2[0]] = pout
#           cnt2[0] += 1
#           if cnt2[0]>ncomb2: ret = -26021
#         if ret<0:
#           free(plms)
#           free(plxs)
#           free(clms)
#           free(clxs)
#           free(mtcs)
#           free(cnts)
#           free(nM1s)
#           free(nm1s)
#           return ret

#   # Find 3-combinations
#   cnt3[0] = 0
#   if comb_n >= 3:
#     for k in range(cnt2[0]):
#       iA = comb2_idx[k,0]
#       iB = comb2_idx[k,1]
#       for j in range(iB+1,len1):
#         ret = _eval_comb3(&plms[iA*n0], &plxs[iA*n0], &mtcs[iA*n0], cnts[iA], nM1s[iA], nm1s[iA],
#                           &plms[iB*n0], &plxs[iB*n0], &mtcs[iB*n0], cnts[iB], nM1s[iB], nm1s[iB],
#                           &plms[j*n0],  &plxs[j*n0],  &mtcs[j*n0],  cnts[j],  nM1s[j],  nm1s[j],
#                           pinfo, n0, clms, clxs, ratio_miss, pmtc, &pout)
#         if ret==0: continue
#         if ret==1:
#           comb3_idx[cnt3[0],0] = iA
#           comb3_idx[cnt3[0],1] = iB
#           comb3_idx[cnt3[0],2] = j
#           comb3_scr[cnt3[0]] = pout
#           cnt3[0] += 1
#           if cnt3[0]>ncomb3: ret = -26031
#         if ret<0:
#           free(plms)
#           free(plxs)
#           free(clms)
#           free(clxs)
#           free(mtcs)
#           free(cnts)
#           free(nM1s)
#           free(nm1s)
#           return ret

#   free(plms)
#   free(plxs)
#   free(clms)
#   free(clxs)
#   free(mtcs)
#   free(cnts)
#   free(nM1s)
#   free(nm1s)

#   return 0
# # -------


# # @cython.cdivision(True)
# # @cython.boundscheck(False)
# # @cython.wraparound(False)
# cdef inline CINT32_t _eval_comb2(CFLOAT32_t* plmA, CFLOAT32_t* plxA, CUINT8_t* mtcsA, CINT32_t cntA, CINT32_t nMA, CINT32_t nmA,
#                                  CFLOAT32_t* plmB, CFLOAT32_t* plxB, CUINT8_t* mtcsB, CINT32_t cntB, CINT32_t nMB, CINT32_t nmB,
#                                  PinfoStrct *pinfo, CINT32_t n0, CFLOAT32_t* clms, CFLOAT32_t* clxs, CFLOAT32_t ratio_miss, CFLOAT32_t pmtc, CFLOAT32_t* pout):
#   cdef CINT32_t k, cnt=0, nM=0, nm, nH, ny=pinfo[0].mmax+1
#   for k in range(cntA):
#     clms[cnt] = plmA[k]
#     clxs[cnt] = plxA[k]
#     cnt += 1
#   for k in range(cntB):
#     clms[cnt] = plmB[k]
#     clxs[cnt] = plxB[k]
#     cnt += 1
  
#   for k in range(n0):
#     if mtcsA[k]:   nM += 1
#     elif mtcsB[k]: nM += 1

#   if nMA+nMB-nM >= 2: return 0

#   nH = min(n0,pinfo[0].nmax)
#   nm = min(n0-nM+nmA+nmB, pinfo[0].mmax)
#   max_miss = max(1,<CINT32_t>round((<CFLOAT32_t>(n0+nMA+nmA+nMB+nmB))/ratio_miss))
#   ret = _calc_prob(clms, clxs, &pinfo[0].Pok_miss_comb[nH*ny], &pinfo[0].Pno_miss_comb[nH*ny], pmtc, nm, cnt, max_miss, pinfo[0].mmax, pout)
#   if ret<0: return ret

#   # if max_dr>0.:
#   #   dr = ((ijks1[idxs1[i],0]-ijks1[idxs1[j],0])**_2+\
#   #         (ijks1[idxs1[i],1]-ijks1[idxs1[j],1])**_2+\
#   #         (ijks1[idxs1[i],2]-ijks1[idxs1[j],2])**_2)**_05
#   #   if dr>max_dr: continue

#   return 1
# # -------

# # @cython.cdivision(True)
# # @cython.boundscheck(False)
# # @cython.wraparound(False)
# cdef inline CINT32_t _eval_comb3(CFLOAT32_t* plmA, CFLOAT32_t* plxA, CUINT8_t* mtcsA, CINT32_t cntA, CINT32_t nMA, CINT32_t nmA,
#                                  CFLOAT32_t* plmB, CFLOAT32_t* plxB, CUINT8_t* mtcsB, CINT32_t cntB, CINT32_t nMB, CINT32_t nmB,
#                                  CFLOAT32_t* plmC, CFLOAT32_t* plxC, CUINT8_t* mtcsC, CINT32_t cntC, CINT32_t nMC, CINT32_t nmC,
#                                  PinfoStrct *pinfo, CINT32_t n0, CFLOAT32_t* clms, CFLOAT32_t* clxs, CFLOAT32_t ratio_miss, CFLOAT32_t pmtc, CFLOAT32_t* pout):
#   cdef CINT32_t k, cnt=0, nM=0, nm, nH, ny=pinfo[0].mmax+1
#   for k in range(cntA):
#     clms[cnt] = plmA[k]
#     clxs[cnt] = plxA[k]
#     cnt += 1
#   for k in range(cntB):
#     clms[cnt] = plmB[k]
#     clxs[cnt] = plxB[k]
#     cnt += 1
#   for k in range(cntC):
#     clms[cnt] = plmC[k]
#     clxs[cnt] = plxC[k]
#     cnt += 1
  
#   for k in range(n0):
#     if mtcsA[k]:   nM += 1
#     elif mtcsB[k]: nM += 1
#     elif mtcsC[k]: nM += 1

#   if nMA+nMB+nMC-nM >= 2: return 0

#   nH = min(n0,pinfo[0].nmax)
#   nm = min(n0-nM+nmA+nmB+nmC, pinfo[0].mmax)
#   max_miss = max(1,<CINT32_t>round((<CFLOAT32_t>(n0+nMA+nmA+nMB+nmB+nMC+nmC))/ratio_miss))
#   ret = _calc_prob(clms, clxs, &pinfo[0].Pok_miss_comb[nH*ny], &pinfo[0].Pno_miss_comb[nH*ny], pmtc, nm, cnt, max_miss, pinfo[0].mmax, pout)
#   if ret<0: return ret

#   # if max_dr>0.:
#   #   dr = ((ijks1[idxs1[i],0]-ijks1[idxs1[j],0])**_2+\
#   #         (ijks1[idxs1[i],1]-ijks1[idxs1[j],1])**_2+\
#   #         (ijks1[idxs1[i],2]-ijks1[idxs1[j],2])**_2)**_05
#   #   if dr>max_dr: continue

#   return 1
# # -------


