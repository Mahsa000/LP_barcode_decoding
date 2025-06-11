cdef CINT32_t path2i[6]
cdef CINT32_t path2j[6]
path2i[:] = [0,1,0,-1,1,-1]
path2j[:] = [0,1,1,-1,0,-1]
cdef CINT32_t path3i[21]
cdef CINT32_t path3j[21]
path3i[:] = [0,1,2,0,1,-1,0,2,-1,0,1,-1,1,2,-1,0,2,-1,1,2,-1]
path3j[:] = [0,1,2,0,2,-1,1,2,-1,1,2,-1,0,1,-1,0,1,-1,0,2,-1]
cdef CINT32_t path4i[36]
cdef CINT32_t path4j[36]
path4i[:] = [0,1,2,3,0,1,2,-1,1,2,3,-1,0,1,2,-1,0,1,2,-1,0,1,3,-1,0,2,3,-1,0,1,3,-1,0,2,3,-1]
path4j[:] = [0,1,2,3,1,2,3,-1,0,1,2,-1,0,2,3,-1,0,1,3,-1,0,1,2,-1,0,1,2,-1,0,2,3,-1,0,1,3,-1]



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def fscore(dataA, dataB, simil, opt):
  assert len(dataA) == simil.shape[0]
  assert len(dataB) == simil.shape[1]

  cdef CINT32_t lenA = len(dataA), lenB = len(dataB), minB

  cdef CFLOAT_t[:] _simil_ = simil.data
  cdef CINT32_t[:] _indptr_ = simil.indptr
  cdef CINT32_t[:] _indices_ = simil.indices

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

  cdef CFLOAT_t min_simil = opt.min_simil
  cdef CFLOAT_t max_dE = opt.max_dE
  cdef CFLOAT_t th_amp = opt.th_amp
  cdef CFLOAT_t wgt_small = opt.wgt_small
  cdef CFLOAT_t min_dscr = opt.min_dscr
  cdef CFLOAT_t fE = opt.fE

  cdef CFLOAT_t n1LL, n1LS, n1SS, n0L, n0S, nS, nL, n1, adE, dscr, ascr, dE_match
  cdef CINT32_t matchedA, matchedB, ievtA, indB, ievtB, iEA, iEB, imatch, iA0, iB0
  cdef CUINT64_t iscr = 0

  dscrData = np.zeros_like(simil.data, dtype=CFLOAT)
  cdef CFLOAT_t[:] _dscrData_ = dscrData

  ascrData = np.zeros_like(simil.data, dtype=CFLOAT)
  cdef CFLOAT_t[:] _ascrData_ = ascrData

  for ievtA in range(lenA):
    iA0 = _cumA_[ievtA]
    for indB in range(_indptr_[ievtA],_indptr_[ievtA+1]):
      if _simil_[iscr] < min_simil:
        iscr += 1
        continue

      ievtB = _indices_[indB]
      iB0 = _cumB_[ievtB]
      matchedA = matchedB = 0
      minB = 0
      n1LL = n1LS = n1SS = n0L = n0S = ascr = 0.

      for iEA in range(_nsA_[ievtA]):
        dE_match = max_dE
        for iEB in range(minB, _nsB_[ievtB]):
          if matchedB & (1 << iEB): continue

          adE = abs(_EsA_[iA0+iEA]-_EsB_[iB0+iEB])
          if adE < dE_match:
            dE_match = adE
            imatch = iEB

        if dE_match < max_dE:
          ascr += dE_match
          matchedA |= (1 << iEA)
          matchedB |= (1 << imatch)
          minB = imatch+1

          if   (_AsA_[iA0+iEA]<th_amp)&(_AsB_[iB0+imatch]<th_amp): n1SS += 2
          elif (_AsA_[iA0+iEA]<th_amp)|(_AsB_[iB0+imatch]<th_amp): n1LS += 2
          else:                                                    n1LL += 2
        else:
          if _AsA_[iA0+iEA] < th_amp: n0S += 1
          else:                       n0L += 1

      for iEB in range(_nsB_[ievtB]):
        if matchedB & (1<<iEB): continue

        if _AsB_[iB0+iEB] < th_amp: n0S += 1
        else:                       n0L += 1

      nS = n1SS + n1LS/2 + n0S
      nL = n1LL + n1LS/2 + n0L
      n1 = n1SS+n1LS+n1LL

      dscr = (n1SS*wgt_small + n1LL + 0.5*n1LS*(1+wgt_small))/(nS*wgt_small + nL)
      if dscr >= min_dscr:
        _dscrData_[iscr] = dscr
        _ascrData_[iscr] = max(EPS, 1-2*ascr/(fE*n1))
      iscr += 1

  mat_ascr = csr_matrix((ascrData, simil.indices.copy(), simil.indptr.copy()), (lenA,lenB))
  mat_ascr.eliminate_zeros()
  mat_dscr = csr_matrix((dscrData, simil.indices.copy(), simil.indptr.copy()), (lenA,lenB))
  mat_dscr.eliminate_zeros()

  return mat_ascr, mat_dscr



# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
def find_AinB(EsA, nsA, EsB, nsB, CFLOAT_t max_dE=1., CINT32_t nmax=30):
  cdef CINT32_t lenA=len(nsA), lenB=len(nsB), iA, iB

  cdef CFLOAT_t[:] _EsA_ = EsA
  cdef CFLOAT_t[:] _EsB_ = EsB
  cdef CINT32_t[:] _nsA_ = nsA
  cdef CINT32_t[:] _nsB_ = nsB

  matchedA = np.full(nmax, -1, dtype=CINT32)
  cdef CINT32_t[:] _matchedA_ = matchedA

  jB0s = np.full(nmax, -1, dtype=CINT32)
  cdef CINT32_t[:] _jB0s_ = jB0s
  jB1s = np.full(nmax, -1, dtype=CINT32)
  cdef CINT32_t[:] _jB1s_ = jB1s
  mmat = np.full((nmax+1,nmax+1), 0., dtype=CFLOAT)
  cdef CFLOAT_t[:,:] _mmat_ = mmat

  cdef CINT32_t ret, iA0=0, iB0

  for jA in range(lenA):
    iB0 = 0
    for jB in range(lenB):
      _matrixAB(_EsA_, iA0, _nsA_[jA], _EsB_, iB0, _nsB_[jB], _mmat_, _jB0s_, _jB1s_, max_dE)
      ret = _walk_matrix(_mmat_, _jB0s_, _jB1s_, _nsA_[jA], _nsB_[jB], _matchedA_)
      if ret < 0: raise RuntimeError(f'Error {ret}!!!')
      iB0 += _nsB_[jB]
    iA0 += _nsA_[jA]

  return matchedA[:_nsA_[jA]], mmat[:_nsA_[jA],:_nsB_[jB]]

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
def find_AinB2(EsA, nsA, EsB, nsB, CFLOAT_t max_dE=1., CINT32_t nmax=30):
  cdef CINT32_t lenA=len(nsA), lenB=len(nsB), iA, iB

  cdef CFLOAT_t[:] _EsA_ = EsA
  cdef CFLOAT_t[:] _EsB_ = EsB
  cdef CINT32_t[:] _nsA_ = nsA
  cdef CINT32_t[:] _nsB_ = nsB

  matchedA = np.full(nmax, -1, dtype=CINT32)
  cdef CINT32_t[:] _matchedA_ = matchedA

  jB0s = np.full(nmax, -1, dtype=CINT32)
  cdef CINT32_t[:] _jB0s_ = jB0s
  jB1s = np.full(nmax, -1, dtype=CINT32)
  cdef CINT32_t[:] _jB1s_ = jB1s
  mmat = np.full((nmax+1,nmax+1), 0., dtype=CFLOAT)
  cdef CFLOAT_t[:,:] _mmat_ = mmat

  cdef CINT32_t ret, iA0=0, iB0

  for jA in range(lenA):
    iB0 = 0
    for jB in range(lenB):
      ret = _full_findAinB(_EsA_, iA0, _nsA_[jA], _EsB_, iB0, _nsB_[jB],
                           _mmat_, _jB0s_, _jB1s_, _matchedA_, max_dE)
      if ret < 0: raise RuntimeError(f'Error {ret}!!!')
      iB0 += _nsB_[jB]
    iA0 += _nsA_[jA]

  return matchedA[:_nsA_[jA]], mmat[:_nsA_[jA],:_nsB_[jB]]

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def matrixAB(EsA, EsB, CFLOAT_t max_dE, CINT32_t nmax=30):
  cdef CINT32_t nA=len(EsA), nB=len(EsB)
  cdef CFLOAT_t[:] _EsA_ = EsA
  cdef CFLOAT_t[:] _EsB_ = EsB

  jB0s = np.full(nmax, -1, dtype=CINT32)
  cdef CINT32_t[:] _jB0s_ = jB0s
  jB1s = np.full(nmax, -1, dtype=CINT32)
  cdef CINT32_t[:] _jB1s_ = jB1s
  mmat = np.full((nmax+1,nmax+1), 0., dtype=CFLOAT)
  cdef CFLOAT_t[:,:] _mmat_ = mmat

  _matrixAB(_EsA_, 0 , nA, _EsB_, 0, nB, _mmat_, _jB0s_, _jB1s_, max_dE)

  return mmat[:nA,:nB]

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef CINT32_t _matrixAB(CFLOAT_t[:] _EsA_, CINT32_t iA0, CINT32_t nA,
                        CFLOAT_t[:] _EsB_, CINT32_t iB0, CINT32_t nB,
                        CFLOAT_t[:,:] _mmat_, CINT32_t[:] _jB0s_, CINT32_t[:] _jB1s_,
                        CFLOAT_t max_dE):

  cdef CINT32_t jA, jB, jB0=0
  cdef CFLOAT_t dE

  for jA in range(nA+1):
    for jB in range(nB+1):
      _mmat_[jA,jB] = 0.

  for jA in range(nA):
    _jB0s_[jA] = -1
    _jB1s_[jA] = -1
    for jB in range(jB0, nB):
      dE = _EsB_[iB0+jB]-_EsA_[iA0+jA]
      if dE > max_dE:
        _mmat_[jA,jB] = 0.
        break

      dE = abs(dE)
      if dE >= max_dE:
        _mmat_[jA,jB] = 0.
      else:
        _mmat_[jA,jB] = max_dE-dE
        _jB1s_[jA] = jB
        if _jB0s_[jA]==-1:
          jB0 = jB
          _jB0s_[jA] = jB

  return 0

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef CINT32_t _walk_matrix(CFLOAT_t[:,:] _mmat_, CINT32_t[:] _iB0s_, CINT32_t[:] _iB1s_,
                           CINT32_t nA, CINT32_t nB, CINT32_t[:] _matchedA_):

  cdef CINT32_t iA=-1, iB=0, jj, nsq, nsq_row, nsq_col
  cdef CINT32_t *sq_match = <CINT32_t *> malloc(MAX_SQAURE_SIZE*sizeof(CINT32_t))
  cdef CFLOAT_t dsum

  while True: # cycling through iA
    iA += 1
    _matchedA_[iA] = -1
    # If there are no more rows left, finish
    if iA >= nA: break
    # If the current row does not have any possible match, go to next row
    if _iB0s_[iA] == -1: continue
    # Start from the highest between the first match of the row and the next available column
    iB = max(_iB0s_[iA], iB)
    # If there are no possible matches left after the start position, go to next row
    if iB > _iB1s_[iA]: continue
    # If there are no more columns left, finish
    if iB >= nB: break

    while True: # cycling through iB
      # If current position has lower direct neighbors, it's the best match
      if (_mmat_[iA+1,iB]<_mmat_[iA,iB]) and (_mmat_[iA,iB+1]<_mmat_[iA,iB]):
        _matchedA_[iA] = iB
        iB += 1
        break
      # If current position has a higher direct neighbor...
      else:
        # If diagonal neighbor is 0, move to the highest direct neighbor
        # (in this case direct neighbors cannot be equal by construction)
        if _mmat_[iA+1,iB+1] == 0:
          # If the highest neighbor is vertical, go to next row
          if _mmat_[iA+1,iB]>_mmat_[iA,iB+1]:
            break
          # If the highest neighbor is horizontal, go to next column
          else:
            iB += 1
            continue
        # If diagonal neighbor is >0, find the minimum square
        else:
          nsq_row = 2
          nsq_col = 2
          while True:
            if   _mmat_[iA+nsq_row,iB+nsq_col-1]>0: nsq_row += 1
            elif _mmat_[iA+nsq_row-1,iB+nsq_col]>0: nsq_col += 1
            else:                                   break

          nsq = max(nsq_row, nsq_col)
          if nsq == 2:
            dsum = _mmat_[iA,iB]+_mmat_[iA+1,iB+1]
            if (_mmat_[iA+1,iB] > _mmat_[iA,iB+1]) and (_mmat_[iA+1,iB] > dsum):
              _matchedA_[iA+1] = iB
            elif (_mmat_[iA,iB+1] > _mmat_[iA+1,iB]) and (_mmat_[iA,iB+1] > dsum):
              _matchedA_[iA] = iB+1
            else:
              _matchedA_[iA] = iB
              _matchedA_[iA+1] = iB+1
            iB += 2
            iA += 1
            break
          elif nsq <= 4:
            ret = _solve_square(_mmat_, iA, iB, nsq, sq_match)
          else:
            ret = -1*nsq

          if ret != 0:
            free(sq_match)
            return ret

          # Update the matches
          for jj in range(nsq):
            if sq_match[jj] == -1: continue
            _matchedA_[iA+jj] = iB+sq_match[jj]
          iB += nsq_col
          iA += nsq_row-1
          break

  free(sq_match)
  return 0

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef CINT32_t _solve_square(CFLOAT_t[:,:] _mmat_, CINT32_t iA0, CINT32_t iB0,
                            CINT32_t n, CINT32_t* _match_):
  cdef CINT32_t *_pathi_
  cdef CINT32_t *_pathj_
  cdef CINT32_t pp, ii, npath, row, col, pm
  cdef CFLOAT_t scr, cs, max_scr=0

  if n==3:
    _pathi_ = path3i
    _pathj_ = path3j
    npath = 7
  elif n==4:
    _pathi_ = path4i
    _pathj_ = path4j
    npath = 9
  else:
    return -1*n

  for pp in range(npath): # Cycle possible paths
    scr = 0
    pm = 1
    for ii in range(n): # Cycle individual positions in the path
      row = _pathi_[pp*n+ii]
      if row == -1: break
      col = _pathj_[pp*n+ii]

      cs = _mmat_[iA0+row,iB0+col]
      if cs == 0: # If any position is not a match, discard the path
        pm = 0
        break

      scr += cs

    if scr < max_scr: continue

    # Update with the current match data
    max_scr = scr

    for ii in range(n): _match_[ii] = -1
    for ii in range(n):
      row = _pathi_[pp*n+ii]
      if row == -1: break
      col = _pathj_[pp*n+ii]
      _match_[row] = col

  return 0

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef CINT32_t _full_findAinB(CFLOAT_t[:] _EsA_, CINT32_t iA0, CINT32_t nA,
                             CFLOAT_t[:] _EsB_, CINT32_t iB0, CINT32_t nB,
                             CFLOAT_t[:,:] _mmat_, CINT32_t[:] _jB0s_, CINT32_t[:] _jB1s_,
                             CINT32_t[:] _matchedA_, CFLOAT_t max_dE):

  cdef CINT32_t jA, jB, jB0=0, jj, nsq, nsq_row, nsq_col, pp, ii, npath, row, col, pm
  cdef CFLOAT_t dE, dsum, scr, cs, nf, max_scr
  cdef CINT32_t *sq_match = <CINT32_t *> malloc(MAX_SQAURE_SIZE*sizeof(CINT32_t))
  cdef CINT32_t *_pathi_
  cdef CINT32_t *_pathj_

  for jA in range(nA+1):
    for jB in range(nB+1):
      _mmat_[jA,jB] = 0.

  # Calc matrixAB
  for jA in range(nA):
    _jB0s_[jA] = -1
    _jB1s_[jA] = -1
    for jB in range(jB0, nB):
      dE = _EsB_[iB0+jB]-_EsA_[iA0+jA]
      if dE > max_dE:
        _mmat_[jA,jB] = 0.
        break

      dE = abs(dE)
      if dE >= max_dE:
        _mmat_[jA,jB] = 0.
      else:
        _mmat_[jA,jB] = max_dE-dE
        _jB1s_[jA] = jB
        if _jB0s_[jA]==-1:
          jB0 = jB
          _jB0s_[jA] = jB


  jA = -1
  jB = 0
  while True: # cycling through iA
    jA += 1
    _matchedA_[jA] = -1
    # If there are no more rows left, finish
    if jA >= nA: break
    # If the current row does not have any possible match, go to next row
    if _jB0s_[jA] == -1: continue
    # Start from the highest between the first match of the row and the next available column
    jB = max(_jB0s_[jA], jB)
    # If there are no possible matches left after the start position, go to next row
    if jB > _jB1s_[jA]: continue
    # If there are no more columns left, finish
    if jB >= nB: break

    while True: # cycling through iB
      # If current position has lower direct neighbors, it's the best match
      if (_mmat_[jA+1,jB]<_mmat_[jA,jB]) and (_mmat_[jA,jB+1]<_mmat_[jA,jB]):
        _matchedA_[jA] = jB
        jB += 1
        break
      # If current position has a higher direct neighbor...
      else:
        # If diagonal neighbor is 0, move to the highest direct neighbor
        # (in this case direct neighbors cannot be equal by construction)
        if _mmat_[jA+1,jB+1] == 0:
          # If the highest neighbor is vertical, go to next row
          if _mmat_[jA+1,jB]>_mmat_[jA,jB+1]:
            break
          # If the highest neighbor is horizontal, go to next column
          else:
            jB += 1
            continue
        # If diagonal neighbor is >0, find the minimum square
        else:
          nsq_row = 2
          nsq_col = 2
          while True:
            if   _mmat_[jA+nsq_row,jB+nsq_col-1]>0: nsq_row += 1
            elif _mmat_[jA+nsq_row-1,jB+nsq_col]>0: nsq_col += 1
            else:                                   break

          nsq = max(nsq_row, nsq_col)
          max_scr = 0
          if nsq <= 4:
            if nsq == 2:
              _pathi_ = path2i
              _pathj_ = path2j
              npath = 3
            elif nsq == 3:
              _pathi_ = path3i
              _pathj_ = path3j
              npath = 7
            elif nsq == 4:
              _pathi_ = path4i
              _pathj_ = path4j
              npath = 9

            for pp in range(npath): # Cycle possible paths
              scr = 0
              pm = 1
              for ii in range(nsq): # Cycle individual positions in the path
                row = _pathi_[pp*nsq+ii]
                if (row == -1) or (row >= nsq_row): break
                col = _pathj_[pp*nsq+ii]
                if (col >= nsq_col): break

                cs = _mmat_[jA+row,jB+col]
                if cs == 0: # If any position is not a match, discard the path
                  pm = 0
                  break
                scr += cs
              if scr < max_scr: continue

              # Update with the current match data
              max_scr = scr
              for ii in range(nsq): sq_match[ii] = -1
              for ii in range(nsq):
                row = _pathi_[pp*nsq+ii]
                if (row == -1) or (row >= nsq_row): break
                col = _pathj_[pp*nsq+ii]
                if (col >= nsq_col): break
                sq_match[row] = col

          else:
            free(sq_match)
            return -1*nsq

          # Update the matches
          for jj in range(nsq):
            if sq_match[jj] == -1: continue
            _matchedA_[jA+jj] = jB+sq_match[jj]
          jB += nsq_col
          jA += nsq_row-1
          break

  free(sq_match)
  return 0



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def find_squares(mmat):
  cdef CFLOAT_t[:,:] _mmat_ = mmat
  cdef CINT32_t nA = mmat.shape[0]
  cdef CINT32_t nB = mmat.shape[1]

  iB0s = np.full(nA,-1,dtype=CINT32)
  cdef CINT32_t[:] _iB0s_ = iB0s
  iB1s = np.full(nA,-1,dtype=CINT32)
  cdef CINT32_t[:] _iB1s_ = iB1s

  cdef CINT32_t iA=-1, iB=0, iB0=0, jj, nsq, nsq_row, nsq_col
  cdef CUINT8_t first

  for iA in range(nA):
    first = 1
    for iB in range(iB0,nB):
      if _mmat_[iA,iB] > 0:
        _iB1s_[iA] = iB
        if first:
          first = 0
          _iB0s_[iA] = iB

  iA = -1
  iB = 0
  squares = []
  while True:
    iA += 1
    if iA >= nA: break
    if _iB0s_[iA] == -1: continue
    iB = max(_iB0s_[iA], iB)
    if iB > _iB1s_[iA]: continue
    if iB >= nB: break

    while True:
      if (_mmat_[iA+1,iB]<_mmat_[iA,iB]) and (_mmat_[iA,iB+1]<_mmat_[iA,iB]):
        iB += 1
        break
      else:
        if _mmat_[iA+1,iB+1] == 0:
          if _mmat_[iA+1,iB]>_mmat_[iA,iB+1]:
            break
          else:
            iB += 1
            continue
        else:
          nsq_row = 2
          nsq_col = 2
          while True:
            if   _mmat_[iA+nsq_row,iB+nsq_col-1]>0: nsq_row += 1
            elif _mmat_[iA+nsq_row-1,iB+nsq_col]>0: nsq_col += 1
            else:                                   break

          nsq = max(nsq_row, nsq_col)
          squares.append((iA,iB,nsq,nsq_row,nsq_col))
          iB += nsq_col
          iA += nsq_row-1
          break

  return squares

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def try_walk(mat):
  cdef CFLOAT_t[:,:] _mat_ = mat
  cdef CINT32_t nrow = mat.shape[0]
  cdef CINT32_t ncol = mat.shape[1]

  matchedA = np.full(nrow, -1, dtype=CINT32)
  cdef CINT32_t[:] _matchedA_ = matchedA

  cdef CINT32_t ret
  cdef CFLOAT_t scr_tmp

  ret = _match_square(_mat_, nrow, ncol, _matchedA_, 0, 0, &scr_tmp)
  if ret < 0: raise RuntimeError(f'Error {ret}!!!')

  return matchedA, scr_tmp
