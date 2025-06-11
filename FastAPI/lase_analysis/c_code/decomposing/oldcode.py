# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# error code -30200
cdef CINT32 _rec_cmpt(CUINT8[:,:] cmpt, CFLOAT64[:] poks, CUINT8* left, CINT32 i0, CINT32 N, CINT32[:,:] combs, CINT32 c0, CINT32 Cmax,
                      CFLOAT64[:] roks, CFLOAT64[:] rnos, CFLOAT64[:] wrgs, CFLOAT64 thr):
  cdef CINT32 i,j,k, idx,cnt, rc0,ret,n=0
  cdef CUINT8 *nlft = <CUINT8*>malloc(N*sizeof(CUINT8))
  cdef CFLOAT64 cwrg=1.

  for i in range(i0+1): nlft[i] = 0

  for i in range(i0+1,N):
    if i>i0+1: cwrg *= 1-poks[i-1]
    if not left[i]: continue

    cnt = 0
    for j in range(i0+1,N):
      if j<=i:
        nlft[j] = 0
      else:
        nlft[j] = cmpt[i,j]*left[j]
        cnt += 1

    combs[c0+n,0] = 1
    combs[c0+n,1] = i
    roks[c0+n] = poks[i]
    rnos[c0+n] = 1-poks[i]
    wrgs[c0+n] = cwrg
    for k in range(i+1,N): wrgs[c0+n] *= 1-poks[k]
    n += 1
    if c0+n>=Cmax:
      free(nlft)
      return -30001

    if cnt==0: continue

    rc0 = c0+n
    ret = _rec_cmpt(cmpt, poks, nlft, i, N, combs, rc0, Cmax, roks, rnos, wrgs, thr)
    if ret<0:
      free(nlft)
      return ret

    idx = rc0
    for j in range(rc0, rc0+ret):
      roks[j] *= poks[i]
      rnos[j] *= 1-poks[i]
      wrgs[j] *= cwrg

      if roks[j]/rnos[j]<thr:
        ret -= 1

      else:
        if idx<j:
          for k in range(combs[j,0]+1): combs[idx,k] = combs[j,k]
          roks[idx] = roks[j]
          rnos[idx] = rnos[j]
          wrgs[idx] = wrgs[j]
        
        combs[idx,combs[idx,0]+1] = i
        combs[idx,0] += 1
        idx += 1
      
    n += ret

  free(nlft)

  return n
# -------



# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# error code -31500
cdef CINT32 _rec_combs(CombsData* cdata, CFLOAT64* poks, CUINT8* left, CINT32 i0, CINT32 c0, CFLOAT64 thr):
  cdef CINT32 i,j,k, idx,cnt, ret,n=0, N=cdata.N
  cdef CUINT8 *nlft = <CUINT8*>malloc(N*sizeof(CUINT8))
  cdef CFLOAT64 cprod=1., pprev=1., cprb_max

  for i in range(i0): nlft[i] = 0

  for i in range(i0,N):
    cprod = pprev*poks[i]
    if cprod<=thr: break

    if left[i]:
      cnt = 0
      for j in range(i0,N):
        if j<=i:
          nlft[j] = 0
        else:
          nlft[j] = cdata.cmpt[i*N+j]*left[j]
          cnt += 1

      if cnt>0:
        ret = _rec_combs(cdata, poks, nlft, i+1, c0+n, thr)
        if ret<0:
          free(nlft)
          return ret

        idx = c0+n
        for j in range(c0+n, c0+n+ret):
          cdata.prbs[j] *= cprod

          if (cdata.prbs[j]<=thr):
            ret -= 1

          else:
            if idx<j:
              for k in range(cdata.combs[j*(N+1)]+1): cdata.combs[idx*(N+1)+k] = cdata.combs[j*(N+1)+k]
              cdata.prbs[idx] = cdata.prbs[j]
            
            cdata.combs[idx*(N+1)+cdata.combs[idx*(N+1)]+1] = i
            cdata.combs[idx*(N+1)] += 1
            idx += 1
        n += ret

      # Add combination of current ok and all following wrong
      for k in range(i+1,N): cprod *= 1-poks[k]
      if cprod>thr:
        if c0+n>=cdata.max_combs:
          free(nlft)
          return -31500

        cdata.combs[(c0+n)*(N+1)] = 1
        cdata.combs[(c0+n)*(N+1)+1] = i
        cdata.prbs[c0+n] = cprod
        n += 1

    pprev *= 1-poks[i]
    
  free(nlft)
  return n
# -------





  # # --- TreeData structure ----------
  # cdef CINT32[:] tedges = ptree.edges
  # cdef CINT32[:] tnodesR = ptree.nodesR
  # cdef CINT32[:] tnodesC = ptree.nodesC
  # cdef CUINT8[:] tsides = ptree.sides
  # cdef CFLOAT64[:] tpsing = ptree.psing
  # cdef CFLOAT64[:] tpfull = ptree.pfull
  # cdef CINT32[:] tptr = ptree.ptr

  # self.tdata = <TreeData*>malloc(sizeof(TreeData))
  # if not self.tdata: raise MemoryError("Failed to allocate memory for TreeData")
  # self.tdata.edges = <TreeEdge*>malloc(self.wopts.tmp_edges*sizeof(TreeEdge))
  # if not self.tdata.edges: raise MemoryError("Failed to allocate memory for TreeData.edges")
  # self.tdata.ptr = <CINT32*>malloc((self.wopts.max_depth+1)*sizeof(CINT32))
  # if not self.tdata.ptr: raise MemoryError("Failed to allocate memory for TreeData.ptr")

  # self.tdata.edict = create_dict()
  # self.tdata.ndicts[0] = create_dict()
  # self.tdata.ndicts[1] = create_dict()

  # self.tdata.ptr[0] = 0
  # for i in range(ptree.depth):
  #   for k in range(tptr[i],tptr[i+1]):
  #     self.tdata.edges[k].edge = tedges[k]
  #     self.tdata.edges[k].nodes[0] = tnodesR[k]
  #     self.tdata.edges[k].nodes[1] = tnodesC[k]
  #     self.tdata.edges[k].psing = tpsing[k]
  #     self.tdata.edges[k].pfull = tpfull[k]
  #     self.tdata.edges[k].side = tsides[k]
  #     self.tdata.edges[k].depth = i

  #     if not contains_key(self.tdata.edict, tedges[k]):
  #       set_item(self.tdata.edict, tedges[k], i)
  #     if not contains_key(self.tdata.ndicts[tsides[k]], self.tdata.edges[k].nodes[tsides[k]]):
  #       if contains_key(self.wopts.blockN[tsides[k]], self.tdata.edges[k].nodes[tsides[k]]):
  #         set_item(self.tdata.ndicts[tsides[k]], self.tdata.edges[k].nodes[tsides[k]], 0)
  #       else:
  #         set_item(self.tdata.ndicts[tsides[k]], self.tdata.edges[k].nodes[tsides[k]], 1)

  #   self.tdata.ptr[i+1] = k+1
  # for k in range(i+2, self.wopts.max_depth+1): self.tdata.ptr[k] = self.tdata.ptr[i+1]

  # self.tdata.isort = <CINT32*>malloc(self.wopts.tmp_edges*sizeof(CINT32))
  # self.tdata.edges_sort = <CINT32*>malloc(self.wopts.tmp_edges*sizeof(CINT32))
  # self.tdata.pvals_sort = <CFLOAT64*>malloc(self.wopts.tmp_edges*sizeof(CFLOAT64))

  # self.tdata.is_edge = <CUINT8>ptree.is_edge
  # self.tdata.depth = ptree.depth
  # self.tdata.max_edges = self.wopts.tmp_edges
  # self.tdata.max_depth = self.wopts.max_depth




# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# error code -31100
# @treefunc
# def tree_calc_pfull(ptree, excl, opts):
#   cdef CINT32 i,k
#   cdef CINT32 N = ptree.ptr[ptree.depth]-1

#   # --- WalkOpts structure ----------
#   cdef WalkOpts wopts
#   wopts.pmin_walk = opts.pmin_walk
#   wopts.pmax_remove = opts.pmax_remove
#   wopts.pmin_comb = opts.pmin_comb
#   wopts.rem_disj = opts.rem_disj
#   wopts.max_edges = opts.max_edges
#   wopts.tmp_edges = opts.tmp_edges
#   wopts.max_combs = opts.max_combs

#   # --- Exclusion matrix ----------
#   cdef CINT32[:] excl_indptr = excl.indptr
#   cdef CINT32[:] excl_indices = excl.indices

#   # --- TreeData structure ----------
#   cdef CINT32[:] tedges = ptree.edges
#   cdef CINT32[:] tnodesR = ptree.nodesR
#   cdef CINT32[:] tnodesC = ptree.nodesC
#   cdef CUINT8[:] tsides = ptree.sides
#   cdef CFLOAT64[:] tpsing = ptree.psing
#   cdef CFLOAT64[:] tpfull = ptree.pfull
#   cdef CINT32[:] tptr = ptree.ptr

#   cdef TreeData tdata
#   cdef TreeEdge[ALLOC_EMAX] edges
#   cdef CINT32[ALLOC_MAX_DEPTH] ptr

#   tdata.edges = edges
#   tdata.ptr = ptr
#   tdata.edict = create_dict()
#   tdata.ndicts[0] = create_dict()
#   tdata.ndicts[1] = create_dict()

#   ptr[0] = 0
#   for i in range(ptree.depth):
#     for k in range(tptr[i],tptr[i+1]):
#       edges[k].edge = tedges[k]
#       edges[k].nodes[0] = tnodesR[k]
#       edges[k].nodes[1] = tnodesC[k]
#       edges[k].psing = tpsing[k]
#       edges[k].pfull = tpfull[k]
#       edges[k].side = tsides[k]
#       edges[k].depth = i

#       if not contains_key(tdata.edict, tedges[k]): set_item(tdata.edict, tedges[k], i)
#       if not contains_key(tdata.ndicts[tsides[k]], edges[k].nodes[tsides[k]]): set_item(tdata.ndicts[tsides[k]], edges[k].nodes[tsides[k]], 1)
#     ptr[i+1] = k+1
#   for k in range(i+2, ptree.depth+1): ptr[k] = ptr[i+1]

#   isort = np.arange(N, dtype=np.int32)
#   edges_sort = np.zeros(N, dtype=np.int32)
#   pvals_sort = np.zeros(N, dtype=np.float64)
#   cdef CINT32[:] _isort_ = isort
#   cdef CINT32[:] _edges_sort_ = edges_sort
#   cdef CFLOAT64[:] _pvals_sort_ = pvals_sort

#   tdata.isort = &_isort_[0]
#   tdata.edges_sort = &_edges_sort_[0]
#   tdata.pvals_sort = &_pvals_sort_[0]

#   tdata.is_edge = <CUINT8>ptree.is_edge
#   tdata.depth = ptree.depth
#   tdata.max_edges = N
#   tdata.max_depth = ptree.depth

#   # --- Combinations arrays ----------
#   cmpt = np.zeros((N,N), dtype=np.uint8)
#   cdef CUINT8[:,:] _cmpt_ = cmpt
#   combs = np.zeros((wopts.max_combs,N), dtype=np.int32)
#   cdef CINT32[:,:] _combs_ = combs
#   pcomb = np.zeros(wopts.max_combs, dtype=np.float64)
#   cdef CFLOAT64[:] _pcomb_ = pcomb
#   left = np.zeros(N, dtype=np.uint8)
#   cdef CUINT8[:] _left_ = left

#   # --- Main logic ----------
#   ret=_tree_calc_pfull(&tdata, excl_indptr, excl_indices, _cmpt_, _combs_, _pcomb_, _left_, wopts.pmin_comb, wopts.max_combs)
#   if ret<0: return ret

#   # --- Update ptree ----------
#   for i in range(tdata.ptr[tdata.depth]): tpfull[i] = tdata.edges[i].pfull

#   return 0
# # -------



# # @cython.cdivision(True)
# # @cython.boundscheck(False)
# # @cython.wraparound(False)
# # error code -31200
# @check_dicts
# def tree_reduce(ptree, excl, CINT32 steps, CFLOAT64 pmin_walk, CFLOAT64 pmax_remove, CFLOAT64 pmin_comb, CUINT8 rem_disj, CINT32 EMAX=50, CINT32 CMAX=np.int32(1e6), CINT32 _TMAX=1000):
#   cdef CINT32 i,j,k, val,ret
#   cdef CINT32 max_depth=ptree.depth+steps

#   assert EMAX <= ALLOC_EMAX
#   assert max_depth+1 <= ALLOC_MAX_DEPTH

#   # --- Exclusion matrix ----------
#   cdef CINT32[:] excl_indptr = excl.indptr
#   cdef CINT32[:] excl_indices = excl.indices

#   # --- ImatData structure ----------
#   cdef CINT32[:] rdata = ptree.imatr.data
#   cdef CINT32[:] rindices = ptree.imatr.indices
#   cdef CINT32[:] rindptr = ptree.imatr.indptr
#   cdef CINT32[:] cdata = ptree.imatc.data
#   cdef CINT32[:] cindices = ptree.imatc.indices
#   cdef CINT32[:] cindptr = ptree.imatc.indptr
#   cdef CFLOAT64[:] prbs = ptree.prbs

#   cdef ImatData[2] imats
#   imats[0].data = &rdata[0]
#   imats[0].indices = &rindices[0]
#   imats[0].indptr = &rindptr[0]
#   imats[0].prbs = &prbs[0]
#   imats[0].nrows = <CINT32>ptree.imatr.shape[0]
#   imats[0].ncols = <CINT32>ptree.imatr.shape[1]

#   imats[1].data = &cdata[0]
#   imats[1].indices = &cindices[0]
#   imats[1].indptr = &cindptr[0]
#   imats[1].prbs = &prbs[0]
#   imats[1].nrows = <CINT32>ptree.imatr.shape[0]
#   imats[1].ncols = <CINT32>ptree.imatr.shape[1]

#   # --- TreeData structure ----------
#   cdef CINT32[:] tedges = ptree.edges
#   cdef CINT32[:] tnodesR = ptree.nodesR
#   cdef CINT32[:] tnodesC = ptree.nodesC
#   cdef CUINT8[:] tsides = ptree.sides
#   cdef CFLOAT64[:] tpsing = ptree.psing
#   cdef CFLOAT64[:] tpfull = ptree.pfull
#   cdef CINT32[:] tptr = ptree.ptr

#   cdef TreeData tdata
#   cdef TreeEdge[ALLOC_EMAX] edges
#   cdef CINT32[ALLOC_MAX_DEPTH] ptr

#   tdata.edges = edges
#   tdata.ptr = ptr
#   tdata.edict = create_dict()
#   tdata.ndicts[0] = create_dict()
#   tdata.ndicts[1] = create_dict()

#   ptr[0] = 0
#   for i in range(ptree.depth):
#     for k in range(tptr[i],tptr[i+1]):
#       edges[k].edge = tedges[k]
#       edges[k].nodes[0] = tnodesR[k]
#       edges[k].nodes[1] = tnodesC[k]
#       edges[k].psing = tpsing[k]
#       edges[k].pfull = 0.
#       edges[k].side = tsides[k]
#       edges[k].depth = i

#       if not contains_key(tdata.edict, tedges[k]): set_item(tdata.edict, tedges[k], i)
#       if not contains_key(tdata.ndicts[tsides[k]], edges[k].nodes[tsides[k]]): set_item(tdata.ndicts[tsides[k]], edges[k].nodes[tsides[k]], i)
#     ptr[i+1] = k+1
#   for k in range(i+2, ptree.depth+1): ptr[k] = ptr[i+1]

#   isort = np.arange(EMAX, dtype=np.int32)
#   edges_sort = np.zeros(EMAX, dtype=np.int32)
#   pvals_sort = np.zeros(EMAX, dtype=np.float64)
#   cdef CINT32[:] _isort_ = isort
#   cdef CINT32[:] _edges_sort_ = edges_sort
#   cdef CFLOAT64[:] _pvals_sort_ = pvals_sort

#   tdata.isort = &_isort_[0]
#   tdata.edges_sort = &_edges_sort_[0]
#   tdata.pvals_sort = &_pvals_sort_[0]

#   tdata.is_edge = <CUINT8>ptree.is_edge
#   tdata.depth = ptree.depth
#   tdata.max_edges = EMAX
#   tdata.max_depth = max_depth
  
#   # --- Combinations arrays ----------
#   cmpt = np.zeros((EMAX+1,EMAX+1), dtype=np.uint8)
#   cdef CUINT8[:,:] _cmpt_ = cmpt
#   combs = np.zeros((CMAX,EMAX+1), dtype=np.int32)
#   cdef CINT32[:,:] _combs_ = combs
#   pcomb = np.zeros(CMAX, dtype=np.float64)
#   cdef CFLOAT64[:] _pcomb_ = pcomb
#   left = np.zeros(EMAX+1, dtype=np.uint8)
#   cdef CUINT8[:] _left_ = left

#   # --- Main logic ----------
#   # print(f'========== START ==========')
#   # print('edges: ', [f'{ee}' for ee in edges[:ptr[2*tdata.depth]]])
#   # print('psing: ', [f'{100*pp:.1f}' for pp in psing[:ptr[2*tdata.depth]]])
#   # print('pfull: ', [f'{100*pp:.1f}' for pp in pfull[:ptr[2*tdata.depth]]])
#   # print('ptr: ', ptr)
#   for i in range(steps+1):
#     # print(f'========== Step #{i} ==========')
#     # --- Remove improbable edges ----------
#     if tdata.depth>1:
#       if tdata.ptr[tdata.depth]-tdata.ptr[1]>EMAX:
#         # print('... remove psing')
#         _tree_remove_psing(&tdata, EMAX, pmax_remove)
#         # print('edges: ', [f'{ee}' for ee in edges[:ptr[2*tdata.depth]]])
#         # print('psing: ', [f'{100*pp:.1f}' for pp in psing[:ptr[2*tdata.depth]]])
#         # print('pfull: ', [f'{100*pp:.1f}' for pp in pfull[:ptr[2*tdata.depth]]])
#         # print('ptr: ', ptr)
#         if tdata.ptr[tdata.depth]-tdata.ptr[1]>EMAX: return -31200
        
#         # if rem_disj: _tree_remove_disj(&tdata)
      
#       # print('... calc pfull')
#       _calc_tree_pfull(&tdata, excl_indptr, excl_indices, _cmpt_, _combs_, _pcomb_, _left_, pmin_comb, CMAX)
#       # print('edges: ', [f'{ee}' for ee in edges[:ptr[2*tdata.depth]]])
#       # print('psing: ', [f'{100*pp:.1f}' for pp in psing[:ptr[2*tdata.depth]]])
#       # print('pfull: ', [f'{100*pp:.1f}' for pp in pfull[:ptr[2*tdata.depth]]])
#       # print('ptr: ', ptr)

#       # print('... remove pfull')
#       _tree_remove_pfull(&tdata)
#       # if rem_disj: _tree_remove_disj(&tdata)
#       # print('edges: ', [f'{ee}' for ee in edges[:ptr[2*tdata.depth]]])
#       # print('psing: ', [f'{100*pp:.1f}' for pp in psing[:ptr[2*tdata.depth]]])
#       # print('pfull: ', [f'{100*pp:.1f}' for pp in pfull[:ptr[2*tdata.depth]]])
#       # print('ptr: ', ptr)

#     if i==steps: break

#     # print('... walk')
#     ret = _walk_fwd(&tdata, imats, pmin_walk)
#     if ret<0: return ret

#     # print('edges: ', [f'{ee}' for ee in edges[:ptr[2*tdata.depth]]])
#     # print('psing: ', [f'{100*pp:.1f}' for pp in psing[:ptr[2*tdata.depth]]])
#     # print('pfull: ', [f'{100*pp:.1f}' for pp in pfull[:ptr[2*tdata.depth]]])
#     # print('ptr: ', ptr)
#     if ret==0: break

#   # print(f'========== FINAL ==========')
#   # print('edges: ', [f'{ee}' for ee in edges[:ptr[2*tdata.depth]]])
#   # print('psing: ', [f'{100*pp:.1f}' for pp in psing[:ptr[2*tdata.depth]]])
#   # print('pfull: ', [f'{100*pp:.1f}' for pp in pfull[:ptr[2*tdata.depth]]])
#   # print('ptr: ', ptr)

#   # --- Update ptree ----------
#   ptree.edges = np.zeros(tdata.ptr[tdata.depth], dtype=np.int32)
#   ptree.nodesR = np.zeros(tdata.ptr[tdata.depth], dtype=np.int32)
#   ptree.nodesC = np.zeros(tdata.ptr[tdata.depth], dtype=np.int32)
#   ptree.sides = np.zeros(tdata.ptr[tdata.depth], dtype=np.uint8)
#   ptree.psing = np.zeros(tdata.ptr[tdata.depth], dtype=np.float64)
#   ptree.pfull = np.zeros(tdata.ptr[tdata.depth], dtype=np.float64)
#   cdef CINT32[:] _edges_ = ptree.edges
#   cdef CINT32[:] _nodesR_ = ptree.nodesR
#   cdef CINT32[:] _nodesC_ = ptree.nodesC
#   cdef CUINT8[:] _sides_ = ptree.sides
#   cdef CFLOAT64[:] _psing_ = ptree.psing
#   cdef CFLOAT64[:] _pfull_ = ptree.pfull

#   for i in range(tdata.ptr[tdata.depth]):
#     _edges_[i] = tdata.edges[i].edge
#     _nodesR_[i] = tdata.edges[i].nodes[0]
#     _nodesC_[i] = tdata.edges[i].nodes[1]
#     _psing_[i] = tdata.edges[i].psing
#     _pfull_[i] = tdata.edges[i].pfull
#     _sides_[i] = tdata.edges[i].side

#   ptree.ptr = np.zeros(max_depth+1, dtype=np.int32)
#   for i in range(max_depth+1): ptree.ptr[i] = tdata.ptr[i]

#   return 0
# # -------

# # @cython.cdivision(True)
# # @cython.boundscheck(False)
# # @cython.wraparound(False)
# # error code ...
# cdef CINT32 _tree_remove_disj(TreeData* tdata):
#   cdef CINT32 i,dd,ax, val0,val1,dd1, cnt0=0,cnt=0, nd0,nd1,ee, ndict0,ndict1

#   _update_dicts(tdata, 1)  

#   # --- Remove disconnected edges ----------
#   cnt = tdata.ptr[2]
#   p0 = tdata.ptr[2]
#   for dd in range(1,tdata.depth):
#     for ax in range(2):
#       ndict0 = ndictC if ax==0 else ndictR
#       ndict1 = ndictR if ax==0 else ndictC

#       for i in range(p0,tdata.ptr[2*dd+ax+1]):
#         ndR = tdata.nodes1[i] if ax==0 else tdata.nodes0[i]
#         ndC = tdata.nodes0[i] if ax==0 else tdata.nodes1[i]
#         if contains_key(edict, tdata.edges[i]):
#           if cnt==i: continue
#           tdata.edges[cnt] = tdata.edges[i]
#           tdata.nodes0[cnt] = tdata.nodes0[i]
#           tdata.nodes1[cnt] = tdata.nodes1[i]
#           tdata.psing[cnt] = tdata.psing[i]
#           tdata.pfull[cnt] = tdata.pfull[i]
#           cnt += 1
#       p0 = tdata.ptr[2*dd+ax+1]
#       tdata.ptr[2*dd+ax+1] = cnt
#   for i in range(2*dd+ax+2, 2*tdata.max_depth+1): tdata.ptr[i] = cnt
  
#   return cnt
# # -------







# # @cython.cdivision(True)
# # @cython.boundscheck(False)
# # @cython.wraparound(False)
# cdef CINT32 _calc_edges_prob(EdgesData* edata, CINT32[:] excl_indptr, CINT32[:] excl_indices, CUINT8[:,:] cmpt, CINT32[:,:] combs, CUINT8[:] left, CFLOAT64[:] prbs, 
#                              CFLOAT64 Cthr, CINT32 Cmax):
#   cdef CINT32 ret,i,j
#   cdef CFLOAT64 psum=1.
#   for i in range(edata.N): psum *= 1-edata.values0[i] # initialize with the probability of all edges being wrong

#   _get_cmpt(excl_indptr, excl_indices, edata, cmpt)

#   for i in range(edata.N): left[i] = 1
  
#   ret = _rec_cmpt(cmpt, edata.values0, &left[0], -1, edata.N, combs, 0, Cmax, prbs, Cthr)
#   if ret<0: return ret

#   for i in range(ret): psum += prbs[i]
#   for i in range(ret): prbs[i] /= psum # normalize probabilities

#   for i in range(edata.N): edata.values1[i] = 0.
#   for i in range(ret):
#     for j in range(1,combs[i,0]+1): edata.values1[combs[i,j]] += prbs[i]
  
#   return 0
# # -------




# def tree_probs(ptree, excl, CINT32 Cmax=int(1e6)):

  
#   # # --- Exclusion matrix ----------
#   # cdef CINT32[:] excl_indptr = excl.indptr
#   # cdef CINT32[:] excl_indices = excl.indices


#   # cmpt = np.zeros((NMAX+1,NMAX+1), dtype=np.uint8)
#   # cdef CUINT8[:,:] _cmpt_ = cmpt
#   # combs = np.zeros((Cmax,NMAX+3), dtype=np.int32)
#   # cdef CINT32[:,:] _combs_ = combs
#   # prbs = np.zeros(Cmax, dtype=np.float64)
#   # cdef CFLOAT64[:] _prbs_ = prbs
#   # left = np.zeros(NMAX+1, dtype=np.uint8)
#   # cdef CUINT8[:] _left_ = left

#   cdef CINT32 tree_max_size = 2*NMAX*ptree.depth

#   tree_edges = np.zeros(2*EMAX, dtype=np.float64)
#   cdef CFLOAT64[:] _edges_values0_ = edges_values0

#   cdef EdgesData edata = {'values0': &_edges_values0_[0], 'values1': &_edges_values1_[0], 'edges': &_edges_edges_[0], 'nodes': &_edges_nodes_[0],
#                           'which': &_edges_which_[0], 'depth': &_edges_depth_[0], 'idx': &_edges_idx_[0], 'isort': &_edges_isort_[0], 'i0': 0, 'N': 0, 'NMAX': 2*EMAX}
#   pass


# ------------------------------ REDUCE FUNCTIONS ------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# # @cython.cdivision(True)
# # @cython.boundscheck(False)
# # @cython.wraparound(False)
# # error code -31000
# def creduce(imatr, imatc, excl, CINT32[:] prows, CINT32[:] pcols, CFLOAT64[:] poks, CINT32[:] idxs, CINT32 max_depth, CFLOAT64 Cthr, CINT32 NMAX=50, CINT32 Cmax=int(1e6)):
#   cdef CINT32 N = <CINT32>len(imatr.data)
#   assert N==len(imatc.data)
#   assert N==len(poks)

#   cdef CINT32 EMAX = 10*NMAX*max_depth

#   # --- Exclusion matrix ----------
#   cdef CINT32[:] excl_indptr = excl.indptr
#   cdef CINT32[:] excl_indices = excl.indices

#   # --- Data structures ----------
#   imatr_data = imatr.data.copy()
#   cdef CINT32[:] _rdata_ = imatr_data
#   cdef CINT32[:] _rindices_ = imatr.indices
#   cdef CINT32[:] _rindptr_ = imatr.indptr

#   r_edges = np.zeros((max_depth+1,10*NMAX), dtype=np.int32)
#   r_nodes = np.zeros((max_depth+1,10*NMAX), dtype=np.int32)
#   r_poks = np.zeros((max_depth+1,10*NMAX), dtype=np.float64)
#   r_cnts = np.zeros(max_depth+1, dtype=np.int32)
#   cdef CINT32[:] _redges_ = np.asarray(r_edges).ravel()
#   cdef CINT32[:] _rnodes_ = np.asarray(r_nodes).ravel()
#   cdef CFLOAT64[:] _rpoks_ = np.asarray(r_poks).ravel()
#   cdef CINT32[:] _rcnts_ = r_cnts

#   imatc_data = imatc.data.copy()
#   cdef CINT32[:] _cdata_ = imatc_data
#   cdef CINT32[:] _cindices_ = imatc.indices
#   cdef CINT32[:] _cindptr_ = imatc.indptr

#   cidx = np.arange(N, dtype=np.int32)
#   cdef CINT32[:] _cidx_ = cidx
#   argsort_I32(&_cdata_[0], &_cidx_[0], N)

#   c_edges = np.zeros((max_depth+1,10*NMAX), dtype=np.int32)
#   c_nodes = np.zeros((max_depth+1,10*NMAX), dtype=np.int32)
#   c_poks = np.zeros((max_depth+1,10*NMAX), dtype=np.float64)
#   c_cnts = np.zeros(max_depth+1, dtype=np.int32)
#   cdef CINT32[:] _cedges_ = np.asarray(c_edges).ravel()
#   cdef CINT32[:] _cnodes_ = np.asarray(c_nodes).ravel()
#   cdef CFLOAT64[:] _cpoks_ = np.asarray(c_poks).ravel()
#   cdef CINT32[:] _ccnts_ = c_cnts

#   cdef ReduceData redat0 = {'indptr': &_rindptr_[0], 'indices': &_rindices_[0], 'data': &_rdata_[0],
#                             'edges': &_redges_[0], 'nodes': &_rnodes_[0], 'poks': &_rpoks_[0], 'edict': create_dict(), 'ndict': create_dict(), 'cnts': &_rcnts_[0],
#                             'depth': max_depth, 'MMAX': 10*NMAX, 'NMAX': NMAX, 'len': 0}
#   cdef ReduceData redat1 = {'indptr': &_cindptr_[0], 'indices': &_cindices_[0], 'data': &_cdata_[0],
#                             'edges': &_cedges_[0], 'nodes': &_cnodes_[0], 'poks': &_cpoks_[0], 'edict': create_dict(), 'ndict': create_dict(), 'cnts': &_ccnts_[0],
#                             'depth': max_depth, 'MMAX': 10*NMAX, 'NMAX': NMAX, 'len': 0}

#   # --- Edges structure ----------
#   edges_values0 = np.zeros(2*EMAX, dtype=np.float64)
#   cdef CFLOAT64[:] _edges_values0_ = edges_values0
#   edges_values1 = np.zeros(2*EMAX, dtype=np.float64)
#   cdef CFLOAT64[:] _edges_values1_ = edges_values1
#   edges_edges = np.zeros(2*EMAX, dtype=np.int32)
#   cdef CINT32[:] _edges_edges_ = edges_edges
#   edges_nodes = np.zeros(2*EMAX, dtype=np.int32)
#   cdef CINT32[:] _edges_nodes_ = edges_nodes
#   edges_which = np.zeros(2*EMAX, dtype=np.int32)
#   cdef CINT32[:] _edges_which_ = edges_which
#   edges_depth = np.zeros(2*EMAX, dtype=np.int32)
#   cdef CINT32[:] _edges_depth_ = edges_depth
#   edges_idx = np.zeros(2*EMAX, dtype=np.int32)
#   cdef CINT32[:] _edges_idx_ = edges_idx
#   edges_isort = np.zeros(2*EMAX, dtype=np.int32)
#   cdef CINT32[:] _edges_isort_ = edges_isort
#   cdef EdgesData edata = {'values0': &_edges_values0_[0], 'values1': &_edges_values1_[0], 'edges': &_edges_edges_[0], 'nodes': &_edges_nodes_[0],
#                           'which': &_edges_which_[0], 'depth': &_edges_depth_[0], 'idx': &_edges_idx_[0], 'isort': &_edges_isort_[0], 'i0': 0, 'N': 0, 'NMAX': 2*EMAX}

#   # --- Probability ----------
#   cmpt = np.zeros((NMAX+1,NMAX+1), dtype=np.uint8)
#   cdef CUINT8[:,:] _cmpt_ = cmpt
#   combs = np.zeros((Cmax,NMAX+3), dtype=np.int32)
#   cdef CINT32[:,:] _combs_ = combs
#   prbs = np.zeros(Cmax, dtype=np.float64)
#   cdef CFLOAT64[:] _prbs_ = prbs
#   left = np.zeros(NMAX+1, dtype=np.uint8)
#   cdef CUINT8[:] _left_ = left

#   # --- Variables ----------
#   cdef CINT32 ret, i,j,k, ee, dd,nn, nidx=<CINT32>len(idxs)
#   cdef CFLOAT64 p0

#   remove = np.zeros(nidx, dtype=np.uint8)
#   cdef CUINT8[:] _remove_ = remove

#   for i in range(nidx):
#     ee = _rdata_[idxs[i]]

#     # --- Initialize ----------
#     _init_redat(&redat0, ee, prows[idxs[i]], poks[idxs[i]])
#     _init_redat(&redat1, ee, pcols[idxs[i]], poks[idxs[i]])

#     for dd in range(max_depth):
#       _walk_fwd(&redat0, &redat1, dd, poks)
#       _walk_fwd(&redat1, &redat0, dd, poks)

#       ret = _get_edges(&redat0, &redat1, &edata, 2*EMAX)
#       if ret<0:
#         delete_all()
#         return ret

#       if edata.N>NMAX: _reduce_edges_N(&redat0, &redat1, &edata, NMAX)

#       ret = _calc_edges_prob(&edata, excl_indptr, excl_indices, _cmpt_, _combs_, _left_, _prbs_, Cthr, Cmax)
#       if ret<0:
#         delete_all()
#         return ret
      
#       if ((edata.values0[edata.i0]<.01) and (edata.values1[edata.i0]<.01)) or ((edata.values0[edata.i0]<.1) and (edata.values1[edata.i0]/edata.values0[edata.i0]<.1)):
#         _rdata_[idxs[i]] = -1
#         _cdata_[_cidx_[idxs[i]]] = -1
#         _remove_[i] = 1
#         break

#       _reduce_edges_P1(&redat0, &redat1, &edata)

#   delete_all()

#   return remove
# # -------

# # @cython.cdivision(True)
# # @cython.boundscheck(False)
# # @cython.wraparound(False)
# cdef inline CINT32 _init_redat(ReduceData* dat, CINT32 edge, CINT32 node, CFLOAT64 pok):
#   cdef CINT32 i,j

#   for i in range(dat.depth+1):
#     for j in range(dat.cnts[i]):
#       dat.edges[i*dat.MMAX+j] = 0
#       dat.nodes[i*dat.MMAX+j] = 0
#     dat.cnts[i] = 0
#   delete_dict(dat.edict)
#   delete_dict(dat.ndict)

#   dat.edges[0] = edge
#   dat.nodes[0] = node
#   dat.poks[0] = pok
#   dat.cnts[0] = 1
#   set_item(dat.ndict, node, 0)
#   set_item(dat.edict, edge, 0)
#   dat.len = 1

#   return 0
# # -------

# # @cython.cdivision(True)
# # @cython.boundscheck(False)
# # @cython.wraparound(False)
# # error code -31100
# cdef CINT32 _get_edges(ReduceData* dat0, ReduceData* dat1, EdgesData* edata, CINT32 NMAX):
#   if NMAX > edata.NMAX: return -31101
  
#   cdef CINT32 i,j,i0,cnt=0

#   for i in range(dat0.depth+1):
#     for j in range(dat0.cnts[i]):
#       if cnt >= NMAX: return -31102
#       edata.values0[cnt] = dat0.poks[i*dat0.MMAX+j]
#       edata.isort[cnt] = cnt
#       cnt += 1
#   for i in range(1,dat1.depth+1):
#     for j in range(dat1.cnts[i]):
#       if cnt >= NMAX: return -31102
#       edata.values0[cnt] = dat1.poks[i*dat1.MMAX+j]
#       edata.isort[cnt] = cnt
#       cnt += 1

#   cdef CINT32 *isort = <CINT32*>malloc(cnt*sizeof(CINT32))

#   argsort_F64(edata.values0, edata.isort, cnt)
#   for i in range(cnt): isort[edata.isort[i]] = cnt-i-1

#   cnt = 0
#   edata.i0 = isort[0]
#   for i in range(dat0.depth+1):
#     for j in range(dat0.cnts[i]):
#       i0 = isort[cnt]
#       edata.values0[i0] = dat0.poks[i*dat0.MMAX+j]
#       edata.values1[i0] = 0.
#       edata.edges[i0] = dat0.edges[i*dat0.MMAX+j]
#       edata.nodes[i0] = dat0.nodes[i*dat0.MMAX+j]
#       edata.which[i0] = 0
#       edata.depth[i0] = i
#       edata.idx[i0] = j
#       edata.isort[i0] = cnt
#       cnt += 1

#   for i in range(1,dat1.depth+1):
#     for j in range(dat1.cnts[i]):
#       i0 = isort[cnt]
#       edata.values0[i0] = dat1.poks[i*dat1.MMAX+j]
#       edata.values1[i0] = 0.
#       edata.edges[i0] = dat1.edges[i*dat1.MMAX+j]
#       edata.nodes[i0] = dat1.nodes[i*dat1.MMAX+j]
#       edata.which[i0] = 1
#       edata.depth[i0] = i
#       edata.idx[i0] = j
#       edata.isort[i0] = cnt
#       cnt += 1
  
#   edata.N = cnt
#   free(isort)

#   return 0
# # -------

# # @cython.cdivision(True)
# # @cython.boundscheck(False)
# # @cython.wraparound(False)
# cdef CINT32 _reduce_edges_N(ReduceData* dat0, ReduceData* dat1, EdgesData* edata, CINT32 NMAX):
#   cdef CINT32 i

#   for i in range(1,dat0.depth+1): dat0.cnts[i] = 0
#   for i in range(1,dat1.depth+1): dat1.cnts[i] = 0

#   for i in range(NMAX):
#     if i>=edata.N: break

#     if edata.depth[i]==0: continue
#     if edata.which[i]==0:
#       dat0.edges[edata.depth[i]*dat0.MMAX+dat0.cnts[edata.depth[i]]] = edata.edges[i]
#       dat0.nodes[edata.depth[i]*dat0.MMAX+dat0.cnts[edata.depth[i]]] = edata.nodes[i]
#       dat0.poks[edata.depth[i]*dat0.MMAX+dat0.cnts[edata.depth[i]]] = edata.values0[i]
#       dat0.cnts[edata.depth[i]] += 1
#     else:
#       dat1.edges[edata.depth[i]*dat1.MMAX+dat1.cnts[edata.depth[i]]] = edata.edges[i]
#       dat1.nodes[edata.depth[i]*dat1.MMAX+dat1.cnts[edata.depth[i]]] = edata.nodes[i]
#       dat1.poks[edata.depth[i]*dat1.MMAX+dat1.cnts[edata.depth[i]]] = edata.values0[i]
#       dat1.cnts[edata.depth[i]] += 1

#   for i in range(dat0.depth+1): dat0.len += dat0.cnts[i]
#   for i in range(dat1.depth+1): dat1.len += dat1.cnts[i]

#   return _get_edges(dat0, dat1, edata, dat0.len+dat1.len-1)

# # -------

# # @cython.cdivision(True)
# # @cython.boundscheck(False)
# # @cython.wraparound(False)
# cdef CINT32 _reduce_edges_P1(ReduceData* dat0, ReduceData* dat1, EdgesData* edata):
#   cdef CINT32 i

#   for i in range(1,dat0.depth+1): dat0.cnts[i] = 0
#   for i in range(1,dat1.depth+1): dat1.cnts[i] = 0

#   for i in range(edata.N):
#     if edata.depth[i]==0: continue
#     if ((edata.values0[i]<.01) and (edata.values1[i]<.01)) or ((edata.values0[i]<.1) and (edata.values1[i]/edata.values0[i]<.1)): continue
#     if edata.which[i]==0:
#       dat0.edges[edata.depth[i]*dat0.MMAX+dat0.cnts[edata.depth[i]]] = edata.edges[i]
#       dat0.nodes[edata.depth[i]*dat0.MMAX+dat0.cnts[edata.depth[i]]] = edata.nodes[i]
#       dat0.poks[edata.depth[i]*dat0.MMAX+dat0.cnts[edata.depth[i]]] = edata.values0[i]
#       dat0.cnts[edata.depth[i]] += 1
#     else:
#       dat1.edges[edata.depth[i]*dat1.MMAX+dat1.cnts[edata.depth[i]]] = edata.edges[i]
#       dat1.nodes[edata.depth[i]*dat1.MMAX+dat1.cnts[edata.depth[i]]] = edata.nodes[i]
#       dat1.poks[edata.depth[i]*dat1.MMAX+dat1.cnts[edata.depth[i]]] = edata.values0[i]
#       dat1.cnts[edata.depth[i]] += 1

#   for i in range(dat0.depth+1): dat0.len += dat0.cnts[i]
#   for i in range(dat1.depth+1): dat1.len += dat1.cnts[i]

#   return 0
# # -------



# # @cython.cdivision(True)
# # @cython.boundscheck(False)
# # @cython.wraparound(False)
# def walk_imat(imatr, imatc, CFLOAT64[:] poks, CINT32 node0, CINT32 node1, CINT32 edge, CINT32 depth, CINT32 NMAX=50):
#   assert len(imatr.data)==len(imatc.data)

#   cdef CINT32[:] _rdata_ = imatr.data
#   cdef CINT32[:] _rindices_ = imatr.indices
#   cdef CINT32[:] _rindptr_ = imatr.indptr

#   r_edges = np.zeros((depth+1,10*NMAX), dtype=np.int32)
#   r_nodes = np.zeros((depth+1,10*NMAX), dtype=np.int32)
#   r_poks = np.zeros((depth+1,10*NMAX), dtype=np.float64)
#   r_cnts = np.zeros((depth+1), dtype=np.int32)
#   cdef CINT32[:] _redges_ = np.asarray(r_edges).ravel()
#   cdef CINT32[:] _rnodes_ = np.asarray(r_nodes).ravel()
#   cdef CFLOAT64[:] _rpoks_ = np.asarray(r_poks).ravel()
#   cdef CINT32[:] _rcnts_ = r_cnts

#   cdef CINT32[:] _cdata_ = imatc.data
#   cdef CINT32[:] _cindices_ = imatc.indices
#   cdef CINT32[:] _cindptr_ = imatc.indptr

#   c_edges = np.zeros((depth+1,10*NMAX), dtype=np.int32)
#   c_nodes = np.zeros((depth+1,10*NMAX), dtype=np.int32)
#   c_poks = np.zeros((depth+1,10*NMAX), dtype=np.float64)
#   c_cnts = np.zeros((depth+1), dtype=np.int32)
#   cdef CINT32[:] _cedges_ = np.asarray(c_edges).ravel()
#   cdef CINT32[:] _cnodes_ = np.asarray(c_nodes).ravel()
#   cdef CFLOAT64[:] _cpoks_ = np.asarray(c_poks).ravel()
#   cdef CINT32[:] _ccnts_ = c_cnts

#   cdef ReduceData redat0 = {'indptr': &_rindptr_[0], 'indices': &_rindices_[0], 'data': &_rdata_[0],
#                             'edges': &_redges_[0], 'nodes': &_rnodes_[0], 'poks': &_rpoks_[0], 'edict': create_dict(), 'ndict': create_dict(), 'cnts': &_rcnts_[0],
#                             'depth': depth, 'MMAX': 10*NMAX, 'NMAX': NMAX, 'len': 0}
#   cdef ReduceData redat1 = {'indptr': &_cindptr_[0], 'indices': &_cindices_[0], 'data': &_cdata_[0],
#                             'edges': &_cedges_[0], 'nodes': &_cnodes_[0], 'poks': &_cpoks_[0], 'edict': create_dict(), 'ndict': create_dict(), 'cnts': &_ccnts_[0],
#                             'depth': depth, 'MMAX': 10*NMAX, 'NMAX': NMAX, 'len': 0}

#   cdef CINT32 ret,dd

#   # --- Initialize ----------
#   redat0.edges[0] = edge
#   redat0.nodes[0] = node0
#   redat0.poks[0] = poks[edge]
#   redat0.cnts[0] = 1
#   set_item(redat0.ndict, node0, 0)
#   set_item(redat0.edict, edge, 0)
#   redat0.len = 1
  
#   redat1.edges[0] = edge
#   redat1.nodes[0] = node1
#   redat1.poks[0] = poks[edge]
#   redat1.cnts[0] = 1
#   set_item(redat1.ndict, node1, 0)
#   set_item(redat1.edict, edge, 0)
#   redat1.len = 1

#   for dd in range(depth):
#     ret = _walk_fwd(&redat0,&redat1,dd,poks)
#     if ret<0: raise RuntimeError(f'Error {ret} in _walk_fwd #0')
#     ret = _walk_fwd(&redat1,&redat0,dd,poks)
#     if ret<0: raise RuntimeError(f'Error {ret} in _walk_fwd #1')

#   delete_all()

#   return [{'edges': [ees[:cc] for ees,cc in zip(r_edges,r_cnts)], 'nodes': [nns[:cc] for nns,cc in zip(r_nodes,r_cnts)]},
#           {'edges': [ees[:cc] for ees,cc in zip(c_edges,c_cnts)], 'nodes': [nns[:cc] for nns,cc in zip(c_nodes,c_cnts)]}]
# # -------



# cdef struct ReduceData:
#   CINT32* indptr
#   CINT32* indices
#   CINT32* data

#   CINT32* edges
#   CINT32* nodes
#   CFLOAT64* poks
#   CINT32* cnts

#   CINT32 ndict
#   CINT32 edict

#   CINT32 depth
#   CINT32 MMAX
#   CINT32 NMAX

#   CINT32 len
# # -------

# cdef struct EdgesData:
#   CFLOAT64* values0
#   CFLOAT64* values1
#   CINT32* edges
#   CINT32* nodes

#   CINT32* which
#   CINT32* depth
#   CINT32* idx
#   CINT32* isort

#   CINT32 i0
#   CINT32 N
#   CINT32 NMAX
# # -------


# # @cython.cdivision(True)
# # @cython.boundscheck(False)
# # @cython.wraparound(False)
# # error code -30000
# def probs_compatible(CUINT8[:,:] cmpt, CFLOAT64[:] poks, CFLOAT64 thr, CINT32 Cmax, CINT32 _NMAX=50):
#   cdef CINT32 N=<CINT32>cmpt.shape[0]
#   assert N==len(poks)
#   assert N<=_NMAX

#   cdef CINT32 i,j, cnt, Nmax=0

#   for i in range(N):
#     cnt = 0
#     for j in range(N):
#       if cmpt[i,j]: cnt += 1
#     if cnt>Nmax: Nmax = cnt
  
#   combs = np.zeros((Cmax,Nmax+2), dtype=np.int32)
#   cdef CINT32[:,:] _combs_ = combs
#   prbs = np.zeros(Cmax, dtype=np.float64)
#   cdef CFLOAT64[:] _prbs_ = prbs

#   cdef CUINT8* left = <CUINT8*>malloc(N*sizeof(CUINT8))
#   for i in range(N): left[i] = 1

#   ret = _rec_cmpt(cmpt, &poks[0], left, -1, N, _combs_, 0, Cmax, _prbs_, thr)
#   if ret<0: return ret

#   pout = np.zeros(N, dtype=np.float64)
#   for i in range(ret):
#     for j in range(1,combs[i,0]+1): pout[combs[i,j]] += prbs[i]

#   return pout
# # -------





# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# # error code ...
# cdef CINT32 _tree_remove_pfull(TreeData* tdata):
#   cdef CINT32 i,dd,cnt,p0

#   cnt = tdata.ptr[1]
#   p0 = tdata.ptr[1]
#   for dd in range(1,tdata.depth):
#     for i in range(p0,tdata.ptr[dd+1]):
#       if ((tdata.edges[i].psing>.1)  and (tdata.edges[i].pfull>tdata.edges[i].psing/100.)) or\
#          ((tdata.edges[i].psing>.01) and (tdata.edges[i].pfull>tdata.edges[i].psing/10.)) or\
#          ((tdata.edges[i].psing<.01) and (tdata.edges[i].pfull>tdata.edges[i].psing)):
#         if cnt<i: tdata.edges[cnt] = tdata.edges[i]
#         cnt += 1
#     p0 = tdata.ptr[dd+1]
#     tdata.ptr[dd+1] = cnt
#   for i in range(dd+2, tdata.max_depth+1): tdata.ptr[i] = cnt

#   return cnt
# # -------

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# # error code -31600
# cdef CINT32 _tree_remove_psing(TreeData* tdata, CINT32 max_edges, CFLOAT64 max_psing):
#   cdef CINT32 N=tdata.ptr[tdata.depth]-1 # total number of valid edges
#   cdef CINT32 N0=tdata.ptr[1]-1 # number of starting edges (0 for node trees, 1 for edge trees)
#   if max_edges < N0: return -31600
#   if N<=max_edges:   return N

#   cdef CFLOAT64 thr
#   cdef CINT32 i,dd,cnt,p0   

#   p0 = tdata.ptr[1]
#   for i in range(N-N0):
#     tdata.isort[i] = i
#     tdata.pvals_sort[i] = tdata.edges[p0+i].psing
#   argsort_F64(tdata.pvals_sort, tdata.isort, N, 1, 1)

#   thr = tdata.pvals_sort[max_edges-N0]
#   if thr>max_psing: return -31601

#   cnt = p0
#   for dd in range(1,tdata.depth):
#     for i in range(p0,tdata.ptr[dd+1]):
#       if tdata.edges[i].psing>thr:
#         if cnt<i: tdata.edges[cnt] = tdata.edges[i]
#         cnt += 1
#     p0 = tdata.ptr[dd+1]
#     tdata.ptr[dd+1] = cnt
#   for i in range(dd+2, tdata.max_depth+1): tdata.ptr[i] = cnt

#   return cnt
# # -------

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# # error code ...
# cdef CINT32 _tree_remove_disj(TreeData* tdata):
#   if tdata.depth==1: return 0

#   cdef CINT32 i,dd,ss, nd0,nd1, cnt0,cnt, p0, val0,val1

#   cdef CINT32[2] ndicts
#   ndicts[0] = create_dict()
#   ndicts[1] = create_dict()

#   for i in range(tdata.ptr[0],tdata.ptr[1]):
#     set_item(ndicts[tdata.edges[i].side], tdata.edges[i].nodes[tdata.edges[i].side], 0)

#   # --- Walk tree from origin node/edge ----------
#   cnt = 0
#   while True:
#     cnt0 = cnt
#     for i in range(tdata.ptr[1], tdata.ptr[tdata.depth]):
#       ss = tdata.edges[i].side
#       nd0 = tdata.edges[i].nodes[1-ss]
#       nd1 = tdata.edges[i].nodes[ss]

#       get_item(ndicts[1-ss], nd0, &val0)
#       get_item(ndicts[ss],   nd1, &val1)

#       if (val0>=0) and (val1>=0):
#         if   (val0>val1+1): set_item(ndicts[1-ss], nd0, val1+1)
#         elif (val1>val0+1): set_item(ndicts[ss],   nd1, val0+1)
      
#       elif val0>=0:
#         set_item(ndicts[ss], nd1, val0+1)
#         cnt += 1
        
#       elif val1>=0:
#         set_item(ndicts[1-ss], nd0, val1+1)
#         cnt += 1
    
#     if cnt == cnt0: break
  
#   # --- Update edges ----------
#   cnt = tdata.ptr[1]
#   p0 = tdata.ptr[1]
#   for dd in range(1,tdata.depth):
#     for i in range(p0,tdata.ptr[dd+1]):
#       get_item(ndicts[tdata.edges[i].side], tdata.edges[i].nodes[tdata.edges[i].side], &val0)
#       if (val0>=0) and (val0<tdata.depth):
#         if cnt<i: tdata.edges[cnt] = tdata.edges[i]
#         cnt += 1
#     p0 = tdata.ptr[dd+1]
#     tdata.ptr[dd+1] = cnt
#   for i in range(dd+2, tdata.max_depth+1): tdata.ptr[i] = cnt
  
#   delete_dict(ndicts[0])
#   delete_dict(ndicts[1])

#   return 0
# # -------



# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# # Error code ...
# @treefunc
# def walk_tree(DataWrapperTree dw, CINT32 steps=1):
#   return _walk_tree(dw, steps)
# # -------

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# # Error code -30000
# cdef CINT32 _walk_tree(DataWrapperTree dw, CINT32 steps):
#   # --- Init variables and checks ----------
#   cdef CINT32 i,j,k,ret
#   cdef CINT32 max_depth=dw.tdata.depth+steps

#   # assert opts.max_edges <= COMBS_EMAX
#   assert max_depth <= dw.wopts.max_depth

#   # --- Main logic ----------
#   for i in range(steps+1):
#     # --- Prune improbable edges ----------
#     if dw.wopts.prune:
#       if dw.tdata.depth>1:
#         if dw.tdata.ptr[dw.tdata.depth]-1 > dw.wopts.max_edges:
#           _tree_remove_psing(dw.tdata, dw.wopts.max_edges, dw.wopts.pmax_remove)
#           if dw.tdata.ptr[dw.tdata.depth]-1 > dw.wopts.max_edges: return -30000

#           if dw.wopts.rem_disj: _tree_remove_disj(dw.tdata)

#         _tree_calc_pfull(dw.tdata, dw.exdata, dw.cdata, dw.wopts.pmin_comb, dw.opts.ratio_comb)
#         _tree_remove_pfull(dw.tdata)
#         if dw.wopts.rem_disj: _tree_remove_disj(dw.tdata)

#     if i==steps: break

#     # --- Walk one step ----------
#     ret = _walk_fwd(dw.tdata, dw.imdata, dw.wopts.blockN, dw.wopts.pmin_walk)
#     if ret<0: return ret
#     if ret==0: break
  
#   return 0
# # -------



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class DataWrapperPrune:
  cdef PruneOpts *opts
  cdef ExclData *exdata
  cdef ImatData *imdata
  cdef TreeData *tdata
  cdef CombsData *cdata
  cdef PsetData *psdata

  def __cinit__(self, poss, opts):
    cdef CINT32 i,k

    self.opts = NULL
    self.exdata = NULL
    self.imdata = NULL
    self.tdata = NULL
    self.cdata = NULL

    # --- PruneOpts structure ----------
    self.opts = <PruneOpts*>malloc(sizeof(PruneOpts))
    if not self.opts: raise MemoryError("Failed to allocate memory for PruneOpts")

    self.opts.pmin_walk = opts.pmin_walk
    self.opts.nmax_walk = opts.nmax_walk
    self.opts.pmin_comb = opts.pmin_comb
    self.opts.ratio_step1 = opts.ratio_step1
    self.opts.ratio_step2 = opts.ratio_step2
    self.opts.max_cycles = opts.max_cycles
    self.opts.idx_max = opts.idx_max

    cdef CINT32[:] blockR = opts.blockR
    self.opts.blockN[0] = create_dict()
    if self.opts.blockN[0]==-1: raise MemoryError("Failed to allocate memory for dict 'blockN_0'")
    for i in range(<CINT32>len(opts.blockR)): set_item(self.opts.blockN[0], blockR[i], 1)

    cdef CINT32[:] blockC = opts.blockC
    self.opts.blockN[1] = create_dict()
    if self.opts.blockN[1]==-1: raise MemoryError("Failed to allocate memory for dict 'blockN_1'")
    for i in range(<CINT32>len(opts.blockC)): set_item(self.opts.blockN[1], blockC[i], 1)

    self.opts.max_edges = opts.max_edges
    self.opts.tmp_edges = opts.tmp_edges
    self.opts.max_combs = opts.max_combs
    self.opts.max_depth = opts.max_depth

    # --- Exclusion matrix ----------
    self.exdata = <ExclData*>malloc(sizeof(ExclData))
    if not self.exdata: raise MemoryError("Failed to allocate memory for ExclData")

    cdef CINT32[:] excl_indptr = poss.excl.indptr
    cdef CINT32[:] excl_indices = poss.excl.indices
    self.exdata.indptr = &excl_indptr[0]
    self.exdata.indices = &excl_indices[0]

    # --- ImatData structure ----------
    cdef CINT32[:] imat_rdata = poss.data
    cdef CINT32[:] imat_rindices = poss.indices
    cdef CINT32[:] imat_rindptr = poss.indptr
    cdef CINT32[:] imat_cdata = poss.cdata
    cdef CINT32[:] imat_cindices = poss.cindices
    cdef CINT32[:] imat_cindptr = poss.cindptr
    cdef CINT32[:] erows = poss.rows
    cdef CINT32[:] ecols = poss.cols
    poss._temp_p0 = poss.vals['p0'].astype(np.float64)
    cdef CFLOAT64[:] imat_prbs = poss._temp_p0

    self.imdata = <ImatData*>malloc(sizeof(ImatData))
    if not self.imdata: raise MemoryError("Failed to allocate memory for ImatData")
    self.imdata.data = <CINT32**>malloc(2*sizeof(CINT32*))
    self.imdata.indices = <CINT32**>malloc(2*sizeof(CINT32*))
    self.imdata.indptr = <CINT32**>malloc(2*sizeof(CINT32*))

    self.imdata.data[0] = &imat_rdata[0]
    self.imdata.indices[0] = &imat_rindices[0]
    self.imdata.indptr[0] = &imat_rindptr[0]
    self.imdata.data[1] = &imat_cdata[0]
    self.imdata.indices[1] = &imat_cindices[0]
    self.imdata.indptr[1] = &imat_cindptr[0]

    self.imdata.edge_row = &erows[0]
    self.imdata.edge_col = &ecols[0]

    self.imdata.prbs = &imat_prbs[0]
    self.imdata.nrows = <CINT32>poss.shape[0]
    self.imdata.ncols = <CINT32>poss.shape[1]
    self.imdata.N = <CINT32>len(poss.data)

    # --- TreeData structure ----------
    self.tdata = <TreeData*>malloc(sizeof(TreeData))
    if not self.tdata: raise MemoryError("Failed to allocate memory for TreeData")
    self.tdata.edges = NULL
    self.tdata.ptr = NULL
    self.tdata.isort = NULL
    self.tdata.edges_sort = NULL
    self.tdata.pvals_sort = NULL

    self.tdata.edges = <TreeEdge*>malloc(self.opts.tmp_edges*sizeof(TreeEdge))
    if not self.tdata.edges: raise MemoryError("Failed to allocate memory for TreeData.edges")
    self.tdata.ptr = <CINT32*>malloc((self.opts.max_depth+1)*sizeof(CINT32))
    if not self.tdata.ptr: raise MemoryError("Failed to allocate memory for TreeData.ptr")

    self.tdata.edict = create_dict()
    if self.tdata.edict==-1: raise MemoryError("Failed to allocate memory for dict 'edict'")
    self.tdata.ndicts[0] = create_dict()
    if self.tdata.ndicts[0]==-1: raise MemoryError("Failed to allocate memory for dict 'ndict_0'")
    self.tdata.ndicts[1] = create_dict()
    if self.tdata.ndicts[1]==-1: raise MemoryError("Failed to allocate memory for dict 'ndict_1'")

    self.tdata.isort = <CINT32*>malloc(self.opts.tmp_edges*sizeof(CINT32))
    self.tdata.edges_sort = <CINT32*>malloc(self.opts.tmp_edges*sizeof(CINT32))
    self.tdata.pvals_sort = <CFLOAT64*>malloc(self.opts.tmp_edges*sizeof(CFLOAT64))

    self.tdata.is_edge = 0
    self.tdata.depth = 0
    self.tdata.max_edges = self.opts.tmp_edges
    self.tdata.max_depth = self.opts.max_depth

    # --- Combinations arrays ----------
    self.cdata = <CombsData*>malloc(sizeof(CombsData))
    if not self.cdata: raise MemoryError("Failed to allocate memory for CombsData")
    self.cdata.cmpt = NULL
    self.cdata.combs = NULL
    self.cdata.prbs = NULL
    self.cdata.prbs2 = NULL
    self.cdata.pset = NULL

    self.cdata.cmpt = <CUINT8*>malloc(self.opts.max_edges*self.opts.max_edges*sizeof(CUINT8))
    if not self.cdata.cmpt: raise MemoryError("Failed to allocate memory for CombsData.cmpt")
    self.cdata.combs = <CINT32*>malloc(self.opts.max_combs*(self.opts.max_edges+1)*sizeof(CINT32))
    if not self.cdata.combs: raise MemoryError("Failed to allocate memory for CombsData.combs")
    self.cdata.prbs = <CFLOAT64*>malloc(self.opts.max_combs*sizeof(CFLOAT64))
    if not self.cdata.prbs: raise MemoryError("Failed to allocate memory for CombsData.prbs")
    self.cdata.prbs2 = <CFLOAT64*>malloc(self.opts.max_combs*sizeof(CFLOAT64))
    if not self.cdata.prbs2: raise MemoryError("Failed to allocate memory for CombsData.prbs2")
    self.cdata.pset = <CFLOAT64*>malloc(self.opts.max_combs*sizeof(CFLOAT64))
    if not self.cdata.pset: raise MemoryError("Failed to allocate memory for CombsData.pset")
    for i in range(self.opts.max_combs): self.cdata.pset[i] = 1.

    self.cdata.max_combs = self.opts.max_combs
    self.cdata.max_edges = self.opts.max_edges
    self.cdata.n_edges = 0

    # --- PsetData structure ----------
    self.psdata = <PsetData*>malloc(sizeof(PsetData))
    if not self.psdata: raise MemoryError("Failed to allocate memory for PsetData")

    self.psdata.dictNR = create_dict()
    if self.psdata.dictNR==-1: raise MemoryError("Failed to allocate memory for dict 'dictNR'")
    self.psdata.dictNC = create_dict()
    if self.psdata.dictNC==-1: raise MemoryError("Failed to allocate memory for dict 'dictNC'")

    self.psdata.max_nodes = self.opts.max_edges

    self.psdata.psetR = <CFLOAT64*>malloc((self.psdata.max_nodes+1)*(self.psdata.max_nodes+1)*sizeof(CFLOAT64))
    if not self.psdata.psetR: raise MemoryError("Failed to allocate memory for PsetData.psetR")
    self.psdata.psetC = <CFLOAT64*>malloc((self.psdata.max_nodes+1)*(self.psdata.max_nodes+1)*sizeof(CFLOAT64))
    if not self.psdata.psetC: raise MemoryError("Failed to allocate memory for PsetData.psetC")

    for i in range(self.psdata.max_nodes+1):
      for j in range(i+1):
        self.psdata.psetR[i*(self.psdata.max_nodes+1)+j] = (<CFLOAT64>binom(j,i))*(opts.pmissR**(i-j))*((1-opts.pmissR)**j)
        self.psdata.psetC[i*(self.psdata.max_nodes+1)+j] = (<CFLOAT64>binom(j,i))*(opts.pmissC**(i-j))*((1-opts.pmissC)**j)

    self.psdata.coresR = <CUINT8*>malloc((self.psdata.max_nodes+1)*sizeof(CUINT8))
    if not self.psdata.coresR: raise MemoryError("Failed to allocate memory for PsetData.coresR")
    self.psdata.coresC = <CUINT8*>malloc((self.psdata.max_nodes+1)*sizeof(CUINT8))
    if not self.psdata.coresC: raise MemoryError("Failed to allocate memory for PsetData.coresC")
    self.psdata.donesR = <CUINT8*>malloc((self.psdata.max_nodes+1)*sizeof(CUINT8))
    if not self.psdata.donesR: raise MemoryError("Failed to allocate memory for PsetData.donesR")
    self.psdata.donesC = <CUINT8*>malloc((self.psdata.max_nodes+1)*sizeof(CUINT8))
    if not self.psdata.donesC: raise MemoryError("Failed to allocate memory for PsetData.donesC")
    self.psdata.edge2R = <CINT32*>malloc((self.cdata.max_edges)*sizeof(CINT32))
    if not self.psdata.edge2R: raise MemoryError("Failed to allocate memory for PsetData.edge2R")
    self.psdata.edge2C = <CINT32*>malloc((self.cdata.max_edges)*sizeof(CINT32))
    if not self.psdata.edge2C: raise MemoryError("Failed to allocate memory for PsetData.edge2C")

  def __dealloc__(self):
    if self.opts: free(self.opts)
    self.opts = NULL
    
    if self.exdata: free(self.exdata)
    self.exdata = NULL

    if self.imdata:
      if self.imdata.data: free(self.imdata.data)
      if self.imdata.indices: free(self.imdata.indices)
      if self.imdata.indptr: free(self.imdata.indptr)
      free(self.imdata)
    self.imdata = NULL

    if self.tdata:
      if self.tdata.edges: free(self.tdata.edges)
      if self.tdata.ptr: free(self.tdata.ptr)
      if self.tdata.isort: free(self.tdata.isort)
      if self.tdata.edges_sort: free(self.tdata.edges_sort)
      if self.tdata.pvals_sort: free(self.tdata.pvals_sort)
      free(self.tdata)
    self.tdata = NULL

    if self.cdata:
      if self.cdata.cmpt: free(self.cdata.cmpt)
      if self.cdata.combs: free(self.cdata.combs)
      if self.cdata.prbs: free(self.cdata.prbs)
      if self.cdata.prbs2: free(self.cdata.prbs2)
      if self.cdata.pset: free(self.cdata.pset)
      free(self.cdata)
    self.cdata = NULL

    if self.psdata:
      if self.psdata.coresR: free(self.psdata.coresR)
      if self.psdata.coresC: free(self.psdata.coresC)
      if self.psdata.donesR: free(self.psdata.donesR)
      if self.psdata.donesC: free(self.psdata.donesC)
      if self.psdata.edge2R: free(self.psdata.edge2R)
      if self.psdata.edge2C: free(self.psdata.edge2C)
      free(self.psdata)
    self.psdata = NULL

    delete_all()
# -------



@basefunc
def prune_large_node(DataWrapper dw, CINT32 node, CUINT8 ax0, CINT32 nkeep, CINT32 nmax, CFLOAT64 pratio):
  cdef CINT32 tidx = dw.allocate_tree()
  cdef CINT32 ret

  ret = _prune_large_node(dw.tdata[tidx], dw.imdata, dw.cdata, dw.exdata, dw.psdata, dw.opts, node, ax0, nkeep, nmax, pratio)

  return (ret,)
# -------

@basefunc
def prune_large_all(DataWrapper dw, CINT32 nlarge, CINT32 nkeep, CINT32 nmax, CFLOAT64 pratio):
  cdef CINT32 tidx = dw.allocate_tree()
  cdef CINT32 node,ret,N
  cdef CUINT8 side

  for side in range(2):
    t0 = time()
    print(f'============================== Side {side} ==============================')
    if side==0: N = dw.imdata.nrows
    else:       N = dw.imdata.ncols

    for node in range(N):
      if node%10000==0: print(f'{node}/{N} [t={time()-t0:.3f}s]')
      if dw.imdata.indptr[side][node+1]-dw.imdata.indptr[side][node]>=nlarge:
        ret = _prune_large_node(dw.tdata[tidx], dw.imdata, dw.cdata, dw.exdata, dw.psdata, dw.opts, node, side, nkeep, nmax, pratio)
        if ret<0: return (ret,)

  return (ret,)
# -------

@basefunc
def prune_edge(DataWrapper dw, CINT32 edge, CFLOAT64 pratio=100.):
  cdef CINT32 ret,rem=0
  cdef CINT32 tidx = dw.allocate_tree()

  if (dw.imdata.prbs[edge]==0.): return 0
  ret = _prune_egde(dw.tdata[tidx], dw.imdata, dw.cdata, dw.exdata, dw.psdata, dw.opts, edge, pratio, 1, &rem)
  if ret<0: return (ret,)
  
  return (rem,)
# -------

@basefunc
def prune_edge_all(DataWrapper dw, CUINT8[:] done, CINT32 max_cycles=5, CINT32 idx_max=0, CFLOAT64 pratio=100.):
  cdef CINT32 i,k,edge,rem,ret
  cdef CINT32 tidx = dw.allocate_tree()

  assert len(done)==dw.imdata.N, 'done array must have length equal to number of edges'

  _p0s = np.zeros(dw.imdata.N, dtype=np.float64)
  cdef CFLOAT64[:] p0s = _p0s
  for i in range(dw.imdata.N): p0s[i] = dw.imdata.prbs[i]

  _isort = np.argsort(_p0s).astype(np.int32)
  cdef CINT32[:] isort = _isort

  idx_max = min(dw.imdata.N, idx_max) if idx_max>0 else dw.imdata.N

  for k in range(max_cycles):
    print(f'===== Cycle {k} =====')
    t0 = time()
    rem = 0
    for i in range(idx_max):
      if i%100000==0: print(f'{i}/{dw.imdata.N} [t={time()-t0:.3f}s]')
      edge = isort[i]
      if (dw.imdata.prbs[edge]==0.) or done[edge]: continue

      ret = _prune_egde(dw.tdata[tidx], dw.imdata, dw.cdata, dw.exdata, dw.psdata, dw.opts, edge, pratio, 0, &rem)
      if ret<0: return (ret,)
      done[edge] = <CUINT8>ret

    print(f'... removed {rem} edges [t={time()-t0:.3f}s]')
    if rem==0: break
  
  return (0,)
# -------


@basefunc
def prune_probable_matches(DataWrapper dw, CINT32 edge, CFLOAT64 pthr, CINT32 nmax):
  cdef CINT32 i,ret
  cdef CINT32 tidx = dw.allocate_tree()

  ret = _prune_probable_matches(dw.tdata[tidx], dw.imdata, dw.cdata, dw.exdata, dw.psdata, dw.opts, edge, pthr, nmax)

  return (0, ret)
# -------


@basefunc
def prune_probable_matches_all(DataWrapper dw, CINT32 max_cycles, CFLOAT64 pthr, CINT32 nmax):
  cdef CINT32 i,c,ret,rem,nno
  cdef CINT32 tidx = dw.allocate_tree()

  _p0s = np.zeros(dw.imdata.N, dtype=np.float64)
  cdef CFLOAT64[:] p0s = _p0s
  for i in range(dw.imdata.N): p0s[i] = dw.imdata.prbs[i]

  _isort = np.argsort(_p0s)[::-1].astype(np.int32)
  cdef CINT32[:] isort = _isort

  _done = np.zeros(dw.imdata.N, dtype=np.uint8)
  cdef CUINT8[:] done = _done

  t0 = time()
  for c in range(max_cycles):
    print(f'==================== Cycle {c} ====================')
    rem = 0
    nno = 0
    for i in range(dw.imdata.N):
      if i%50000==0: print(f'{i}/{dw.imdata.N} [t={time()-t0:.3f}s]')
      if done[i]: continue

      ret = _prune_probable_matches(dw.tdata[tidx], dw.imdata, dw.cdata, dw.exdata, dw.psdata, dw.opts, isort[i], pthr, nmax)
      if ret>=0:
        rem += ret
        done[i] = 1
      else:
        nno += 1
        done[i] = 0
    
    print(f'--> removed {rem} edges, skipped {nno} [t={time()-t0:.3f}s]')
    if rem==0: break

  return (0,)
# -------





# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef inline CINT32 _prune_large_node(TreeData* tdata, ImatData* imdata, CombsData* cdata, ExclData *exdata, PsetData *psdata, OptsData* opts,
                                     CINT32 node, CUINT8 ax0, CINT32 nkeep, CINT32 nmax, CFLOAT64 pratio) except -32010:
  cdef CINT32 i,j,k, ee,ret=0
  cdef CFLOAT64 pmax, pcur
  cdef CINT32 p0 = imdata.indptr[ax0][node]
  cdef CINT32 N  = imdata.indptr[ax0][node+1]-p0
  if N<=0: return -32011

  cdef CINT32* isort = <CINT32*>malloc(N*sizeof(CINT32))
  cdef CFLOAT64* pvals = <CFLOAT64*>malloc(N*sizeof(CFLOAT64))

  k = 0
  for i in range(N):
    if imdata.prbs[imdata.data[ax0][p0+i]]==0.: continue
    isort[i] = i
    pvals[i] = imdata.prbs[imdata.data[ax0][p0+i]]
    k += 1
  argsort_F64(pvals, isort, k, 1, 1)

  if k>nkeep:
    for i in range(nkeep,k):
      ee = imdata.data[ax0][p0+isort[i]]
      if imdata.prbs[ee]==0.: continue

      ret = _init_from_edge(tdata, imdata, ee)
      if ret<0: break

      ret = _walk_fwd_sort(tdata, imdata, opts, nmax)
      if ret<0: break
      ret = _walk_fwd_sort(tdata, imdata, opts, nmax)
      if ret<0: break
      ret = _tree_calc_pfull2(tdata, cdata, exdata, psdata, opts, 1)
      if ret<0: break

      pcur = tdata.edges[0][1].pfull2
      pmax = 0.
      for j in range(tdata.counts[1]):
        if (tdata.edges[1][j].nodes[ax0]==node) and (tdata.edges[1][j].pfull2>pmax): pmax = tdata.edges[1][j].pfull2

      if pcur<pmax/pratio: imdata.prbs[ee] = 0.
  
  free(isort)
  free(pvals)
  return ret
# -------

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef inline CINT32 _prune_egde(TreeData* tdata, ImatData* imdata, CombsData* cdata, ExclData* exdata, PsetData* psdata, OptsData *opts,
                               CINT32 edge, CFLOAT64 pratio, CUINT8 error, CINT32* rem) except -32020:
  cdef CINT32 i,ret
  cdef CFLOAT64 thr

  ret = _init_from_edge(tdata, imdata, edge)
  if ret<0: return ret

  ret = _walk_fwd(tdata, imdata, opts)
  if ret<0:  return ret if error else 0
  if ret==0: return 1

  if tdata.n_edges > cdata.max_edges: return -32021 if error else 0 # too many edges to calculate pfull

  ret = _walk_fwd_sort(tdata, imdata, opts, 3)
  if ret<0: return ret if error else 0

  if tdata.n_edges > cdata.max_edges:  return -32022 if error else 0 # too many edges to calculate pfull

  ret = _tree_calc_pfull2(tdata, cdata, exdata, psdata, opts, 1)
  if ret<0: return ret if error else 0

  if tdata.edges[0][1].pfull2==0.: return -32023 if error else 0
  thr = tdata.edges[0][1].pfull2/pratio

  for i in range(tdata.counts[1]):
    if tdata.edges[1][i].pfull2<thr:
      imdata.prbs[tdata.edges[1][i].edge] = 0.
      rem[0] += 1

  return 1
# -------



# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef inline CINT32 _prune_probable_matches(TreeData* tdata, ImatData* imdata, CombsData* cdata, ExclData* exdata, PsetData* psdata, OptsData *opts,
                                           CINT32 edge, CFLOAT64 pthr, CINT32 nmax) except -32040:
  if imdata.prbs[edge]==0.: return 0

  cdef CINT32 i,j,ret

  ret = _init_from_edge(tdata, imdata, edge)
  if ret<0: return ret

  ret = _walk_fwd(tdata, imdata, opts)
  if ret<=0:  return ret

  if tdata.n_edges > cdata.max_edges: return -32041

  ret = _walk_fwd_sort(tdata, imdata, opts, nmax)
  if ret<0: return ret

  if tdata.n_edges > cdata.max_edges:  return -32042

  ret = _walk_fwd_sort(tdata, imdata, opts, nmax)
  if ret<0: return ret

  if tdata.n_edges > cdata.max_edges:  return -32043

  ret = _tree_calc_pfull2(tdata, cdata, exdata, psdata, opts, 2)
  if ret<0: return ret

  ret = 0
  if tdata.edges[0][1].pfull2>=pthr:
    for i in range(tdata.counts[1]):
      ee1 = tdata.edges[1][i].edge
      for j in range(exdata.indptr[edge],exdata.indptr[edge+1]):
        if exdata.indices[j]<ee1: continue
        if exdata.indices[j]>ee1: break
        
        if imdata.prbs[ee1]>0.: ret += 1
        imdata.prbs[ee1] = 0.
  
  return ret
# -------
