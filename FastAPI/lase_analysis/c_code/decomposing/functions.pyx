# cython: profile=True

cimport cython

import numpy as np
cimport numpy as np

from scipy.sparse import coo_array

from libc.stdlib cimport malloc, free
from libc.math cimport round
from numbers import Number
from time import time

ctypedef np.uint8_t CUINT8
ctypedef np.uint16_t CUINT16
ctypedef np.int32_t CINT32
ctypedef np.uint64_t CUINT64
ctypedef np.int64_t CINT64
ctypedef np.float32_t CFLOAT32
ctypedef np.float64_t CFLOAT64

cdef CFLOAT32 _INF = np.finfo('f4').max
cdef CINT32 _MAXINT32 = np.iinfo('i4').max


# ------------------------------ C IMPORTS -------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

cdef extern from "c_dict.h":
  CINT32 create_dict()
  void set_item(CINT32 did, CINT32 key, CINT32 value)
  CINT32 get_item(CINT32 did, CINT32 key, CINT32 *value)
  void delete_item(CINT32 did, CINT32 key)
  CINT32 contains_key(CINT32 did, CINT32 key)
  void delete_dict(CINT32 did)
  void delete_all()
  CINT32 ERR_NOTIN
# ----

cdef extern from "c_argsort.h":
  CINT32 argsort_F64(CFLOAT64* array, CINT32* indices, CINT32 length, CUINT8 reverse, CUINT8 inplace)
  CINT32 argsort_I32(CINT32* array, CINT32* indices, CINT32 length, CUINT8 reverse, CUINT8 inplace)
# ----

# ------------------------------ STRUCTURES ------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

cdef struct ExclData:
  CINT32* indices
  CINT32* indptr
# -------

cdef struct ImatData:
  CINT32** data
  CINT32** indices
  CINT32** indptr

  CINT32* edge_row
  CINT32* edge_col

  CFLOAT64* prbs
  
  CINT32 nrows
  CINT32 ncols
  CINT32 N
# -------

cdef struct TreeEdge:
  CINT32 edge
  CINT32[2] nodes # [0]: row, [1]: col
  CUINT8 side # index of arrival node

  CFLOAT64 psing
  CFLOAT64 pfull
  CFLOAT64 pfull2
  
  CINT32 depth
# -------

cdef struct TreeData:
  TreeEdge** edges
  CINT32* counts
  CINT32 n_edges
  TreeEdge** eptr

  CINT32[2] ndicts
  CINT32 edict
  
  CINT32* isort
  CINT32* edges_sort
  CFLOAT64* pvals_sort
  TreeEdge** eptr_sort

  CUINT8 is_edge
  CINT32 depth
  CINT32 max_edges
  CINT32 max_depth
# -------

cdef struct CombsData:
  CUINT8* cmpt
  CINT32* combs
  CFLOAT64* prbs
  CFLOAT64* prbs2
  CFLOAT64* pset

  CINT32 max_combs
  CINT32 max_edges
  CINT32 n_edges
  CINT32 n_combs
# -------

cdef struct CombsData2:
  CUINT8* cmpt
  CFLOAT64* poks
  CFLOAT64* prbs

  CINT32 dictR
  CINT32 dictC

  CFLOAT64 pmsR
  CFLOAT64 pmsC
  CFLOAT64 ptot
  CFLOAT64 pmax0
  CFLOAT64 pmax
  CFLOAT64 ratio

  CINT32 core
  CINT32 max_edges
  CINT32 n_edges
  CINT32 n_miss
  CINT32 n_tot
# -------

cdef struct PsetData:
  CFLOAT64** psets
  CINT32 max_nodes
  CINT32 N

  CINT32 dictNR
  CINT32 dictNC

  CUINT8** cores
  CUINT8** dones
  CINT32** edge2idx
# -------

cdef struct OptsData:
  CFLOAT64 pmin_walk
  CFLOAT64 pmin_comb
  CFLOAT64 ratio_comb

  CFLOAT64 pmsR
  CFLOAT64 pmsC
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
# error code -1000
cdef CINT32 binom(CINT32 k, CINT32 n) except -1000:
  if k>n:  return -1000
  if n==0: return 0
  if (k==0) or (k==n): return 1

  cdef CINT32 i,val=1

  k = min(k,n-k) # C(k,n) = C(n-k,n)
  for i in range(1,k+1): val = val*(n-i+1)//i
  return val
# -------

# ------------------------------ DECORATORS ------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef class DataWrapper:
  cdef OptsData *opts
  cdef ExclData *exdata
  cdef ImatData *imdata
  cdef TreeData **tdata
  cdef CombsData *cdata
  cdef CombsData2 *cdata2
  cdef PsetData *psdata

  cdef CINT32 ntree
  cdef CINT32 max_ntree
  cdef CINT32 tmp_edges
  cdef CINT32 max_edges
  cdef CINT32 max_depth

  def __cinit__(self, poss, opts, ptree=None):
    cdef CINT32 i
    self.ntree = 0
    self.max_ntree = 5
    self.tdata = <TreeData**>malloc(self.max_ntree*sizeof(TreeData*))
    if not self.tdata: raise MemoryError("Failed to allocate memory for TreeData")
    for i in range(self.max_ntree): self.tdata[i] = NULL

    self.tmp_edges = opts.tmp_edges
    self.max_edges = opts.max_edges
    self.max_depth = opts.max_depth

    # --- WalkOpts structure ----------
    self.opts = <OptsData*>malloc(sizeof(OptsData))
    if not self.opts: raise MemoryError("Failed to allocate memory for OptsData")

    self.opts.pmin_walk = opts.pmin_walk
    self.opts.pmin_comb = opts.pmin_comb
    self.opts.ratio_comb = opts.ratio_comb
    self.opts.pmsR = opts.pmsR
    self.opts.pmsC = opts.pmsC

    # --- Exclusion matrix ----------
    self.exdata = <ExclData*>malloc(sizeof(ExclData))
    if not self.exdata: raise MemoryError("Failed to allocate memory for ExclData")

    cdef CINT32[:] excl_indptr = poss.excl.indptr
    cdef CINT32[:] excl_indices = poss.excl.indices
    self.exdata.indptr = &excl_indptr[0]
    self.exdata.indices = &excl_indices[0]

    # --- ImatData structure ----------
    self.imdata = <ImatData*>malloc(sizeof(ImatData))
    if not self.imdata: raise MemoryError("Failed to allocate memory for ImatData")
    self.imdata.data = NULL
    self.imdata.indices = NULL
    self.imdata.indptr = NULL

    self.imdata.nrows = <CINT32>poss.imatr.shape[0]
    self.imdata.ncols = <CINT32>poss.imatr.shape[1]
    self.imdata.N = <CINT32>len(poss.imatr.data)

    poss._temp_p0 = poss.vals['p0'].astype(np.float64)
    cdef CFLOAT64[:] imat_prbs = poss._temp_p0
    self.imdata.prbs = &imat_prbs[0]

    cdef CINT32[:] erows = poss.rows
    self.imdata.edge_row = &erows[0]

    cdef CINT32[:] ecols = poss.cols
    self.imdata.edge_col = &ecols[0]

    # Init data
    cdef CINT32[:] imat_rdata = poss.imatr.data
    cdef CINT32[:] imat_cdata = poss.imatc.data
    self.imdata.data = <CINT32**>malloc(2*sizeof(CINT32*))
    if not self.imdata.data: raise MemoryError("Failed to allocate memory for ImatData.data")
    self.imdata.data[0] = &imat_rdata[0]
    self.imdata.data[1] = &imat_cdata[0]

    # Init indices
    cdef CINT32[:] imat_rindices = poss.imatr.indices
    cdef CINT32[:] imat_cindices = poss.imatc.indices
    self.imdata.indices = <CINT32**>malloc(2*sizeof(CINT32*))
    if not self.imdata.indices: raise MemoryError("Failed to allocate memory for ImatData.indices")
    self.imdata.indices[0] = &imat_rindices[0]
    self.imdata.indices[1] = &imat_cindices[0]

    # Init indptr
    cdef CINT32[:] imat_rindptr = poss.imatr.indptr
    cdef CINT32[:] imat_cindptr = poss.imatc.indptr
    self.imdata.indptr = <CINT32**>malloc(2*sizeof(CINT32*))
    if not self.imdata.indptr: raise MemoryError("Failed to allocate memory for ImatData.indptr")
    self.imdata.indptr[0] = &imat_rindptr[0]
    self.imdata.indptr[1] = &imat_cindptr[0]

    # --- Combinations arrays ----------
    self.cdata = <CombsData*>malloc(sizeof(CombsData))
    if not self.cdata: raise MemoryError("Failed to allocate memory for CombsData")
    self.cdata.cmpt = NULL
    self.cdata.combs = NULL
    self.cdata.prbs = NULL
    self.cdata.prbs2 = NULL
    self.cdata.pset = NULL

    self.cdata.max_combs = opts.max_combs
    self.cdata.max_edges = opts.max_edges
    self.cdata.n_edges = 0

    # Init arrays
    self.cdata.cmpt = <CUINT8*>malloc(self.cdata.max_edges*self.cdata.max_edges*sizeof(CUINT8))
    if not self.cdata.cmpt: raise MemoryError("Failed to allocate memory for CombsData.cmpt")

    self.cdata.combs = <CINT32*>malloc(self.cdata.max_combs*(self.cdata.max_edges+1)*sizeof(CINT32))
    if not self.cdata.combs: raise MemoryError("Failed to allocate memory for CombsData.combs")

    self.cdata.prbs = <CFLOAT64*>malloc(self.cdata.max_combs*sizeof(CFLOAT64))
    if not self.cdata.prbs: raise MemoryError("Failed to allocate memory for CombsData.prbs")

    self.cdata.prbs2 = <CFLOAT64*>malloc(self.cdata.max_combs*sizeof(CFLOAT64))
    if not self.cdata.prbs2: raise MemoryError("Failed to allocate memory for CombsData.prbs2")

    self.cdata.pset = <CFLOAT64*>malloc(self.cdata.max_combs*sizeof(CFLOAT64))
    if not self.cdata.pset: raise MemoryError("Failed to allocate memory for CombsData.pset")
    
    # --- Combinations2 arrays ----------
    self.cdata2 = <CombsData2*>malloc(sizeof(CombsData2))
    if not self.cdata2: raise MemoryError("Failed to allocate memory for CombsData2")
    self.cdata2.cmpt = NULL
    self.cdata2.poks = NULL
    self.cdata2.prbs = NULL
    self.cdata2.dictR = create_dict()
    self.cdata2.dictC = create_dict()

    self.cdata2.max_edges = opts.max_edges
    self.cdata2.n_edges = 0
    self.cdata2.n_miss = 0
    self.cdata2.n_tot = 0
    self.cdata2.core = 0

    self.cdata2.pmsR = self.opts.pmsR
    self.cdata2.pmsC = self.opts.pmsC
    self.cdata2.ptot = 0.
    self.cdata2.pmax0 = 0.
    self.cdata2.pmax = 0.
    self.cdata2.ratio = self.opts.ratio_comb

    # Init arrays
    self.cdata2.cmpt = <CUINT8*>malloc(self.cdata2.max_edges*self.cdata2.max_edges*sizeof(CUINT8))
    if not self.cdata2.cmpt: raise MemoryError("Failed to allocate memory for CombsData2.cmpt")

    self.cdata2.prbs = <CFLOAT64*>malloc(self.cdata.max_edges*sizeof(CFLOAT64))
    if not self.cdata2.prbs: raise MemoryError("Failed to allocate memory for CombsData2.prbs")

    # --- PsetData structure ----------
    self.psdata = <PsetData*>malloc(sizeof(PsetData))
    if not self.psdata: raise MemoryError("Failed to allocate memory for PsetData")
    self.psdata.psets = NULL
    self.psdata.cores = NULL
    self.psdata.dones = NULL
    self.psdata.edge2idx = NULL

    assert opts.psetsR.shape[0]==opts.psetsR.shape[1], "PsetsR must be a square matrix"
    assert opts.psetsR.shape[0]==opts.psetsC.shape[0], "PsetsR and PsetsC must have the same number of rows"
    assert opts.psetsR.shape[1]==opts.psetsC.shape[1], "PsetsR and PsetsC must have the same number of columns"

    self.psdata.max_nodes = opts.psetsR.shape[0]-1
    self.psdata.N = 0

    # Init psets
    cdef CFLOAT64[:] psetsR = opts.psetsR.ravel('C')
    cdef CFLOAT64[:] psetsC = opts.psetsC.ravel('C')

    self.psdata.psets = <CFLOAT64**>malloc(2*sizeof(CFLOAT64*))
    if not self.psdata.psets: raise MemoryError("Failed to allocate memory for PsetData.psets")
    self.psdata.psets[0] = &psetsR[0]
    self.psdata.psets[1] = &psetsC[0]

    # Init cores
    self.psdata.cores = <CUINT8**>malloc(2*sizeof(CUINT8*))
    if not self.psdata.cores: raise MemoryError("Failed to allocate memory for PsetData.cores")
    self.psdata.cores[0] = NULL
    self.psdata.cores[1] = NULL
    self.psdata.cores[0] = <CUINT8*>malloc(self.psdata.max_nodes*sizeof(CUINT8))
    if not self.psdata.cores[0]: raise MemoryError("Failed to allocate memory for PsetData.cores[0]")
    self.psdata.cores[1] = <CUINT8*>malloc(self.psdata.max_nodes*sizeof(CUINT8))
    if not self.psdata.cores[1]: raise MemoryError("Failed to allocate memory for PsetData.cores[1]")

    # Init dones
    self.psdata.dones = <CUINT8**>malloc(2*sizeof(CUINT8*))
    if not self.psdata.dones: raise MemoryError("Failed to allocate memory for PsetData.dones")
    self.psdata.dones[0] = NULL
    self.psdata.dones[1] = NULL
    self.psdata.dones[0] = <CUINT8*>malloc(self.psdata.max_nodes*sizeof(CUINT8))
    if not self.psdata.dones[0]: raise MemoryError("Failed to allocate memory for PsetData.dones[0]")
    self.psdata.dones[1] = <CUINT8*>malloc(self.psdata.max_nodes*sizeof(CUINT8))
    if not self.psdata.dones[1]: raise MemoryError("Failed to allocate memory for PsetData.dones[1]")

    # Init edge2idx
    self.psdata.edge2idx = <CINT32**>malloc(2*sizeof(CINT32*))
    if not self.psdata.edge2idx: raise MemoryError("Failed to allocate memory for PsetData.edge2idx")
    self.psdata.edge2idx[0] = NULL
    self.psdata.edge2idx[1] = NULL
    self.psdata.edge2idx[0] = <CINT32*>malloc(self.cdata.max_edges*sizeof(CINT32))
    if not self.psdata.edge2idx[0]: raise MemoryError("Failed to allocate memory for PsetData.edge2idx[0]")
    self.psdata.edge2idx[1] = <CINT32*>malloc(self.cdata.max_edges*sizeof(CINT32))
    if not self.psdata.edge2idx[1]: raise MemoryError("Failed to allocate memory for PsetData.edge2idx[1]")

    # Init dicts
    self.psdata.dictNR = create_dict()
    if self.psdata.dictNR==-1: raise MemoryError("Failed to allocate memory for dict 'dictNR'")
    self.psdata.dictNC = create_dict()
    if self.psdata.dictNC==-1: raise MemoryError("Failed to allocate memory for dict 'dictNC'")
  # -------

  def allocate_tree(self):
    if self.ntree>=self.max_ntree: return -1

    # --- TreeData structure ----------
    self.tdata[self.ntree] = <TreeData*>malloc(sizeof(TreeData))
    if not self.tdata[self.ntree]: raise MemoryError("Failed to allocate memory for TreeData")
    cdef TreeData* tdata = self.tdata[self.ntree]

    tdata.edges = NULL
    tdata.eptr = NULL
    tdata.counts = NULL
    tdata.isort = NULL
    tdata.edges_sort = NULL
    tdata.pvals_sort = NULL
    
    tdata.max_edges = max(self.tmp_edges, self.max_edges+1)
    tdata.max_depth = self.max_depth
    tdata.is_edge = 0
    tdata.depth = 0
    tdata.n_edges = 0

    # Init arrays
    tdata.edges = <TreeEdge**>malloc((tdata.max_depth+1)*sizeof(TreeEdge*))
    if not tdata.edges: raise MemoryError("Failed to allocate memory for TreeData.edges")
    for i in range(tdata.max_depth+1): tdata.edges[i] = NULL
    for i in range(tdata.max_depth+1):
      tdata.edges[i] = <TreeEdge*>malloc((tdata.max_edges)*sizeof(TreeEdge))
      if not tdata.edges[i]: raise MemoryError(f"Failed to allocate memory for TreeData.edges[{i}]")

    tdata.eptr = <TreeEdge**>malloc((tdata.max_depth+1)*(tdata.max_edges)*sizeof(TreeEdge*))
    if not tdata.eptr: raise MemoryError("Failed to allocate memory for TreeData.eptr")

    tdata.counts = <CINT32*>malloc((tdata.max_depth+1)*sizeof(CINT32))
    if not tdata.counts: raise MemoryError("Failed to allocate memory for TreeData.counts")

    tdata.isort = <CINT32*>malloc(tdata.max_edges*sizeof(CINT32))
    if not tdata.isort: raise MemoryError("Failed to allocate memory for TreeData.isort")

    tdata.edges_sort = <CINT32*>malloc(tdata.max_edges*sizeof(CINT32))
    if not tdata.edges_sort: raise MemoryError("Failed to allocate memory for TreeData.edges_sort")

    tdata.pvals_sort = <CFLOAT64*>malloc(tdata.max_edges*sizeof(CFLOAT64))
    if not tdata.pvals_sort: raise MemoryError("Failed to allocate memory for TreeData.pvals_sort")

    tdata.eptr_sort = <TreeEdge**>malloc((tdata.max_depth+1)*(tdata.max_edges)*sizeof(TreeEdge*))
    if not tdata.eptr_sort: raise MemoryError("Failed to allocate memory for TreeData.eptr_sort")

    # Init dicts
    tdata.edict = create_dict()
    if tdata.edict==-1: raise MemoryError("Failed to allocate memory for dict 'edict'")
    tdata.ndicts[0] = create_dict()
    if tdata.ndicts[0]==-1: raise MemoryError("Failed to allocate memory for dict 'ndicts[0]'")
    tdata.ndicts[1] = create_dict()
    if tdata.ndicts[1]==-1: raise MemoryError("Failed to allocate memory for dict 'ndicts[1]'")

    self.ntree += 1
    return self.ntree-1
  # -------

  def init_from_tree(self, CINT32 tidx, ptree):
    if tidx >= self.ntree: return -1

    cdef TreeData* tdata = self.tdata[tidx]
    if ptree.depth > tdata.max_depth: return -2
    if ptree.ptr[ptree.depth] > tdata.max_edges: return -3

    cdef CINT32 i,j,p0

    cdef CINT32[:] tptr = ptree.ptr
    cdef CINT32[:] tedges = ptree.edges
    cdef CINT32[:] tnodesR = ptree.nodesR
    cdef CINT32[:] tnodesC = ptree.nodesC
    cdef CUINT8[:] tsides = ptree.sides
    cdef CFLOAT64[:] tpsing = ptree.psing
    cdef CFLOAT64[:] tpfull = ptree.pfull
    cdef CFLOAT64[:] tpfull2 = ptree.pfull2

    delete_dict(tdata.edict)
    delete_dict(tdata.ndicts[0])
    delete_dict(tdata.ndicts[1])

    tdata.n_edges = -1
    for i in range(ptree.depth):
      p0 = tptr[i]
      for j in range(p0,tptr[i+1]):
        tdata.edges[i][j-p0].edge = tedges[j]
        tdata.edges[i][j-p0].nodes[0] = tnodesR[j]
        tdata.edges[i][j-p0].nodes[1] = tnodesC[j]
        tdata.edges[i][j-p0].psing = tpsing[j]
        tdata.edges[i][j-p0].pfull = tpfull[j]
        tdata.edges[i][j-p0].pfull2 = tpfull2[j]
        tdata.edges[i][j-p0].side = tsides[j]
        tdata.edges[i][j-p0].depth = i
        tdata.n_edges += 1
        
        if (tedges[j]>=0) and not contains_key(tdata.edict, tedges[j]):
          set_item(tdata.edict, tedges[j], i)
        if not contains_key(tdata.ndicts[0], tnodesR[j]):
          if (i==0) and tnodesR[j]>=0: set_item(tdata.ndicts[0], tnodesR[j], 0)
          elif tsides[j]==0:           set_item(tdata.ndicts[0], tnodesR[j], i)
          else:                        set_item(tdata.ndicts[0], tnodesR[j], i+1)
        if not contains_key(tdata.ndicts[1], tnodesC[j]):
          if (i==0) and tnodesC[j]>=0: set_item(tdata.ndicts[1], tnodesC[j], 0)
          elif tsides[j]==1:           set_item(tdata.ndicts[1], tnodesC[j], i)
          else:                        set_item(tdata.ndicts[1], tnodesC[j], i+1)

      tdata.counts[i] = tptr[i+1]-p0
    for j in range(i+1, tdata.max_depth): tdata.counts[j] = 0

    tdata.depth = ptree.depth-1
  # -------

  def update_ptree(self, CINT32 tidx, ptree):
    if tidx >= self.ntree: return -1
    cdef TreeData* tdata = self.tdata[tidx]

    ptree.edges = np.zeros(tdata.n_edges+1, dtype=np.int32)
    ptree.nodesR = np.zeros(tdata.n_edges+1, dtype=np.int32)
    ptree.nodesC = np.zeros(tdata.n_edges+1, dtype=np.int32)
    ptree.sides = np.zeros(tdata.n_edges+1, dtype=np.uint8)
    ptree.psing = np.zeros(tdata.n_edges+1, dtype=np.float64)
    ptree.pfull = np.zeros(tdata.n_edges+1, dtype=np.float64)
    ptree.pfull2 = np.zeros(tdata.n_edges+1, dtype=np.float64)

    cdef CINT32[:] _edges_ = ptree.edges
    cdef CINT32[:] _nodesR_ = ptree.nodesR
    cdef CINT32[:] _nodesC_ = ptree.nodesC
    cdef CUINT8[:] _sides_ = ptree.sides
    cdef CFLOAT64[:] _psing_ = ptree.psing
    cdef CFLOAT64[:] _pfull_ = ptree.pfull
    cdef CFLOAT64[:] _pfull2_ = ptree.pfull2

    cdef CINT32 i,j,k=0
    for i in range(tdata.depth+1):
      for j in range(tdata.counts[i]):
        _edges_[k] = tdata.edges[i][j].edge
        _nodesR_[k] = tdata.edges[i][j].nodes[0]
        _nodesC_[k] = tdata.edges[i][j].nodes[1]
        _psing_[k] = tdata.edges[i][j].psing
        _pfull_[k] = tdata.edges[i][j].pfull
        _pfull2_[k] = tdata.edges[i][j].pfull2
        _sides_[k] = tdata.edges[i][j].side
        k += 1

    ptree.ptr = np.zeros(tdata.depth+2, dtype=np.int32)
    for i in range(1,tdata.depth+2): ptree.ptr[i] = ptree.ptr[i-1]+tdata.counts[i-1]
  # -------

  def get_combs(self):
    cdef CINT32 i,j,M=self.cdata.max_edges+1
    cdef CINT32 cc=self.cdata.n_combs

    _combs = np.zeros((cc, self.cdata.max_edges+1), dtype=np.int32)
    cdef CINT32[:,:] combs = _combs
    _cprbs = np.zeros(cc, dtype=np.float64)
    cdef CFLOAT64[:] cprbs = _cprbs
    _cpset = np.zeros(cc, dtype=np.float64)
    cdef CFLOAT64[:] cpset = _cpset

    for i in range(cc):
      cprbs[i] = self.cdata.prbs[i]
      cpset[i] = self.cdata.pset[i]
      combs[i,0] = self.cdata.combs[i*M]
      for j in range(1,self.cdata.combs[i*M]+1): combs[i,j] = self.cdata.combs[i*M+j]

    return {'combs': _combs, 'prbs': _cprbs, 'pset': _cpset}
  # -------

  def __dealloc__(self):
    delete_all()

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
      for i in range(self.max_ntree):
        if self.tdata[i]:
          if self.tdata[i].edges:
            for d in range(self.tdata[i].max_depth+1):
              if self.tdata[i].edges[d]: free(self.tdata[i].edges[d])
            free(self.tdata[i].edges)
          if self.tdata[i].eptr: free(self.tdata[i].eptr)
          if self.tdata[i].counts: free(self.tdata[i].counts)
          if self.tdata[i].isort: free(self.tdata[i].isort)
          if self.tdata[i].edges_sort: free(self.tdata[i].edges_sort)
          if self.tdata[i].pvals_sort: free(self.tdata[i].pvals_sort)
          if self.tdata[i].eptr_sort: free(self.tdata[i].eptr_sort)
          free(self.tdata[i])
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

    if self.cdata2:
      if self.cdata2.cmpt: free(self.cdata2.cmpt)
      if self.cdata2.prbs: free(self.cdata2.prbs)
      free(self.cdata2)
    self.cdata2 = NULL

    if self.psdata:
      if self.psdata.psets: free(self.psdata.psets)
      if self.psdata.cores:
        if self.psdata.cores[0]: free(self.psdata.cores[0])
        if self.psdata.cores[1]: free(self.psdata.cores[1])
      if self.psdata.dones:
        if self.psdata.dones[0]: free(self.psdata.dones[0])
        if self.psdata.dones[1]: free(self.psdata.dones[1])
      if self.psdata.edge2idx:
        if self.psdata.edge2idx[0]: free(self.psdata.edge2idx[0])
        if self.psdata.edge2idx[1]: free(self.psdata.edge2idx[1])
      free(self.psdata)
    self.psdata = NULL

# -------

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
def treefunc(func):
  def wrapper(poss, ptree, opts, *args, **kwds):
    cdef CINT32 ret, tidx
    dw = DataWrapper(poss, opts)
    try:
      tidx = dw.allocate_tree()
      dw.init_from_tree(tidx, ptree)
      ret = func(dw, tidx, *args, **kwds)
      if ret>=0: dw.update_ptree(tidx, ptree)
    finally:
      del dw

    if ret<0: raise RuntimeError(f'Error {ret} in {func.__name__}', ret)
    return ret

  return wrapper
# -------

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
def basefunc(func):
  def wrapper(poss, opts, *args, **kwds):
    dw = DataWrapper(poss, opts)
    try:
      ret = func(dw, *args, **kwds)
    finally:
      del dw

    if ret[0]<0: raise RuntimeError(f'Error {ret[0]} in {func.__name__}', ret[0])
    return ret

  return wrapper
# -------

# ----------------------------- SUPPORT FUNCTIONS ------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

cdef CINT32 _copy_tree(TreeData* torig, TreeData* tcopy) except -30000:
  cdef CINT32 i,j,d

  tcopy.n_edges = torig.n_edges
  tcopy.depth = torig.depth
  tcopy.max_depth = torig.max_depth
  tcopy.max_edges = torig.max_edges
  tcopy.is_edge = torig.is_edge

  delete_dict(tcopy.edict)
  delete_dict(tcopy.ndicts[0])
  delete_dict(tcopy.ndicts[1])

  for d in range(torig.depth+1):
    tcopy.counts[d] = torig.counts[d]
    for i in range(torig.counts[d]):
      tcopy.edges[d][i].edge = torig.edges[d][i].edge
      tcopy.edges[d][i].nodes[0] = torig.edges[d][i].nodes[0]
      tcopy.edges[d][i].nodes[1] = torig.edges[d][i].nodes[1]
      tcopy.edges[d][i].psing = torig.edges[d][i].psing
      tcopy.edges[d][i].pfull = torig.edges[d][i].pfull
      tcopy.edges[d][i].pfull2 = torig.edges[d][i].pfull2
      tcopy.edges[d][i].side = torig.edges[d][i].side
      tcopy.edges[d][i].depth = torig.edges[d][i].depth
      if not contains_key(tcopy.edict, torig.edges[d][i].edge): set_item(tcopy.edict, torig.edges[d][i].edge, d)
      if not contains_key(tcopy.ndicts[torig.edges[d][i].side], torig.edges[d][i].nodes[torig.edges[d][i].side]):
        set_item(tcopy.ndicts[torig.edges[d][i].side], torig.edges[d][i].nodes[torig.edges[d][i].side], d)

  return 0
# -------

cdef CINT32 _init_from_node(TreeData* tdata, ImatData* imdata, CINT32 nd0, CUINT8 ax0) except -30010:
  tdata.edges[0][0].edge = -1
  tdata.edges[0][0].nodes[ax0] = nd0
  tdata.edges[0][0].nodes[1-ax0] = -1
  tdata.edges[0][0].side = ax0
  tdata.edges[0][0].psing = 0.
  tdata.edges[0][0].pfull = 0.
  tdata.edges[0][0].pfull2 = 0.
  tdata.edges[0][0].depth = 0

  delete_dict(tdata.ndicts[0])
  delete_dict(tdata.ndicts[1])
  delete_dict(tdata.edict)
  set_item(tdata.ndicts[ax0], nd0, 0)

  tdata.counts[0] = 1
  for j in range(1, tdata.max_depth): tdata.counts[j] = 0
  tdata.depth = 0
  tdata.n_edges = 0
  tdata.is_edge = 0

  return 0
# -------

cdef CINT32 _init_from_edge(TreeData* tdata, ImatData* imdata, CINT32 ee) except -30020:
  tdata.edges[0][0].edge = ee
  tdata.edges[0][0].nodes[0] = imdata.edge_row[ee]
  tdata.edges[0][0].nodes[1] = imdata.edge_col[ee]
  tdata.edges[0][0].side = 0
  tdata.edges[0][0].psing = imdata.prbs[ee]
  tdata.edges[0][0].pfull = 0.
  tdata.edges[0][0].pfull2 = 0.
  tdata.edges[0][0].depth = 0

  tdata.edges[0][1].edge = ee
  tdata.edges[0][1].nodes[0] = imdata.edge_row[ee]
  tdata.edges[0][1].nodes[1] = imdata.edge_col[ee]
  tdata.edges[0][1].side = 1
  tdata.edges[0][1].psing = imdata.prbs[ee]
  tdata.edges[0][1].pfull = 0.
  tdata.edges[0][1].pfull2 = 0.
  tdata.edges[0][1].depth = 0
  
  delete_dict(tdata.ndicts[0])
  delete_dict(tdata.ndicts[1])
  delete_dict(tdata.edict)
  set_item(tdata.ndicts[0], imdata.edge_row[ee], 0)
  set_item(tdata.ndicts[1], imdata.edge_col[ee], 0)
  set_item(tdata.edict, ee, 0)

  tdata.counts[0] = 2
  for j in range(1, tdata.max_depth): tdata.counts[j] = 0
  tdata.depth = 0
  tdata.n_edges = 1
  tdata.is_edge = 1

  return 0
# -------

# ------------------------------ EXCLUSION MATRIX ------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
# Error code -30000
def calc_excl_symm(poss):
  cdef CINT32[:] rindices = poss.rindices
  cdef CINT32[:] rindptr = poss.rindptr
  _rdata = np.arange(len(poss.rindices), dtype=np.int32)
  cdef CINT32[:] rdata = _rdata

  cdef CINT32[:] cindices = poss.cindices
  cdef CINT32[:] cdata = poss.cdata
  cdef CINT32[:] cindptr = poss.cindptr

  cdef CINT32 rr,cc,i,j,i0, N=<CINT32>poss.shape[0], M=<CINT32>poss.shape[1]
  cdef CINT64 iseq=0, OMAX=<CINT64>np.sum(np.diff(poss.rindptr)**2) + <CINT64>np.sum(np.diff(poss.cindptr)**2)

  _outrows = np.zeros(OMAX, dtype=np.int32)
  cdef CINT32[:] outrows = _outrows
  _outcols = np.zeros(OMAX, dtype=np.int32)
  cdef CINT32[:] outcols = _outcols

  # --- Check along rows ----------
  for rr in range(N):
    i0 = rindptr[rr]
    for i in range(i0,rindptr[rr+1]):
      for j in range(i+1,rindptr[rr+1]):
        if iseq+1>=OMAX: return -30001
        outrows[iseq] = rdata[i]
        outcols[iseq] = rdata[j]
        iseq += 1
        outrows[iseq] = rdata[j]
        outcols[iseq] = rdata[i]
        iseq += 1
    
  # --- Check along columns ----------
  for cc in range(M):
    i0 = cindptr[cc]
    for i in range(i0,cindptr[cc+1]):
      for j in range(i+1,cindptr[cc+1]):
        if iseq+1>=OMAX: return -30002
        outrows[iseq] = cdata[i]
        outcols[iseq] = cdata[j]
        iseq += 1
        outrows[iseq] = cdata[j]
        outcols[iseq] = cdata[i]
        iseq += 1

  return coo_array((np.ones(iseq, dtype=np.uint8), (outrows[:iseq], outcols[:iseq])), shape=(len(poss),len(poss))).tocsr()
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
# Error code -30010
def calc_excl_in(poss, CFLOAT32[:] Es0, CUINT8[:] ns0, CFLOAT32[:] Es1, CUINT8[:] ns1, CFLOAT32 max_dE, CFLOAT32 density):
  _cidx0 = np.insert(np.cumsum(ns0,dtype=np.int32),0,0)
  cdef CINT32[:] cidx0 = _cidx0
  _cidx1 = np.insert(np.cumsum(ns1,dtype=np.int32),0,0)
  cdef CINT32[:] cidx1 = _cidx1

  cdef CINT32[:] rindices = poss.rindices
  cdef CINT32[:] rindptr = poss.rindptr
  _rdata = np.arange(len(poss.rindices), dtype=np.int32)
  cdef CINT32[:] rdata = _rdata

  cdef CINT32[:] cindices = poss.cindices
  cdef CINT32[:] cdata = poss.cdata
  cdef CINT32[:] cindptr = poss.cindptr

  cdef CINT32[:] rows = poss.rows
  cdef CINT32[:] cols = poss.cols

  cdef CINT32 CMAX = max(np.diff(poss.rindptr).max(), np.diff(poss.cindptr).max())
  cdef CINT32 MMAX = max(np.max(ns0), np.max(ns1))

  cdef _mtc0 = np.zeros((CMAX,MMAX), dtype=np.uint8)
  cdef CUINT8[:] mtc0 = _mtc0.ravel('C')
  cdef _mtc1 = np.zeros((CMAX,MMAX), dtype=np.uint8)
  cdef CUINT8[:] mtc1 = _mtc1.ravel('C')
  cdef _dEs0 = np.zeros(MMAX, dtype=np.float32)
  cdef CFLOAT32[:] dEs0 = _dEs0
  cdef _dEs1 = np.zeros(MMAX, dtype=np.float32)
  cdef CFLOAT32[:] dEs1 = _dEs1
  cdef _nx0 = np.zeros(CMAX, dtype=np.int32)
  cdef CINT32[:] nx0 = _nx0
  cdef _nx1 = np.zeros(CMAX, dtype=np.int32)
  cdef CINT32[:] nx1 = _nx1

  _skip = np.zeros(CMAX, dtype=np.uint8)
  cdef CUINT8[:] skip = _skip

  cdef CINT32 rr,cc,i,j,k,i0,nm,nx, N=<CINT32>poss.shape[0], M=<CINT32>poss.shape[1]
  cdef CINT64 iseq=0, OMAX=<CINT64>round((np.sum(np.diff(poss.rindptr)**2) + np.sum(np.diff(poss.cindptr)**2))*density)

  _outrows = np.zeros(OMAX, dtype=np.int32)
  cdef CINT32[:] outrows = _outrows
  _outcols = np.zeros(OMAX, dtype=np.int32)
  cdef CINT32[:] outcols = _outcols

  # --- Check along rows ----------
  for rr in range(N):
    i0 = rindptr[rr]
    for i in range(i0,rindptr[rr+1]):
      skip[i-i0] = 0
      cc = rindices[i]
      _match_pair(Es0, cidx0[rr], ns0[rr], dEs0, &mtc0[MMAX*(i-i0)],
                  Es1, cidx1[cc], ns1[cc], dEs1, &mtc1[MMAX*(i-i0)], max_dE, &nm, &nx0[i-i0], &nx1[i-i0])
      
      # Match with nx0<2: (rr0,cc0) is incompatible with all other (rr0,cc_i), i.e. all other matches on same row
      if nx0[i-i0]<2:
        skip[i-i0] = 1
        for j in range(i0,rindptr[rr+1]):
          if j==i: continue
          if iseq+1>=OMAX: return -30011
          outrows[iseq] = rdata[i]
          outcols[iseq] = rdata[j]
          iseq += 1
          outrows[iseq] = rdata[j]
          outcols[iseq] = rdata[i]
          iseq += 1
    
    for i in range(i0,rindptr[rr+1]):
      if skip[i-i0]: continue
      for j in range(i+1,rindptr[rr+1]):
        if skip[j-i0]: continue

        nx = 0
        for k in range(ns0[rr]):
          if (mtc0[MMAX*(i-i0)+k]==0) and (mtc0[MMAX*(j-i0)+k]==0): nx += 1
        # if the best improvement from the combination is less than 2 lines, then the matcehs are incompatible
        if min(nx0[i-i0],nx0[j-i0])-nx < 2:
          if iseq+1>=OMAX: return -30012
          outrows[iseq] = rdata[i]
          outcols[iseq] = rdata[j]
          iseq += 1
          outrows[iseq] = rdata[j]
          outcols[iseq] = rdata[i]
          iseq += 1
  
  # --- Check along columns ----------
  for cc in range(M):
    i0 = cindptr[cc]
    for i in range(i0,cindptr[cc+1]):
      skip[i-i0] = 0
      rr = cindices[i]
      _match_pair(Es0, cidx0[rr], ns0[rr], dEs0, &mtc0[MMAX*(i-i0)],
                  Es1, cidx1[cc], ns1[cc], dEs1, &mtc1[MMAX*(i-i0)], max_dE, &nm, &nx0[i-i0], &nx1[i-i0])
      
      # Match with nx1<2: (rr0,cc0) is incompatible with all other (rr_i,cc0), i.e. all other matches on same column
      if nx1[i-i0]<2:
        skip[i-i0] = 1
        for j in range(i0,cindptr[cc+1]):
          if j==i: continue
          if iseq+1>=OMAX: return -30013
          outrows[iseq] = cdata[i]
          outcols[iseq] = cdata[j]
          iseq += 1
          outrows[iseq] = cdata[j]
          outcols[iseq] = cdata[i]
          iseq += 1
    
    for i in range(i0,cindptr[cc+1]):
      if skip[i-i0]: continue
      for j in range(i+1,cindptr[cc+1]):
        if skip[j-i0]: continue

        nx = 0
        for k in range(ns1[cc]):
          if (mtc1[MMAX*(i-i0)+k]==0) and (mtc1[MMAX*(j-i0)+k]==0): nx += 1
        # if the best improvement from the combination is less than 2 lines, then the matcehs are incompatible
        if min(nx1[i-i0],nx1[j-i0])-nx < 2:
          if iseq+1>=OMAX: return -30014
          outrows[iseq] = cdata[i]
          outcols[iseq] = cdata[j]
          iseq += 1
          outrows[iseq] = cdata[j]
          outcols[iseq] = cdata[i]
          iseq += 1

  return coo_array((np.ones(iseq, dtype=np.uint8), (outrows[:iseq], outcols[:iseq])), shape=(len(poss),len(poss))).tocsr()
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
# Error code -30100
cdef inline CINT32 _find_lines_pairs(CFLOAT32[:] Es0, CINT32 cidx0, CUINT8 _n0, CFLOAT32[:] dEs0, CUINT8* mtc0,
                                     CFLOAT32[:] Es1, CINT32 cidx1, CUINT8 _n1, CFLOAT32[:] dEs1, CUINT8* mtc1) except -30100:
  
  cdef CINT32 i,j, n0=<CINT32>_n0, n1=<CINT32>_n1
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
      mtc0[i] = <CUINT8>j
      do0 = False
    if do1 and (abs(curr)<=bottom):
      dEs1[j] = curr
      mtc1[j] = <CUINT8>i
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
      if bottom < abs(Es1[cidx1+j+1]-Es0[cidx0+i+1]): return -30101

    i += 1
    j += 1
    do0=do1=True

  for i in range(n0):
    if (mtc0[i]>=_n1) or (mtc0[i]<0): return -30102
  for i in range(n1):
    if (mtc1[i]>=_n0) or (mtc1[i]<0): return -30103

  return 0
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
# Error code -30110
cdef inline CINT32 _match_pair(CFLOAT32[:] Es0, CINT32 cidx0, CUINT8 _n0, CFLOAT32[:] dEs0, CUINT8* mtc0, 
                               CFLOAT32[:] Es1, CINT32 cidx1, CUINT8 _n1, CFLOAT32[:] dEs1, CUINT8* mtc1,
                               CFLOAT32 max_dE, CINT32* nm, CINT32* nx0, CINT32* nx1) except -30110:

  cdef CINT32 k, ret

  ret = _find_lines_pairs(Es0, cidx0, _n0, dEs0, mtc0, Es1, cidx1, _n1, dEs1, mtc1)
  if ret < 0: return ret

  cdef CUINT8 *done = <CUINT8*>malloc(<CINT32>_n1*sizeof(CUINT8))
  for k in range(<CINT32>_n1): done[k] = 0

  nx0[0] = 0
  nx1[0] = 0
  nm[0] = 0
  for k in range(<CINT32>_n0):
    if (abs(dEs0[k])<max_dE) and (mtc1[mtc0[k]]==k):
      done[mtc0[k]] = <CUINT8>1
      nm[0] += 1
      mtc0[k] = 1
    else:
      nx0[0] += 1
      mtc0[k] = 0

  for k in range(<CINT32>_n1):
    if done[k]:
      mtc1[k] = 1
    else:
      mtc1[k] = 0
      nx1[0] += 1
  
  free(done)
  return 0
# -------

# ---------------------------- WALK/PRUNE FUNCTIONS ----------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32 _walk_fwd(TreeData* tdata, ImatData* imdata, OptsData* opts) except -31000:
  cdef CINT32 i,j,k, nd0,nd1,ee, val, d=tdata.depth
  cdef CUINT8 ax0

  if d>=tdata.max_depth: return -31001
  
  k = tdata.counts[d+1]
  for i in range(tdata.counts[d]):
    ax0 = tdata.edges[d][i].side
    nd0 = tdata.edges[d][i].nodes[ax0]
    get_item(tdata.ndicts[ax0], nd0, &val)
    if val==ERR_NOTIN: return -31002 # starting node not found
    if val!=d:        continue      # node is not at current depth

    for j in range(imdata.indptr[ax0][nd0],imdata.indptr[ax0][nd0+1]):
      ee = imdata.data[ax0][j]
      if ee<0: continue
      if contains_key(tdata.edict, ee): continue
      if imdata.prbs[ee]<=opts.pmin_walk: continue

      if k>=tdata.max_edges: return -31003
      nd1 = imdata.indices[ax0][j]
      tdata.edges[d+1][k].edge = ee
      tdata.edges[d+1][k].nodes[ax0] = nd0
      tdata.edges[d+1][k].nodes[1-ax0] = nd1
      tdata.edges[d+1][k].psing = imdata.prbs[ee]
      tdata.edges[d+1][k].pfull = 0.
      tdata.edges[d+1][k].pfull2 = 0.
      tdata.edges[d+1][k].side = 1-ax0
      tdata.edges[d+1][k].depth = tdata.depth

      set_item(tdata.edict, ee, d+1)
      if not contains_key(tdata.ndicts[1-ax0], nd1): set_item(tdata.ndicts[1-ax0], nd1, d+1)
      k += 1
      tdata.n_edges += 1

  tdata.depth += 1
  j = k-tdata.counts[d+1]
  tdata.counts[d+1] = k
  
  return j
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32 _walk_fwd_sort(TreeData* tdata, ImatData* imdata, OptsData* opts, CINT32 nmax) except -31010:
  cdef CINT32 i,j,j0,k,p0,cnt, nd0,nd1,ee0,ee1, val,
  cdef CINT32 d=tdata.depth
  cdef CUINT8 ax0

  if d>=tdata.max_depth: return -31011

  k = tdata.counts[d+1]
  # Add the best nmax edges from each node plus eventual edges between the nodes at starting depth
  for i in range(tdata.counts[d]):
    ax0 = tdata.edges[d][i].side
    nd0 = tdata.edges[d][i].nodes[ax0]
    ee0 = tdata.edges[d][i].edge
    
    get_item(tdata.ndicts[ax0], nd0, &val)
    if val==ERR_NOTIN: return -31012 # starting node not found
    if val!=d:         continue      # already walked from this node

    cnt = 0
    for j0 in range(imdata.indptr[ax0][nd0],imdata.indptr[ax0][nd0+1]):
      ee1 = imdata.data[ax0][j0]
      if (ee1<0) or (ee1==ee0) or (imdata.prbs[ee1]<=opts.pmin_walk) or (contains_key(tdata.edict, ee1)): continue
      tdata.isort[cnt] = cnt
      tdata.edges_sort[cnt] = j0
      tdata.pvals_sort[cnt] = imdata.prbs[ee1]
      cnt += 1

    argsort_F64(tdata.pvals_sort, tdata.isort, cnt, 1, 1)

    for j in range(cnt):
      j0 = tdata.edges_sort[tdata.isort[j]]
      ee1 = imdata.data[ax0][j0]
      nd1 = imdata.indices[ax0][j0]
      if (j>=nmax) and not contains_key(tdata.ndicts[1-ax0], nd1): continue
      if k>=tdata.max_edges: return -31013
      tdata.edges[d+1][k].edge = ee1
      tdata.edges[d+1][k].nodes[ax0] = nd0
      tdata.edges[d+1][k].nodes[1-ax0] = nd1
      tdata.edges[d+1][k].psing = imdata.prbs[ee1]
      tdata.edges[d+1][k].pfull = 0.
      tdata.edges[d+1][k].pfull2 = 0.
      tdata.edges[d+1][k].side = 1-ax0
      tdata.edges[d+1][k].depth = d+1

      set_item(tdata.edict, ee1, d+1)
      if not contains_key(tdata.ndicts[1-ax0], nd1): set_item(tdata.ndicts[1-ax0], nd1, d+1)
      k += 1
      tdata.n_edges += 1

  tdata.depth += 1
  j = k-tdata.counts[d+1]
  tdata.counts[d+1] = k
  
  return j
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32 _walk_node(TreeData* tdata, ImatData* imdata, OptsData *opts, CINT32 nd0, CUINT8 ax0) except -31020:
  cdef CINT32 i,j,cnt,d0,d1, nd1,ee

  get_item(tdata.ndicts[ax0], nd0, &d0)
  if d0==ERR_NOTIN: return -31021 # starting node not found
  if d0>=tdata.max_depth: return -31022

  cnt = 0
  for i in range(imdata.indptr[ax0][nd0],imdata.indptr[ax0][nd0+1]):
    ee = imdata.data[ax0][i]
    if ee<0: continue
    if contains_key(tdata.edict, ee): continue
    if imdata.prbs[ee]<=opts.pmin_walk: continue

    nd1 = imdata.indices[ax0][i]
    get_item(tdata.ndicts[1-ax0], nd1, &d1)
    if d1==ERR_NOTIN:
      d1 = d0+1
      set_item(tdata.ndicts[1-ax0], nd1, d1)
    else:
      d1 = min(d1+1, d0+1)
    
    j = tdata.counts[d1]
    if j>=tdata.max_edges: return -31023

    tdata.edges[d1][j].edge = ee
    tdata.edges[d1][j].nodes[ax0] = nd0
    tdata.edges[d1][j].nodes[1-ax0] = nd1
    tdata.edges[d1][j].psing = imdata.prbs[ee]
    tdata.edges[d1][j].pfull = 0.
    tdata.edges[d1][j].pfull2 = 0.
    tdata.edges[d1][j].side = 1-ax0
    tdata.edges[d1][j].depth = d1

    set_item(tdata.edict, ee, d1)
    tdata.counts[d1] += 1
    tdata.n_edges += 1
    cnt += 1

  if tdata.depth < d0+1: tdata.depth = d0+1
  
  return cnt
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32 _walk_node_sort(TreeData* tdata, ImatData* imdata, OptsData* opts, CINT32 nd0, CUINT8 ax0, CINT32 nmax) except -31030:
  cdef CINT32 i,j,k,idx,d0,d1, nd1,ee1,cnt

  get_item(tdata.ndicts[ax0], nd0, &d0)
  if d0==ERR_NOTIN:       return -31031 # starting node not found
  if d0>=tdata.max_depth: return -31032

  # j = tdata.counts[d+1]
  k = 0
  for idx in range(imdata.indptr[ax0][nd0],imdata.indptr[ax0][nd0+1]):
    ee1 = imdata.data[ax0][idx]
    if (ee1<0) or (imdata.prbs[ee1]<=opts.pmin_walk): continue

    tdata.isort[k] = k
    tdata.edges_sort[k] = idx
    tdata.pvals_sort[k] = imdata.prbs[ee1]
    k += 1

  argsort_F64(tdata.pvals_sort, tdata.isort, k, 1, 1)

  cnt = 0
  for i in range(k):
    idx = tdata.edges_sort[tdata.isort[i]]
    ee1 = imdata.data[ax0][idx]
    if (contains_key(tdata.edict, ee1)): continue

    nd1 = imdata.indices[ax0][idx]
    get_item(tdata.ndicts[1-ax0], nd1, &d1)
    if (i>=nmax) and d1==ERR_NOTIN: continue

    if d1==ERR_NOTIN:
      d1 = d0+1
      set_item(tdata.ndicts[1-ax0], nd1, d1)
    else:
      d1 = min(d1+1, d0+1)
    
    j = tdata.counts[d1]
    if j>=tdata.max_edges: return -31033

    tdata.edges[d1][j].edge = ee1
    tdata.edges[d1][j].nodes[ax0] = nd0
    tdata.edges[d1][j].nodes[1-ax0] = nd1
    tdata.edges[d1][j].psing = imdata.prbs[ee1]
    tdata.edges[d1][j].pfull = 0.
    tdata.edges[d1][j].pfull2 = 0.
    tdata.edges[d1][j].side = 1-ax0
    tdata.edges[d1][j].depth = d1

    set_item(tdata.edict, ee1, d1)
    tdata.counts[d1] += 1
    tdata.n_edges += 1
    cnt += 1

  if tdata.depth < d0+1: tdata.depth = d0+1

  return cnt
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32 _add_edge(TreeData* tdata, ImatData* imdata, OptsData* opts, CINT32 ee, CINT32 *nd0, CINT32* nd1) except -31040:
  if contains_key(tdata.edict, ee): return 0 # edge already present

  cdef CINT32 i, ndR,ndC, dR,dC,d0,d1, ret=1
  cdef CUINT8 ax0
  ndR = imdata.edge_row[ee]
  ndC = imdata.edge_col[ee]

  get_item(tdata.ndicts[0], ndR, &dR)
  get_item(tdata.ndicts[0], ndC, &dC)

  if (dR==ERR_NOTIN) and (dC==ERR_NOTIN):
    return -31041 # edge doesn't connect to current tree
  
  if dR==ERR_NOTIN:
    dR = _MAXINT32
    ret = 2
  if dC==ERR_NOTIN:
    dC = _MAXINT32
    ret = 2

  d0 = min(dR,dC)
  ax0 = dR>dC
  d1 = min(d0+1, max(dR,dC))

  if ax0: nd0[0]=ndC; nd1[0]=ndR
  else:   nd0[0]=ndR; nd1[0]=ndC

  i = tdata.counts[d1]
  if i>=tdata.max_edges: return -31042

  tdata.edges[d1][i].edge = ee
  tdata.edges[d1][i].nodes[ax0] = nd0[0]
  tdata.edges[d1][i].nodes[1-ax0] = nd1[0]
  tdata.edges[d1][i].psing = imdata.prbs[ee]
  tdata.edges[d1][i].pfull = 0.
  tdata.edges[d1][i].pfull2 = 0.
  tdata.edges[d1][i].side = 1-ax0
  tdata.edges[d1][i].depth = d1

  set_item(tdata.edict, ee, d1)
  if dR!=dC: set_item(tdata.ndicts[1-ax0], nd1[0], d1)

  tdata.counts[d1] += 1
  tdata.n_edges += 1

  if tdata.depth < d1: tdata.depth = d1
  
  return ret
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32 _tree_calc_pfull(TreeData* tdata, CombsData* cdata, ExclData* exdata, PsetData* psdata, OptsData* opts, CINT32 core_depth) except -31110:
  cdef CINT32 i,j,d,ret

  ret = _tree_calc_combs(tdata, cdata, exdata, opts)
  if ret<0: return ret

  ret = _calc_pset_core(tdata, cdata, psdata, opts, core_depth)
  if ret<0: return ret

  cdef CFLOAT64 psum=0.,psum2=0.
  for i in range(cdata.n_combs):
    cdata.prbs2[i] = cdata.prbs[i]*cdata.pset[i]
    psum += cdata.prbs[i]
    psum2 += cdata.prbs2[i]

  # Normalize probabilities
  if (psum>0.) and (psum2>0.):
    for i in range(cdata.n_combs):
      if psum>0.: cdata.prbs[i] /= psum
      if psum2>0: cdata.prbs2[i] /= psum2
  elif psum>0:
    for i in range(cdata.n_combs): cdata.prbs[i] /= psum
  elif psum2>0.:
    for i in range(cdata.n_combs): cdata.prbs2[i] /= psum2

  cdef CINT32 M=cdata.max_edges+1
  for d in range(tdata.depth+1):
    for i in range(tdata.counts[d]):
      tdata.edges[d][i].pfull = 0.
      # tdata.edges[d][i].pfull2 = 0.

  for i in range(cdata.n_combs):
    for j in range(1,cdata.combs[i*M]+1):
      tdata.eptr[cdata.combs[i*M+j]][0].pfull += cdata.prbs2[i]
      # tdata.eptr[cdata.combs[i*M+j]][0].pfull += cdata.prbs[i]
      # tdata.eptr[cdata.combs[i*M+j]][0].pfull2 += cdata.prbs2[i]

  return 0
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32 _tree_calc_pfull2(TreeData* tdata, CombsData2* cdata, ExclData* exdata, OptsData* opts, CINT32 core_depth) except -31100:
  cdef CINT32 i,j,d,ret

  cdata.core = core_depth
  if cdata.core>tdata.depth: return -31101
  ret = _tree_calc_combs2(tdata, cdata, exdata, opts)
  if ret<0: return ret

  # for i in range(cdata.n_tot): cdata.prbs[i] /= cdata.ptot # normalize probabilities

  # cdef CINT32 M=cdata.max_edges+1
  # for d in range(tdata.depth+1):
  #   for i in range(tdata.counts[d]):
  #     tdata.edges[d][i].pfull = 0.
  #     tdata.edges[d][i].pfull2 = 0.

  # for i in range(cdata.n_combs):
  #   for j in range(1,cdata.combs[i*M]+1):
  #     tdata.eptr[cdata.combs[i*M+j]][0].pfull += cdata.prbs[i]

  return 0
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32 _tree_calc_combs(TreeData* tdata, CombsData* cdata, ExclData* exdata, OptsData* opts) except -31120:
  cdef CINT32 i,j,d,ret,tmp,idxs=0
  cdef CINT32 N=0
  cdef CFLOAT64 pmax=0.

  if tdata.n_edges>cdata.max_edges: return -31121

  # Sort edges in descending order of probability
  for d in range(tdata.depth+1):
    for i in range(tdata.counts[d]):
      if d==0 and i==0: continue
      tdata.isort[N] = N
      tdata.pvals_sort[N] = tdata.edges[d][i].psing
      tdata.eptr[N] = &tdata.edges[d][i]
      N += 1
  if N!=tdata.n_edges: return -31122

  argsort_F64(tdata.pvals_sort, tdata.isort, N, 1, 1)
  for i in range(N): tdata.edges_sort[i] = tdata.eptr[tdata.isort[i]][0].edge

  # Get compatibility matrix
  cdata.n_edges = N
  ret = _get_cmpt(cdata, exdata, tdata.edges_sort)

  # Calculate combinations
  cdef CUINT8 *left = <CUINT8*>malloc(N*sizeof(CUINT8))
  for i in range(N): left[i] = 1

  # Initialize with empty combination
  cdata.combs[0] = 0
  cdata.prbs[0] = 1.
  for i in range(N): cdata.prbs[0] *= 1-tdata.pvals_sort[i]

  cdef CINT32 rc=0
  print(f'Start rec combs: N={N}')
  t0 = time()
  ret = _rec_combs(cdata, tdata.pvals_sort, tdata.isort, left, 0, 1, 1., opts.pmin_comb, opts.ratio_comb, &pmax, &idxs, &rc)
  free(left)
  print(f'End rec combs: {ret} combs, {rc} recursions, dt={time()-t0}')
  if ret<0: return ret

  cdata.n_combs = ret+1
  return 0
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32 _tree_calc_combs2(TreeData* tdata, CombsData2* cdata, ExclData* exdata, OptsData* opts) except -31120:
  cdef CINT32 i,j,d,ret,tmp,idxs=0
  cdef CINT32 Ne=0, Nm=0, Nt

  if tdata.n_edges>cdata.max_edges: return -31121

  # Sort edges in descending order of probability
  for d in range(tdata.depth+1):
    for i in range(tdata.counts[d]):
      if d==0 and i==0: continue
      tdata.isort[Ne] = Ne
      tdata.pvals_sort[Ne] = tdata.edges[d][i].psing
      tdata.eptr[Ne] = &tdata.edges[d][i]
      Ne += 1
  if Ne!=tdata.n_edges: return -31122

  argsort_F64(tdata.pvals_sort, tdata.isort, Ne, 1, 1)
  for i in range(Ne):
    tdata.eptr_sort[i] = tdata.eptr[tdata.isort[i]]
    # tdata.edges_sort[i] = tdata.eptr[tdata.isort[i]][0].edge

  delete_dict(cdata.dictR)
  delete_dict(cdata.dictC)

  # Add core nodes miss-edges
  Nt = Ne
  for d in range(cdata.core+1):
    for i in range(tdata.counts[d]):
      if d==0 and i==0: continue
      if not contains_key(cdata.dictR, tdata.edges[d][i].nodes[0]):
        if Nt >= cdata.max_edges: return -31123
        set_item(cdata.dictR, tdata.edges[d][i].nodes[0], Nt)
        tdata.pvals_sort[Nt] = cdata.pmsR
        Nt += 1
      if not contains_key(cdata.dictC, tdata.edges[d][i].nodes[1]):
        if Nt >= cdata.max_edges: return -31124
        set_item(cdata.dictC, tdata.edges[d][i].nodes[1], Nt)
        tdata.pvals_sort[Nt] = cdata.pmsC
        Nt += 1
  
  cdata.poks = tdata.pvals_sort
  Nm = Nt-Ne
  # Get compatibility matrix
  cdata.n_edges = Ne
  cdata.n_miss = Nm
  cdata.n_tot = Nt
  ret = _get_cmpt2(cdata, exdata, tdata.eptr_sort)

  # Initialize for recursive function
  cdef CUINT8 *left = <CUINT8*>malloc(Nt*sizeof(CUINT8))
  if not left: return -31125
  
  cdata.ptot = 1.
  for i in range(Nt):
    left[i] = 1
    cdata.prbs[i] = 0.
    if i>=Ne: cdata.ptot *= tdata.pvals_sort[i]
  
  cdata.pmax0 = 0.
  cdata.pmax = 0.
  # Start recursive function
  cdef CINT32 rc=0
  print(f'Start rec combs: Ne={Ne}, Nm={Nm}, Nt={Nt}')
  t0 = time()
  ret = _rec_combs2(cdata, left, &idxs, 0, 1., &rc)
  free(left)
  print(f'End rec combs: {ret} combs, {rc} recursions, dt={time()-t0}')
  if ret<0: return ret

  # Update tree
  for i in range(Ne): tdata.eptr_sort[i][0].pfull2 = cdata.prbs[i]/cdata.ptot

  return 0
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32 _get_cmpt(CombsData* cdata, ExclData* exdata, CINT32* edges) except -31200:
  cdef CINT32 i,j,k, N=cdata.n_edges

  for i in range(N):
    cdata.cmpt[i*N+i] = 0
    for j in range(i+1,N):
      cdata.cmpt[i*N+j] = 1
      cdata.cmpt[j*N+i] = 1
      for k in range(exdata.indptr[edges[i]],exdata.indptr[edges[i]+1]):
        if exdata.indices[k]<edges[j]: continue
        if exdata.indices[k]>edges[j]: break

        cdata.cmpt[i*N+j] = 0
        cdata.cmpt[j*N+i] = 0
        break

  return 0
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32 _get_cmpt2(CombsData2* cdata, ExclData* exdata, TreeEdge** eptr) except -31200:
  cdef CINT32 i,j,k, Ne=cdata.n_edges, Nm=cdata.n_miss, Nt=cdata.n_tot
  cdef TreeEdge* ei
  cdef TreeEdge* ej
  cdef CINT32 val1,val2

  for i in range(Ne):
    cdata.cmpt[i*Nt+i] = 0
    ei = eptr[i]
    # Check compatibility of current edge with full-edges
    for j in range(i+1,Ne):
      cdata.cmpt[i*Nt+j] = 1
      cdata.cmpt[j*Nt+i] = 1

      ej = eptr[j]
      for k in range(exdata.indptr[ei[0].edge],exdata.indptr[ei[0].edge+1]):
        if exdata.indices[k]<ej[0].edge: continue
        if exdata.indices[k]>ej[0].edge: break
        cdata.cmpt[i*Nt+j] = 0
        cdata.cmpt[j*Nt+i] = 0
        break
    # Check compatibility of currente edge with miss-edges
    for j in range(Ne,Nt):
      get_item(cdata.dictR, ei[0].nodes[0], &val1)
      get_item(cdata.dictC, ei[0].nodes[1], &val2)
      if ((val1!=ERR_NOTIN) and (val1==j)) or ((val2!=ERR_NOTIN) and (val2==j)):
        cdata.cmpt[i*Nt+j] = 0
        cdata.cmpt[j*Nt+i] = 0
      else:
        cdata.cmpt[i*Nt+j] = 1
        cdata.cmpt[j*Nt+i] = 1

  for i in range(Ne,Nt):
    cdata.cmpt[i*Nt+i] = 0
    for j in range(i+1, Nt):
      cdata.cmpt[i*Nt+j] = 1
      cdata.cmpt[j*Nt+i] = 1

  return 0
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32 _rec_combs(CombsData* cdata, CFLOAT64* poks, CINT32* isort, CUINT8* left, CINT32 i0, CINT32 c0, CFLOAT64 cprb, CFLOAT64 thr, CFLOAT64 ratio, CFLOAT64* pmax, CINT32* idxs, CINT32* rc) except -31210:
  rc[0] += 1
  cdef CINT32 i,j, idx,cnt, ret,n=0, N=cdata.n_edges, M=cdata.max_edges+1
  cdef CFLOAT64 cprod=1.

  cdef CUINT8 *nlft = <CUINT8*>malloc(N*sizeof(CUINT8))
  if not nlft: return -31211
  cdef CINT32 *nidxs = <CINT32*>malloc((idxs[0]+2)*sizeof(CINT32))
  if not nidxs:
    free(nlft)
    return -31212

  # Copy current combination in a new idx array with one more element
  for i in range(idxs[0]+1): nidxs[i] = idxs[i]
  nidxs[0] = nidxs[0]+1
  if nidxs[0]>N:
    free(nlft)
    free(nidxs)
    return -31213

  for i in range(i0): nlft[i] = 0

  # Cycle through remaining edges
  for i in range(i0,N):
    cprod = cprb*poks[i]
    if (cprod<=thr) or (cprod<=pmax[0]*ratio): break

    if left[i]:
      nidxs[nidxs[0]] = isort[i]
      cnt = 0
      for j in range(i0,N):
        if j<=i:
          nlft[j] = 0
        else:
          nlft[j] = cdata.cmpt[i*N+j]*left[j]
          cnt += 1

      if cnt>0:
        ret = _rec_combs(cdata, poks, isort, nlft, i+1, c0+n, cprod, thr, ratio, pmax, nidxs, rc)
        if ret<0:
          free(nlft)
          free(nidxs)
          return ret
        n += ret

      # Add combination of current ok and all following wrong
      for j in range(i+1,N): cprod *= 1-poks[j]
      if (cprod>thr) and (cprod>pmax[0]*ratio):
        if c0+n>=cdata.max_combs:
          free(nlft)
          free(nidxs)
          return -31214

        for j in range(nidxs[0]+1): cdata.combs[(c0+n)*M+j] = nidxs[j]
        cdata.prbs[c0+n] = cprod
        cdata.prbs2[c0+n] = cprod
        cdata.pset[c0+n] = 1.
        if cprod>pmax[0]: pmax[0] = cprod
        n += 1

    cprb *= 1-poks[i]
    
  free(nlft)
  free(nidxs)
  return n
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32 _rec_combs2(CombsData2* cdata, CUINT8* left, CINT32* idxs, CINT32 i0, CFLOAT64 cprb, CINT32* rc) except -31210:
  rc[0] += 1
  cdef CINT32 i,j, cnt, n=0, ret, Ne=cdata.n_edges, Nt=cdata.n_tot
  cdef CFLOAT64 cprod0=1., cprod=1.

  cdef CUINT8 *nlft = <CUINT8*>malloc(Nt*sizeof(CUINT8))
  if not nlft: return -31211
  cdef CINT32 *nidxs = <CINT32*>malloc((idxs[0]+2)*sizeof(CINT32))
  if not nidxs:
    free(nlft)
    return -31212

  # Copy current combination in a new idx array with one more element
  for i in range(idxs[0]+1): nidxs[i] = idxs[i]
  nidxs[0] = nidxs[0]+1
  if nidxs[0]>Nt:
    free(nlft)
    free(nidxs)
    return -31213

  for i in range(i0): nlft[i] = 0
  # Cycle through remaining edges
  for i in range(i0,Ne):
    cprod0 = cprb*cdata.poks[i]
    if cprod0<=cdata.pmax0: break

    nlft[i] = 0
    if left[i]:
      nidxs[nidxs[0]] = i
      cnt = 0
      # Check compatibility with all following edges
      for j in range(i+1,Nt):
        nlft[j] = cdata.cmpt[i*Nt+j]*left[j]
        if j<Ne: cnt += nlft[j]

      # If there is at least a compatible (full) edge, continue recursion
      if cnt>0:
        ret = _rec_combs2(cdata, nlft, nidxs, i+1, cprod0, rc)
        if ret<0:
          free(nlft)
          free(nidxs)
          return ret
        n += ret

      # Add current combination
      for j in range(i+1,Ne): cprod0 *= 1-cdata.poks[j]
      cprod = cprod0
      for j in range(Ne,Nt):
        if nlft[j]: cprod *= cdata.poks[j]
        else:       cprod *= 1-cdata.poks[j]
      
      if (cprod0>cdata.pmax0) and (cprod>cdata.pmax):
        # print('ADD COMBINATION')
        for j in range(1,nidxs[0]+1): cdata.prbs[nidxs[j]] += cprod
        for j in range(Ne,Nt):
          if nlft[j]: cdata.prbs[j] += cprod
        cdata.ptot += cprod
        if cprod0*cdata.ratio>cdata.pmax0: cdata.pmax0 = cprod0*cdata.ratio
        if cprod*cdata.ratio >cdata.pmax:  cdata.pmax  = cprod*cdata.ratio

        n += 1

    cprb *= 1-cdata.poks[i]

  free(nlft)
  free(nidxs)
  return n
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32 _calc_pset_core(TreeData* tdata, CombsData* cdata, PsetData* psdata, OptsData* opts, CINT32 core_depth) except -31220:
  # pset is calculated by trying to maximize the number of core nodes that are matched in each combination
  # Core nodes are those nodes that are connected to at least one edge with depth <= 1
  cdef CINT32 i,j,val, nR=0, nC=0, ncR=0, ncC=0, ncmR=0, ncmC=0, idxR, idxC
  cdef CINT32 M=cdata.max_edges+1
  cdef CINT32 P=psdata.max_nodes+1
  cdef TreeEdge *edge

  delete_dict(psdata.dictNR)
  delete_dict(psdata.dictNC)

  if cdata.n_edges>psdata.max_nodes: return -31221

  # Assign nodes to core or periphery
  for i in range(cdata.n_edges):
    psdata.cores[0][i] = 0
    psdata.cores[1][i] = 0

  for i in range(tdata.n_edges):
    edge = tdata.eptr[i]
    get_item(psdata.dictNR, edge[0].nodes[0], &val)
    if val==ERR_NOTIN:
      set_item(psdata.dictNR, edge[0].nodes[0], nR)
      val = nR
      nR += 1 # <------------

    if edge[0].depth <= core_depth: psdata.cores[0][val] = 1
    psdata.edge2idx[0][i] = val

    get_item(psdata.dictNC, edge[0].nodes[1], &val)
    if val==ERR_NOTIN:
      set_item(psdata.dictNC, edge[0].nodes[1], nC)
      val = nC
      nC += 1
      
    if edge[0].depth <= core_depth: psdata.cores[1][val] = 1
    psdata.edge2idx[1][i] = val

  # Count nodes in the core
  for i in range(nR):
    if psdata.cores[0][i]: ncR += 1
  for i in range(nC):
    if psdata.cores[1][i]: ncC += 1

  if (ncR > psdata.max_nodes) or (ncC > psdata.max_nodes): return -31222

  # Calc pset for each combination
  for i in range(cdata.n_combs):
    ncmR = 0
    ncmC = 0
    for j in range(nR): psdata.dones[0][j] = 0
    for j in range(nC): psdata.dones[1][j] = 0

    for j in range(1,cdata.combs[i*M]+1):
      idxR = psdata.edge2idx[0][cdata.combs[i*M+j]]
      if not psdata.dones[0][idxR]:
        psdata.dones[0][idxR] = 1
        if psdata.cores[0][idxR]: ncmR += 1

      idxC = psdata.edge2idx[1][cdata.combs[i*M+j]]
      if not psdata.dones[1][idxC]:
        psdata.dones[1][idxC] = 1
        if psdata.cores[1][idxC]: ncmC += 1

    cdata.pset[i] = psdata.psets[0][ncR*P+ncmR]*psdata.psets[1][ncC*P+ncmC]

  return 0
# -------

# --------------------------- INDIVIDUAL OPERATIONS ----------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

@treefunc
def tree_walk_fwd(DataWrapper dw, CINT32 tidx):
  return _walk_fwd(dw.tdata[tidx], dw.imdata, dw.opts)
# -------

@treefunc
def tree_walk_fwd_sort(DataWrapper dw, CINT32 tidx, CINT32 nmax):
  return _walk_fwd_sort(dw.tdata[tidx], dw.imdata, dw.opts, nmax)
# -------

@treefunc
def tree_walk_node(DataWrapper dw, CINT32 tidx, CINT32 node, CUINT8 side):
  return _walk_node(dw.tdata[tidx], dw.imdata, dw.opts, node, side)
# -------

@treefunc
def tree_walk_node_sort(DataWrapper dw, CINT32 tidx, CINT32 node, CUINT8 side, CINT32 nmax):
  return _walk_node_sort(dw.tdata[tidx], dw.imdata, dw.opts, node, side, nmax)
# -------

@treefunc
def tree_calc_pfull(DataWrapper dw, CINT32 tidx, CINT32 core_depth):
  return _tree_calc_pfull(dw.tdata[tidx], dw.cdata, dw.exdata, dw.psdata, dw.opts, core_depth)
# -------

@treefunc
def tree_calc_pfull2(DataWrapper dw, CINT32 tidx, CINT32 core_depth):
  return _tree_calc_pfull2(dw.tdata[tidx], dw.cdata2, dw.exdata, dw.opts, core_depth)
# -------

# ------------------------------ FULL ALGORITHMS -------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

@basefunc
def prune_improbable_all(DataWrapper dw, CFLOAT64 pratio=100., CINT32 nmax=3):
  cdef CINT32 i,rem=0,ret
  cdef CINT32 tidx = dw.allocate_tree()

  _p0s = np.zeros(dw.imdata.N, dtype=np.float64)
  cdef CFLOAT64[:] p0s = _p0s
  for i in range(dw.imdata.N): p0s[i] = dw.imdata.prbs[i]

  _isort = np.argsort(_p0s).astype(np.int32)
  cdef CINT32[:] isort = _isort

  t0 = time()
  for i in range(dw.imdata.N):
    if i%100000==0: print(f'{i}/{dw.imdata.N} [t={time()-t0:.3f}s]')

    ret = _prune_improbable(dw.tdata[tidx], dw.imdata, dw.cdata, dw.exdata, dw.psdata, dw.opts, isort[i], pratio, nmax)
    if ret<0: return (ret,)
    rem += ret

  print(f'--> removed {rem} edges [t={time()-t0:.3f}s]')
  return (0,)
# -------

@basefunc
def prune_probable_all(DataWrapper dw, CUINT8[:] done, CINT32 max_cycles, CFLOAT64 ratio, CINT32 nmax):
  assert len(done)==dw.imdata.N, 'done array must have length equal to number of edges'

  cdef CINT32 i,c,m,ret,rem,nno
  cdef CINT32 tidx = dw.allocate_tree()

  _p0s = np.zeros(dw.imdata.N, dtype=np.float64)
  cdef CFLOAT64[:] p0s = _p0s
  for i in range(dw.imdata.N): p0s[i] = dw.imdata.prbs[i]

  _isort = np.argsort(_p0s)[::-1].astype(np.int32)
  cdef CINT32[:] isort = _isort

  t0 = time()
  for c in range(max_cycles):
    print(f'-------------------- Cycle {c} --------------------')
    rem = 0
    nno = 0
    for i in range(dw.imdata.N):
      if i%50000==0: print(f'{i}/{dw.imdata.N} [t={time()-t0:.3f}s]')
      if done[i]: continue

      ret = _prune_probable(dw.tdata[tidx], dw.imdata, dw.cdata, dw.exdata, dw.psdata, dw.opts, isort[i], ratio, nmax)
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

@basefunc
def walk_all(DataWrapper dw, CINT32 node0, CUINT8 side0, CINT32 nmax):

  _edges = np.full(dw.imdata.N, -1, dtype=np.int32)
  cdef CINT32[:] edges = _edges
  _eadd = np.zeros(dw.imdata.N, dtype=np.uint8)
  cdef CUINT8[:] eadd = _eadd

  _pfs = np.full(dw.imdata.N, -1., dtype=np.float64)
  cdef CFLOAT64[:] pfs = _pfs

  cdef CINT32 i,j,cnt,ret,ee
  cdef CINT32 tidx = dw.allocate_tree()
  cdef CFLOAT64 pf

  cnt = 0
  for i in range(dw.imdata.indptr[side0][node0],dw.imdata.indptr[side0][node0+1]):
    edges[cnt] = dw.imdata.data[side0][i]
    eadd[edges[cnt]] = 1
    # print(f'... add {edges[cnt]} at pos {cnt}')
    cnt += 1

  # print(f'N: {dw.imdata.N}')
  for i in range(dw.imdata.N):
    # print(f'---- edge {edges[i]} ----')
    if edges[i]<0: break

    ret = _solve_edge(dw.tdata[tidx], dw.imdata, dw.cdata, dw.exdata, dw.psdata, dw.opts, edges[i], nmax, &pfs[i])
    if ret<0: continue

    # print(f'walked tree with {dw.tdata[tidx].counts[1]} edges at depth 1')
    for j in range(dw.tdata[tidx].counts[1]):
      ee = dw.tdata[tidx].edges[1][j].edge
      if eadd[ee]: continue
      # print(f'... add {ee} at pos {cnt}')
      eadd[ee] = 1
      edges[cnt] = ee
      cnt += 1

    print(i,cnt,f'{100*i/cnt:.1f}%')
  
  return (0,_edges[:cnt],_pfs[:cnt])
# -------

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef inline CINT32 _prune_improbable(TreeData* tdata, ImatData* imdata, CombsData* cdata, ExclData* exdata, PsetData* psdata, OptsData *opts,
                                    CINT32 edge, CFLOAT64 pratio, CINT32 nmax) except -32030:
  if imdata.prbs[edge]==0.: return 0

  cdef CINT32 i,ret
  cdef CFLOAT64 pthr=0.

  ret = _init_from_edge(tdata, imdata, edge)
  if ret<0: return ret

  ret = _walk_fwd_sort(tdata, imdata, opts, nmax)
  if ret<0:  return ret
  if ret==0: return 0

  if tdata.n_edges > cdata.max_edges: return -32031

  ret = _walk_fwd_sort(tdata, imdata, opts, nmax)
  if ret<0: return ret

  if tdata.n_edges > cdata.max_edges:  return -32032

  ret = _tree_calc_pfull(tdata, cdata, exdata, psdata, opts, 1)
  if ret<0: return ret

  for i in range(tdata.counts[1]):
    if (tdata.edges[1][i].pfull2>pthr): pthr = tdata.edges[1][i].pfull2
  pthr /= pratio

  if pthr==0. and tdata.edges[0][1].pfull2==0.: return -32033

  if tdata.edges[0][1].pfull2<pthr:
    imdata.prbs[edge] = 0.
    return 1

  return 0
# -------

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef inline CINT32 _prune_probable(TreeData* tdata, ImatData* imdata, CombsData* cdata, ExclData* exdata, PsetData* psdata, OptsData *opts,
                                            CINT32 edge, CFLOAT64 ratio, CINT32 nmax) except -32040:
  if imdata.prbs[edge]==0.: return 0

  cdef CINT32 i,j,ret,nd
  cdef CUINT8 ax,comp
  cdef CFLOAT64 pm,px,po

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

  ret = _tree_calc_pfull(tdata, cdata, exdata, psdata, opts, 2)
  if ret<0: return ret

  pm = min(tdata.edges[0][1].pfull2, .99999)
  if pm==0.:
    imdata.prbs[edge] = 0.
    return 1

  ret = 0
  for i in range(tdata.counts[1]):
    px = min(tdata.edges[1][i].pfull2, .99999)
    # If edge is certainly wrong (p=0), remove it
    if px==0.:
      imdata.prbs[tdata.edges[1][i].edge] = 0.
      ret += 1
      continue

    # Check if edge is compatible, and in case  skip it
    comp = 1
    for j in range(exdata.indptr[edge],exdata.indptr[edge+1]):
      if exdata.indices[j]<tdata.edges[1][i].edge: continue
      if exdata.indices[j]>tdata.edges[1][i].edge: break
      comp = 0
    if comp: continue

    # Find best match for second node
    po = px
    ax = tdata.edges[1][i].side
    nd = tdata.edges[1][i].nodes[ax]
    for j in range(tdata.counts[2]):
      if tdata.edges[2][j].nodes[ax]==nd: po = max(po, tdata.edges[2][j].pfull2)
    po = min(po, .99999)

    # Calculate probability of combination
    if (pm*po*(1-px))/((1-pm)*(1-po)*px) > ratio:
      imdata.prbs[tdata.edges[1][i].edge] = 0.
      ret += 1
  
  return ret
# -------

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef inline CINT32 _solve_edge(TreeData* tdata, ImatData* imdata, CombsData* cdata, ExclData* exdata, PsetData* psdata, OptsData *opts,
                               CINT32 edge, CINT32 nmax, CFLOAT64* pf) except -32050:
  if imdata.prbs[edge]==0.:
    pf[0] = 0.
    return 0

  cdef CINT32 ret

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

  ret = _tree_calc_pfull(tdata, cdata, exdata, psdata, opts, 2)
  if ret<0: return ret

  pf[0] = tdata.edges[0][1].pfull2
  return 0
# -------
