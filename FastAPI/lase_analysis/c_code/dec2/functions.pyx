# cython: profile=True

cimport cython

import numpy as np
cimport numpy as np
from scipy.sparse import coo_array

from libc.stdlib cimport malloc, free
from libc.math cimport pow
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

cdef struct PossData:
  CINT32** data
  CINT32** indices
  CINT32** indptr

  CINT32** nodes

  CFLOAT64* prbs
  CFLOAT64* pfulls
  CFLOAT64[2] pms

  CINT32* excl_indices
  CINT32* excl_indptr

  CINT32* isort
  CINT32* sort_edges
  CFLOAT64* sort_pvals

  CINT32[2] shape
  CINT32 N
# -------

cdef struct TreeEdge:
  CINT32 edge
  TreeNode** nodes # [0]: row, [1]: col
  CINT32 depth

  CFLOAT64 psing
  CFLOAT64 pfull

  CINT32 idx
# -------

cdef struct TreeNode:
  CINT32 node
  CUINT8 axis
  CINT32 depth

  CFLOAT64 pmiss0
  CFLOAT64 pmissF
  CUINT8 done

  CINT32 idx
# -------

cdef struct TreeData:
  TreeEdge* edges
  CINT32 n_edges
  CINT32 edict

  TreeNode* nodes
  CINT32 n_nodes
  CINT32 ndict[2]

  CINT32 depth # current depth, i.e. index of last non-zero element in counts array

  CINT32 max_edges # max number of edges per depth, i.e. length (on 2nd dimension) of edges array --> valid index is [0,max_edges)
  CINT32 max_nodes # max number of nodes per depth, i.e. length (on 2nd dimension) of nodes array --> valid index is [0,max_nodes)

  PossData* poss
# -------

cdef struct CombsData:
  CUINT8* cmpt
  CINT32* isort
  CFLOAT64* poks
  CFLOAT64* prbs
  CUINT8* left
  CINT32* idxs

  CFLOAT64 ptot
  CFLOAT64 pmax0
  CFLOAT64 pmax
  CFLOAT64 K # geom mean of poks to keep products from underflowing

  CINT32 max_edges
  CINT32 max_nodes
  CINT32 n_edges
  CINT32 n_tot

  # opts
  CFLOAT64 ratio

  CUINT64 recs
  CFLOAT64* combs
  CUINT64 nc
  CUINT64 max_combs
# -------

cdef struct SolveData:
  CFLOAT64* pfs
  CINT32* nfs

  CFLOAT64* pms
  CINT32* nms

  CINT32 n_edges
  CINT32 n_nodes
  CINT32 max_poss
# -------

cdef struct MatchData:
  CFLOAT64* pfulls
  CINT32* isort

  CFLOAT64* pmtcs
  
  CINT32 ne
  CINT32 nn0
  CINT32 nn1
  CINT32 N
# -------

# ------------------------------ DECORATORS ------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef class DataWrapper:
  cdef TreeData **tdata
  cdef CombsData *cdata
  cdef PossData *pdata
  cdef SolveData *sdata
  cdef MatchData *mdata
  
  cdef CINT32 MAX_TDATA

  def __cinit__(self):
    self.MAX_TDATA = 10

    self.tdata = <TreeData**>malloc(self.MAX_TDATA*sizeof(TreeData*))
    if not self.tdata: raise MemoryError("Failed to allocate memory for TreeData")
    for i in range(self.MAX_TDATA): self.tdata[i] = NULL

    self.cdata = NULL
    self.pdata = NULL
    self.sdata = NULL
    self.mdata = NULL
  # -------

  def init_pdata(self, poss):
    # --- PossData structure ----------
    self.pdata = <PossData*>malloc(sizeof(PossData))
    self.pdata.data = NULL
    self.pdata.indices = NULL
    self.pdata.indptr = NULL
    self.pdata.nodes = NULL
    self.pdata.isort = NULL
    self.pdata.sort_edges = NULL
    self.pdata.sort_pvals = NULL
    self.pdata.pfulls = NULL

    self.pdata.pms[0] = poss.pmissR
    self.pdata.pms[1] = poss.pmissC
    self.pdata.shape[0] = <CINT32>poss.imatr.shape[0]
    self.pdata.shape[1] = <CINT32>poss.imatr.shape[1]
    self.pdata.N = <CINT32>len(poss.imatr.data)

    poss._temp_p0 = poss.vals['p0'].astype(np.float64)
    cdef CFLOAT64[:] imat_prbs = poss._temp_p0
    self.pdata.prbs = &imat_prbs[0]

    # Init data
    cdef CINT32[:] imat_rdata = poss.imatr.data
    cdef CINT32[:] imat_cdata = poss.imatc.data
    self.pdata.data = <CINT32**>malloc(2*sizeof(CINT32*))
    if not self.pdata.data: raise MemoryError("Failed to allocate memory for PossData.data")
    self.pdata.data[0] = &imat_rdata[0]
    self.pdata.data[1] = &imat_cdata[0]

    # Init indices
    cdef CINT32[:] imat_rindices = poss.imatr.indices
    cdef CINT32[:] imat_cindices = poss.imatc.indices
    self.pdata.indices = <CINT32**>malloc(2*sizeof(CINT32*))
    if not self.pdata.indices: raise MemoryError("Failed to allocate memory for PossData.indices")
    self.pdata.indices[0] = &imat_rindices[0]
    self.pdata.indices[1] = &imat_cindices[0]

    # Init indptr
    cdef CINT32[:] imat_rindptr = poss.imatr.indptr
    cdef CINT32[:] imat_cindptr = poss.imatc.indptr
    self.pdata.indptr = <CINT32**>malloc(2*sizeof(CINT32*))
    if not self.pdata.indptr: raise MemoryError("Failed to allocate memory for PossData.indptr")
    self.pdata.indptr[0] = &imat_rindptr[0]
    self.pdata.indptr[1] = &imat_cindptr[0]

    # Init nodes
    cdef CINT32[:] nodes_rows = poss.rows
    cdef CINT32[:] nodes_cols = poss.cols
    self.pdata.nodes = <CINT32**>malloc(2*sizeof(CINT32*))
    if not self.pdata.nodes: raise MemoryError("Failed to allocate memory for PossData.nodes")
    self.pdata.nodes[0] = &nodes_rows[0]
    self.pdata.nodes[1] = &nodes_cols[0]

    # Init sorting
    # ... isort
    self.pdata.isort = <CINT32*>malloc(self.pdata.N*sizeof(CINT32))
    if not self.pdata.isort: raise MemoryError("Failed to allocate memory for PossData.isort")
    # ... sort_edges
    self.pdata.sort_edges = <CINT32*>malloc(self.pdata.N*sizeof(CINT32))
    if not self.pdata.sort_edges: raise MemoryError("Failed to allocate memory for PossData.sort_edges")
    # ... sort_pvals
    self.pdata.sort_pvals = <CFLOAT64*>malloc(self.pdata.N*sizeof(CFLOAT64))
    if not self.pdata.sort_pvals: raise MemoryError("Failed to allocate memory for PossData.sort_pvals")

    # Init excl
    cdef CINT32[:] excl_indptr = poss.excl.indptr
    cdef CINT32[:] excl_indices = poss.excl.indices
    self.pdata.excl_indptr = &excl_indptr[0]
    self.pdata.excl_indices = &excl_indices[0]
  # -------
  def dealloc_pdata(self):
    if self.pdata:
      if self.pdata.data: free(self.pdata.data)
      if self.pdata.indices: free(self.pdata.indices)
      if self.pdata.indptr: free(self.pdata.indptr)
      if self.pdata.nodes: free(self.pdata.nodes)
      if self.pdata.isort: free(self.pdata.isort)
      if self.pdata.sort_edges: free(self.pdata.sort_edges)
      if self.pdata.sort_pvals: free(self.pdata.sort_pvals)
      free(self.pdata)
    self.pdata = NULL
  # -------

  def init_tdata(self, CINT32 tidx, CINT32 max_edges, CINT32 max_nodes, tree=None):
    if tidx >= self.MAX_TDATA: raise RuntimeError(f'Max number of TreeData is {self.MAX_TDATA}')
    if self.tdata[tidx]:       raise RuntimeError(f'TreeData {tidx} already initialized')

    # --- TreeData structure ----------
    self.tdata[tidx] = <TreeData*>malloc(sizeof(TreeData))
    if not self.tdata[tidx]: raise MemoryError("Failed to allocate memory for TreeData")
    
    if not self.pdata: raise RuntimeError("PossData has to be initialized before TreeData")
    self.tdata[tidx].poss = self.pdata

    self.tdata[tidx].n_edges = 0
    self.tdata[tidx].n_nodes = 0

    self.tdata[tidx].edict = create_dict()
    if self.tdata[tidx].edict==-1: raise MemoryError("Failed to allocate memory for dict 'edict'")
    self.tdata[tidx].ndict[0] = create_dict()
    if self.tdata[tidx].ndict[0]==-1: raise MemoryError("Failed to allocate memory for dict 'ndict[0]'")
    self.tdata[tidx].ndict[1] = create_dict()
    if self.tdata[tidx].ndict[1]==-1: raise MemoryError("Failed to allocate memory for dict 'ndict[1]'")

    self.tdata[tidx].max_edges = max_edges
    self.tdata[tidx].max_nodes = max_nodes

    self.tdata[tidx].edges = NULL
    self.tdata[tidx].nodes = NULL
    
    # Init arrays
    cdef CINT32 i
    # ... edges
    self.tdata[tidx].edges = <TreeEdge*>malloc(self.tdata[tidx].max_edges*sizeof(TreeEdge))
    if not self.tdata[tidx].edges: raise MemoryError("Failed to allocate memory for TreeData.edges")
    for i in range(self.tdata[tidx].max_edges): self.tdata[tidx].edges[i].nodes = NULL
    for i in range(self.tdata[tidx].max_edges):
      self.tdata[tidx].edges[i].nodes = <TreeNode**>malloc(2*sizeof(TreeNode*))
      if not self.tdata[tidx].edges[i].nodes: raise MemoryError("Failed to allocate memory for TreeData.edges.nodes")
    # ... nodes
    self.tdata[tidx].nodes = <TreeNode*>malloc(self.tdata[tidx].max_nodes*sizeof(TreeNode))
    if not self.tdata[tidx].nodes: raise MemoryError("Failed to allocate memory for TreeData.nodes")

    if not tree is None: _init_from_tree(self.tdata[tidx], tree)
  # -------
  def dealloc_tdata(self, CINT32 tidx):
    if self.tdata[tidx]:
      if self.tdata[tidx].edges:
        for i in range(self.tdata[tidx].max_edges):
          if self.tdata[tidx].edges[i].nodes: free(self.tdata[tidx].edges[i].nodes)
        free(self.tdata[tidx].edges)
      if self.tdata[tidx].nodes: free(self.tdata[tidx].nodes)
      free(self.tdata[tidx])
    self.tdata[tidx] = NULL
  # -------

  def init_cdata(self, CINT32 max_edges, CINT32 max_nodes):
    # --- Combinations arrays ----------
    self.cdata = <CombsData*>malloc(sizeof(CombsData))
    if not self.cdata: raise MemoryError("Failed to allocate memory for CombsData")
    self.cdata.cmpt = NULL
    self.cdata.isort = NULL
    self.cdata.poks = NULL
    self.cdata.prbs = NULL
    self.cdata.left = NULL
    self.cdata.idxs = NULL
    self.cdata.combs = NULL

    self.cdata.ptot = 0.
    self.cdata.pmax0 = 0.
    self.cdata.pmax = 0.

    self.cdata.max_edges = max_edges
    self.cdata.max_nodes = max_nodes
    self.cdata.n_edges = 0
    self.cdata.n_tot = 0

    # Init arrays
    cdef CINT32 n = self.cdata.max_edges+self.cdata.max_nodes

    self.cdata.cmpt = <CUINT8*>malloc(n*self.cdata.max_edges*sizeof(CUINT8))
    if not self.cdata.cmpt: raise MemoryError("Failed to allocate memory for CombsData.cmpt")

    self.cdata.isort = <CINT32*>malloc(n*sizeof(CINT32))
    if not self.cdata.isort: raise MemoryError("Failed to allocate memory for CombsData.isort")

    self.cdata.poks = <CFLOAT64*>malloc(n*sizeof(CFLOAT64))
    if not self.cdata.poks: raise MemoryError("Failed to allocate memory for CombsData.poks")

    self.cdata.prbs = <CFLOAT64*>malloc(n*sizeof(CFLOAT64))
    if not self.cdata.prbs: raise MemoryError("Failed to allocate memory for CombsData.prbs")

    self.cdata.left = <CUINT8*>malloc(n*self.cdata.max_edges*sizeof(CUINT8))
    if not self.cdata.left: raise MemoryError("Failed to allocate memory for CombsData.left")

    self.cdata.idxs = <CINT32*>malloc(self.cdata.max_edges*sizeof(CINT32))
    if not self.cdata.idxs: raise MemoryError("Failed to allocate memory for CombsData.idxs")

  # -------
  def dealloc_cdata(self):
    if self.cdata:
      if self.cdata.cmpt: free(self.cdata.cmpt)
      if self.cdata.isort: free(self.cdata.isort)
      if self.cdata.poks: free(self.cdata.poks)
      if self.cdata.prbs: free(self.cdata.prbs)
      if self.cdata.left: free(self.cdata.left)
      if self.cdata.idxs: free(self.cdata.idxs)
      free(self.cdata)
    self.cdata = NULL
  # -------

  def init_sdata(self, CINT32 n_edges, CINT32 n_nodes, CINT32 max_poss):
    # --- SolveData structure ----------
    self.sdata = <SolveData*>malloc(sizeof(SolveData))
    if not self.sdata: raise MemoryError("Failed to allocate memory for SolveData")

    self.sdata.n_edges = n_edges
    self.sdata.n_nodes = n_nodes
    self.sdata.max_poss = max_poss

    self.sdata.pfs = NULL
    self.sdata.nfs = NULL
    self.sdata.pms = NULL
    self.sdata.nms = NULL

    # Init arrays
    self.sdata.pfs = <CFLOAT64*>malloc(self.sdata.n_edges*self.sdata.max_poss*sizeof(CFLOAT64))
    if not self.sdata.pfs: raise MemoryError("Failed to allocate memory for SolveData.pfs")

    self.sdata.nfs = <CINT32*>malloc(self.sdata.n_edges*sizeof(CINT32))
    if not self.sdata.nfs: raise MemoryError("Failed to allocate memory for SolveData.nfs")

    self.sdata.pms = <CFLOAT64*>malloc(self.sdata.n_nodes*self.sdata.max_poss*sizeof(CFLOAT64))
    if not self.sdata.pms: raise MemoryError("Failed to allocate memory for SolveData.pms")

    self.sdata.nms = <CINT32*>malloc(self.sdata.n_nodes*sizeof(CINT32))
    if not self.sdata.nms: raise MemoryError("Failed to allocate memory for SolveData.nms")

  # -------
  def dealloc_sdata(self):
    if self.sdata:
      if self.sdata.pfs: free(self.sdata.pfs)
      if self.sdata.nfs: free(self.sdata.nfs)
      if self.sdata.pms: free(self.sdata.pms)
      if self.sdata.nms: free(self.sdata.nms)
      free(self.sdata)
    self.sdata = NULL
  # -------

  def init_mdata(self, CINT32 N):
    # --- MatchData structure ----------
    self.mdata = <MatchData*>malloc(sizeof(MatchData))
    if not self.mdata: raise MemoryError("Failed to allocate memory for MatchData")

    self.mdata.pfulls = NULL
    self.mdata.isort = NULL
    self.mdata.pmtcs = NULL

    self.mdata.ne = 0
    self.mdata.nn0 = 0
    self.mdata.nn1 = 0
    self.mdata.N = N

    # Init arrays
    self.mdata.pfulls = <CFLOAT64*>malloc(self.mdata.N*sizeof(CFLOAT64))
    if not self.mdata.pfulls: raise MemoryError("Failed to allocate memory for MatchData.pfulls")

    self.mdata.isort = <CINT32*>malloc(self.mdata.N*sizeof(CINT32))
    if not self.mdata.isort: raise MemoryError("Failed to allocate memory for MatchData.isort")

    self.mdata.pmtcs = <CFLOAT64*>malloc(self.mdata.N*sizeof(CFLOAT64))
    if not self.mdata.pmtcs: raise MemoryError("Failed to allocate memory for MatchData.pmtcs")
  # -------
  def dealloc_mdata(self):
    if self.mdata:
      if self.mdata.pfulls: free(self.mdata.pfulls)
      if self.mdata.isort: free(self.mdata.isort)
      if self.mdata.pmtcs: free(self.mdata.pmtcs)
      free(self.mdata)
    self.mdata = NULL
  # -------

  def __dealloc__(self):
    cdef CINT32 i

    delete_all()

    for i in range(self.MAX_TDATA): self.dealloc_tdata(i)
    free(self.tdata)
    self.tdata = NULL

    self.dealloc_pdata()
    self.dealloc_cdata()
    self.dealloc_sdata()
    self.dealloc_mdata()
  # -------
# -------

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
def dwrap(func):
  def wrapper(**kwds):
    dw = DataWrapper()
    try:     (ret, fret) = func(dw,**kwds)
    finally: del dw

    if ret<0: raise RuntimeError(f'Error {ret} in {func.__name__}', ret)
    return fret

  return wrapper
# -------

# ----------------------------- SUPPORT FUNCTIONS ------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

cdef CINT32 _init_from_node(TreeData* tdata, CINT32 node, CUINT8 axis) except -30010:
  tdata.depth = 0

  # --- Initialize nodes ----------------------------------
  delete_dict(tdata.ndict[0])
  delete_dict(tdata.ndict[1])

  tdata.nodes[0].node = node
  tdata.nodes[0].axis = axis
  tdata.nodes[0].depth = 0
  tdata.nodes[0].pmiss0 = tdata.poss.pms[axis]
  tdata.nodes[0].pmissF = .0
  tdata.nodes[0].done = 0
  tdata.nodes[0].idx = -1

  set_item(tdata.ndict[axis], node, 0)

  tdata.n_nodes = 1

  # --- Initialize edges ----------------------------------
  delete_dict(tdata.edict)
  tdata.n_edges = 0

  return 0
# -------

cdef CINT32 _init_from_edge(TreeData* tdata, CINT32 edge) except -30020:
  cdef CUINT8 ax
  tdata.depth = 0

  # --- Initialize nodes ----------------------------------
  delete_dict(tdata.ndict[0])
  delete_dict(tdata.ndict[1])

  for ax in range(2):
    tdata.nodes[ax].node = tdata.poss.nodes[ax][edge]
    tdata.nodes[ax].axis = ax
    tdata.nodes[ax].depth = 0
    tdata.nodes[ax].pmiss0 = tdata.poss.pms[ax]
    tdata.nodes[ax].pmissF = .0
    tdata.nodes[ax].done = 0
    tdata.nodes[ax].idx = -1

    set_item(tdata.ndict[ax], tdata.nodes[ax].node, ax)
  tdata.n_nodes = 2

  # --- Initialize edges ----------------------------------
  delete_dict(tdata.edict)

  tdata.edges[0].edge = edge
  tdata.edges[0].nodes[0] = &tdata.nodes[0]
  tdata.edges[0].nodes[1] = &tdata.nodes[1]
  tdata.edges[0].depth = 0
  tdata.edges[0].psing = tdata.poss.prbs[edge]
  tdata.edges[0].pfull = .0
  tdata.edges[0].idx = -1

  set_item(tdata.edict, edge, 0)
  tdata.n_edges = 1

  return 0
# -------

cdef CINT32 _init_from_tree(TreeData* tdata, object tree) except -30030:
  cdef CINT32 i,val
  cdef CINT32 ne=<CINT32>len(tree.edges)
  cdef CINT32 nn=<CINT32>len(tree.nodes)
  cdef CINT32 dmax=<CINT32>tree.depth
  cdef CUINT8 ax

  if ne>tdata.max_edges: raise RuntimeError(f'Tree has {ne} edges, max is {tdata.max_edges}')
  if nn>tdata.max_nodes: raise RuntimeError(f'Tree has {nn} nodes, max is {tdata.max_nodes}')

  # --- Init nodes ----------------------------------
  delete_dict(tdata.ndict[0])
  delete_dict(tdata.ndict[1])

  for i in range(nn):
    tdata.nodes[i].node = <CINT32>tree.nodes[i].node
    tdata.nodes[i].axis = <CUINT8>tree.nodes[i].axis
    tdata.nodes[i].depth = <CINT32>tree.nodes[i].depth
    tdata.nodes[i].pmiss0 = <CFLOAT64>tree.nodes[i].pmiss0
    tdata.nodes[i].pmissF = <CFLOAT64>tree.nodes[i].pmissF
    tdata.nodes[i].done = tdata.nodes[i].depth<dmax
    tdata.nodes[i].idx = -1
    set_item(tdata.ndict[tdata.nodes[i].axis], tdata.nodes[i].node, i)
  
  # --- Init edges ----------------------------------
  delete_dict(tdata.edict)
  for i in range(ne):
    tdata.edges[i].edge = <CINT32>tree.edges[i].edge
    for ax in range(2):
      get_item(tdata.ndict[ax], tree.edges[i].nodes[ax].node, &val)
      if val==ERR_NOTIN: raise RuntimeError(f'Node {tree.edges[i].nodes[ax].node} not in tree')
      tdata.edges[i].nodes[ax] = &tdata.nodes[val]
    tdata.edges[i].depth = <CINT32>tree.edges[i].depth
    tdata.edges[i].psing = <CFLOAT64>tree.edges[i].psing
    tdata.edges[i].pfull = <CFLOAT64>tree.edges[i].pfull
    tdata.edges[i].idx = -1
    set_item(tdata.edict, tdata.edges[i].edge, i)
  
  tdata.n_edges = ne
  tdata.n_nodes = nn
  tdata.depth = dmax

  return 0
# -------

cdef CINT32 _update_tree(TreeData* tdata, object tree) except -30040:
  cdef CINT32 i
  cdef TreeEdge* ee
  cdef TreeNode* nn

  edata = []
  for i in range(tdata.n_edges):
    ee = &tdata.edges[i]
    edata.append({'edge': ee[0].edge, 'node0': ee[0].nodes[0].node, 'node1': ee[0].nodes[1].node, 'depth': ee[0].depth,
                  'psing': ee[0].psing, 'pfull': ee[0].pfull})
  
  ndata = []
  for i in range(tdata.n_nodes):
    nn = &tdata.nodes[i]
    ndata.append({'node': nn[0].node, 'axis': nn[0].axis, 'depth': nn[0].depth, 'pmiss0': nn[0].pmiss0, 'pmissF': nn[0].pmissF})
  
  tree.update(edata, ndata)

  return 0
# -------

cdef inline CINT32 _calc_stats(CFLOAT64* vals, CINT32 n, CFLOAT64* mean, CFLOAT64* std) except -30050:
  cdef CINT32 i
  mean[0] = 0.
  std[0] = 0.
  
  if n==0: return 0

  for i in range(n):
    mean[0] += vals[i]
    std[0] += vals[i]**2
  mean[0] /= n
  std[0] = np.sqrt(max(std[0]/n-mean[0]**2,0))   
  
  return 0
# -------

# ------------------------------ EXCLUSION MATRIX ------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
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

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
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

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
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

# ------------------------------- WALK FUNCTIONS -------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32 _walk_fwd(TreeData* tdata, CFLOAT64 pmin=0.) except -31000:
  cdef CINT32 i,j, nd0,nd1,ee,dd0,dd1, val,nwalk=0
  cdef CINT32 ne = tdata.n_edges
  cdef CINT32 nn = tdata.n_nodes
  cdef CUINT8 ax0,ax1

  for i in range(nn):
    if tdata.nodes[i].done: continue
    nd0 = tdata.nodes[i].node
    ax0 = tdata.nodes[i].axis
    dd0 = tdata.nodes[i].depth
    ax1 = 1-ax0

    for j in range(tdata.poss.indptr[ax0][nd0],tdata.poss.indptr[ax0][nd0+1]):
      ee = tdata.poss.data[ax0][j]
      if ee<0: continue
      if contains_key(tdata.edict, ee): continue
      if tdata.poss.prbs[ee]<=pmin: continue

      if ne>=tdata.max_edges: return -31002

      nd1 = tdata.poss.indices[ax0][j]
      get_item(tdata.ndict[ax1], nd1, &val)
      if val==ERR_NOTIN:
        if nn>=tdata.max_nodes: return -31003
        dd1 = dd0+1
        tdata.nodes[nn].node = nd1
        tdata.nodes[nn].axis = ax1
        tdata.nodes[nn].depth = dd1
        tdata.nodes[nn].pmiss0 = tdata.poss.pms[ax1]
        tdata.nodes[nn].pmissF = .0
        tdata.nodes[nn].done = 0
        tdata.nodes[nn].idx = -1
        set_item(tdata.ndict[ax1], nd1, nn)
        val = nn
        nn += 1
      else:
        dd1 = min(tdata.nodes[val].depth, dd0)+1

      tdata.edges[ne].edge = ee
      tdata.edges[ne].nodes[ax0] = &tdata.nodes[i]
      tdata.edges[ne].nodes[ax1] = &tdata.nodes[val]
      tdata.edges[ne].depth = dd1
      tdata.edges[ne].psing = tdata.poss.prbs[ee]
      tdata.edges[ne].pfull = 0.
      tdata.edges[ne].idx = -1
      set_item(tdata.edict, ee, ne)
      ne += 1
      nwalk += 1
    
    tdata.nodes[i].done = 1
  
  tdata.n_nodes = nn
  tdata.n_edges = ne
  tdata.depth += 1
  return nwalk
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32 _walk_fwd_sort(TreeData* tdata, CFLOAT64 pmin, CINT32 nmax) except -31020:
  cdef CINT32 i,j,j0,cnt, nd0,nd1,ee,dd0,dd1, val,nwalk=0
  cdef CINT32 ne = tdata.n_edges
  cdef CINT32 nn = tdata.n_nodes
  cdef CUINT8 ax0,ax1

  for i in range(nn):
    if tdata.nodes[i].done: continue
    nd0 = tdata.nodes[i].node
    ax0 = tdata.nodes[i].axis
    dd0 = tdata.nodes[i].depth
    ax1 = 1-ax0

    cnt = 0
    for j in range(tdata.poss.indptr[ax0][nd0],tdata.poss.indptr[ax0][nd0+1]):
      ee = tdata.poss.data[ax0][j]
      if contains_key(tdata.edict, ee) or (tdata.poss.prbs[ee]<=pmin): continue
      if cnt>=tdata.poss.N: return -31023
      tdata.poss.isort[cnt] = cnt
      tdata.poss.sort_edges[cnt] = j
      tdata.poss.sort_pvals[cnt] = tdata.poss.prbs[ee]
      cnt += 1
    argsort_F64(tdata.poss.sort_pvals, tdata.poss.isort, cnt, 1, 1)

    for j0 in range(cnt):
      j = tdata.poss.sort_edges[tdata.poss.isort[j0]]
      ee = tdata.poss.data[ax0][j]
      nd1 = tdata.poss.indices[ax0][j]
      get_item(tdata.ndict[ax1], nd1, &val)
      if (j0>=nmax) and val==ERR_NOTIN: continue

      if ne>=tdata.max_edges: return -31021

      if val==ERR_NOTIN:
        if nn>=tdata.max_nodes: return -31022
        dd1 = dd0+1
        tdata.nodes[nn].node = nd1
        tdata.nodes[nn].axis = ax1
        tdata.nodes[nn].depth = dd1
        tdata.nodes[nn].pmiss0 = tdata.poss.pms[ax1]
        tdata.nodes[nn].pmissF = 0.
        tdata.nodes[nn].done = 0
        tdata.nodes[nn].idx = -1
        set_item(tdata.ndict[ax1], nd1, nn)
        val = nn
        nn += 1
      else:
        dd1 = min(tdata.nodes[val].depth, dd0)+1

      tdata.edges[ne].edge = ee
      tdata.edges[ne].nodes[ax0] = &tdata.nodes[i]
      tdata.edges[ne].nodes[ax1] = &tdata.nodes[val]
      tdata.edges[ne].depth = dd1
      tdata.edges[ne].psing = tdata.poss.prbs[ee]
      tdata.edges[ne].pfull = 0.
      tdata.edges[ne].idx = -1
      set_item(tdata.edict, ee, ne)
      ne += 1
      nwalk += 1
    
    tdata.nodes[i].done = 1
  
  tdata.n_nodes = nn
  tdata.n_edges = ne
  tdata.depth += 1
  return nwalk
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32 _walk_multi(TreeData* tdata, CFLOAT64 pmin, CINT32 nwalk, CINT32* steps) except -31010:
  cdef CINT32 ntot=0,ret
  steps[0] = 0

  if nwalk==0: return 0
  if nwalk<0:  nwalk = _MAXINT32

  while steps[0]<nwalk:
    ret = _walk_fwd(tdata, pmin)
    if ret<0:  return ret
    if ret==0: break
    steps[0] += 1
    ntot += ret
  
  return ntot
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32 _walk_upto(TreeData* tdata, CFLOAT64 pmin, CINT32 max_edges) except -31030:
  cdef CINT32 i,cnt,ret

  cnt = 0
  while True:
    ret = _walk_fwd(tdata, pmin)
    if (ret==0) or (ret==-31002): break
    cnt += 1
    if ret<0:  return ret
  
  if ret<0:
    if tdata.edges[0].depth==0: _init_from_edge(tdata, tdata.edges[0].edge)
    else:                       _init_from_node(tdata, tdata.nodes[0].node, tdata.nodes[0].axis)
    for i in range(cnt):
      ret = _walk_fwd(tdata, pmin)
      if ret<0:  return ret

  return _check_complete(tdata, pmin)
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32 _check_complete(TreeData* tdata, CFLOAT64 pmin) except -31040:
  cdef CINT32 i,j,nd0,ee
  cdef CUINT8 ax0

  for i in range(tdata.n_nodes):
    if tdata.nodes[i].done: continue
    nd0 = tdata.nodes[i].node
    ax0 = tdata.nodes[i].axis

    for j in range(tdata.poss.indptr[ax0][nd0],tdata.poss.indptr[ax0][nd0+1]):
      ee = tdata.poss.data[ax0][j]
      if ee<0: continue
      if contains_key(tdata.edict, ee): continue
      if tdata.poss.prbs[ee]<=pmin: continue
      return 0
  
  return 1
# -------

# ------------------------------ PFULL FUNCTIONS -------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32 _calc_pfull(TreeData* tdata, CombsData* cdata, CINT32 core_depth, CFLOAT64 ratio) except -31120:
  cdef CINT32 i,d, ret, Ne,Nt
  cdef CUINT8 ax
  cdef CFLOAT64 pm,K,ex

  cdata.ratio = ratio

  # if tdata.n_edges>cdata.max_edges: return -31121
  if tdata.n_nodes>cdata.max_nodes: return -31122

  # Sort edges in descending order of probability
  # Ne = tdata.n_edges
  Ne = 0
  for i in range(tdata.n_edges):
    if tdata.edges[i].psing>0.:
      if Ne>=cdata.max_edges: return -31121
      cdata.isort[Ne] = i
      cdata.poks[Ne] = tdata.edges[i].psing
      Ne += 1
    else:
      tdata.edges[i].pfull = 0.

  argsort_F64(cdata.poks, cdata.isort, Ne, 1, 1)

  # Add core nodes miss-edges
  Nt = Ne
  for i in range(tdata.n_nodes):
    if tdata.nodes[i].depth <= core_depth:
      tdata.nodes[i].idx = Nt
      cdata.poks[Nt] = tdata.nodes[i].pmiss0
      Nt += 1
    else:
      tdata.nodes[i].idx = -1
  
  cdata.n_edges = Ne
  cdata.n_tot = Nt
 
  # Get compatibility matrix
  ret = _get_cmpt(cdata, tdata)

  # Initialize for recursive function
  cdata.ptot = 1.
  for i in range(Nt):
    cdata.left[i] = 1
    cdata.prbs[i] = 0.
    if i<Ne:  cdata.ptot *= (1-cdata.poks[i])
    else:     cdata.ptot *= cdata.poks[i]
  for i in range(Ne,Nt): cdata.prbs[i] = cdata.ptot

  cdata.pmax0 = 0.
  cdata.pmax = 0.
  cdata.recs = 0
  
  # Start recursive function
  # print(f'Start rec combs: Ne={cdata.n_edges}, Nt={cdata.n_tot}')
  # t0 = time()
  ret = _rec_combs(cdata, 0, 1., 0)
  # print(f'End rec combs: {ret} combs, {cdata.recs} recursions, dt={time()-t0}')
  if ret<0: return ret
  if cdata.ptot==0.: return -31123

  # Update tree
  for i in range(Ne):
    tdata.edges[cdata.isort[i]].pfull = cdata.prbs[i]/cdata.ptot
  for i in range(tdata.n_nodes):
    if tdata.nodes[i].idx>=0: tdata.nodes[i].pmissF = cdata.prbs[tdata.nodes[i].idx]/cdata.ptot

  return 0
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32 _get_cmpt(CombsData* cdata, TreeData* tdata) except -31200:
  cdef CINT32 i,j,k, Ne=cdata.n_edges, Nt=cdata.n_tot
  cdef TreeEdge* ei
  cdef TreeEdge* ej

  for i in range(Ne):
    cdata.cmpt[i*Nt+i] = 0
    ei = &tdata.edges[cdata.isort[i]]
    # Check compatibility of current edge with full-edges
    for j in range(i+1,Ne):
      cdata.cmpt[i*Nt+j] = 1
      cdata.cmpt[j*Nt+i] = 1

      ej = &tdata.edges[cdata.isort[j]]
      for k in range(tdata.poss.excl_indptr[ei[0].edge],tdata.poss.excl_indptr[ei[0].edge+1]):
        if tdata.poss.excl_indices[k]<ej[0].edge: continue
        if tdata.poss.excl_indices[k]>ej[0].edge: break
        cdata.cmpt[i*Nt+j] = 0
        cdata.cmpt[j*Nt+i] = 0
        break

    # Check compatibility of currente edge with miss-edges
    for j in range(Ne,Nt): cdata.cmpt[i*Nt+j] = 1

    j = ei[0].nodes[0].idx
    if j>=0: cdata.cmpt[i*Nt+j] = 0
    j = ei[0].nodes[1].idx
    if j>=0: cdata.cmpt[i*Nt+j] = 0

  return 0
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32 _rec_combs(CombsData* cdata, CINT32 i0, CFLOAT64 cprb, CINT32 lvl) except -31210:
  cdata.recs += 1

  cdef CINT32 i,j, cnt, n=0, ret, Ne=cdata.n_edges, Nt=cdata.n_tot
  cdef CFLOAT64 cprod0=1., cprod=1.

  # Cycle through remaining edges
  for i in range(i0,Ne):
    cprod0 = cprb*cdata.poks[i]
    if cprod0<=cdata.pmax0: break

    if cdata.left[lvl*Nt+i]:
      cdata.idxs[lvl] = i
      cnt = 0
      # Check compatibility with all following edges
      for j in range(i+1,Nt):
        cdata.left[(lvl+1)*Nt+j] = cdata.cmpt[i*Nt+j]*cdata.left[lvl*Nt+j]
        if j<Ne: cnt += cdata.left[(lvl+1)*Nt+j]

      # If there is at least a compatible (full) edge, continue recursion
      if cnt>0:
        ret = _rec_combs(cdata, i+1, cprod0, lvl+1)
        if ret<0: return ret
        n += ret

      # Add current combination
      for j in range(i+1,Ne): cprod0 *= (1-cdata.poks[j])
      cprod = cprod0
      for j in range(Ne,Nt):
        if cdata.left[(lvl+1)*Nt+j]: cprod *= cdata.poks[j]
        else:                        cprod *= (1-cdata.poks[j])
      
      if (cprod0>cdata.pmax0) and (cprod>cdata.pmax):
        for j in range(lvl+1): cdata.prbs[cdata.idxs[j]] += cprod
        for j in range(Ne,Nt):
          if cdata.left[(lvl+1)*Nt+j]: cdata.prbs[j] += cprod
        cdata.ptot += cprod
        if cdata.combs: # --- ADD ---
          if cdata.nc>=cdata.max_combs: return -31211
          cdata.combs[cdata.nc] = cprod
          cdata.nc += 1
        if cprod0*cdata.ratio>cdata.pmax0: cdata.pmax0 = cprod0*cdata.ratio
        if cprod*cdata.ratio >cdata.pmax:  cdata.pmax  = cprod*cdata.ratio

        n += 1

    cprb *= (1-cdata.poks[i])

  return n
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32 _solve_tree_pfull(TreeData* tdata, TreeData* ttemp, CombsData* cdata, SolveData* sdata, CINT32 depth, CFLOAT64 pmin, CFLOAT64 ratio) except -32000:
  if depth < 3: return -32001
  if ttemp.max_edges != cdata.max_edges: return -32002
  if tdata.n_edges > sdata.n_edges: return -32003
  if tdata.n_nodes > sdata.n_nodes: return -32004

  cdef CINT32 ret

  # --- Solve for tree smaller than max_edges ----------
  if tdata.n_edges<=cdata.max_edges:
    ret = _calc_pfull(tdata, cdata, tdata.depth, ratio)
    if ret==-31123: return 0 # ptot == 0
    if ret<0: return ret
    return 1
  
  # --- Solve for tree larger than max_edges ----------
  cdef CINT32 i,j,d,_depth, ee,nn
  cdef CFLOAT64 mean

  for i in range(tdata.n_nodes):
    # --- Find partial tree ----------
    _depth = depth
    while True:
      _init_from_node(ttemp, tdata.nodes[i].node, tdata.nodes[i].axis)
      for d in range(_depth):
        ret = _walk_fwd(ttemp, pmin)
        if ret==-31002: # too many edges
          _depth -= 1
          break
        if ret<0: return ret
        if ret==0: return -32005
      else:
        break

    if _depth<3: continue

    # --- Calculate full probabilities ----------
    ret = _calc_pfull(ttemp, cdata, _depth-1, ratio)
    if ret==-31123: continue # ptot == 0
    if ret<0: return ret

    # --- Update core edges probabilities ----------
    for j in range(ttemp.n_edges):
      if ttemp.edges[j].depth>_depth-1: continue
      get_item(tdata.edict, ttemp.edges[j].edge, &ee)
      if ee==ERR_NOTIN: continue

      if sdata.nfs[ee]>=sdata.max_poss: return -32006
      sdata.pfs[ee*sdata.max_poss+sdata.nfs[ee]] = ttemp.edges[j].pfull
      sdata.nfs[ee] += 1

    # --- Update core nodes probabilities ----------
    for j in range(ttemp.n_nodes):
      if ttemp.nodes[j].depth>_depth-2: continue
      get_item(tdata.ndict[ttemp.nodes[j].axis], ttemp.nodes[j].node, &nn)
      if nn==ERR_NOTIN: continue

      if sdata.nms[nn]>=sdata.max_poss: return -32007
      sdata.pms[nn*sdata.max_poss+sdata.nms[nn]] = ttemp.nodes[j].pmissF
      sdata.nms[nn] += 1

  # --- Update tree edges pfull ----------
  for i in range(tdata.n_edges):
    if sdata.nfs[i]>0:
      mean = 0.
      for j in range(sdata.nfs[i]): mean += sdata.pfs[i*sdata.max_poss+j]
      mean /= sdata.nfs[i]
    else:
      mean = np.nan
    tdata.edges[i].pfull = mean

  # --- Update tree nodes pmissF ----------
  for i in range(tdata.n_nodes):
    if sdata.nms[i]>0:
      mean = 0.
      for j in range(sdata.nms[i]): mean += sdata.pms[i*sdata.max_poss+j]
      mean /= sdata.nms[i]
    else:
      mean = np.nan
    tdata.nodes[i].pmissF = mean

  return 2
# -------

# ------------------------------ PRUNE FUNCTIONS -------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32 _prune_edge_improbable(TreeData* tdata, CombsData* cdata, CINT32 edge, CINT32 nmax, CFLOAT64 pfull_ratio, CFLOAT64 thr_ratio, CUINT8* removed) except -31500:
  if tdata.poss.prbs[edge]==0.: return 0
  
  cdef CINT32 i,j,ret
  cdef CFLOAT64 thr=0.
  cdef CUINT8 comp

  ret = _init_from_edge(tdata, edge)
  if ret<0: return ret

  ret = _walk_fwd_sort(tdata, 0., nmax)
  if ret<=0: return ret
  ret = _walk_fwd_sort(tdata, 0., nmax)
  if ret<0: return ret

  ret = _calc_pfull(tdata, cdata, 1, pfull_ratio)
  if ret<0: return ret

  for i in range(tdata.n_edges):
    if tdata.edges[i].depth!=1: continue

    comp = 1
    for j in range(tdata.poss.excl_indptr[tdata.edges[0].edge],tdata.poss.excl_indptr[tdata.edges[0].edge+1]):
      if tdata.poss.excl_indices[j]<tdata.edges[i].edge: continue
      if tdata.poss.excl_indices[j]>tdata.edges[i].edge: break
      comp = 0
      break
    if comp: continue

    if tdata.edges[i].pfull>thr: thr=tdata.edges[i].pfull

  if tdata.edges[0].pfull<thr*thr_ratio:
    tdata.poss.prbs[tdata.edges[0].edge] = 0.
    removed[tdata.edges[0].edge] = 1
    return 1
  else:
    return 0
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32 _prune_edge_probable(TreeData* tdata, CombsData* cdata, CINT32 edge, CINT32 nmax, CFLOAT64 pfull_ratio, CFLOAT64 thr_ratio, CUINT8* removed, CFLOAT64* ratios) except -31510:
  if tdata.poss.prbs[edge]==0.: return 0
  
  cdef CINT32 i,j,ret
  cdef CFLOAT64 p0,pe,p1
  cdef CUINT8 comp,ax1
  cdef TreeNode* node1

  ret = _init_from_edge(tdata, edge)
  if ret<0: return ret

  ret = _walk_fwd(tdata, 0.)
  if ret<=0: return ret
  ret = _walk_fwd_sort(tdata, 0., nmax)
  if ret<0: return ret
  ret = _walk_fwd_sort(tdata, 0., nmax)
  if ret<0: return ret

  ret = _calc_pfull(tdata, cdata, 2, pfull_ratio)
  if ret<0: return ret

  p0 = min(tdata.edges[0].pfull, .99999)
  if p0==0.:
    tdata.poss.prbs[tdata.edges[0].edge] = 0.
    return 1
  
  ret = 0
  for i in range(tdata.n_edges):
    if (tdata.edges[i].depth!=1): continue
    pe = min(tdata.edges[i].pfull, .99999)
    # Remove the edge if it's certainly wrong (p=0)
    if pe==0.:
      tdata.poss.prbs[tdata.edges[i].edge] = 0.
      removed[tdata.edges[i].edge] = 1
      ret += 1
      continue
    
    # Check if edge is compatible with base edge, and in case skip it
    comp = 1
    for j in range(tdata.poss.excl_indptr[tdata.edges[0].edge],tdata.poss.excl_indptr[tdata.edges[0].edge+1]):
      if tdata.poss.excl_indices[j]<tdata.edges[i].edge: continue
      if tdata.poss.excl_indices[j]>tdata.edges[i].edge: break
      comp = 0
      break
    if comp: continue

    # Find best edge out of second node (node1)
    if tdata.edges[0].nodes[0]==tdata.edges[i].nodes[0]:
      node1 = tdata.edges[i].nodes[1]
      ax1 = 1
    elif tdata.edges[0].nodes[1]==tdata.edges[i].nodes[1]:
      node1 = tdata.edges[i].nodes[0]
      ax1 = 0
    else:
      return -31511
    # p1 = node1[0].pmissF
    p1 = pe

    for j in range(tdata.n_edges):
      if (tdata.edges[j].depth!=2) or (tdata.edges[j].nodes[ax1]!=node1): continue
      if tdata.edges[j].pfull>p1: p1=tdata.edges[j].pfull
    p1 = min(p1, .99999)

    # Remove the edge if it's probably wrong
    ratios[tdata.edges[i].edge] = (p0*(1-pe)*p1)/((1-p0)*pe*(1-p1))
    if (p0*(1-pe)*p1)/((1-p0)*pe*(1-p1)) > thr_ratio:
      tdata.poss.prbs[tdata.edges[i].edge] = 0.
      removed[tdata.edges[i].edge] = 1
      ret += 1
  
  return ret
# -------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32 _prune_edge_probable2(TreeData* tdata, CombsData* cdata, CINT32 edge, CINT32 nmax, CFLOAT64 pfull_ratio, CFLOAT64 pmatch, CUINT8* removed) except -31510:
  if tdata.poss.prbs[edge]==0.: return 0
  
  cdef CINT32 i,j,ret
  cdef CUINT8 comp

  ret = _init_from_edge(tdata, edge)
  if ret<0: return ret

  ret = _walk_fwd(tdata, 0.)
  if ret<=0: return ret
  ret = _walk_fwd_sort(tdata, 0., nmax)
  if ret<0: return ret
  ret = _walk_fwd_sort(tdata, 0., nmax)
  if ret<0: return ret

  ret = _calc_pfull(tdata, cdata, 2, pfull_ratio)
  if ret<0: return ret

  if tdata.edges[0].pfull<pmatch: return 0
  
  ret = 0
  for i in range(tdata.n_edges):
    if (tdata.edges[i].depth!=1): continue
    
    # Check if edge is compatible with base edge, and in case skip it
    comp = 1
    for j in range(tdata.poss.excl_indptr[tdata.edges[0].edge],tdata.poss.excl_indptr[tdata.edges[0].edge+1]):
      if tdata.poss.excl_indices[j]<tdata.edges[i].edge: continue
      if tdata.poss.excl_indices[j]>tdata.edges[i].edge: break
      comp = 0
      break
    if comp: continue

    tdata.poss.prbs[tdata.edges[i].edge] = 0.
    removed[tdata.edges[i].edge] = 1
    ret += 1
  
  return ret
# -------

# ------------------------------ MATCH FUNCTIONS -------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef CINT32 _match_poss(TreeData* tdata, CombsData* cdata, MatchData* mdata, CINT32 min_depth, CFLOAT64 ratio) except -31600:
  cdef CINT32 i,j,idx,nd,ret
  cdef CUINT8 ax

  for i in range(mdata.N):
    idx = mdata.isort[i]
    if mdata.pfulls[idx]==0.: continue

    if idx<mdata.ne: # full edge
      # Remove non-compatible edges
      for j in range(tdata.poss.excl_indptr[idx],tdata.poss.excl_indptr[idx+1]):
        mdata.pfulls[tdata.poss.excl_indices[j]] = 0.
      
      # Remove non-matching for the nodes
      mdata.pfulls[mdata.ne+tdata.poss.nodes[0][idx]] = 0.
      mdata.pfulls[mdata.ne+mdata.nn0+tdata.poss.nodes[1][idx]] = 0.
    
    else:
      if idx-mdata.ne-mdata.nn0>=0:
        nd = idx-mdata.ne-mdata.nn0
        ax = 1
      else:
        nd = idx-mdata.ne
        ax = 0
      
      # Remove edges from node
      for j in range(tdata.poss.indptr[ax][nd],tdata.poss.indptr[ax][nd+1]):
        mdata.pfulls[tdata.poss.data[ax][j]] = 0.
  
  return 0
# -------

# ------------------------------ PYTHON INTERFACE ------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

@dwrap
def tree_walk_fwd(DataWrapper dw, poss, tree, CINT32 max_edges, CINT32 max_nodes, CFLOAT64 pmin):
  # --- Init DataWrapper ----------
  dw.init_pdata(poss)
  dw.init_tdata(0, max_edges, max_nodes, tree=tree)

  # --- Main logic ----------
  cdef CINT32 ret
  ret = _walk_fwd(dw.tdata[0], pmin)
  if ret<0: return (ret,{})

  # --- Return logic ----------
  _update_tree(dw.tdata[0], tree)

  # --- Deallocate ----------
  dw.dealloc_pdata()
  dw.dealloc_tdata(0)

  return (0,{'edges+': ret})
# -------

@dwrap
def tree_walk_fwd_sort(DataWrapper dw, poss, tree, CINT32 max_edges, CINT32 max_nodes, CFLOAT64 pmin, CINT32 nmax):
  # --- Init DataWrapper ----------
  dw.init_pdata(poss)
  dw.init_tdata(0, max_edges, max_nodes, tree=tree)

  # --- Main logic ----------
  cdef CINT32 ret
  ret = _walk_fwd_sort(dw.tdata[0], pmin, nmax)
  if ret<0: return (ret,{})

  # --- Return logic ----------
  _update_tree(dw.tdata[0], tree)

  # --- Deallocate ----------
  dw.dealloc_pdata()
  dw.dealloc_tdata(0)

  return (0,{'edges+': ret})
# -------

@dwrap
def tree_walk_multi(DataWrapper dw, poss, tree, CINT32 max_edges, CINT32 max_nodes, CINT32 nwalk, CFLOAT64 pmin):
  # --- Init DataWrapper ----------
  dw.init_pdata(poss)
  dw.init_tdata(0, max_edges, max_nodes, tree=tree)

  # --- Main logic ----------
  cdef CINT32 ret, steps
  ret = _walk_multi(dw.tdata[0], pmin, nwalk, &steps)
  if ret<0: return (ret,{})

  # # --- Return logic ----------
  _update_tree(dw.tdata[0], tree)

  # --- Deallocate ----------
  dw.dealloc_pdata()
  dw.dealloc_tdata(0)

  return (0, {'steps': steps, 'edges+': ret})
# -------

@dwrap
def tree_walk_upto(DataWrapper dw, poss, tree, CINT32 max_edges, CINT32 max_nodes, CFLOAT64 pmin):
  # --- Init DataWrapper ----------
  dw.init_pdata(poss)
  dw.init_tdata(0, max_edges, max_nodes, tree=tree)

  # --- Main logic ----------
  cdef CINT32 ret, steps
  ret = _walk_upto(dw.tdata[0], pmin, max_edges)
  if ret<0: return (ret,{})

  # # --- Return logic ----------
  _update_tree(dw.tdata[0], tree)

  # --- Deallocate ----------
  dw.dealloc_pdata()
  dw.dealloc_tdata(0)

  return (0,{})
# -------

@dwrap
def tree_calc_pfull(DataWrapper dw, poss, tree, CINT32 core_depth, CFLOAT64 ratio, CINT32 max_edges=50):
  if len(tree.edges)>max_edges: raise RuntimeError(f'Tree has too many edges ({len(tree.edges)}), max is {max_edges}!')

  # --- Init DataWrapper ----------
  dw.init_pdata(poss)
  dw.init_tdata(0, <CINT32>len(tree.edges), <CINT32>len(tree.nodes), tree=tree)
  dw.init_cdata(<CINT32>len(tree.edges), <CINT32>len(tree.nodes))

  # --- Main logic ----------
  cdef CINT32 ret
  ret = _calc_pfull(dw.tdata[0], dw.cdata, core_depth, ratio)
  if ret<0: return (ret,{})

  # --- Return logic ----------
  _update_tree(dw.tdata[0], tree)

  # --- Deallocate ----------
  dw.dealloc_pdata()
  dw.dealloc_tdata(0)
  dw.dealloc_cdata()

  return (0,{})
# -------

@dwrap
def tree_check_complete(DataWrapper dw, poss, tree, CFLOAT64 pmin):
  # --- Init DataWrapper ----------
  dw.init_pdata(poss)
  dw.init_tdata(0, <CINT32>len(tree.edges), <CINT32>len(tree.nodes), tree=tree)

  # --- Main logic ----------
  cdef CINT32 ret
  ret = _check_complete(dw.tdata[0], pmin)
  if ret<0: return (ret,{})

  # --- Deallocate ----------
  dw.dealloc_pdata()
  dw.dealloc_tdata(0)

  return (0,{'complete': ret})
# -------

@dwrap
def solve_tree(DataWrapper dw, poss, tree, CINT32 max_edges, CINT32 max_poss, CFLOAT64 pmin, CINT32 depth, CFLOAT64 ratio):
  cdef CINT32 ret,i,j
  cdef CFLOAT64 mean,std

  # --- Init DataWrapper ----------
  dw.init_pdata(poss)
  dw.init_tdata(0, <CINT32>len(tree.edges), <CINT32>len(tree.nodes), tree=tree)
  dw.init_tdata(1, max_edges, 2*max_edges, tree=None)
  dw.init_cdata(max_edges, 2*max_edges)
  dw.init_sdata(<CINT32>len(tree.edges), <CINT32>len(tree.nodes), max_poss)
  
  cdef TreeData* tdata = dw.tdata[0]
  cdef TreeData* ttemp = dw.tdata[1]

  if tdata.n_edges != dw.sdata.n_edges: raise RuntimeError(f'Error initializing SolveData edges: {tdata.n_edges} != {dw.sdata.n_edges}')
  if tdata.n_nodes != dw.sdata.n_nodes: raise RuntimeError(f'Error initializing SolveData nodes: {tdata.n_nodes} != {dw.sdata.n_nodes}')

  for i in range(tdata.n_edges): dw.sdata.nfs[i] = 0
  for i in range(tdata.n_nodes): dw.sdata.nms[i] = 0

  # --- Main logic ----------
  ret = _solve_tree_pfull(tdata, ttemp, dw.cdata, dw.sdata, depth, pmin, ratio)
  if ret<0: return (ret,{})

  # --- Return logic ----------
  edges = {}
  for i in range(dw.sdata.n_edges):
    _calc_stats(&dw.sdata.pfs[i*dw.sdata.max_poss], dw.sdata.nfs[i], &mean, &std)
    edges[tdata.edges[i].edge] = (mean,std,dw.sdata.nfs[i])
  
  nodes = [{},{}]
  for i in range(dw.sdata.n_nodes):
    _calc_stats(&dw.sdata.pms[i*dw.sdata.max_poss], dw.sdata.nms[i], &mean, &std)
    nodes[tdata.nodes[i].axis][tdata.nodes[i].node] = (mean,std,dw.sdata.nms[i])
 
  _update_tree(tdata, tree)

  # --- Deallocate ----------
  dw.dealloc_pdata()
  dw.dealloc_tdata(0)
  dw.dealloc_tdata(1)
  dw.dealloc_cdata()
  dw.dealloc_sdata()

  return (0, {'complete': ret, 'edges': edges, 'nodes': nodes})
# -------

@dwrap
def cluster_poss(DataWrapper dw, poss, CINT32 max_edges, CFLOAT64 pmin):
  # --- Init DataWrapper ----------
  dw.init_pdata(poss)
  dw.init_tdata(0, max_edges, 2*max_edges, tree=None)
  cdef TreeData* tdata = dw.tdata[0]

  # --- Main logic ----------
  cdef CINT32 i,ret,clst
  cdef CUINT8 ax

  _edges = np.full(len(poss), -1, dtype=np.int32)
  _nodes = np.full((2,max(poss.shape)), -1, dtype=np.int32)
  _ns = np.array(poss.shape, dtype=np.int32)
  cdef CINT32[:] edges = _edges
  cdef CINT32[:,:] nodes = _nodes
  cdef CINT32[:] ns = _ns

  clst = 0
  for ax in range(2):
    for i in range(ns[ax]):
      if nodes[ax,i]>=0: continue

      _init_from_node(tdata, i, ax)
      while True:
        ret = _walk_fwd(tdata, pmin)
        if ret<0: return (ret,{})
        if ret==0: break
      
      for j in range(tdata.n_edges): edges[tdata.edges[j].edge] = clst
      for j in range(tdata.n_nodes): nodes[tdata.nodes[j].axis,tdata.nodes[j].node] = clst
      clst += 1
  
  # --- Deallocate ----------
  dw.dealloc_pdata()
  dw.dealloc_tdata(0)

  return (0,{'edges': _edges, 'nodes0': _nodes[0,:ns[0]], 'nodes1': _nodes[1,:ns[1]]})
# -------

@dwrap
def solve_poss(DataWrapper dw, poss, CINT32 tree_edges, CINT32 comb_edges, CINT32 max_poss, CINT32 depth, CFLOAT64 ratio):
  cdef CINT32 i

  # --- Init DataWrapper ----------
  dw.init_pdata(poss)
  dw.init_tdata(0, tree_edges, 2*tree_edges, tree=None)
  dw.init_tdata(1, comb_edges, 2*comb_edges, tree=None)
  dw.init_cdata(comb_edges, 2*comb_edges)
  dw.init_sdata(tree_edges, 2*tree_edges, max_poss)

  cdef TreeData* tdata = dw.tdata[0]
  cdef TreeData* ttemp = dw.tdata[1]

  # --- Main logic ----------
  _edges = np.full(len(poss), -1, dtype=np.int32)
  _nodes = np.full((2,max(poss.shape)), -1, dtype=np.int32)
  _ns = np.array(poss.shape, dtype=np.int32)
  cdef CINT32[:] edges = _edges
  cdef CINT32[:,:] nodes = _nodes
  cdef CINT32[:] ns = _ns

  cdef CINT32 j,ret,steps,clst
  cdef CUINT8 ax
  cdef CFLOAT64 mean,std

  clst = 0
  out_edges = {}
  out_nodes = [{},{}]
  for ax in range(2):
    for i in range(ns[ax]):
      if nodes[ax,i]>=0: continue

      # --- Find cluster ----------
      ret = _init_from_node(tdata, i, ax)
      if ret<0: return (ret,{})
      ret = _walk_multi(tdata, 0., -1, &steps)
      if ret<0: return (ret,{})
      
      for j in range(tdata.n_edges): edges[tdata.edges[j].edge] = clst
      for j in range(tdata.n_nodes): nodes[tdata.nodes[j].axis,tdata.nodes[j].node] = clst
      clst += 1

      # --- Solve tree ----------
      for j in range(tdata.n_edges): dw.sdata.nfs[j] = 0
      for j in range(tdata.n_nodes): dw.sdata.nms[j] = 0

      ret = _solve_tree_pfull(tdata, ttemp, dw.cdata, dw.sdata, depth, 0., ratio)
      if ret<0: return (ret,{})

      # --- Update output ----------
      if ret==0:
        for j in range(tdata.n_edges): out_edges[tdata.edges[j].edge] = (0.,0.,-2)
        for j in range(tdata.n_nodes): out_nodes[tdata.nodes[j].axis][tdata.nodes[j].node] = (1.,0.,-2)

      elif ret==1:
        for j in range(tdata.n_edges): out_edges[tdata.edges[j].edge] = (tdata.edges[j].pfull,0.,-1)
        for j in range(tdata.n_nodes): out_nodes[tdata.nodes[j].axis][tdata.nodes[j].node] = (tdata.nodes[j].pmissF,0.,-1)
      
      elif ret==2:
        for j in range(tdata.n_edges):
          _calc_stats(&dw.sdata.pfs[j*dw.sdata.max_poss], dw.sdata.nfs[j], &mean, &std)
          out_edges[tdata.edges[j].edge] = (mean,std,dw.sdata.nfs[j])
        
        for j in range(tdata.n_nodes):
          _calc_stats(&dw.sdata.pms[j*dw.sdata.max_poss], dw.sdata.nms[j], &mean, &std)
          out_nodes[tdata.nodes[j].axis][tdata.nodes[j].node] = (mean,std,dw.sdata.nms[j])
        
      else:
        raise RuntimeError(f'Unexpected return value from _solve_tree_pfull: {ret}!')

  # --- Deallocate ----------
  dw.dealloc_pdata()
  dw.dealloc_tdata(0)
  dw.dealloc_tdata(1)
  dw.dealloc_cdata()
  dw.dealloc_sdata()

  return (0, {'clusters': clst, 'edges': out_edges, 'nodes': out_nodes})
# -------

@dwrap
def match_poss(DataWrapper dw, poss, solve, CINT32 max_edges, CINT32 min_depth, CFLOAT64 ratio, CUINT8 match_miss):
  ne = len(poss)
  nn0,nn1 = poss.shape

  # --- Init DataWrapper ----------
  dw.init_pdata(poss)
  dw.init_tdata(0, max_edges, 2*max_edges, tree=None)
  dw.init_cdata(max_edges, 2*max_edges)
  dw.init_mdata(<CINT32>(ne+nn0+nn1))
  
  # --- Main logic ----------
  for i in range(dw.mdata.N): dw.mdata.pfulls[i] = 0.
  for ee,val in solve['edges'].items():    dw.mdata.pfulls[ee] = <CFLOAT64>val[0]
  for nn,val in solve['nodes'][0].items(): dw.mdata.pfulls[ne+nn] = <CFLOAT64>val[0] if match_miss else 0.
  for nn,val in solve['nodes'][1].items(): dw.mdata.pfulls[ne+nn0+nn] = <CFLOAT64>val[0] if match_miss else 0.
  dw.mdata.ne = <CINT32>ne
  dw.mdata.nn0 = <CINT32>nn0
  dw.mdata.nn1 = <CINT32>nn1

  for i in range(dw.mdata.N): dw.mdata.isort[i] = i
  argsort_F64(dw.mdata.pfulls, dw.mdata.isort, dw.mdata.N, 1, 0)

  ret = _match_poss(dw.tdata[0], dw.cdata, dw.mdata, min_depth, ratio)

  # # --- Return logic ----------
  mtcs = []
  for i in range(dw.mdata.N):
    if dw.mdata.pfulls[i]==0.: continue
    if i<ne: mtcs.append((poss.rows[i],poss.cols[i], dw.mdata.pfulls[i]))
    elif i<ne+nn0: mtcs.append((i-ne,-1, dw.mdata.pfulls[i]))
    else: mtcs.append((-1,i-ne-nn0, dw.mdata.pfulls[i]))

  # --- Deallocate ----------
  dw.pdata.pfulls = NULL
  dw.dealloc_pdata()
  dw.dealloc_tdata(0)
  dw.dealloc_cdata()
  dw.dealloc_mdata()

  return (0,{'mtcs': mtcs})
# -------

@dwrap
def prune_improbable(DataWrapper dw, poss, CINT32 max_edges, CINT32 nmax, CFLOAT64 pfull_ratio, CFLOAT64 thr_ratio):
  # --- Init DataWrapper ----------
  dw.init_pdata(poss)
  dw.init_tdata(0, max_edges, 2*max_edges, tree=None)
  dw.init_cdata(max_edges, 2*max_edges)

  # --- Main logic ----------
  _isort = np.argsort(poss.vals['p0']).astype(np.int32)
  cdef CINT32[:] isort = _isort
  _removed = np.zeros(len(poss), dtype=np.uint8)
  cdef CUINT8[:] removed = _removed
  _errors = np.zeros(len(poss), dtype=np.int32)
  cdef CINT32[:] errors = _errors

  cdef CINT32 i,ret,cnt=0

  t0 = time()
  for i in range(dw.tdata[0].poss.N):
    if i%100000==0: print(f'{i}/{dw.tdata[0].poss.N} [t={time()-t0:.3f}s]')

    ret = _prune_edge_improbable(dw.tdata[0], dw.cdata, isort[i], nmax, pfull_ratio, thr_ratio, &removed[0])
    if (ret==-31002) or (ret==-31021) or (ret==-31123) or (ret==-31501):
      errors[isort[i]] = ret
      continue # (too many edges to calculate pfull) or (ptot==0)
    if ret<0: return (ret,{})
    cnt += ret
  
  print(f'--> removed {cnt} edges [t={time()-t0:.3f}s]')

  # --- Deallocate ----------
  dw.dealloc_pdata()
  dw.dealloc_tdata(0)
  dw.dealloc_cdata()

  return (0,{'removed': _removed.astype(bool), 'errors': _errors})
# -------

@dwrap
def prune_probable(DataWrapper dw, poss, CINT32 max_edges, CINT32 nmax, CFLOAT64 pfull_ratio, CFLOAT64 thr_ratio):
  # --- Init DataWrapper ----------
  dw.init_pdata(poss)
  dw.init_tdata(0, max_edges, 2*max_edges, tree=None)
  dw.init_cdata(max_edges, 2*max_edges)

  # --- Main logic ----------
  _isort = np.argsort(poss.vals['p0'])[::-1].astype(np.int32)
  cdef CINT32[:] isort = _isort
  _removed = np.zeros(len(poss), dtype=np.uint8)
  cdef CUINT8[:] removed = _removed
  _ratios = np.zeros(len(poss), dtype=np.float64)
  cdef CFLOAT64[:] ratios = _ratios
  _errors = np.zeros(len(poss), dtype=np.int32)
  cdef CINT32[:] errors = _errors

  cdef CINT32 i,ret,cnt=0

  t0 = time()
  for i in range(dw.tdata[0].poss.N):
    if i%100000==0: print(f'{i}/{dw.tdata[0].poss.N} [t={time()-t0:.3f}s]')

    ret = _prune_edge_probable(dw.tdata[0], dw.cdata, isort[i], nmax, pfull_ratio, thr_ratio, &removed[0], &ratios[0])
    if (ret==-31002) or (ret==-31021) or (ret==-31123) or (ret==-31501):
      errors[isort[i]] = ret
      continue # (too many edges to calculate pfull) or (ptot==0)
    if ret<0: return (ret,{})
    cnt += ret
  
  print(f'--> removed {cnt} edges [t={time()-t0:.3f}s]')

  # --- Deallocate ----------
  dw.dealloc_pdata()
  dw.dealloc_tdata(0)
  dw.dealloc_cdata()

  return (0,{'removed': _removed.astype(bool), 'ratios': _ratios, 'errors': _errors})
# -------

@dwrap
def prune_probable2(DataWrapper dw, poss, CINT32 max_edges, CINT32 nmax, CFLOAT64 pfull_ratio, CFLOAT64 pmatch):
  # --- Init DataWrapper ----------
  dw.init_pdata(poss)
  dw.init_tdata(0, max_edges, 2*max_edges, tree=None)
  dw.init_cdata(max_edges, 2*max_edges)

  # --- Main logic ----------
  _isort = np.argsort(poss.vals['p0'])[::-1].astype(np.int32)
  cdef CINT32[:] isort = _isort
  _removed = np.zeros(len(poss), dtype=np.uint8)
  cdef CUINT8[:] removed = _removed
  _errors = np.zeros(len(poss), dtype=np.int32)
  cdef CINT32[:] errors = _errors

  cdef CINT32 i,ret,cnt=0

  t0 = time()
  for i in range(dw.tdata[0].poss.N):
    if i%100000==0: print(f'{i}/{dw.tdata[0].poss.N} [t={time()-t0:.3f}s]')

    ret = _prune_edge_probable2(dw.tdata[0], dw.cdata, isort[i], nmax, pfull_ratio, pmatch, &removed[0])
    if (ret==-31002) or (ret==-31021) or (ret==-31123) or (ret==-31501):
      errors[isort[i]] = ret
      continue # (too many edges to calculate pfull) or (ptot==0)
    if ret<0: return (ret,{})
    cnt += ret
    # if i>100000: break
  
  print(f'--> removed {cnt} edges [t={time()-t0:.3f}s]')

  # --- Deallocate ----------
  dw.dealloc_pdata()
  dw.dealloc_tdata(0)
  dw.dealloc_cdata()

  return (0,{'removed': _removed.astype(bool), 'errors': _errors})
# -------

# ----------------------------------- TESTS ------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

@dwrap
def test_pfull(DataWrapper dw, poss, tree, CINT32 max_edges, CINT32 core_depth, CFLOAT64 ratio, CUINT64 ncombs):
  if len(tree.edges)>max_edges: raise RuntimeError(f'Tree has too many edges ({len(tree.edges)}), max is {max_edges}!')

  # --- Init DataWrapper ----------
  dw.init_pdata(poss)
  dw.init_tdata(0, <CINT32>len(tree.edges), <CINT32>len(tree.nodes), tree=tree)
  dw.init_cdata(<CINT32>len(tree.edges), <CINT32>len(tree.nodes))

  _combs = np.zeros(ncombs, dtype=np.float64)
  cdef CFLOAT64[:] combs = _combs
  dw.cdata.combs = &combs[0]
  dw.cdata.nc = 0
  dw.cdata.max_combs = ncombs

  # --- Main logic ----------
  cdef CINT32 ret
  ret = _calc_pfull(dw.tdata[0], dw.cdata, core_depth, ratio)
  if ret<0: return (ret,{})

  # --- Return logic ----------
  _update_tree(dw.tdata[0], tree)

  # --- Deallocate ----------
  dw.dealloc_pdata()
  dw.dealloc_tdata(0)
  dw.dealloc_cdata()

  return (0,{'combs': _combs[:dw.cdata.nc]})
# -------
