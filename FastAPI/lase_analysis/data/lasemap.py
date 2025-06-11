import numpy as np
import pandas as pd

from copy import deepcopy

from scipy.spatial import ConvexHull
from sklearn.cluster import dbscan
from matplotlib.path import Path as PltPath
from matplotlib.patches import PathPatch

from .lasebase import LaseData


class LaseData_Map(LaseData):
  def __init__(self, *args, **kwds):
    super().__init__(*args, **kwds)

class LaseData_Map_Confocal(LaseData_Map):
  def __init__(self, *args, **kwds):
    super().__init__(*args, **kwds)

    if 'info' in kwds: self.info = kwds['info']

  def copy(self):
    obj = super().copy()
    obj.info = deepcopy(self.info)
    return obj

  def subset(self, mids, fids=None, name=None):
    obj = super().subset(mids, fids, name)
    obj.info = deepcopy(self.info)
    return obj
    
  # def find_cluster(self, Es0, max_dE=1., max_dR=10., min_prc=.75,
  #                  min_ana=.6, min_dig=.6, min_tot=.6, max_nin=0, check_hull=False):
  #   idx = np.where(np.logical_or.reduce([np.abs(self.lines.E-E)<max_dE for E in Es0]))[0]
  #   _,cids = dbscan(self.lines.iloc[idx][['i','j']].values,
  #                   eps=max_dR, min_samples=int(np.floor(min_prc*len(Es0))))
  #   ucids = np.unique(cids[cids>=0])

  #   if len(ucids) == 0: return []

  #   poss = []
  #   _scrs = []
  #   for cc in ucids:
  #     ll = self.lines.iloc[idx[cids==cc]]

  #     # Calculate scores
  #     EsC = np.sort(ll.E.values)
  #     _,_,ana,dig = align_single(Es0, EsC, max_dE=max_dE, thr_A=1., wgt_small=1., pmode='01')
  #     tot = 0.5*(dig+ana)

  #     if check_hull:
  #       # Check hull
  #       lcrd = ll[['i','j']].values
  #       i0,j0 = np.floor(np.min(lcrd,axis=0)).astype(int)
  #       i1,j1 = np.ceil(np.max(lcrd,axis=0)).astype(int)
  #       if lcrd.shape[0] >= 3:
  #         try:
  #           hull = ConvexHull(lcrd)
  #           hull_path = PltPath(lcrd[hull.vertices])

  #           alns = self.lines.loc[lambda df: (df.i>=i0)&(df.i<=i1)&(df.j>=j0)&(df.j<=j1)].drop(ll.index)
  #           acrd = alns[['i','j']].values
  #           nin = np.sum([int(hull_path.contains_point(pp)) for pp in acrd])
  #         except:
  #           return []
  #       else:
  #         nin = -1
  #     else:
  #       nin = -1

  #     if (dig<min_dig) or (ana<min_ana) or (tot<min_tot) or (nin>max_nin): continue
  #     poss.append((ll.index.values,ana,dig,tot,nin))
  #     _scrs.append(-tot)

  #   return [poss[ii] for ii in np.argsort(_scrs)]
