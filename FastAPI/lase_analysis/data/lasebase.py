import numpy as np
import pandas as pd
import numbers

from copy import copy, deepcopy
from operator import xor

from ..utils.uid import UID
from ..utils.constants import MIDS


# ------------------------------------------------------------------------------
# ------------------------------- BASE DATA TYPE -------------------------------
# ------------------------------------------------------------------------------

class LaseData:
  def __init__(self, mlt, lns, pks=None, fts=None, name='none', **kwds):
    self.name = name

    self.mlt = mlt
    self.lns = lns
    self.pks = pks
    self.fts = fts

    self.uids = [UID(uu) for uu in np.unique(self.mlt.index.values&UID.MASK_UID)]
    self._check_consistency()

    self.mid2fid = None

  def _check_consistency(self):
    if np.any(~self.lns.mid.isin(self.mlt.index)):
      raise ValueError('Some of the lines have a non-existing mid!')

    if not (self.pks is None) and np.any(~self.pks.lid.isin(self.lns.index)):
        raise ValueError('Some of the peaks have a non-existing lid!')


  @property
  def peaks(self):
    return self.pks

  @property
  def lines(self):
    return self.lns

  @property
  def multi(self):
    return self.mlt

  @property
  def feats(self):
    return self.fts


  def __add__(self, other):
    if not self.__class__ == other.__class__:
      raise TypeError('The two objects to be added are from different classes!')

    if len(np.intersect1d(self.uids, other.uids)) > 0:
      raise ValueError('The two objects to be added have uid(s) in common!')

    if xor(self.pks is None, other.pks is None):
      raise ValueError('Incompatible addends. One declares peaks and the other one does not!')

    if xor(self.fts is None, other.fts is None):
      raise ValueError('Incompatible addends. One declares feats and the other one does not!')

    mlt = pd.concat((self.mlt, other.mlt), axis=0)
    lns = pd.concat((self.lns, other.lns), axis=0)
    pks = None if self.pks is None else pd.concat((self.pks, other.pks), axis=0)
    fts = None if self.fts is None else pd.concat((self.fts, other.fts), axis=0)

    name = (self.name+'|'+other.name) if self.name!=other.name else self.name
    obj = self.__class__(mlt, lns, pks, fts, name)
    obj.mid2fid = None if self.mid2fid is None else pd.concat((self.mid2fid, other.mid2fid), axis=0)

    return obj

  def __radd__(self, other):
    if other == 0: return self.copy()
    return other.__add__(self)


  def copy(self):
    obj = self.__class__(self.mlt.copy(), self.lns.copy(), copy(self.pks), copy(self.fts), self.name)
    obj.mid2fid = copy(self.mid2fid)
    return obj

  def reset_uid(self):
    mid_map = {mid: ii for ii,mid in enumerate(self.mlt.index)}
    self.mlt.index = np.uint64([mid_map[mm] for mm in self.mlt.index])
    self.lns['mid'] = np.uint64([mid_map[mm] for mm in self.lns.mid])

    lid_map = {lid: ii for ii,lid in enumerate(self.lns.index)}
    self.lns.index = np.uint64([lid_map[ll] for ll in self.lns.index])

    if not self.pks is None:
      self.pks['lid'] = np.uint64([lid_map[ll] for ll in self.pks.lid])
      pid_map = {pid: ii for ii,pid in enumerate(self.pks.index)}
      self.pks.index = np.uint64([pid_map[pp] for pp in self.pks.index])

    if not self.fts is None:
      fid_map = {fid: ii for ii,fid in enumerate(self.fts.index)}
      self.fts.index = np.uint64([fid_map[ff] for ff in self.fts.index])

    if not self.mid2fid is None:
      self.mid2fid.index = np.uint64([mid_map[mm] for mm in self.mid2fid.index])
      self.mid2fid[:] = np.uint64([fid_map[ff] for ff in self.mid2fid.values])

    self.uids = [UID(uu) for uu in np.unique(self.mlt.index.values&UID.MASK_UID)]
    self._check_consistency()

  def set_uid(self, val=None):
    if val is None:                              uid = UID()
    elif isinstance(val, (numbers.Number, str)): uid = UID(val)
    elif isinstance(val, UID):                   uid = val
    else:
      raise TypeError(f"Invalid type for 'uid' ({type(val)})")

    self.reset_uid()

    self.mlt.index += uid.full
    self.lns.mid += uid.full
    self.lns.index += uid.full

    if not self.pks is None:
      self.pks.lid += uid.full
      self.pks.index += uid.full

    if not self.fts is None:
      self.fts.index += uid.full
    
    if not self.mid2fid is None:
      self.mid2fid.index += uid.full
      self.mid2fid[:] += uid.full

    self.uids = [UID(uu) for uu in np.unique(self.mlt.index.values&UID.MASK_UID)]
    self._check_consistency()

  def sample(self, N):
    mids = np.random.choice(self.mlt.index.values, N, replace=False)
    return self.subset(mids)

  def subset(self, mids, fids=None, name=None, docopy=False):
    mlt = self.mlt.loc[mids]
    lns = self.lns[self.lns.mid.isin(mlt.index)]
    pks = None if self.pks is None else\
          self.pks[self.pks.lid.isin(lns.index)]
    if (self.fts is None) or (fids is None): fts = None
    else:                                    fts = self.fts.loc[fids]

    if name is None: name = self.name+'>sub'
    obj = self.__class__(mlt=mlt.copy(), lns=lns.copy(), pks=copy(pks), fts=copy(fts), name=name)\
          if docopy else self.__class__(mlt=mlt, lns=lns, pks=pks, fts=fts, name=name)

    if not self.mid2fid is None: obj.mid2fid = self.mid2fid[self.mid2fid.index.isin(mids)]

    return obj

  def filter_uids(self, uids):
    if not hasattr(uids,'__iter__'): uids = [uids]

    pks = None if self.pks is None else \
          self.pks[np.isin(self.pks.index.values&UID.MASK_UID,uids)]
    fts = None if self.fts is None else \
          self.fts[np.isin(self.fts.index.values&UID.MASK_UID,uids)]
    lns = self.lns[np.isin(self.lns.index.values&UID.MASK_UID,uids)]
    mlt = self.mlt[np.isin(self.mlt.index.values&UID.MASK_UID,uids)]

    obj = self.__class__(mlt,lns,pks,fts,self.name+'>flt')
    
    if not self.mid2fid is None: obj.mid2fid = self.mid2fid[self.mid2fid.index.isin(mlt.index)]

    return 




















# DTYPE_DEFAULTS = {np.dtype('int32'): 0, np.dtype('uint64'): 0, np.dtype('int64'): 0,
#                   np.dtype('<f8'): np.nan, np.dtype('<f4'): np.nan, np.dtype('float32'): np.nan,
#                   np.dtype('O'): None}


  # def set_features(self, feats, dflt=None):
  #   self.feats = {}
  #   self.multi = self.multi.reindex(columns=self.MLT_COLUMNS.keys()).astype(self.MLT_COLUMNS)
  #   # if np.any(~feats.index[feats.index>0].isin(self.multi.index)):
  #   #   raise ValueError("Some mids in the 'feats' DataFrame are not present in the multi data!")

  #   self.feats = {key: dtype for key,dtype in feats.dtypes.items()}
  #   MLT_FEAT = {**self.MLT_COLUMNS, **self.feats}

  #   idx0 = ~self.multi.index.isin(feats.index)
  #   if np.any(idx0):
  #     if dflt is None:
  #       dflt = {key: DTYPE_DEFAULTS[dtype] for key,dtype in self.feats.items()}

  #     df0 = pd.DataFrame(dflt, index=self.multi.index[idx0])
  #     feats = pd.concat([feats,df0])


  #   idx0 = ~feats.index.isin(self.multi.index)
  #   if np.any(idx0):
  #     # idx0 = (feats.index == 0)
  #     # mids0 = np.arange(1,idx0.sum()+1,dtype=np.uint64) + self.multi.index.max()
  #     # feats.index.values[idx0] = mids0

  #     df0 = pd.DataFrame(self.MLT_DEFAULTS, index=feats.index[idx0]).astype(self.MLT_COLUMNS)
  #     self.multi = pd.concat([self.multi,df0])

  #   self.multi = self.multi.join(feats).reindex(columns=MLT_FEAT.keys()).astype(MLT_FEAT)

