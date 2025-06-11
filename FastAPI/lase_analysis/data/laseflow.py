import numpy as np
import pandas as pd

from copy import deepcopy

from .lasebase import LaseData
from ..utils.constants import K_nm2meV as KE
# from ..files.flow import FileFCS


class LaseData_Flow(LaseData):
  def __init__(self, *args, **kwds):
    super().__init__(*args, **kwds)


class LaseData_Flow_Guava(LaseData_Flow):
  FWHME0 = 0.995
  PKS_COLUMNS = {'t': float, 'a': float, 'wl': float, 'fwhm': float, 'E': float,
                 'fwhmE': float, 'ph': float, 'ispt': int, 'lid': np.uint64}
  LNS_COLUMNS = {'t': float, 'dt': float, 'a': float, 'wl': float, 'dwl': float,
                 'E': float, 'dE': float, 'ph': float, 'n': int, 'mid': np.uint64}
  MLT_COLUMNS = {'t': float, 'dt': float, 'n': int}
  MLT_DEFAULTS = {'t': np.nan, 'dt': np.nan, 'n': 0}

  def __init__(self, *args, **kwds):
    super().__init__(*args, **kwds)

    if 'info' in kwds: self.info = kwds['info']

  def split(self, docopy=False):
    out = {}
    for info in self.info.split():
      out[info.name] = self.subset(mids=self.mlt.index[(self.mlt.t>=info.t0)&(self.mlt.t<info.t1)],
                                   fids=self.fts.index[(self.fts.t>=info.t0)&(self.fts.t<info.t1)], docopy=docopy)
      out[info.name].info = info
    return out

  def copy(self):
    obj = super().copy()
    obj.info = deepcopy(self.info)
    return obj

  def subset(self, mids, fids=None, name=None, docopy=False):
    obj = super().subset(mids, fids, name, docopy)
    obj.info = deepcopy(self.info)
    # obj.mid2fid = self.mid2fid.loc[mids]
    return obj

class LaseData_Flow_LaseV3(LaseData_Flow):
  def __init__(self, *args, **kwds):
    super().__init__(*args, **kwds)

  # @classmethod
  # def from_csv(cls, fpath, wl_axis, ncols=30):
  #   ldf = pd.read_csv(fpath)
  #   ldf.ID = np.uint64(ldf.ID+1000000*ldf.ID1)
  #   mid_dat = ldf.ID.values
  #   n_dat = ldf.Count.values
  #   wl_dat = ldf[[f'H_{ii}' for ii in range(ncols)]].values
  #   a_dat = ldf[[f'X_{ii}' for ii in range(ncols)]].values

  #   dat_lns = []
  #   dat_mlt = []
  #   for mid,n,wls,As in zip(mid_dat,n_dat,wl_dat,a_dat):
  #     cn = min(n,ncols)
  #     dat_lns.extend([(wls[ii],As[ii],mid) for ii in range(cn)])
  #     dat_mlt.append((mid,cn))

  #   lns = pd.DataFrame(dat_lns, columns=['wl','a','mid'])
  #   mlt = pd.DataFrame(dat_mlt, columns=['mid','n']).set_index('mid')

  #   lns.wl = np.interp(lns.wl, np.arange(2048), wl_axis)
  #   lns['E'] = KE/lns.wl

  #   return cls(mlt,lns,None,None,None)

  # def import_fluo_csv(self, fpath, rename=None, **kwds):
  #   df = pd.read_csv(fpath, **kwds)
  #   if not rename is None:
  #     df = df[rename.keys()]
  #     df.rename(columns=rename, inplace=True)

  #   df = df[df.ID1>=0]
  #   df['ID'] = np.uint64(df['ID']+1e6*df['ID1'])
  #   df = df.drop(columns='ID1').astype({'nLP': int}).set_index('ID')

  #   self.fts = df

  # def import_fluo_fcs(self, fpath, fpath_comp=None, colname='name'):
    # fcs = FileFCS(fpath)
    # df = fcs.df(colname)

    # if not fpath_comp is None:
    #   fcsC = FileFCS(fpath_comp)
    #   dfC = fcsC.df(colname)
    #   df = df[['ID','ID1','Time','LP_Count_Cutoff','LP_Count']]

    #   dfC = dfC[[lab for lab in dfC.columns if not lab in df.columns]]
    #   df = pd.concat([df,dfC],axis=1)

    # df = df[df.ID1>=0]
    # df['ID'] = np.uint64(df['ID']+1e6*df['ID1'])
    # df = df.drop(columns=['ID1', 'LP_Count']).rename(columns={'LP_Count_Cutoff': 'nLP'})\
    #        .astype({'nLP': int}).set_index('ID')

    # self.fts = df