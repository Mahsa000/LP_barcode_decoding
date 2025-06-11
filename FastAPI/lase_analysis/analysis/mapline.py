import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist
import scipy.ndimage as ndi
from skimage.segmentation import watershed
from skimage.color import hsv2rgb
from skimage.measure import label as ski_label, regionprops
from sklearn.cluster import SpectralClustering

import matplotlib.pyplot as plt
import matplotlib

import warnings

from ..c_code.utils import pks2limgs
from ..utils.functions import ravel, unravel, condensed_to_square
from ..utils.constants import K_nm2meV as KE, COLS


def _format_axis(ax):
  ax.axes.xaxis.set_visible(False)
  ax.axes.yaxis.set_visible(False)
  ax.axis('equal')

def _find_label(x):
  t = x[x>0]
  if len(t) == 0: return 0
  m = np.median(t)
  return 0 if m == 1.5 else int(m)

# ------------------------------------------------------------------------------
# -------------------------------- LINE CLASSES --------------------------------
# ------------------------------------------------------------------------------

class MapLine:
  FWHME0 = 0.615

  def __init__(self, pks, lns, init=False):
    self.pks = pks
    self.lns = lns

    self.imgA = None
    self.imgE = None
    self.pids = None

    if init: self._calc_images()

  def __getattr__(self, key):
    return self.lns[key].values[0]

  def __getitem__(self, key):
    return self.lns[key].values[0]

  def _calc_images(self, crd0=None, crd1=None):
    crds = self.pks[['i','j','k']].values.astype(np.int32)
    Es = self.pks.E.values
    As = (self.pks.ph.values)**0.5
    pids = self.pks.index.values

    pnt0 = np.min(crds, axis=0)
    if crd0 is None:        crd0 = pnt0
    elif np.any(crd0>pnt0): raise ValueError(f'crd0 {crd0} not compatible with data (pnt0 {pnt0})!')

    pnt1 = np.max(crds, axis=0)
    if crd1 is None:        crd1 = pnt1
    elif np.any(crd1<pnt1): raise ValueError(f'crd1 {crd1} not compatible with data (pnt1 {pnt1})!')

    imgE, imgA, pos = pks2limgs(crds, crd0.astype(np.int32), crd1.astype(np.int32), Es, As)
    self.imgE = LineImageE(imgE, crd0[::-1])
    self.imgA = LineImageA(imgA, crd0[::-1])
    self.pids = pd.Series(self.pks.index.values, index=pos)

  @classmethod
  def from_data(cls, ldat, lid, **kwds):
    return cls(ldat.peaks[ldat.peaks.lid==lid], ldat.lines.loc[[lid]], **kwds)

  @classmethod
  def from_peaks(cls, pks, **kwds):
    return cls(pks, cls._line_info_df(pks), **kwds)

  @staticmethod
  def _line_info_df(pks, lid=0):
    E0 = np.average(pks.E, weights=1/np.power(0.1 + np.abs(pks.fwhmE-MapLine.FWHME0), 2))
    return pd.DataFrame({'i': np.average(pks.i, weights=pks.a),
                         'j': np.average(pks.j, weights=pks.a),
                         'k': np.average(pks.k, weights=pks.a),
                         'a': pks.a.sum(),
                         'wl': KE/E0,
                         'dwl': np.std(pks.wl.values),
                         'E': E0,
                         'dE': np.std(pks.E.values),
                         'ph': pks.ph.sum(),
                         'n': pks.shape[0],
                         'peri': pks.i.max()-pks.i.min()+pks.j.max()-pks.j.min(),
                         'mid': 0}, index=pd.Index([lid], dtype=np.uint64)).astype(COLS.LNS)

  @property
  def center_of_mass(self):
    return self.imgA.center_of_mass

  @property
  def lid(self):
    return self.lns.index.values[0]

  @lid.setter
  def lid(self, val):
    self.lns.set_index(pd.Series(np.uint64(val),name='lid'))


  def reframe(self, new0, new1):
    self._calc_images(new0[::-1],new1[::-1])


  def split(self, pids, **kwds):
    return [MapLine.from_peaks(self.pks.loc[pp], **kwds) for pp in pids]


  def hsv(self, mask=None, E0=None, dE=1.5, max_ampl=2000):
    hue = np.full_like(self.imgE, 0.)
    sat = np.full_like(hue, 1.)
    val = np.full_like(hue, 0.)

    mask = ~np.isnan(self.imgE)*(True if mask is None else mask.astype(bool))

    if E0 is None: E0 = 0.5*(np.nanmin(self.imgE)+np.nanmax(self.imgE))

    hue[mask] = np.clip(.83*(0.5*(self.imgE[mask]-(E0-dE))/dE), 0., .83)
    val[mask] = np.clip(self.imgA[mask]/max_ampl, .15, 1.)

    return np.stack([hue,sat,val], axis=-1)

  def rgb(self, **kwds):
    return hsv2rgb(self.hsv(**kwds))

  def plot_img(self, ncols=6, **kwds):
    Nk = self.imgE.shape[0]
    nrows = np.ceil(Nk/ncols).astype(int)
    rgb = self.rgb(**kwds)

    fig, ax = plt.subplots(nrows,ncols)
    if ncols*nrows==1: ax=[ax]
    else:              ax = ax.flatten()
    fig.set_size_inches(5*ncols,4.5*nrows)

    for k in range(Nk):
      ax[k].imshow(rgb[k,:,:,:])
      _format_axis(ax[k])

    plt.show()

  def plot_wls(self, E0=None, dE=1.5):
    fig, ax = plt.subplots()
    fig.set_size_inches(15,6)

    if E0 is None: E0=0.5*(self.pks.E.min()+self.pks.E.max())

    ax.scatter(self.pks.E, self.pks.fwhmE, c=np.log10(self.pks.ph),
               cmap='plasma', s=5, vmin=4., vmax=7., alpha=1.)
    ax.set_xlim(E0-dE, E0+dE)
    ax.set_ylim(0,2.)
    ax.axvline(self.E, ls='--', color='green')
    ax.axvline(self.E-self.dE, ls='--', color='lime')
    ax.axvline(self.E+self.dE, ls='--', color='lime')

    plt.show()



class LineImage:
  NGH_DELTA_2D = [(0,0),(1,0),(-1,0),(0,1),(0,-1)]
  NGH_DELTA_3D = [(0,0,0),(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

  def __init__(self, data, crd0=None, na=np.nan):
    self.dim = len(data.shape)
    if not self.dim in (2,3):
      raise ValueError('Data must be 2- or 3- dimensional!')

    self.data = data
    if crd0 is None: crd0 = np.zeros(self.dim, dtype=int)
    self.crd0 = np.array(crd0)
    self._na = na


  def __getattr__(self, key):
    return getattr(self.data, key)

  def __getitem__(self, key):
    return self.data[key]

  def copy(self):
    return self.__class__(self.data.copy(), self.crd0, self._na)

  @property
  def crd1(self):
    return self.crd0 + self.shape - 1


  def get_closest(self, pnt):
    crds = np.array(np.where(self.data)).T
    return crds[np.argmin(np.sum((crds-pnt)**2, axis=1)**0.5)]

  def get_neighbors(self, pnt, na=None, mask=None, orig=False):
    if   self.dim == 2: return self._get_neighbors2D(pnt,na,mask,orig)
    elif self.dim == 3: return self._get_neighbors3D(pnt,na,mask,orig)

  def _get_neighbors3D(self, pnt, na=None, mask=None, orig=False):
    vals = []
    na = self._na if na is None else na

    for (a,b,c) in LineImage.NGH_DELTA_3D[0 if orig else 1:]:
      nxt = (pnt[0]+a, pnt[1]+b, pnt[2]+c)
      if (nxt[0]<0) or (nxt[0]>=self.shape[0]) or\
         (nxt[1]<0) or (nxt[1]>=self.shape[1]) or\
         (nxt[2]<0) or (nxt[2]>=self.shape[2]):
        vals.append(na)
      else:
        vals.append(self.data[nxt] if ((mask is None) or (mask[nxt])) else na)

    return np.array(vals)

  def _get_neighbors2D(self, pnt, na=None, mask=None, orig=False):
    vals = []
    na = self._na if na is None else na

    for (a,b) in LineImage.NGH_DELTA_2D[0 if orig else 1:]:
      nxt = (pnt[0]+a, pnt[1]+b)
      if (nxt[0]<0) or (nxt[0]>=self.shape[0]) or\
         (nxt[1]<0) or (nxt[1]>=self.shape[1]):
        vals.append(na)
      else:
        vals.append(self.data[nxt] if ((mask is None) or (mask[nxt])) else na)

    return np.array(vals)


  def mask(self, msk):
    if len(msk.shape) == 1:
      dat = self.full(self.prod(self.shape), self._na, dtype=self.data.dtype)
      dat[msk] = self.data.flatten()[msk]
      dat = dat.reshape(self.shape)
    elif msk.shape == self.shape:
      dat = self.data*(msk>0)
      dat[dat==0] = self._na
    elif len(msk.shape) == len(self.shape)-1:
      dat = self.data*(msk>0)[np.newaxis,:,:]
      dat[dat==0] = self._na
    else:
      raise ValueError(f'Shape of mask {msk.shape} not compatible with the image!')

    return self.__class__(dat, crd0=self.crd0)

  def reframe(self, new0, new1):
    new0 = np.array(new0)
    new1 = np.array(new1)
    dat = np.full((new1-new0+1), self._na, dtype=self.data.dtype)
    shift0 = self.crd0-new0

    dat[shift0[0]:shift0[0]+self.shape[0],
        shift0[1]:shift0[1]+self.shape[1],
        shift0[2]:shift0[2]+self.shape[2]] = self.data

    return self.__class__(dat, new0, na=self._na)



class LineImageA(LineImage):
  def __init__(self, data, crd0, na=0.):
    super().__init__(data=data, crd0=crd0, na=na)
    self._cmf = None
    self._cmi = None

  def __add__(self, other):
    if other == 0: return self.copy()
    if not isinstance(other, LineImageA):
      raise TypeError(f'Cannot sum with instance of class {type(other)}!')
    if np.any(self.shape != other.shape) or np.any(self.crd0 != other.crd0):
      raise ValueError('Shape/origin of the two images do not match!')

    return LineImageA(self.data+other.data, crd0=self.crd0)

  def __radd__(self, other):
    if other == 0: return self.copy()
    return other.__add__(self)

  @property
  def center_of_mass(self):
    if self._cmf is None:
      self._cmf = ndi.center_of_mass(self.data)
    return self._cmf

  @property
  def center_pixel(self):
    if self._cmi is None:
      cm = self.center_of_mass
      self._cmi = np.round(cm).astype(int)
      if self.data[tuple(self._cmi)]==0.: self._cmi = self.get_closest(cm)

    return self._cmi

  def threshold(self, val):
    return LineImageA(self.data*(self.data>val), self.crd0)

  def reframe(self, *args, **kwds):
    obj = super().reframe(*args, **kwds)
    obj._cmf = None
    obj._cmi = None
    return obj



class LineImageE(LineImage):
  def __init__(self, data, crd0, na=np.nan):
    super().__init__(data=data, crd0=crd0, na=na)
    self._dists = None

  def distances(self):
    ok = np.where(~np.isnan(self.flatten()))[0]

    dists = []
    for ii, idx in enumerate(ok):
      k,j,i = unravel(idx, self.shape)
      for a,b,c in self.NGH_DELTA_3D[1:]:
        nxt = (k+a,j+b,i+c)
        if (nxt[0]>=0) and (nxt[0]<self.shape[0]) and \
           (nxt[1]>=0) and (nxt[1]<self.shape[1]) and \
           (nxt[2]>=0) and (nxt[2]<self.shape[2]) and\
           ~np.isnan(self[nxt]):
            dists.append((idx, np.abs(self[k,j,i]-self[nxt]), ravel(nxt, self.shape)))

    self._dists = np.array(dists,dtype=np.dtype([('idx',int),('dE',float),('nxt',int)]))
    return self._dists[np.argsort(self._dists['dE'])]



class LineSplitter:
  def __init__(self, opt):
    self.opt = opt
    self._SptClst = SpectralClustering(n_clusters=2)


    self._thr = 10.
    self._solid_left = .66
    self._watershed_line = True
    self._dE_split = .2

  def __getattr__(self, key):
    return self.opt[key]

  # ------------------------------ Line splitting ------------------------------

  def find_splits(self, ln, splits=None):
    splits = None
    regions = None
    Amasks = []
    Emasks = []

    # # ---------- Find splittable regions (by wavelength) ----------
    # if ln.dE>self._dE_split:
    #   labs = self._SptClst.fit(ln.pks[['E','fwhmE']].values).labels_
    #   simg = np.zeros_like(ln.imgA.data)
    #   for ii in range(2):
    #     pks = ln.pks.iloc[labs==ii]
    #     pks = pks[pks.ph>pks.ph.max()/10.]
    #     E = pks.E.mean()
    #     dE = pks.E.std()
    #     pks = pks[(pks.E>E-dE)&(pks.E<E+dE)]
    #     pos = ln.pids[ln.pids.isin(pks.index.values)].index.values
    #     for pp in pos: simg[unravel(pp, ln.imgA.data.shape)] = ii+1
    #
    #   splits = np.apply_along_axis(_find_label, axis=0, arr=simg)
    #   if np.max(ski_label(splits))>2: splits[splits>0] = 1

    if (splits is None) or (np.max(splits)==1):
      # ---------- Check large mass shift ----------
      if not self.mass_shift is None:
        cms = np.array([ndi.center_of_mass(pln) for pln in ln.imgA.data if np.any(pln)])
        dcms = cms[0]-cms[-1]
        if (np.sum(dcms**2))**0.5 > self.mass_shift:
          return SplitAnalysis(splits, regions, Amasks, Emasks)

      # ---------- Find splittable regions (by intensity) ----------
      imgs = ln.imgA.mean(axis=0)
      imgs[imgs<np.max(imgs)/self._thr] = 0.

      sigmas = self.lapl_sigma
      try:    iter(sigmas)
      except: sigmas = [sigmas]

      for sig in sigmas:
        lapl = -ndi.gaussian_laplace(imgs, sigma=sig)
        lapl[lapl<0] = 0.
        splits = watershed(-lapl, mask=lapl>0, watershed_line=self._watershed_line)
        if np.max(splits)>1: break

    # ---------- Check regions ----------
    todel = []
    Es = []
    inns = ~self._find_borders(splits)
    for ll in range(1,np.max(splits)+1):
      msk = (splits==ll)*inns
      if np.sum(msk) < 3: msk = (splits==ll)
      Amsk = ln.imgA.mask(msk)

      if Amsk.sum() == 0.:
        todel.append(ll)
        continue

      Amasks.append(Amsk)
      Emasks.append(ln.imgE.mask(msk))
      try:    neigh = Emasks[-1].get_neighbors(Amsk.center_pixel, na=np.nan, orig=True)
      except: neigh = np.full(6, np.nan)
      if np.all(np.isnan(neigh)): neigh = Emasks[-1].data.flatten()
      Es.append(np.nanmean(neigh))

    for ll in np.sort(todel)[::-1]:
      splits[splits==ll] = 0
      splits[splits>ll] -= 1

    Nr = len(Amasks)
    if Nr <= 1: return SplitAnalysis(splits, regions, Amasks, Emasks)

    # ---------- Look for regions close in space and energy ----------
    dists = pdist(np.array([msk.center_of_mass for msk in Amasks])[:,1:], metric='euclidean')
    isort = np.argsort(dists)

    merge = []
    for idx in isort:
      if dists[idx] > self.dr_close: break
      row,col = condensed_to_square(idx, Nr)
      if np.abs(Es[row]-Es[col]) > self.dE_close: continue

      for ii, mrg in enumerate(merge):
        if (row+1 in mrg):
          merge[ii].append(col+1)
          break
        elif (col+1 in mrg):
          merge[ii].append(row+1)
          break
      else:
        merge.append([row+1,col+1])

    # ---------- Merge eventual regions ----------
    if len(merge) > 0:
      new_idx = Nr+1
      for mrg in merge:
        splits[np.isin(splits, mrg)] = new_idx
        Amasks.append(ln.imgA.mask(splits==new_idx))
        Emasks.append(ln.imgE.mask(splits==new_idx))
        new_idx += 1

      for mm in np.sort(np.concatenate(merge))[::-1]:
        del Amasks[mm-1]
        del Emasks[mm-1]

      for ii, idx in enumerate(np.sort(np.unique(splits))):
        splits[splits==idx] = ii

    if np.max(splits)<=1: return SplitAnalysis(splits, regions, Amasks, Emasks)

    dists = ln.imgE.distances()

    # ---------- Find regions ----------
    while True:
      seeds = []
      for ll in range(len(Amasks)):
        Amsk = Amasks[ll].threshold(Amasks[ll].max()/self._thr)
        pos = np.where(Amsk.flatten())[0]

        sds = [Amasks[ll].center_pixel]
        for a,b,c in LineImage.NGH_DELTA_3D[1:]:
          nxt = (sds[0][0]+a,sds[0][1]+b,sds[0][2]+c)
          if (nxt[0]>=0) and (nxt[0]<Amsk.shape[0]) and \
             (nxt[1]>=0) and (nxt[1]<Amsk.shape[1]) and \
             (nxt[2]>=0) and (nxt[2]<Amsk.shape[2]) and\
             Amsk[nxt]!=0.:
              sds.append(nxt)

        sarr = [[val for val in [ravel(ss, Amsk.shape) for ss in sds]
                if val in ln.pids.index.values]]

        cdst = dists[np.isin(dists['idx'],pos)&np.isin(dists['nxt'],pos)]
        seeds.append(self._walk_by_E(cdst, sarr, self.dE_seed)[0])

      regions = self._walk_by_E(dists, seeds, self.dE_reg)

      # ---------- Check small regions to remove ----------
      todel = [ll for ll,reg in enumerate(regions) if len(reg) < self.area_small]
      if len(todel) == 0: break

      for ll in np.sort(todel)[::-1]:
        del Amasks[ll]
        del Emasks[ll]
        splits[splits==ll+1] = 0

    if len(regions) <= 1: return SplitAnalysis(splits, regions, Amasks, Emasks)

    # ---------- Check for left-out pixels ----------
    left = (ln.imgA.data>0).flatten()
    left[np.concatenate(regions)] = 0
    if np.sum(left) == 0: return SplitAnalysis(splits, regions, Amasks, Emasks)

    left_idx = np.where(left)[0]
    left = left.reshape(ln.imgA.shape)

    # ---------- Look for independent regions ----------
    lab = ski_label(left)
    N = np.max(lab)

    for ll in range(N):
      mask = lab==ll+1
      prp = regionprops(mask.astype(np.uint8))[0]
      area = prp.area
      if area < self.area_left: continue
      bbox = prp.bbox
      flat = False
      for ii in range(3):
        if bbox[ii+3]-bbox[ii]>1: continue
        prp = regionprops(mask[bbox[0]:bbox[3],bbox[1]:bbox[4],bbox[2]:bbox[5]]\
                          .astype(np.uint8).squeeze())[0]
        flat = True
        break

      Es = ln.imgE*mask
      Es = Es[Es>0]
      Es = Es[~np.isnan(Es)]
      dE = np.std(Es)

      if (prp.solidity>self._solid_left) and (dE<self.dE_left):
        regions.append(np.where(mask.flatten())[0])

    left_idx = np.setdiff1d(left_idx, np.concatenate(regions))

    # ---------- Assign remaining peaks ----------
    if len(left_idx) > 0:
      regions = self._walk_by_E(dists, regions, max_dE=np.inf)

    return SplitAnalysis(splits, regions, Amasks, Emasks)

  @staticmethod
  def _find_borders(splt):
    brd = np.zeros_like(splt, dtype=bool)
    splt = LineImage(splt, (0,0))
    for row in range(splt.shape[0]):
      for col in range(splt.shape[1]):
        if splt[row,col]==0.: continue
        ngh = np.array(splt.get_neighbors((row,col), na=0))
        brd[row,col] = np.any((ngh!=0)&(ngh!=splt[row,col]))

    return brd

  @staticmethod
  def _walk_by_E(dists, seeds, max_dE=.25):
    dists = dists[~np.isin(dists['nxt'],np.concatenate(seeds))]
    dists = dists[dists['dE']<max_dE]

    bucks = [list(sd) for sd in seeds]

    while True:
      for val in dists:
        for ib, bck in enumerate(bucks):
          if val['idx'] in bck: break
        else:
           continue

        bucks[ib].append(val['nxt'])
        dists = dists[~(dists['nxt']==val['nxt'])]
        break
      else:
        break

    return bucks



class SplitAnalysis:
  def __init__(self, splits, regions, Amasks, Emasks):
    self.splits = splits
    self.regions = regions
    self.Amasks = Amasks
    self.Emasks = Emasks

  @property
  def n(self):
    return 1 if self.regions is None else len(self.regions)

  def split_pids(self, ln):
    return [ln.pids.values] if self.regions is None else\
           [ln.pids.loc[rr].values for rr in self.regions]

# class LineRefiner:
#   def __init__(self, line):
#     self.line = line
#
#     # self.imgA = imgA
#     # self.imgE = imgE
#     # self.pids = pids
#     # self.crd0 = crd0
#     #
#     # self._E0 = 0.5*(np.nanmin(imgE)+np.nanmax(imgE))
#     self.lapl = None
#
#     self.splits = None
#     self.dists = None
#
#     self.Amasks = None
#     self.Emasks = None
#     self.seeds = None
#     self.regions = None
#     #
#     # self.splittable = False
#     # self.additional = False
#
#   @property
#   def nregs(self):
#     return len(self.Amasks)
#
#   @property
#   def imgA(self):
#     return self.line.imgA
#
#   @property
#   def imgE(self):
#     return self.line.imgE
#
#   # -------------------------- Splits identification ---------------------------
#
#   def check_large_mass_shift(self, max_dr=4.5):
#     cms = np.array([ndi.center_of_mass(pln) for pln in self.imgA.data])
#     dcms = cms[0]-cms[-1]
#     return (np.sum(dcms**2))**0.5 > max_dr
#
#   def find_splits(self, sig=1.3, max_dist=3., max_dE=.3):
#     # ---------- Find splittable regions ----------
#     imgs = self.imgA.mean(axis=0)
#     imgs[imgs<np.max(imgs)/10.] = 0.
#
#     self.lapl = -ndi.gaussian_laplace(imgs, sigma=sig)
#     self.lapl[self.lapl<0] = 0.
#     self.splits = watershed(-self.lapl, mask=self.lapl>0, watershed_line=True)
#
#     # ---------- Check regions ----------
#     self.Amasks = []
#     self.Emasks = []
#     todel = []
#     Es = []
#     inns = ~self._find_borders(self.splits)
#
#     for ll in range(1,np.max(self.splits)+1):
#       msk = (self.splits==ll)*inns
#       if np.sum(msk) < 3: msk = (self.splits==ll)
#       Amsk = self.imgA.mask(msk)
#
#       if Amsk.sum() == 0.:
#         todel.append(ll)
#         continue
#
#       self.Amasks.append(Amsk)
#       self.Emasks.append(self.imgE.mask(msk))
#       Es.append(np.nanmean(self.Emasks[-1].get_neighbors(Amsk.center_pixel, na=np.nan, orig=True)))
#
#     for ll in np.sort(todel)[::-1]:
#       self.splits[self.splits==ll] = 0
#       self.splits[self.splits>ll] -= 1
#
#     # ---------- Look for regions close in space and energy ----------
#     Nr = len(self.Amasks)
#     dists = pdist(np.array([msk.center_of_mass for msk in self.Amasks])[:,1:], metric='euclidean')
#     isort = np.argsort(dists)
#
#     merge = []
#     for idx in isort:
#       if dists[idx] > max_dist: break
#
#       row,col = condensed_to_square(idx, Nr)
#       if np.abs(Es[row]-Es[col]) > max_dE: continue
#
#       for ii, mrg in enumerate(merge):
#         if (row+1 in mrg):
#           merge[ii].append(col+1)
#           break
#         elif (col+1 in mrg):
#           merge[ii].append(row+1)
#           break
#       else:
#         merge.append([row+1,col+1])
#
#     if len(merge) > 0:
#       new_idx = Nr+1
#       for mrg in merge:
#         self.splits[np.isin(self.splits, mrg)] = new_idx
#         self.Amasks.append(self.imgA.mask(self.splits==new_idx))
#         self.Emasks.append(self.imgE.mask(self.splits==new_idx))
#         new_idx += 1
#
#       for mm in np.sort(np.concatenate(merge))[::-1]:
#         del self.Amasks[mm-1]
#         del self.Emasks[mm-1]
#
#       for ii, idx in enumerate(np.sort(np.unique(self.splits))):
#         self.splits[self.splits==idx] = ii
#
#   @staticmethod
#   def _find_borders(splt):
#     brd = np.zeros_like(splt, dtype=bool)
#     splt = LineImage(splt, (0,0))
#     for row in range(splt.shape[0]):
#       for col in range(splt.shape[1]):
#         if splt[row,col]==0.: continue
#         ngh = np.array(splt.get_neighbors((row,col), na=0))
#         brd[row,col] = np.any((ngh!=0)&(ngh!=splt[row,col]))
#
#     return brd
#
#   # ---------------------------- Regions formation -----------------------------
#
#   def find_regions(self, thr=10., seed_dE=.25, reg_dE=.5, min_area=6, left_area=10,
#                    min_solid=.66, left_dE=.33):
#     dists = self.imgE.distances()
#
#     while True:
#       # ---------- Find regions ----------
#       self.seeds = []
#       for ll in range(self.nregs):
#         Amsk = self.Amasks[ll].threshold(self.Amasks[ll].max()/thr)
#         pos = np.where(Amsk.flatten())[0]
#
#         sds = [self.Amasks[ll].center_pixel]
#         for a,b,c in LineImage.NGH_DELTA_3D[1:]:
#           nxt = (sds[0][0]+a,sds[0][1]+b,sds[0][2]+c)
#           if (nxt[0]>=0) and (nxt[0]<Amsk.shape[0]) and \
#              (nxt[1]>=0) and (nxt[1]<Amsk.shape[1]) and \
#              (nxt[2]>=0) and (nxt[2]<Amsk.shape[2]) and\
#              Amsk[nxt]!=0.:
#               sds.append(nxt)
#
#         sarr = [[val for val in [ravel(ss, Amsk.shape) for ss in sds]
#                 if val in self.line.pids.index.values]]
#
#         cdst = dists[np.isin(dists['idx'],pos)&np.isin(dists['nxt'],pos)]
#         self.seeds.append(self.walk_by_E(cdst, sarr, seed_dE)[0])
#
#       self.regions = self.walk_by_E(dists, self.seeds, reg_dE)
#
#       # ---------- Check regions ----------
#       todel = []
#       for ll, reg in enumerate(self.regions):
#         if len(reg) >= min_area: continue
#         todel.append(ll)
#
#       if len(todel) == 0: break
#
#       for ll in np.sort(todel)[::-1]:
#         del self.Amasks[ll]
#         del self.Emasks[ll]
#         self.splits[self.splits==ll+1] = 0
#
#     left = (self.imgA.data>0).flatten()
#     left[np.concatenate(self.regions)] = 0
#     if np.sum(left) == 0: return
#     left_idx = np.where(left)[0]
#     left = left.reshape(self.imgA.shape)
#
#     # ---------- Look for independent regions ----------
#     lab = ski_label(left)
#     N = np.max(lab)
#
#     for ll in range(N):
#       mask = lab==ll+1
#       prp = regionprops(mask.astype(np.uint8))[0]
#       area = prp.area
#       if area < left_area: continue
#       bbox = prp.bbox
#       flat = False
#       for ii in range(3):
#         if bbox[ii+3]-bbox[ii]>1: continue
#         prp = regionprops(mask[bbox[0]:bbox[3],bbox[1]:bbox[4],bbox[2]:bbox[5]]\
#                           .astype(np.uint8).squeeze())[0]
#         flat = True
#         break
#
#       Es = self.imgE*mask
#       Es = Es[Es>0]
#       Es = Es[~np.isnan(Es)]
#       dE = np.std(Es)
#
#       if (prp.solidity>min_solid) and (dE<left_dE):
#         self.regions.append(np.where(mask.flatten())[0])
#
#     left_idx = np.setdiff1d(left_idx, np.concatenate(self.regions))
#
#     # ---------- Assign remaining peaks ----------
#     if len(left_idx) == 0: return
#     self.regions = self.walk_by_E(dists, self.regions, max_dE=np.inf)
#
#   def check_split(self, max_dr=4.):
#     if len(self.regions)<2: return None
#
#     cms = np.array([ndi.center_of_mass(pln) for pln in self.imgA.data])
#     dcms = cms[0]-cms[-1]
#     dr = (np.sum(dcms**2))**0.5
#
#     if dr<max_dr: return True
#     else:         return False
#
#
#   def get_split_pids(self):
#
#   def get_new_base(self):
#     return pd.concat([pd.Series(np.uint64((ii+1)<<32),
#                                 index=self.line.pids.loc[reg].values)
#                       for ii,reg in enumerate(self.regions)])
#
#   def get_new_lids(self):
#     return {np.uint64((ii+1)<<32)+(IDX_MASK&self.line.lid):\
#             self.line.pids.loc[reg].values for ii,reg in enumerate(self.regions)}
#
#   def get_new_lns(self):
#     nls = self.get_new_lids()
#     lns = [LaseData._line_info_df(self.line.pks.loc[pp], self.line.fwhmE0, ll)
#             for ll,pp in nls.items()]
#     for ll in lns: ll.mid = self.line.mid
#     return lns
#
#   @staticmethod
#   def _calc_dists(imgE):
#     ok = np.where(~np.isnan(imgE.flatten()))[0]
#
#     dists = []
#     for ii, idx in enumerate(ok):
#       k,j,i = unravel(idx, imgE.shape)
#       for a,b,c in LineImage.NGH_DELTA_3D[1:]:
#         nxt = (k+a,j+b,i+c)
#         if (nxt[0]>=0) and (nxt[0]<imgE.shape[0]) and \
#            (nxt[1]>=0) and (nxt[1]<imgE.shape[1]) and \
#            (nxt[2]>=0) and (nxt[2]<imgE.shape[2]) and\
#            ~np.isnan(imgE[nxt]):
#             dists.append((idx, np.abs(imgE[k,j,i]-imgE[nxt]), ravel(nxt, imgE.shape)))
#
#     dists = np.array(dists,dtype=np.dtype([('idx',int),('dE',float),('nxt',int)]))
#     return dists[np.argsort(dists['dE'])]
#
#   @staticmethod
#   def walk_by_E(dists, seeds, max_dE=.25):
#     dists = dists[~np.isin(dists['nxt'],np.concatenate(seeds))]
#     dists = dists[dists['dE']<max_dE]
#
#     bucks = [list(sd) for sd in seeds]
#
#     while True:
#       for val in dists:
#         for ib, bck in enumerate(bucks):
#           if val['idx'] in bck: break
#         else:
#            continue
#
#         bucks[ib].append(val['nxt'])
#         dists = dists[~(dists['nxt']==val['nxt'])]
#         break
#       else:
#         break
#
#     return bucks
#
#
#   # ---------------------------- Plotting functions ----------------------------
#
#   def plot_Es(self, **kwds):
#     rgb = self.rgb(**kwds)
#
#     fig, ax = plt.subplots(1,self.shape[0])
#     if self.shape[0] == 1: ax=[ax]
#     fig.set_size_inches(5*self.shape[0],4)
#
#     for kk in range(len(ax)):
#       ax[kk].imshow(rgb[kk,:,:,:])
#       self._format_axis(ax[kk])
#
#     plt.show()
#
#   def plot_splits(self):
#     splt = self.splits
#     Nr = np.max(splt)
#
#     fig, ax = plt.subplots(1,3)
#     fig.set_size_inches(15,4)
#
#     ax[0].imshow(self._lapl, cmap='hot', vmin=0, vmax=np.max(self._lapl))
#     ax[1].imshow(splt, cmap='hot', vmin=0, vmax=np.max(splt))
#     ax[2].imshow(self._find_borders(splt), cmap='hot', vmin=0, vmax=np.max(splt))
#     for a in ax: self._format_axis(a)
#
#     # if not self.centers is None:
#     #   for ctr in self.centers: ax[1].scatter(ctr[2],ctr[1],s=100,color='gray')
#
#     plt.show()
#
#   def plot_seeds(self):
#     Ns = len(self.seeds)
#
#     fig, ax = plt.subplots(Ns,self.shape[0])
#     fig.set_size_inches(5*self.shape[0],4*Ns)
#
#     for ii, sds in enumerate(self.seeds):
#       seed_mask = np.zeros(np.prod(self.shape), dtype=int)
#       seed_mask[sds] = 1
#       seed_mask = seed_mask.reshape(self.shape)
#       rgb = self.rgb(mask=seed_mask)
#
#       for kk, pln in enumerate(rgb):
#         ax[ii,kk].imshow(pln)
#         self._format_axis(ax[ii,kk])
#
#     plt.show()
#
#   def plot_regions(self):
#     fig, ax = plt.subplots(self.nregs,self.shape[0])
#     fig.set_size_inches(5*self.shape[0],4*self.nregs)
#
#     for ii, reg in enumerate(self.regions):
#       mask = np.zeros(np.prod(self.shape), dtype=int)
#       mask[reg] = 1
#       mask = mask.reshape(self.shape)
#       rgb = self.rgb(mask=mask)
#
#       for kk, pln in enumerate(rgb):
#         ax[ii,kk].imshow(pln)
#         self._format_axis(ax[ii,kk])
#
#     plt.show()
#
#   def plot_left(self):
#     used = np.concatenate(self.regions)
#
#     mask = ~np.isnan(self.imgE.flatten())
#     mask[used] = False
#     mask = mask.reshape(self.shape)
#     rgb = self.rgb(mask=mask)
#
#     fig, ax = plt.subplots(1,self.shape[0])
#     fig.set_size_inches(5*self.shape[0],4)
#     for kk, pln in enumerate(rgb):
#       ax[kk].imshow(pln)
#       self._format_axis(ax[kk])
#     plt.show()
