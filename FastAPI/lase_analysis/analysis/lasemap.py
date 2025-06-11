import numpy as np
import pandas as pd

from scipy.spatial.distance import squareform
from sklearn.cluster import dbscan

from joblib import Parallel, delayed
from os import cpu_count
from itertools import combinations
from time import time
from copy import deepcopy

from .mapline import MapLine, LineSplitter
from ..data import LaseData_Map_Confocal
from ..data.spectrum import Spectrum, PeakFitOpts
# from ..matching.scoring import find_dEs
from ..utils.logging import logger
from ..utils.constants import LIDS, MIDS, COLS, SDTP, K_nm2meV as KE
# from ..utils.unionfind import UnionFind


class LaseAnalysis_Map:
  FWHME0 = 0.615

  def __init__(self, lfile, gname, pks, lns, mlt):
    self.lfile = lfile
    self.name = gname
    
    self.pks = pks
    self.lns = lns
    self.mlt = mlt

    self.info = deepcopy(lfile.info[self.name])

  @property
  def peaks(self):
    return self.pks

  @property
  def lines(self):
    return self.lns

  @property
  def multi(self):
    return self.mlt

  def save_analysis(self, analysis='base', overwrite=False):
    self.lfile.save_analysis(self, analysis, overwrite)

  def get_data(self, peaks=True):
    if (self.lns is None) or (self.mlt is None):
      raise RuntimeError('Analysis does not contain lines and/or multi data, cannot return data class!')
    
    pks = self.peaks[self.peaks.lid.isin(self.lns.index.values)] if peaks else None
    return LaseData_Map_Confocal(multi=self.mlt, lines=self.lns, peaks=pks, name=self.name, feats=None)

  # def save_lite(self, fpath=None, peaks=True):
    # self.lfile.save_lite(self, fpath, peaks)

  # ------------------------------ Peaks analysis ------------------------------

  def find_peaks(self, popt=None, **kwds):
    """
    Analyze spectra in the .map file to extract information about peaks.

    Args:
      popt: PeakFitOpt object to specify parameters of peak fitting. If None, create
      a new object.

      kwds: additional parameters to initialize the PeakFitOpt object (if opt is None);
      can be a combination of 'med_filter', 'threshold', 'window', 'bounds' and
      'saturation' (see Spectrum reference for more details).
    """
    if not self.name in self.lfile.info: raise RuntimeError(f'Group {self.name} not present in the map file!')

    logger.info(f'---------- Find peaks ----------')
    t0 = time()
    if popt is None: popt = PeaksOpts(**kwds)

    ginfo = self.lfile.info[self.name]
    with self.lfile.read() as f:
      nspectra, npixels = f['data']['spectra'].shape
      x = f['data']['wl_axis'][:].astype(np.float64)
      gidxs = f['data']['coordinates'][:,4]

    cidx = np.where(gidxs==ginfo.id)[0]
    nn = len(cidx)

    # Create list of chunks indexes
    idx_list = [cidx] if popt.chunk is None else\
               [cidx[np.arange(ii*popt.chunk, min((ii+1)*popt.chunk, nn))]
                for ii in range(np.ceil(nn/popt.chunk).astype(int))]

    ffit = LaseAnalysis_Map._analyze_spt if popt.method==1 else LaseAnalysis_Map._analyze_spt2
    clst = int(1.5*popt.window/(x[npixels//2]-x[npixels//2-1]))
    fopt = [popt.med_filter, popt.threshold, popt.window, self.FWHME0, popt.gain, popt.saturation] if popt.method==1 else\
           [popt.med_filter, popt.threshold, popt.prominence, popt.distance, clst, popt.saturation, popt.window, self.FWHME0, popt.gain]
    pool = Parallel(cpu_count()-2) if popt.pool is True else popt.pool

    fit = []
    logger.info(f'Analyze spectra, method #{popt.method} [t={time()-t0:.3f}s]')
    for ic, idx in enumerate(idx_list):
      logger.info(f'.. Analyzing chunk {ic+1}/{len(idx_list)} [t={time()-t0:.3f}s]')
      with self.lfile.read() as f:
        spectra = f['data']['spectra'][idx,:].astype(np.float64)

      if pool: ret = pool(delayed(ffit)(x,y,idx[ii],*fopt) for ii, y in enumerate(spectra))
      else:    ret = [ffit(x,y,idx[ii],*fopt) for ii,y in enumerate(spectra)]

      # Merge all peaks from analysis in a single dataframe and sort based on ispt
      if len(ret)>0:
        fit.append(pd.DataFrame(np.concatenate(ret), columns=COLS.FIT)\
                    .dropna(axis=0, how='any').astype({'ispt': int}))

    if len(fit) == 0:
      pks = None
    else:
      fit = pd.concat(fit, axis=0, ignore_index=True)\
            .astype({'a': np.float32, 'wl': np.float32, 'fwhm': np.float32, 'ph': np.float32})
      # Check duplicates
      hsh = self._pks_hash(fit)
      uhs,cnt = np.unique(hsh, return_counts=True)
      if np.any(cnt>1):
        drop = np.concatenate([np.where(hsh==uu)[0][1:] for uu in uhs[cnt>1]])
        fit = fit.drop(index=fit.index[drop]).reset_index(drop=True)
      # Add coordinates, lids and energy values
      crd = self._merged_coordinates(fit.ispt.values)
      lid = pd.Series(LIDS.NON, index=fit.index, name='lid', dtype=np.uint64)
      pks = pd.concat([crd, fit, lid], axis='columns')
      pks['E'] = (KE/pks['wl']).astype(np.float32)
      pks['fwhmE'] = (pks['fwhm']*pks['E']/pks['wl']).astype(np.float32)
      # Sort and reindex
      pks = pks.sort_values(['ispt','E']).reset_index(drop=True)
      pks.index = pks.index.rename('pid').astype(np.uint64)
      pks = pks.reindex(columns=list(COLS.MPKS)).astype(COLS.MPKS)

    self.pks = pks

    logger.info(f'Done! [t={time()-t0:.3f}s]')

  @staticmethod
  def _analyze_spt(x, y, ispt, medflt, thr, wdw, fwhm0, gain, sat):
    y,_ = Spectrum.spt_filter(y, medflt)
    midx = Spectrum.spt_maxima(y, thr)
    return Spectrum.spt_fit(x,y,ispt,midx, wdw,fwhm0,gain,sat)

  @staticmethod
  def _analyze_spt2(x, y, ispt, medflt, thr, prm, dst, clst, sat, wdw, fwhm0, gain):
    y,_ = Spectrum.spt_filter(y, medflt)
    midx = Spectrum.spt_maxima2(y, thr, prm, dst, clst, sat)
    return Spectrum.spt_fit2(x,y,ispt,midx, wdw,fwhm0,gain,sat)

  def _merged_coordinates(self, ispt):
    with self.lfile.read() as f:
      crd = f['data']['coordinates'][:,:]
    crd = crd[ispt,:]
    gidx = np.unique(crd[:,4])
    if len(gidx) > 1: raise RuntimeError('Bad coordinates grouping (>1 groups)')
    gidx = gidx[0]

    info = self.lfile.info[self.name]
    if info.areas > 1:
      gmap = info.gmap
      crd[:,0] += gmap[crd[:,3],1]*info.scan_area[0]
      crd[:,1] += gmap[crd[:,3],0]*info.scan_area[1]

    return pd.DataFrame(crd[:,:3], columns=['i','j','k'], dtype=int)

  def pks_hash(self):
    return self._pks_hash(self.pks)

  @staticmethod
  def _pks_hash(pks):
    return (pks.ispt.values.astype(np.uint32).astype(np.uint64) << 32) + \
           (pks.wl.values.astype(np.float32).view(np.uint32).astype(np.uint64))

  # ------------------------------ Lines analysis ------------------------------

  def find_lines(self, lopt=None, **kwds):
    logger.info(f'---------- Find lines ----------')
    if lopt is None: lopt = LinesOpts(**kwds)

    self.lns = self._empty_lns()
    self.mlt = None
    if len(self.pks) == 0: return

    t0 = time()
    # --- Filtering step -------------------------------------------------------
    logger.info(f'.. filter peaks [t0=0s]')
    flt = pd.Series(False, index=self.pks.index) if lopt.bad is None else lopt.bad
    for key, val in lopt.filter():
      flt = flt | (self.pks[key] < val[0]) | (self.pks[key] > val[1])

    self.pks.loc[flt, 'lid'] = LIDS.FLT
    # fpks: remaining peaks in the group after filtering noise
    fpks = self.pks.loc[~flt]
    logger.info(f'.. .. filtered peaks: {len(fpks)}/{len(self.pks)} [t={time()-t0:.3f}s]')
    if len(fpks.index) == 0: return

    # --- Clustering step ------------------------------------------------------
    logger.info(f'.. cluster peaks [t={time()-t0:.3f}s]')
    n_jobs = cpu_count()-2 if lopt.pool is True else None
    _, ids = dbscan(fpks[['i','j','k','E']].to_numpy()/lopt.scale[None,:],
                    eps=lopt.eps, min_samples=lopt.min_samples, n_jobs=n_jobs)
    isok = ids>=0
    self.pks.loc[fpks.index[isok], 'lid'] = np.uint64(ids[isok])
    self.pks.loc[fpks.index[~isok], 'lid'] = LIDS.UNC
    logger.info(f'.. .. clustered peaks: {len(np.unique(ids[isok]))} clusters found | '+\
                f'{np.sum(~isok)} unclustered peaks [t={time()-t0:.3f}s]')

    # --- Calculation step -----------------------------------------------------
    logger.info(f'.. calculate lines [t={time()-t0:.3f}s]')
    if lopt.pool is None: lopt.pool = len(self.pks)>30000
    pool = Parallel(cpu_count()-2) if lopt.pool is True else lopt.pool

    okpks = self.pks[self.pks.lid < LIDS.NON]
    if len(okpks)==0: return

    if pool is False:
      groups = okpks.groupby('lid')
      out = [self._lines_info(pks,self.FWHME0) for _,pks in groups]
      self.lns = pd.DataFrame(np.concatenate(out), index=list(groups.groups))\
                             .rename_axis('lid').astype(COLS.MLNS) if len(out)>0 else\
                 self._empty_lns()
    else:
      ulids = okpks.lid.unique()
      gsize = int(len(ulids)//pool.n_jobs)
      lgrps = np.split(ulids, np.arange(1,pool.n_jobs)*gsize)
      pgrps = [okpks[okpks.lid.isin(lls)].copy() for lls in lgrps]

      out = pool(delayed(self._calc_lines_pool)(gg,self.FWHME0) for gg in pgrps)
      self.lns = pd.concat(out, axis=0)

    logger.info(f'.. Reindexing [t={time()-t0:.3f}]')
    # --- Check lines hash uniqueness ------------------------------------------
    hsh = self.lns_hash()
    uhs,cnt = np.unique(hsh, return_counts=True)
    if np.any(cnt>1):
      drop = self.lns.index[np.concatenate([np.where(hsh==uu)[0][1:] for uu in uhs[cnt>1]])]
      self.lns.drop(index=drop, inplace=True)
      self.pks.replace({'lid': {dd: LIDS.DRP for dd in drop}}, inplace=True)
      warnings.warn('Duplicate line hash observed!')

    self._reindex_lines()

    logger.info(f'Done! [t={time()-t0:.3f}s]')

  @staticmethod
  def _lines_info(pks, fwhmE0):
    E0 = np.average(pks.E, weights=1/np.power(0.1 + np.abs(pks.fwhmE-fwhmE0), 2))
    return np.array([(np.average(pks.i, weights=pks.a).astype(np.float32),
                      np.average(pks.j, weights=pks.a).astype(np.float32),
                      np.average(pks.k, weights=pks.a).astype(np.float32),
                      pks.a.sum().astype(np.float32),
                      (KE/E0).astype(np.float32),
                      np.std(pks.wl.values).astype(np.float32),
                      E0.astype(np.float32),
                      np.std(pks.E.values).astype(np.float32),
                      pks.ph.sum().astype(np.float32),
                      pks.shape[0],
                      pks.i.max()-pks.i.min()+pks.j.max()-pks.j.min(),
                      MIDS.NON)], dtype=SDTP.MLNS)

  @staticmethod
  def _calc_lines_pool(pks,fwhm):
    grps = pks.groupby('lid')
    out = [LaseAnalysis_Map._lines_info(pks,fwhm) for _,pks in grps]
    return pd.DataFrame(np.concatenate(out), index=list(grps.groups))\
             .rename_axis('lid').astype(COLS.MLNS) if len(out)>0 else\
           LaseAnalysis_Map._empty_lns()

  def _reindex_lines(self):
    if self.lns is None: return

    self.lns.sort_values(by=['k','j','i','E'], inplace=True)
    convert = {lid: np.uint64(ii) for ii,lid in enumerate(self.lns.index)}
    for ll in LIDS.BADS: convert[ll] = ll

    self.lns.index = pd.Index([convert[val] for val in self.lns.index], dtype=np.uint64, name='lid')
    self.pks['lid'] = np.array([convert[val] for val in self.pks.lid], dtype=np.uint64)

  @staticmethod
  def _empty_lns():
    df = pd.DataFrame(columns=COLS.MLNS).astype(COLS.MLNS)
    df.index = df.index.rename('lid').astype(np.uint64)
    return df

  def lns_hash(self):
    return self._lns_hash(self.lns)

  @staticmethod
  def _lns_hash(lns):
    return (lns.i.values.astype(np.uint16).astype(np.uint64) << 48) + \
           (lns.j.values.astype(np.uint16).astype(np.uint64) << 32) + \
           (lns.wl.values.astype(np.float32).view(np.uint32).astype(np.uint64))

  # ----------------------------- Lines operations -----------------------------

  def merge_lines(self, merge, reindex=False):
    if len(merge) == 0: return

    lids_old = []
    mids_old = []
    all_lids = list(sum(merge,()))
    lns = self.lns.loc[all_lids].copy()
    pks = self.pks[self.pks.lid.isin(lns.index)].copy()

    for lids in merge:
      pks.loc[pks.lid.isin(lids), 'lid'] = lids[0]
      lids_old.extend(lids)

      mids = lns.loc[list(lids),'mid'].unique()
      lns.loc[lns.mid.isin(mids),'mid'] = mids[0]
      mids_old.extend(mids)

    self.pks.loc[pks.index,'lid'] = pks.lid
    self.lns.loc[lns.index,'mid'] = lns.mid

    grps = self.pks[self.pks.lid.isin(lids_old)].groupby('lid')

    new_lns = pd.DataFrame(np.concatenate([self._lines_info(pp,self.FWHME0) for _,pp in grps]),
                           index=list(grps.groups)).astype(COLS.MLNS)
    new_lns.index = new_lns.index.rename('lid').astype(np.uint64)
    new_lns['mid'] = self.lns.loc[new_lns.index, 'mid']
    self.lns = pd.concat([self.lns.drop(index=lids_old),new_lns])

    grps = self.lns[self.lns.mid.isin(mids_old)].groupby('mid')
    new_mlt = pd.DataFrame(np.concatenate([self._multi_info(ll) for _,ll in grps]),
                           index=list(grps.groups)).astype(COLS.MMLT)
    new_mlt.index = new_mlt.index.rename('mid').astype(np.uint64)
    self.mlt = pd.concat([self.mlt.drop(index=mids_old),new_mlt])

    if reindex:
      self._reindex_lines()
      self._reindex_multi()

  def remove_lines(self, remove, reindex=False):
    all_pids = self.pks.index[self.pks.lid.isin(remove)].values
    self.pks.loc[all_pids,'lid'] = np.uint64(0)
    self.lns.drop(remove, inplace=True)

    if reindex: self._reindex_lines()

  def split_lines(self, split, reindex=False):
    clid = max(np.uint64(0), self.pks[~self.pks.lid.isin(LIDS.BADS)].lid.max())
    olids = np.empty(len(split), dtype=np.uint64)
    nlids = np.empty(len(split), dtype=np.uint64)
    mids = np.empty(len(split), dtype=np.uint64)
    # Assign new lids to peaks of splitted lines
    for ii,pids in enumerate(split):
      clid += np.uint64(1)
      olid = self.pks.loc[pids, 'lid'].unique()
      if len(olid)!=1:
        raise RuntimeError(f'Split peaks belong to different lines! {olid}')
      olid = olid[0]

      olids[ii] = olid
      self.pks.loc[pids, 'lid'] = clid
      nlids[ii] = clid
      mids[ii] = self.lns.loc[olid,'mid']

    olids = np.unique(olids)

    # Check that all peaks from splitted lines have been assigned
    isold = self.pks.lid.isin(olids)
    if isold.any():
      logger.warning(f'Incomplete splitting [{isold.sum()} peaks]!!')
      self.pks.loc[isold,'lid'] = LIDS.INC

    # Calculate new splitted lines
    cpks = self.pks.loc[np.concatenate(split)]
    grps = cpks[~cpks.isin(LIDS.BADS)].groupby('lid')
    new_lns = pd.DataFrame(np.concatenate([self._lines_info(pp,self.FWHME0) for _,pp in grps]),
                           index=list(grps.groups)).astype(COLS.MLNS)
    new_lns['mid'] = pd.Series(mids, index=nlids)
    self.lns = pd.concat([self.lns.drop(olids, axis=0),new_lns])

    if reindex: self._reindex_lines()

  def get_maplines(self, lids=None, **kwds):
    if lids is None: lids = self.lns.index.values

    try:    iter(lids)
    except: lids = [lids]

    pks = self.pks[self.pks.lid.isin(lids)]
    return [MapLine(pp, self.lns.loc[[ll]], **kwds) for ll,pp in pks.groupby('lid')]

  # --------------------------- Multiplets analysis ----------------------------

  def find_multi(self, mopt=None, **kwds):
    logger.info(f'---------- Find multi ----------')
    if self.lns is None: raise RuntimeError('Lines have not been calculated yet!')
    if mopt is None: mopt = MultiOpts(**kwds)

    self.mlt = self._empty_mlt()
    if len(self.lns) == 0: return

    t0 = time()
    logger.info(f'.. find multiplets [t=0s]')

    glns = self.lns.sort_index()
    gpks = self.pks[self.pks.lid<LIDS.NON].sort_values('lid')
    _, cid_list = dbscan(glns[['i','j','k']].to_numpy(),
                         eps=mopt.lns_dist, min_samples=1, metric='euclidean')

    temp = gpks[['lid','ispt','a']].to_records()
    # Split the peaks information in a list of group of peaks with the same lid
    ret = np.array(np.split(temp[['ispt','a']],
          np.cumsum(np.unique(temp.lid, return_counts=True)[1])[:-1]), dtype=object)
    ucid = np.unique(cid_list)
    logger.info(f'.. .. found {len(ucid)} preliminary multiplets [t={time()-t0:.3f}s]')

    mid0 = np.uint64(0)
    mids = np.full(len(glns.index), 0, dtype=np.uint64)
    for cid in ucid:
      _idx = np.where(cid_list==cid)[0]
      # Clusters with only one line are kept as is
      if len(_idx) == 1:
        mids[_idx] = mid0
        mid0 += 1
      # Clusters with more than one line are further analyzed by calculating
      # the overlap between peaks of each pair of lines
      else:
        try:
          ret[_idx]
        except:
          print(cid)
          print(ret.shape)
          print(_idx)
        ovlp = squareform(np.array([self._peaks_overlap(*pair)\
                                    for pair in combinations(ret[_idx],2)]))
        np.fill_diagonal(ovlp,1)

        _, _mid = dbscan(1-ovlp, eps=1-mopt.min_overlap, min_samples=1, metric='precomputed')
        mids[_idx] = mid0+_mid
        mid0 += np.unique(_mid).shape[0]

    logger.info(f'.. .. found {len(np.unique(mids))} final multiplets [t={time()-t0:.3f}s]')
    self.lns.loc[glns.index, 'mid'] = mids

    grps = self.lns.groupby('mid')
    self.lns.mid = grps.ngroup()
    self.mlt = pd.DataFrame(np.concatenate([self._multi_info(ll) for _,ll in grps])).astype(COLS.MMLT)
    self._reindex_multi()
    logger.info(f'Done! [t={time()-t0:.3f}s]')

  def _reindex_multi(self):
    if self.mlt is None: return

    self.mlt.sort_values(by=['k','j','i'], inplace=True)

    convert = {mid: np.uint64(ii) for ii, mid in enumerate(self.mlt.index.values)}

    self.mlt.index = pd.Index([convert[val] for val in self.mlt.index.values],
                              dtype=np.uint64, name='mid')
    self.lns['mid'] = np.array([convert[val] for val in self.lns.mid.values],
                               dtype=np.uint64)

  @staticmethod
  # def _multi_info(lns):
  #   return np.array([(lns.i.mean().astype(np.float32), lns.j.mean().astype(np.float32),
  #                     lns.k.mean().astype(np.float32), len(lns.index))],
  #                   dtype=SDTP.MMLT)

  def _multi_info(lns):
    return np.array([(np.float32(lns.i.mean()),
                      np.float32(lns.j.mean()),
                      np.float32(lns.k.mean()),
                      len(lns.index))],
                    dtype=SDTP.MMLT)

  @staticmethod
  def _peaks_overlap(pks0, pks1):
      a0 = pks0.a/max(pks0.a)
      a1 = pks1.a/max(pks1.a)

      _, idx0, idx1 = np.intersect1d(pks0.ispt, pks1.ispt, return_indices=True)

      return min(1, np.sum(np.minimum(a0[idx0], a1[idx1]))/min(np.sum(a0), np.sum(a1)))

  @staticmethod
  def _empty_mlt():
    df = pd.DataFrame(columns=COLS.MMLT).astype(COLS.MMLT)
    df.index = df.index.rename('mid').astype(np.uint64)
    return df

  # ----------------------------- Multi operations -----------------------------

  def merge_multi(self, merge, max_dE=1., shift_border=False):
    merge_lns = UnionFind()
    for mrg in merge:
      if shift_border:
        ijs = self.mlt.loc[mrg,['i','j']].values
        areas = self._calc_area(ijs)
        ints = [self.lns.ph[self.lns.mid==mm].sum() for mm in mrg]
        base = np.argmax(ints)
        shift = areas!=areas[base]
        dijs = (ijs - ijs[base]).astype(int)
        dijs[~shift,:] = 0

        for mm,dd in zip(mrg,dijs):
          pids = self.pks.index[self.pks.lid.isin(self.lns.index[self.lns.mid==mm])]
          self.pks.loc[pids,'i'] -= dd[0]
          self.pks.loc[pids,'j'] -= dd[1]
      
      for ii in range(len(mrg)-1):
        E0s = self.lns.E[self.lns.mid==mrg[ii]].sort_values()
        E1s = self.lns.E[self.lns.mid==mrg[ii+1]].sort_values()
        _,_,mtc0,mtc1 = find_dEs(E0s.values.astype(np.float32), E1s.values.astype(np.float32))

        pairs = np.flatnonzero(mtc1[mtc0]==np.arange(len(mtc0)))
        for jj in pairs:
          de = E0s.values[jj]-E1s.values[mtc0[jj]]
          if np.abs(de)<max_dE: merge_lns.union(E0s.index[jj], E1s.index[mtc0[jj]])
    merge_lns = [tuple(sorted(mm)) for mm in merge_lns.sets.values()]

    self.merge_lines(merge_lns, reindex=True)

  def _calc_area(self, ij):
    nareas = self.info.gmap.max(axis=0)+1
    return (ij[:,0]//self.info.scan_area[0] + nareas[1]*(ij[:,1]//self.info.scan_area[1])).astype(int)
  
  # ------------------------------ Remove border -------------------------------

  def _duplicate_score(des, opts):
    jump = ~des.same_area(opts['Dij'])

    sall = des.score(opts['max_dE'])
    rall = des.dijks
    tall = .5*(sall+np.clip((opts['rng1'][1]-(rall-opts['rng1'][0]))/opts['rng1'][1],0,1))
    tall[~jump] = .5*(sall[~jump]+np.clip((opts['rng2'][1]-(rall[~jump]-opts['rng2'][0]))/opts['rng2'][1],0,1))

    return tall

  # ------------------------------- Refinements --------------------------------

  def refine(self, ropt=None, **kwds):
    if self.lns is None:
      raise RuntimeError('Lines analysis must be present to perform refinemnt!')

    logger.info(f'---------- Refinements ----------')
    if len(self.lns) == 0: return
    if ropt is None: ropt = RefineOpts(**kwds)

    self.mlt = None

    for ref in ropt.pre_refinements:
      logger.info(f'Refinement (pre): {ref}')
      self._apply_refinement(ref, ropt)

    self._reindex_lines()
    for ref in ropt.post_refinements:
      logger.info(f'Refinement (post): {ref}')
      self.find_multi(mopt=ropt.mlt_options)
      self._apply_refinement(ref, ropt)

    self.find_multi(mopt=ropt.mlt_options)

    reindex = False
    for ref in ropt.final_refinements:
      reindex = True
      logger.info(f'Refinement (final): {ref}')
      self._apply_refinement(ref, ropt)
    if reindex:
      self._reindex_lines()
      self._reindex_multi()

  def single_refinement(self, name, opt):
    if isinstance(opt, dict): opt = RefineOpts(**opt)
    self._apply_refinement(name, opt)
    self._reindex_lines()
    self._reindex_multi()

  def _apply_refinement(self, func_name, opt):
    ii = 0
    t0 = time()
    self._ref_cache = {}

    while True:
      self._ref_cache['cycle'] = ii
      logger.info(f'.. cycle {ii} [t={time()-t0:.3f}s]')
      mrg, rmv, spl = getattr(self, 'find_'+func_name)(opt[func_name])
      if   len(mrg) > 0: self.merge_lines(mrg, reindex=False)
      elif len(rmv) > 0: self.remove_lines(rmv, reindex=False)
      elif len(spl) > 0: self.split_lines(spl, reindex=False)
      else:              break
      ii += 1

    self._ref_cache = None
    logger.info(f'.. done! [t={time()-t0:.3f}s]')

  def find_splittable_lines(self, opt):
    if opt['max_peri'] is None: opt['max_peri'] = self.lines.peri.max()
    llns = self.lines[(self.lines.peri>=opt['min_peri'])&\
                      (self.lines.peri<=opt['max_peri'])]

    if not self._ref_cache is None:
      if 'checked' in self._ref_cache:
        llns = llns[~llns.index.isin(self._ref_cache['checked'])]
      self._ref_cache['checked'] = llns.index.values

    splits = []
    lines = self.get_maplines(llns.index.values, init=True)
    LS = LineSplitter(opt)
    for ln in lines:
      try:    spl = LS.find_splits(ln)
      except: continue
      if spl.n>1: splits.extend(spl.split_pids(ln))

    return [],[],splits

# ------------------------------------------------------------------------------
# ------------------------------ OPTIONS CLASSES -------------------------------
# ------------------------------------------------------------------------------

class PeaksOpts:
  OPTS = {'pool': False, 'chunk': 100000}

  def __init__(self, **kwds):
    for opt,val in PeaksOpts.OPTS.items():
      setattr(self, opt, kwds[opt] if (opt in kwds) else val)
    
    for opt,val in PeakFitOpts.OPTS.items():
      setattr(self, opt, kwds[opt] if (opt in kwds) else val)

  @property
  def fit_options(self):
    return PeakFitOpts(**{key: getattr(self, key) for key in PeakFitOpts.OPTS})

class LinesOpts:
  OPTS = {'pool': None, 'a': None, 'wl': None, 'fwhm': None, 'E': None, 'fwhmE': None, 'ph': None,
          'i': None, 'j': None, 'k': None, 'bad': None, 'eps': 3, 'min_samples': 2, 'scale': [1, 1, 1, 0.5]}
  FILTER_OPT = ['a', 'wl', 'fwhm', 'E', 'fwhmE', 'ph', 'i', 'j', 'k']

  def __init__(self, **kwds):
    for opt, val in LinesOpts.OPTS.items():
      val = kwds[opt] if (opt in kwds) else val
      if opt=='scale': val=np.array(val)
      setattr(self, opt, val)

  def _filter_dict(self):
    return {key: self.__getattribute__(key) for key in self.FILTER_OPT}

  def filter(self):
    return {key: val for key, val in self._filter_dict().items() if not val is None}.items()


class RefineOpts:
  PRE_REFINEMENTS = {
    'splittable_lines': {'min_peri': 8, 'max_peri': None, 'mass_shift': 4.5,
                         'lapl_sigma': [1.1,1.5], 'dr_close': 3., 'dE_close': .3, 'dE_seed': .25,
                         'dE_reg': .5, 'area_small': 6, 'area_left': 10, 'dE_left':.33},}
  MLT_OPTIONS = {'multiplets': {'lns_dist': 5, 'min_overlap': 0.33}}
  POST_REFINEMENTS = {}
  FINAL_REFINEMENTS = {}

  def __init__(self, **kwds):
    for DICT_OPT in (self.PRE_REFINEMENTS, self.POST_REFINEMENTS,
                     self.FINAL_REFINEMENTS, self.MLT_OPTIONS):
      for ref, dflt in DICT_OPT.items():
        for opt, val in dflt.items():
          if (ref in kwds) and (kwds[ref] is None):
            setattr(self, ref, False)
          else:
            setattr(self, ref, True)
            setattr(self, ref+'_'+opt, kwds[ref][opt] if (ref in kwds)\
                                       and (opt in kwds[ref]) else val)

  def __getitem__(self, ref):
    for category in (self.PRE_REFINEMENTS, self.POST_REFINEMENTS, self.FINAL_REFINEMENTS):
      if ref in category:
        options = category[ref]
        break
    else:
      raise ValueError(f'Refinement {ref} not available!')

    return {opt: getattr(self, ref+'_'+opt) for opt in options}

  @property
  def pre_refinements(self):
    return [ref for ref in self.PRE_REFINEMENTS if getattr(self,ref)]

  @property
  def mlt_options(self):
    return MultiOpts(**{key: getattr(self, 'multiplets_'+key) for key in self.MLT_OPTIONS['multiplets']})

  @property
  def post_refinements(self):
    return [ref for ref in self.POST_REFINEMENTS if getattr(self,ref)]

  @property
  def final_refinements(self):
    return [ref for ref in self.FINAL_REFINEMENTS if getattr(self,ref)]


class MultiOpts:
  def __init__(self, **kwds):
    for opt, val in RefineOpts.MLT_OPTIONS['multiplets'].items():
      setattr(self, opt, kwds[opt] if (opt in kwds) else val)

  def __getitem__(self, key):
    return getattr(self, key)


# class DuplicateOpts:
#   OPTS = {'max_dE': 1., 'max_dist': 25, 'rng_jump': (5.,10.), 'rng_same': (0.,8.), 'min_scr': .7}

#   def __init__(self, **kwds):
#     for opt, val in DuplicateOpts.OPTS.items():
#       setattr(self, opt, kwds[opt] if (opt in kwds) else val)



  # def get_data(self, peaks=True):
  #   if (self.mlt is None) or (self.lns is None): raise RuntimeError('Must be fully analysed to get LaseData')
  #   return LaseData_Map_Confocal(self.mlt, self.lns, self.pks if peaks else None, None)

  # def save_lite(self, fpath, analysis='base'):
  #   with h5.File(fpath, 'a') as f:
  #     if not analysis in f: f.create_group(analysis)
  #     if self.name in f[analysis]: del f[analysis][self.name]

  #     grp = f[analysis].create_group(self.name)

  #     if (self.lns is None) or (self.mlt is None):
  #       raise RuntimeError('Data must be fully analysed before saving lite file!')

  #     lns = self.lns.drop(columns=['wl','dwl','peri','n'])
  #     mlt = self.mlt

  #     grp.create_dataset('lns', data=lns.reset_index().to_records(False))
  #     grp.create_dataset('mlt', data=mlt.reset_index().to_records(False))





# {'border_doubles': {'min_n': 2, 'xpix': 512, 'ypix': 512, 'dpix': 15,
#                     'max_dE': 1., 'min_dig': .51, 'min_ana': .6, 'scale': [1.,1.,1.]},}

  # def find_border_doubles(self, opt):
  #   scale = np.array(opt['scale'])
  #   dx,dy = opt['dpix']/opt['xpix'],opt['dpix']/opt['ypix']
  #   mlt = self.mlt[((np.abs((self.mlt.i/opt['xpix'])-np.round(self.mlt.i/opt['xpix']))<dx)|\
  #                  (np.abs((self.mlt.j/opt['ypix'])-np.round(self.mlt.j/opt['ypix']))<dy))&
  #                  (self.mlt.n>=opt['min_n'])]
  #   lns = self.lns[self.lns.mid.isin(mlt.index.values)]
  #   mids = mlt.index.values

  #   display(mlt)
  #   ijks = mlt[['i','j','k']].values*scale[None,:]
    
  #   _,clst = dbscan(ijks, eps=opt['dpix']*scale[0], min_samples=1)

  #   merge = []
  #   for cc in np.unique(clst):
  #     cidxs = np.where(clst==cc)[0]
  #     if len(cidxs)<2: continue

  #     combs = combinations(cidxs,2)
  #     cnt = 0
  #     for i0,i1 in combs:
  #       cnt += 1
  #       dr = (np.sum((ijks[i0]-ijks[i1])**2))**0.5
  #       if dr>opt['dpix']: continue
  #       if (ijks[i0,0]//opt['xpix'],ijks[i0,1]//opt['ypix']) == (ijks[i1,0]//opt['xpix'],ijks[i1,1]//opt['ypix']):
  #         if (np.abs(ijks[i0,0]/opt['xpix']-np.round(ijks[i0,0]/opt['xpix'])) > 2./opt['xpix']) and\
  #            (np.abs(ijks[i0,1]/opt['ypix']-np.round(ijks[i0,1]/opt['ypix'])) > 2./opt['ypix']) and\
  #            (np.abs(ijks[i1,0]/opt['xpix']-np.round(ijks[i1,0]/opt['xpix'])) > 2./opt['xpix']) and\
  #            (np.abs(ijks[i1,1]/opt['ypix']-np.round(ijks[i1,1]/opt['ypix'])) > 2./opt['ypix']):
  #           continue

  #       lns0 = lns[lns.mid==mids[i0]]
  #       Es0 = lns0.E.values
  #       As0 = lns0.a.values
  #       lids0 = lns0.index.values
  #       isort0 = np.argsort(Es0)
  #       lns1 = lns[lns.mid==mids[i1]]
  #       Es1 = lns1.E.values
  #       As1 = lns1.a.values
  #       lids1 = lns1.index.values
  #       isort1 = np.argsort(Es1)
  #       mtc0,mtc1,ana,dig = align_single(Es0[isort0], Es1[isort1], As0[isort0], As1[isort1],
  #                                        max_dE=opt['max_dE'], thr_A=1., wgt_small=1., pmode='01')

  #       if (dig<opt['min_dig']) or (ana<opt['min_ana']): continue
  #       for l0,l1 in enumerate(mtc0):
  #         if l1<0: continue
  #         merge.append((lids0[isort0[l0]],lids1[isort1[l1]]))

  #   return merge,[],[]




  # @staticmethod
  # def _line_info_df(pks, fwhmE0, lid=0):
  #   E0 = np.average(pks.E, weights=1/np.power(0.1 + np.abs(pks.fwhmE-fwhmE0), 2))
  #   return pd.DataFrame({'i': np.average(pks.i, weights=pks.a),
  #                        'j': np.average(pks.j, weights=pks.a),
  #                        'k': np.average(pks.k, weights=pks.a),
  #                        'a': pks.a.sum(),
  #                        'wl': KE/E0,
  #                        'dwl': np.std(pks.wl.values),
  #                        'E': E0,
  #                        'dE': np.std(pks.E.values),
  #                        'ph': pks.ph.sum(),
  #                        'n': pks.shape[0],
  #                        'peri': pks.i.max()-pks.i.min()+pks.j.max()-pks.j.min(),
  #                        'mid': 0}, index=pd.Index([lid], dtype=np.uint64)).astype(COLS.MLNS)
