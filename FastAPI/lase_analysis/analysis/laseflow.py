import numpy as np
import pandas as pd

from scipy.signal import convolve, correlate, find_peaks, peak_widths
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import dbscan
from scipy.ndimage import median_filter

from collections import defaultdict
from itertools import groupby

from joblib import Parallel, delayed
from os import cpu_count
from time import time
from copy import deepcopy

import warnings
import matplotlib
import matplotlib.pyplot as plt

from ..data.spectrum import Spectrum, PeakFitOpts
from ..c_code.utils import line_ovlp, follow_lines
from ..utils.constants import LIDS, MIDS, FIDS, SDTP, COLS, K_nm2meV as KE
from ..utils.logging import logger
from ..utils.datastr import CondensedMatrix
from ..utils.functions import condensed_to_square


EPS = 1e-6
IDX_MASK = np.uint64(2**32-1)


class LaseAnalysis_Flow:
  FWHME0 = 0.995
  FLUO_DT = 1e-5
  DT = 0.00010002

  def __init__(self, lfile, aname, pks, lns, mlt, flu):
    if not aname in lfile.info:
      raise RuntimeError(f'Acquisition {aname} not present in the flw file!')

    self.lfile = lfile
    self.name = aname
    
    self.pks = pks
    self.lns = lns
    self.mlt = mlt
    self.flu = flu

    self.dt = 0.
    self.match_lf = None

    self.info = deepcopy(self.lfile.info[self.name])
    self._follow = None
    self._comp = None

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
  def fluos(self):
    return self.flu

  @property
  def _streaks(self):
    if self._follow is None: return None
    return self._follow[self._follow.streak_1st == True]

  def save_analysis(self, analysis, overwrite=False):
    self.lfile.save_analysis(self, analysis, overwrite)

  # ------------------------------ Data retrieval ------------------------------

  def get_spectrum(self, ispt):
    return self.lfile.get_spectrum(self.name, ispt)

  # ----------------------------- Synch operations -----------------------------

  def find_synch(self, sopt=None, **kwds):
    logger.info(f'---------- Find synch ----------')
    t0 = time()

    if sopt is None: sopt = SynchOpts(**kwds)
    logger.info(f'Synch splits [t={time()-t0:.3f}s]')
    dts = []
    tlims = self.info.tlims
    self.dt = 0
    for ii,tl in enumerate(tlims):
      tlase, lase = self.trace_spt(tlims=tl)
      daq = sum([np.maximum(self.trace_daq(ch, tlims=tl),0.) for ch in sopt.chs])
      tdaq = self.trace_tdaq(tlims=tl)
      
      if len(tlase)==0: continue
      
      try:    dt = self.calculate_dt(tlase, lase, tdaq, daq, xlims=sopt.xlims, update=False)
      except: dt = None
      if dt is None:
        logger.info(f'.. chunk {ii}/{len(tlims)}: CANNOT SYNCH!!! [t={time()-t0:.3f}]')
        continue
      dts.append(dt)

      if sopt.plot:
        _,ax = plt.subplots(figsize=(20,4))
        ax.plot(tlase-dt, lase/lase.max(), alpha=.8)
        ax.plot(tdaq, daq/daq.max(), alpha=.8)
        ax.set_title(f'dt={dt:.3f}')
        ax.set_ylim(-0.05,1.05)
        plt.show()

      logger.info(f'.. chunk {ii}/{len(tlims)}: dt={dt:.6f}s [t={time()-t0:.3f}]')

    self.dt = np.nanmedian(dts)
    logger.info(f'Done: dt={self.dt:.3f}s [t={time()-t0:.6f}]')

  def trace_spt(self, chunk=100000, tlims=None):
    with self.lfile.read() as f:
      digi = f['Exp_0'][self.name][f'DAQ_Channel_DIGITAL'][:]
      ispt = f['Exp_0'][self.name]['spectIndex'][:].flatten()
      dat = f['Exp_0'][self.name]['spectList']

      if len(ispt)==0: return np.array([]), np.array([])

      ratio = np.mean(np.diff(np.where((np.diff(digi) != 0))[0]))

      tspt = ispt*ratio*self.FLUO_DT
      i0 = 0         if tlims is None else np.argmax(tspt >= tlims[0])
      i1 = len(ispt) if tlims is None else np.argmax(tspt >= min(tlims[1],tspt[-1]))

      ispt = ispt[i0:i1]
      n = len(ispt)

      if n==0: return np.array([]), np.array([])

      index = ispt-(ispt[0]-1)
      tlase = np.arange(ispt[0]-1, ispt[-1]+1, dtype=float)*ratio*self.FLUO_DT-self.dt

      lase = np.zeros_like(tlase, dtype=float)
      for ii in range(n//chunk+1):
        start = ii*chunk
        end = min(start+chunk,n)
        lase[index[start:end]] = np.sum(np.abs(dat[i0+start:i0+end,:]), axis=1)

    return tlase, lase

  def trace_daq(self, channel, tlims=None):
    with self.lfile.read() as f:
      fdat = f['Exp_0'][self.name][f'DAQ_Channel_{self.info.daq_table[channel]}']
      i0 = 0         if (tlims is None) else\
           max(0, np.ceil((tlims[0])/self.FLUO_DT).astype(int))
      i1 = len(fdat) if (tlims is None) else\
           min(len(fdat), np.ceil((tlims[1])/self.FLUO_DT).astype(int))

      fluo = fdat[i0:i1]

    return fluo

  def trace_tdaq(self, tlims=None):
    with self.lfile.read() as f:
      ndat = len(f['Exp_0'][self.name][f'DAQ_Channel_0'])

    i0 = 0    if (tlims is None) else max(0, np.ceil((tlims[0])/self.FLUO_DT).astype(int))
    i1 = ndat if (tlims is None) else min(ndat, np.ceil((tlims[1])/self.FLUO_DT).astype(int))

    return np.arange(i0,i1)*self.FLUO_DT

  def calculate_dt(self, tlase, lase, tfluo, fluo, xlims=[-35000,-5000], update=False):
    fluo = np.abs(fluo/np.max(fluo))
    lase = np.abs(lase/np.max(lase))
    ll = len(lase)
    if ll+xlims[1]<0: return None

    fluo_int = np.interp(tlase, tfluo, fluo)
    delta = np.argmax(correlate(fluo_int, lase, mode='full')[ll+xlims[0]:ll+xlims[1]])+xlims[0]
    dt = tlase[np.abs(delta)]-tlase[1]

    if update: self.dt = dt
    return dt

  # ----------------------------- Lines operations -----------------------------

  def merge_lines(self, merge, reindex=False):
    if len(merge) == 0: return
    if not hasattr(merge[0], '__iter__'): merge = [merge]

    logger.info('Init merge')

    all_merge = np.concatenate(merge)
    lns = self.lns.loc[all_merge]
    pks = self.pks[self.pks.lid.isin(lns.index)].copy()
    # mlt = self.mlt.loc[lns.mid.unique()]

    lids_old = []
    mids_old = []
    logger.info('Merging')
    for lids in merge:
      print(lids[0])
      pks.loc[pks.lid.isin(lids), 'lid'] = lids[0]
      lids_old.extend(lids)

      mids = lns.loc[list(lids),'mid'].unique()
      lns.loc[lns.mid.isin(mids),'mid'] = mids[0]
      mids_old.extend(mids)
    
    self.pks.loc[pks.index,'lid'] = pks.lid
    self.lns.loc[lns.index,'mid'] = lns.mid

    logger.info('Calculate')
    grps = self.pks[self.pks.lid.isin(lids_old)].groupby('lid')
    new_lns = pd.DataFrame(np.concatenate([self._lines_info(pp,self.FWHME0) for _,pp in grps]),
                           index=list(grps.groups)).astype(COLS.FLNS)
    new_lns['mid'] = self.lns.loc[new_lns.index, 'mid']
    self.lns = pd.concat([self.lns.drop(index=lids_old),new_lns])

    if not self.mlt is None:
      grps = self.lns[self.lns.mid.isin(mids_old)].groupby('mid')
      new_mlt = pd.DataFrame(np.concatenate([self._multi_info(ll) for _,ll in grps]),
                            index=list(grps.groups)).astype(COLS.FMLT)
      self.mlt = pd.concat([self.mlt.drop(index=mids_old),new_mlt])

    self._follow = None

    if reindex:
      self._reindex_lines()
      if not self.mlt is None: self._reindex_multi()

  def remove_lines(self, remove, reindex=False):
    all_pids = self.pks.index[self.pks.lid.isin(remove)].values
    all_mids = None if (self.mlt is None) else self.lns.loc[remove,'mid'].values
    
    self.pks.loc[all_pids,'lid'] = LIDS.NON
    self.lns.drop(remove, inplace=True)
    if reindex: self._reindex_lines()

    if not all_mids is None:
      grps = self.lns[self.lns.mid.isin(all_mids)].groupby('mid')
      new_mlt = pd.DataFrame(np.concatenate([self._multi_info(ll) for _,ll in grps]),
                            index=list(grps.groups)).astype(COLS.FMLT)
      self.mlt = pd.concat([self.mlt.drop(index=all_mids),new_mlt])
      if reindex: self._reindex_multi()
    
    self._follow = None

  def split_lines(self, split, reindex=False):
    old_lids = self.pks.loc[np.concatenate(split), 'lid'].unique()
    clid = max(np.uint64(0), self.pks.lid.max())
    for pids in split:
      clid += 1
      self.pks.loc[pids,'lid'] = clid

    grps = self.pks.loc[np.concatenate(split)].groupby('lid')
    new_lns = pd.DataFrame(np.concatenate([self._line_info(pp,self.FWHME0) for _,pp in grps]),
                           index=list(grps.groups)).astype(COLS.FLNS)
    new_lns['mid'] = self.lns.loc[new_lns.index, 'mid']
    self.lns = pd.concat([self.lns.drop(index=lids_old),new_lns])
    if reindex: self._reindex_lines()

    self._follow = None

  # ------------------------------ Peaks analysis ------------------------------

  def find_peaks(self, popt=None, **kwds):
    logger.info(f'---------- Find peaks ----------')
    if self.dt==0: warnings.warn('Data not synchronized yet!')
    t0 = time()

    if not 'threshold' in kwds: kwds['threshold'] = 15
    if not 'saturation' in kwds: kwds['saturation'] = 3867
    if popt is None: popt = PeaksOpts(**kwds)

    # Check spectra information
    logger.info(f'Load data [t={time()-t0:.3f}]')
    with self.lfile.read() as f:
      nspectra, npixels = f['Exp_0'][self.name]['spectList'].shape

      digi = f['Exp_0'][self.name][f'DAQ_Channel_DIGITAL'][:]
      ispt = f['Exp_0'][self.name]['spectIndex'][:].flatten()
      ratio = np.mean(np.diff(np.where((np.diff(digi) != 0))[0]))
      tspt = ispt*ratio*self.FLUO_DT-self.dt

      x = f['Exp_0']['wl_axis'][:].astype(np.float64) if 'wl_axis' in f['Exp_0'] else np.arange(npixels)

    tlims = self.info.tlims
    idxs = np.concatenate([np.where((tspt>=t0)&(tspt<t1))[0] for t0,t1,_ in tlims])
    nidxs = len(idxs)

    # Create list of chunks indexes
    idx_list = [idxs] if popt.chunk is None else\
               [idxs[np.arange(ii*popt.chunk, min((ii+1)*popt.chunk, nidxs))]
                for ii in range(np.ceil(nidxs/popt.chunk).astype(int))]

    # Analyze data by chunk
    logger.info(f'Analyze data (method #{popt.method}) [t={time()-t0:.3f}]')

    ffit = LaseAnalysis_Flow._analyze_spt if popt.method==1 else LaseAnalysis_Flow._analyze_spt2
    clst = int(1.5*popt.window/(x[npixels//2]-x[npixels//2-1]))
    fopt = [popt.med_filter, popt.threshold, popt.window, self.FWHME0, popt.gain, popt.saturation] if popt.method==1 else\
           [popt.med_filter, popt.threshold, popt.prominence, popt.distance, clst, popt.saturation, popt.window, self.FWHME0, popt.gain]
    pool = Parallel(cpu_count()-2) if popt.pool is True else popt.pool

    fit = []
    for ic, idx in enumerate(idx_list):
      logger.info(f'.. Analyzing chunk {ic+1}/{len(idx_list)} [t={time()-t0:.3f}]')

      with self.lfile.read() as f:
        spectra = f['Exp_0'][self.name]['spectList'][idx,:].astype(np.float64)
      if popt.flip: spectra = np.fliplr(spectra)

      if pool: ret = pool(delayed(ffit)(x,y,idx[ii],*fopt) for ii, y in enumerate(spectra))
      else:    ret = [ffit(x,y,idx[ii],*fopt) for ii,y in enumerate(spectra)]

      # Merge all peaks from analysis in a single dataframe and sort based on ispt
      if len(ret)>0:
        fit.append(pd.DataFrame(np.concatenate(ret), columns=COLS.FIT)\
                    .dropna(axis=0, how='any').astype({'ispt': int}))
        
    
    if len(fit)==0:
      pks = None
    else:
      fit = pd.concat(fit, axis=0).reset_index(drop=True)\
            .astype({'a': np.float32, 'wl': np.float32, 'fwhm': np.float32, 'ph': np.float32})
      # Check duplicates
      hsh = self._pks_hash(fit)
      uhs,cnt = np.unique(hsh, return_counts=True)
      if np.any(cnt>1):
        drop = np.concatenate([np.where(hsh==uu)[0][1:] for uu in uhs[cnt>1]])
        fit = fit.drop(index=fit.index[drop]).reset_index(drop=True)
      # Add times, lids and energy values
      t = pd.Series(tspt[np.array(fit.ispt.values)], index=fit.index, name='t', dtype=np.float64)
      lid = pd.Series(LIDS.NON, index=fit.index, name='lid', dtype=np.uint64)
      pks = pd.concat([t,fit,lid], axis='columns')
      pks['E'] = (KE/pks['wl']).astype(np.float32)
      pks['fwhmE'] = (pks['fwhm']*pks['E']/pks['wl']).astype(np.float32)
      # Sort and reindex
      pks = pks.sort_values(['ispt','E']).reset_index(drop=True)
      pks.index = pks.index.rename('pid').astype(np.uint64)
      pks = pks.reindex(columns=list(COLS.FPKS)).astype(COLS.FPKS)
    
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
  
  @staticmethod
  def empty_peaks():
    return pd.DataFrame(columns=list(COLS.FPKS), index=pd.Index([], dtype=np.uint64)).astype(COLS.FPKS)

  def pks_hash(self):
    return self._pks_hash(self.pks)

  @staticmethod
  def _pks_hash(pks):
    return (pks.ispt.values.astype(np.uint32).astype(np.uint64) << 32) + \
           (pks.wl.values.astype(np.float32).view(np.uint32).astype(np.uint64))

  # ------------------------------ Fluo analysis -------------------------------

  def find_fluos(self, fopt=None, **kwds):
    logger.info(f'---------- Find fluo ----------')
    t0 = time()
    if fopt is None: fopt = FluoOpts(**kwds)

    tlims = self.info.tlims
    tdaq = self.trace_tdaq()

    fluo = []
    logger.info(f'Analyze traces [t={time()-t0:.3f}s]')
    for ii,(st0,st1,slab) in enumerate(tlims):
      logger.info(f'.. Analyzing split {slab} ({ii+1}/{len(tlims)}) [t={time()-t0:.3f}s]')

      i0 = np.argmax(tdaq>st0) if st0<tdaq[-1] else len(tdaq)
      i1 = np.argmax(tdaq>st1) if st1<tdaq[-1] else len(tdaq)
      x = tdaq[i0:i1]

      # Load data
      logger.info(f'.. .. Load and pre-condition [t={time()-t0:.3f}s]')

      fdats = {}
      with self.lfile.read() as f:
        for ch in fopt.channels:
          y = f['Exp_0'][self.name][f'DAQ_Channel_{self.info.daq_table[ch]}'][i0:i1]
          if not fopt.background_prc is None: y -= np.percentile(y, fopt.background_prc)
          if not fopt.medfilt_wdw is None:    y -= np.interp(x, x[::fopt.medfilt_del], median_filter(y[::fopt.medfilt_del], fopt.medfilt_wdw))
          if not fopt.smooth is None:         y = convolve(y, np.full(fopt.smooth[ch],1/fopt.smooth[ch]), mode='same')
          fdats[ch] = y

      trig = sum([fdats[ch] for ch in fopt.trigger])
      dx = x[1]-x[0]

      # Find peaks
      logger.info(f'.. .. Find peaks [t={time()-t0:.3f}]')
      prm = np.maximum(fopt.prom_min, fopt.prom_mlt*trig)
      centers, pinfo = find_peaks(trig, prominence=prm, wlen=fopt.wlen, width=fopt.width, rel_height=fopt.rel_height)

      # Remove bad peaks (small excursion between consecutive peaks)
      ratio = []
      for ii in range(len(centers)-1):
        ymax = np.max(trig[centers[ii]:centers[ii+1]])
        ymin = np.min(trig[centers[ii]:centers[ii+1]])
        ratio.append(0.5*(ymax-ymin)/(ymax+ymin) if ymax+ymin!=0 else 1.)
      bad = np.where(np.array(ratio) < .01)[0]

      centers = np.delete(centers, bad)
      lefts = np.floor(np.delete(pinfo['left_ips'], bad)).astype(int)
      rights = np.ceil(np.delete(pinfo['right_ips'], bad)).astype(int)
      ret_widths = peak_widths(trig, centers, rel_height=.25)
      ilefts = np.floor(ret_widths[2]).astype(int)
      irights = np.ceil(ret_widths[3]).astype(int)

      if len(centers) == 0:
        fluo.append(self.empty_fluo(list(fopt.channels)))
        continue

      # Split peaks intervals so that they do not overlap
      for ii in range(len(centers)):
        if (ii < len(centers)-1) and (rights[ii] > lefts[ii+1]):
          rights[ii] = lefts[ii+1] = centers[ii]+np.argmin(trig[centers[ii]:centers[ii+1]])
        if (ii < len(centers)-1) and (irights[ii] > ilefts[ii+1]):
          irights[ii] = centers[ii]+np.argmin(trig[centers[ii]:centers[ii+1]])-1
          lefts[ii+1] = irights[ii]+2

      # Estimate parameters
      logger.info(f'.. .. Calculate peaks [t={time()-t0:.3f}]')

      fits = {'t': [], 'tl': [], 'tr': []}
      for ch in fdats: fits.update({f'{ch}-A': [], f'{ch}-H': [], f'{ch}-L': []})

      for (l,lin,r,rin) in zip(lefts,ilefts,rights,irights):
        fits['t'].append(x[l+np.argmax(trig[l:r])])
        fits['tl'].append(x[l])
        fits['tr'].append(x[r])
        for ch, y in fdats.items():
          fits[ch+'-A'].append(max(EPS, np.abs(np.sum(y[l:r])))*dx)
          fits[ch+'-H'].append(max(EPS, np.abs(np.max(y[lin:rin]))))
          fits[ch+'-L'].append(0.939*fits[ch+'-A'][-1]/fits[ch+'-H'][-1])

      fits = pd.DataFrame(fits, index=pd.Index(np.arange(len(fits['t'])), dtype=np.uint64), dtype=np.float64)
      fluo.append(fits)

    self.flu = pd.concat(fluo, axis=0).sort_values(by=['t']).reset_index(drop=True)
    self.flu.index.rename('fid', inplace=True)
    # Check duplicates
    logger.info(f'Check hashes [t={time()-t0:.3f}]')
    hsh = self.flu_hash()
    uhs,cnt = np.unique(hsh, return_counts=True)
    if np.any(cnt>1):
      drop = np.concatenate([np.where(hsh==uu)[0][1:] for uu in uhs[cnt>1]])
      self.flu = self.flu.drop(index=self.flu.index[drop]).reset_index(drop=True)
      self.flu.index.rename('fid', inplace=True)
      warnings.warn('Duplicate fluo hash observed!')
    # Compensation
    if not ((fopt.comp is None) or (fopt.comp is False)):
      logger.info(f'Compensate [t={time()-t0:.3f}]')
      self.flu = self.compensate(self.flu, fopt.comp)
      self._comp = fopt.comp

    logger.info(f'Done! [t={time()-t0:.3f}]')

  @staticmethod
  def empty_fluo(channels):
    cols = ['t', 'tl', 'tr'] + sum([[ch+suff for suff in ['-A', '-H', '-L']]
                                    for ch in channels],[])
    return pd.DataFrame(columns=cols, index=pd.Index([], dtype=np.uint64), dtype=np.float64)

  @staticmethod
  def compensate(flu, comp):
    cmat = comp.to_numpy().T
    new = comp.index.values
    old = comp.columns.values

    ret = flu[['t','tl','tr']].copy()
    ret[new] = flu[old].to_numpy() @ cmat
    return ret

  def flu_hash(self):
    return self._flu_hash(self.flu)

  @staticmethod
  def _flu_hash(flu):
    return (flu.tl.values.astype(np.float32).view(np.uint32).astype(np.uint64) << 32) + \
           (flu.tr.values.astype(np.float32).view(np.uint32).astype(np.uint64))

  # ------------------------------ Lines analysis ------------------------------

  def find_lines(self, lopt=None, **kwds):
    logger.info(f'---------- Find lines ----------')

    self.lns = self._empty_lns()
    self.mlt = None
    if len(self.pks)==0: return

    if lopt is None: lopt = LinesOpts(**kwds)

    # --- Filtering step -------------------------------------------------------
    t0 = time()
    logger.info(f'.. Filtering [t=0s]')
    flt = np.zeros(len(self.pks), dtype=bool) if lopt.bad is None else lopt.bad
    for key, val in lopt.filter():
      flt = flt | (self.pks[key].values<val[0]) | (self.pks[key].values>val[1])

    self.pks.loc[flt, 'lid'] = LIDS.FLT
    # fpks: remaining peaks in the group after filtering noise
    fpks = self.pks[~flt]
    if len(fpks.index)==0: return

    # --- Clustering step ------------------------------------------------------
    logger.info(f'.. Clustering [t={time()-t0:.3f}]')

    ids = self._cluster_lines(fpks, lopt.max_ndt, lopt.max_dE, lopt.min_samples)
    ids64 = np.full_like(ids, LIDS.UNC, dtype=np.uint64)
    mask = ids>=0
    ids64[mask] = ids[mask].astype(np.uint64)
    self.pks.loc[fpks.index, 'lid'] = ids64

    # --- Calculation step -----------------------------------------------------
    logger.info(f'.. Calculating [t={time()-t0:.3f}]')
    if lopt.pool is None: lopt.pool = len(self.pks)>30000
    pool = Parallel(cpu_count()-2) if lopt.pool is True else lopt.pool

    okpks = self.pks[self.pks.lid.values < LIDS.NON]
    if len(okpks)==0: return

    if pool is False:
      groups = okpks.groupby('lid')
      out = [self._lines_info(pks,self.FWHME0) for _,pks in groups]
      self.lns = pd.DataFrame(np.concatenate(out), index=list(groups.groups))\
                             .rename_axis('lid').astype(COLS.FLNS)
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

    logger.info(f'Done [t={time()-t0:.3f}]')

  @staticmethod
  def _cluster_lines(pks, max_ndt, max_dE, min_sample):
    pks_sort = pks.t.argsort('t')
    pks = pks.iloc[pks_sort]
    ts = np.sort(pks.t.unique())

    splt = np.append(np.where(np.diff(pks.t.values) != 0)[0]+1, len(pks))

    prev = 0
    pii0 = 0
    open_clst = []
    closed = []
    # bad = []

    for ii, curr in enumerate(splt):
      ct = ts[ii]

      cpks = pks.iloc[prev:curr]
      Es = cpks.E.values
      cn = len(Es)
      piis = np.arange(cn)+pii0

      assigned = np.zeros(cn,dtype=bool)
      to_remove = []
      for cc in open_clst:
        if ct-cc[1] > max_ndt*LaseAnalysis_Flow.DT:
          if len(cc[0]) >= min_sample: closed.append(cc[0])
          # else:                         bad.extend(cc[0])
          to_remove.append(cc)
      for cc in to_remove: open_clst.remove(cc)

      if len(open_clst) > 0:
        pwls = np.array([cc[2] for cc in open_clst])[:,None]
        dst = np.abs(cdist(pwls,Es[:,None]))
        isort = np.argsort(dst.flatten())
        for idx in isort:
          ip = idx//cn
          ic = idx%cn
          if dst[ip,ic] > max_dE: continue
          open_clst[ip][0].append(piis[ic])
          open_clst[ip][1] = ct
          open_clst[ip][2] = Es[ic]
          dst[ip,:] = np.inf
          dst[:,ic] = np.inf
          assigned[ic] = True

      for _pi,_E in zip(piis[~assigned],Es[~assigned]):
        open_clst.append([[_pi], ct, _E])

      prev=curr
      pii0 += len(piis)

    for cc in open_clst:
      if len(cc[0]) >= min_sample: closed.append(cc[0])

    lids = np.full(len(pks),-1,dtype=int)
    for ii,cc in enumerate(closed): lids[cc] = ii
      # lids[cc] = np.uint64((1<<32)+ii)

    return lids[np.argsort(pks_sort)]

  @staticmethod
  def _lines_info(pks, fwhmE0):
    E0 = np.average(pks.E, weights=1/np.power(0.1 + np.abs(pks.fwhmE-fwhmE0), 2))
    t0 = pks.t.min()
    t1 = pks.t.max()
    return np.array([(np.float32(0.5*(t0+t1)),
                      np.float32(t1-t0+LaseAnalysis_Flow.DT),
                      np.float32(pks.a.mean()),
                      np.float32(KE/E0),
                      np.float32(pks.wl.std()),
                      np.float32(E0),
                      np.float32(pks.E.std()),
                      np.float32(pks.ph.max()),
                      pks.shape[0],
                      MIDS.NON)], dtype=SDTP.FLNS)

  @staticmethod
  def _calc_lines_pool(pks,fwhm):
    grps = pks.groupby('lid')
    out = [LaseAnalysis_Flow._lines_info(pks,fwhm) for _,pks in grps]

    return pd.DataFrame(np.concatenate(out), index=list(grps.groups)).rename_axis('lid').astype(COLS.FLNS)\
           if len(out)>0 else LaseAnalysis_Flow._empty_lns()
    
  def _reindex_lines(self):
    if self.lns is None: return

    self.lns.sort_values(by=['t','E'], inplace=True)
    convert = {lid: np.uint64(ii) for ii,lid in enumerate(self.lns.index)}
    for ll in LIDS.BADS: convert[ll] = ll

    self.lns.index = pd.Index([convert[val] for val in self.lns.index], dtype=np.uint64, name='lid')
    self.pks['lid'] = np.array([convert[val] for val in self.pks.lid], dtype=np.uint64)

  @staticmethod
  def _empty_lns():
    df = pd.DataFrame(columns=COLS.FLNS).astype(COLS.FLNS)
    df.index = df.index.rename('lid').astype(np.uint64)
    return df

  def lns_hash(self):
    return self._lns_hash(self.lns)

  @staticmethod
  def _lns_hash(lns):
    return (lns.t.values.astype(np.float32).view(np.uint32).astype(np.uint64) << 32) + \
           (lns.wl.values.astype(np.float32).view(np.uint32).astype(np.uint64))

  # --------------------------- Multiplets analysis ----------------------------

  def find_multi(self, mopt=None, **kwds):
    logger.info(f'---------- Find multi ----------')
    t0 = time()

    self.mlt = self._empty_mlt()
    if len(self.lns)==0: return

    if mopt is None: mopt = MultiOpts(**kwds)

    # --- Clustering step ------------------------------------------------------
    logger.info(f'.. Clustering [t={time()-t0:.3f}]')

    tmat = np.array([self.lns.t.values-self.lns.dt.values/2,
                     self.lns.t.values+self.lns.dt.values/2]).T
    dist = line_ovlp(tmat, mopt.max_out)
    dist.data = 1-dist.data
    _, mids = dbscan(self._sort_by_data(dist), eps=mopt.eps, min_samples=mopt.min_samples,
                     metric='precomputed')

    self.lns['mid'] = mids
    
    # --- Calculation step -----------------------------------------------------
    logger.info(f'.. Calculating [t={time()-t0:.3f}]')

    groups = self.lns.groupby('mid')
    if len(groups) == 0: return

    out = [self._multi_info(lns) for _,lns in groups]
    self.mlt = pd.DataFrame(np.concatenate(out), index=list(groups.groups))\
                           .rename_axis('mid').astype(COLS.FMLT)

    logger.info(f'.. Reindexing [t={time()-t0:.3f}]')
    self._reindex_multi()

    logger.info(f'Done [t={time()-t0:.3f}]')

  @staticmethod
  def _multi_info(lns):
    t0 = np.min(lns.t.values-lns.dt.values/2)
    t1 = np.max(lns.t.values+lns.dt.values/2)
    return np.array([(np.float32(0.5*(t0+t1)),
                      np.float32(t1-t0),
                      len(lns))], dtype=SDTP.FMLT)

  @staticmethod
  def _sort_by_data(orig):
    graph = orig.copy()
    # if each sample has the same number of provided neighbors
    row_nnz = np.diff(graph.indptr)
    if row_nnz.max() == row_nnz.min():
      n_samples = graph.shape[0]
      distances = graph.data.reshape(n_samples, -1)

      order = np.argsort(distances, kind="mergesort")
      order += np.arange(n_samples)[:, None] * row_nnz[0]
      order = order.ravel()
      graph.data = graph.data[order]
      graph.indices = graph.indices[order]

    else:
      for start, stop in zip(graph.indptr, graph.indptr[1:]):
        order = np.argsort(graph.data[start:stop], kind="mergesort")
        graph.data[start:stop] = graph.data[start:stop][order]
        graph.indices[start:stop] = graph.indices[start:stop][order]

    return graph

  def _reindex_multi(self):
    if self.mlt is None: return

    self.multi.sort_values(by=['t'], inplace=True)
    convert = {mid: np.uint64(ii) for ii,mid in enumerate(self.mlt.index)}

    self.mlt.index = pd.Index([convert[val] for val in self.mlt.index], dtype=np.uint64, name='mid')
    self.lns['mid'] = np.array([convert[val] for val in self.lns.mid], dtype=np.uint64)

  @staticmethod
  def _empty_mlt():
    df = pd.DataFrame(columns=COLS.FMLT).astype(COLS.FMLT)
    df.index = df.index.rename('mid').astype(np.uint64)
    return df

  # ----------------------------- Lines refinement -----------------------------

  def refine_lines(self, ropt=None, **kwds):
    if self.lns is None:
      raise RuntimeError('Lines analysis must be present to perform refinemnt!')
    
    logger.info(f'---------- Refinements ----------')
    t0 = time()
    if len(self.lns) == 0: return
    self.mlt = None
    self._follow = None

    if ropt is None: ropt = RefineOpts(**kwds)
    logger.info(f'Follow lines [t={time()-t0:.3f}]')
    self.follow_lines()

    logger.info(f'Refinement pre [t={time()-t0:.3f}]')
    for ref in ropt.pre_refinements:
      logger.info(f'.. find {ref} [t={time()-t0:.3f}]')
      self._apply_refinement(ref, ropt)

    logger.info(f'Refinement post [t={time()-t0:.3f}]')
    for ref in ropt.post_refinements:
      logger.info(f'.. find {ref} [t={time()-t0:.3f}]')
      self.find_multi(mopt=ropt.mlt_options)
      self._apply_refinement(ref, ropt)

    self._reindex_lines()
    logger.info(f'Find multiplets [t={time()-t0:.3f}]')
    self.find_multi(mopt=ropt.mlt_options)

  def find_small_closeby(self, n_max=2, dt_max=1.1e-3):
    small = self.lns[self.lns.n <= n_max]
    fol = self._follow.loc[small.index]
    fol = pd.concat([fol.loc[fol.dt_next < dt_max].lid_next,
                     fol.loc[fol.dt_prev < dt_max].lid_prev])\
            .drop_duplicates().sort_index()

    merge = []
    for lid0, lid1 in fol.items():
      for lids in merge:
        if (lid0 in lids) or (lid1 in lids):
          lids.update((lid0,lid1))
          break
      else:
        merge.append(set((lid0,lid1)))

    return [[np.uint64(val) for val in ss] for ss in merge], [], []

  def find_dense_streaks(self, n_min=3, ratio_max=[.1, 1.]):
    strk = self._streaks[self._streaks.lns_next >= n_min]

    ns = np.sort(strk.lns_next.unique())
    if len(ns) == 0: return [], [], []
    all_ns = np.arange(ns[0],ns[-1]+1)
    n_out = len(all_ns)-len(ratio_max)

    dat = ratio_max+[ratio_max[-1]]*n_out if n_out > 0 else ratio_max[:len(all_ns)]
    rat_map = {_n: _r for _n, _r in zip(all_ns, dat)}
    ratio_max = pd.Series(data=[rat_map[val] for val in strk.lns_next.values],
                          index=strk.index)

    ratio = strk.blk_next/(self.DT*strk.pks_next*strk.lns_next)

    strk = strk[ratio <= ratio_max]
    nxt = self._follow.lid_next.values
    is1 = self._follow.streak_1st.values
    idx = self._follow.index.values
    return [self._get_streak_fast(nxt, is1, idx, lid) for lid in strk.index.values], [], []

  def find_long_lines(self, dt_max=.5):
    return [], np.uint64(self.lns[self.lns.dt>=dt_max].index), []

  def find_multi_broken_lines(self, dE_max=.5):
    merge = []
    for mid, lns in self.lns.groupby('mid'):
      if len(lns) <= 1: continue

      Es = lns.E.values
      isort = np.argsort(Es)
      close = np.where(np.diff(Es[isort]) < dE_max)[0]

      for idx in close:
        lid0, lid1 = lns.index[isort[idx:idx+2]]
        for lids in merge:
          if (lid0 in lids) or (lid1 in lids):
            lids.update((lid0,lid1))
            break
        else:
          merge.append(set((lid0,lid1)))

    return [[np.uint64(val) for val in ss] for ss in merge], [], []

  def _apply_refinement(self, func_name, opt):
    ii = 0
    t0 = time()
    while True:
      logger.info(f'.. .. Cycle #{ii} [tc={time()-t0:.1f}s]')
      ii += 1
      mrg, rmv, spl = getattr(self, 'find_'+func_name)(**opt[func_name])
      if   len(mrg) > 0: self.merge_lines(mrg)
      elif len(rmv) > 0: self.remove_lines(rmv)
      elif len(spl) > 0: self.split_lines(spl)
      else:              break
      self._reindex_lines()
      self.follow_lines(**opt.flu_options)

  # --------------------------- Lase/Fluo refinement ---------------------------

  def match_lasefluo(self):
    logger.info(f'---------- Match lase/fluo ----------')
    t0 = time()

    self.mlt = self.mlt.reindex(columns=COLS.FMLT)
    # self.multi['fid'] = np.uint64(FIDS.NON)

    logger.info(f'Fluo matching [t={time()-t0:.3f}]')
    tlims = self.info.tlims
    mtcs = []
    for ii,(tl0,tl1,_) in enumerate(tlims):
      logger.info(f'.. split #{ii}/{len(tlims)} [t={time()-t0:.3f}]')
      pks = self.pks[(self.pks.t>tl0)&(self.pks.t<=tl1)]
      lns = self.lns[self.lns.index.isin(pks.lid.unique())]
      mlt = self.mlt[self.mlt.index.isin(lns.mid.unique())]
      flu = self.flu[(self.flu.t>tl0)&(self.flu.t<=tl1)]

      mtc = self.fluo_matching(pks,lns,mlt,flu)
      # self.multi.loc[mtc.index, 'fid'] = np.uint64(mtc.values)
      mtcs.append(mtc)

    logger.info(f'Update [t={time()-t0:.3f}]')
    mtc = pd.concat(mtcs)
    mtc = mtc[mtc<FIDS.NON]
    # self.multi.loc[mtc.index,  self.flu.columns[3:]] = self.flu.loc[mtc.values, self.flu.columns[3:]].values
    self.match_lf = mtc
    logger.info(f'Done! [t={time()-t0:.3f}]')

  @staticmethod
  def fluo_matching(pks,lns,mlt,flu):
    mt0s = mlt.t.values - mlt['dt'].values/2
    mt1s = mlt.t.values + mlt['dt'].values/2
    mids = mlt.index.values

    fts = flu.t.values
    mtc = []
    for mid,t0,t1 in zip(mids,mt0s,mt1s):
      out = (fts>=t0) & (fts<=t1)
      n = out.sum()
      if n==0:
        mtc.append(FIDS.UNM)
      elif n==1:
        mtc.append(flu.index[np.argmax(out)])
      else:
        fidxs = np.where(out)[0]
        lpk = pks[pks.lid.isin(lns[lns.mid==mid].index.values)]
        tlns = np.average(lpk.t, weights=lpk.ph)
        tdaq = fts[fidxs]
        dist = np.abs(tdaq-tlns)
        imin = np.argmin(dist)
        mtc.append(flu.index[fidxs[imin]])
    
    return pd.Series(mtc, index=mids, dtype=np.uint64)

  def refine_lasefluo(self):
    logger.info(f'---------- Refine lase/fluo ----------')
    t0 = time()

    rem_pids = []
    rem_lids = []
    rem_mids = []

    new_pks = []
    new_lns = []
    new_mlt = []

    ref_mids = self.multi[self.multi.n>=5].index.values
    logger.info(f'Check multi [t={time()-t0:.3f}]')
    nn = len(ref_mids)
    dn = max(1,int(nn/10))
    for ii,mid in enumerate(ref_mids):
      if ii%dn==0: logger.info(f'.. multi {ii}/{nn} [t={time()-t0:.3f}]')

      ldat = self.get_mltdata(mid, min_flu=2)
      if ldat is None: continue
      
      ldat.refine()
      if not ldat._changed: continue

      rem_pids.extend(ldat._orig_pids)
      rem_lids.extend(ldat._orig_lids)
      rem_mids.extend(ldat._orig_mids)

      new_pks.append(ldat.pks)
      new_lns.append(ldat.lns)
      new_mlt.append(ldat.mlt)

    logger.info(f'Reindex [t={time()-t0:.3f}]')
    if len(new_pks)>0:
      orig_pids = self.pks.index.values

      new_pks = pd.concat(new_pks)
      new_lns = pd.concat(new_lns)
      new_mlt = pd.concat(new_mlt)

      self.pks.drop(rem_pids, inplace=True)
      self.lns.drop(rem_lids, inplace=True)
      self.mlt.drop(rem_mids, inplace=True)

      self.pks = pd.concat([self.pks, new_pks]).loc[orig_pids]
      self.lns = pd.concat([self.lns, new_lns])
      self.mlt = pd.concat([self.mlt, new_mlt])

      self._reindex_lines()
      self._reindex_multi()

    logger.info(f'Done! [t={time()-t0:.3f}]')

  def get_mltdata(self, mid, min_flu=0):
    mlt = self.mlt.loc[[mid]]
    t0 = mlt.t.values[0]-mlt['dt'].values[0]/2
    t1 = mlt.t.values[0]+mlt['dt'].values[0]/2
    isin = (self.flu.t>t0)&(self.flu.t<t1)
    nflu = isin.sum()

    if nflu<min_flu: return None
    
    flu = self.flu[isin]
    lns = self.lns[self.lns.mid==mid]
    pks = self.pks[self.pks.lid.isin(lns.index.values)]
    return MltData(pks,lns,mlt,flu,fwhmE0=self.FWHME0)

  # ---------------------------- Streaks operations ----------------------------

  def follow_lines(self, dE_max=1., dt_max=.1):
    self._follow = follow_lines(self.lns, dE_max, dt_max)
    return self._follow

  def get_streak(self, lid0):
    lids = [lid0]
    nline = self._follow.loc[lid0]
    while nline.lid_next > 0:
      nline = self._follow.loc[nline.lid_next]
      if nline.streak_1st == True: break
      lids.append(nline.name)
    return self.lns.loc[lids]

  @staticmethod
  def _get_streak_fast(nxt, is1, idx, lid):
    lids = []
    while True:
        lids.append(idx[lid])
        lid = nxt[lid]
        if (lid==0) or is1[lid]: break
    return np.array(lids, dtype=np.uint64)
  
# ------------------------------------------------------------------------------
# ------------------------------ MLTDATA CLASSES -------------------------------
# ------------------------------------------------------------------------------

class MltLine:
  def __init__(self, ts=None, iline=None, bline=None, eline=None, fline=None):
    self.ts = ts
    self.int = iline
    self.bool = bline
    self.E = eline
    self.fwhmE = fline

  def __and__(self, other):
    # if np.sum(self.bool&other.bool) > 0:
    #   logger.warning('WARNING: merging overlapping lines!!!')

    return MltLine(self.ts, np.maximum(self.int,other.int), self.bool|other.bool,
                   None if ((self.E is None) or (other.E is None)) else np.maximum(self.E,other.E),
                   None if ((self.fwhmE is None) or (other.fwhmE is None)) else np.maximum(self.fwhmE,other.fwhmE))

  def __add__(self, other):
    return self.copy() if other==0 else\
           MltLine(self.ts, np.maximum(self.int,other.int), self.bool|other.bool)

  def __radd__(self, other):
    return self.__add__(other)

  def copy(self):
    return MltLine(self.ts, self.int.copy(), self.bool.copy(),
                   self.E.copy() if not self.E is None else None,
                   self.fwhmE.copy() if not self.fwhmE is None else None)

  @classmethod
  def from_pks(cls, pks, ts):
    idxt = np.searchsorted(ts, pks.t.values)

    ilines = np.zeros(len(ts))
    ilines[idxt] = np.log10(pks.a.values)/np.log10(pks.a.max())

    blines = ilines > 0.

    elines = np.zeros(len(ts))
    elines[idxt] = pks.E.values

    flines = np.zeros(len(ts))
    flines[idxt] = pks.fwhmE.values

    return cls(ts, ilines, blines, elines, flines)

  @classmethod
  def empty(cls, ts):
    return MltLine(ts, np.zeros_like(ts), np.zeros_like(ts, dtype=bool), None, None)

  @property
  def isempty(self):
    return not np.any(self.bool)

  @property
  def n(self):
    return np.sum(self.bool)

  def add_int(self, other):
    self.int = np.maximum(self.int,other.int)
    self.bool = self.int>0.

  def overlap(self, other):
    return np.sum(self.bool&other.bool) > 0

  def tidx(self, t):
    return np.where(self.ts==t)[0][0]

  def mask(self, idx0, idx1=None):
    if idx1 is None: idx1 = len(self.ts)
    mask0 = np.zeros_like(self.ts)
    mask0[idx0:idx1] = 1.
    return MltLine(self.ts, self.int*mask0, (self.bool*mask0).astype(bool),
                   self.E*mask0 if not self.E is None else None,
                   self.fwhmE*mask0 if not self.fwhmE is None else None)

  def distIH(self, other):
    return DataRefiner.fdistIH(self.int, other.int if isinstance(other, MltLine) else other)


class MltData:
  def __init__(self, peaks, lines, multi, flow=None, alns=None, fwhmE0=None):
    self.pks = peaks
    self.lns = lines
    self.mlt = multi
    self.mid = multi.index.values[0]
    self.fwhmE0 = LaseAnalysis_Flow.FWHME0 if fwhmE0 is None else fwhmE0

    self._ts = None
    self._calc_ts()

    self.alns = alns if not alns is None else\
                {lid: MltLine.from_pks(peaks[peaks.lid==lid], self._ts)
                 for lid in lines.index.values}

    self.flu = flow
    self._calc_fmasks()

    self._orig_pids = self.pks.index.values
    self._orig_lids = self.lns.index.values
    self._orig_mids = self.mlt.index.values

    self._changed = False
    self._iscopy = False

  def _do_copy(self):
    if self._iscopy: return
    self.pks = self.pks.copy()
    self.lns = self.lns.copy()
    self.mlt = self.mlt.copy()
    self._iscopy = True

  def _calc_ts(self):
    ts = np.sort(self.pks.t.unique())
    all_ts = [ts[0]]
    for t in ts[1:]:
      voids = int(np.round(((t-all_ts[-1])-DataRefiner.DT)/DataRefiner.DT))
      if voids>0: all_ts.extend([all_ts[-1]+(ii+1)*DataRefiner.DT for ii in range(voids)])
      all_ts.append(t)
    self._ts = np.array(all_ts)

  def _calc_fmasks(self):
    tflu = self.flu[['tl','tr']].values # + DataRefiner.DT
    keep = [ii for ii, (tl,tr) in enumerate(tflu)\
            if (np.sum(self._ts-tr<0) > 1) and (np.sum(self._ts-tl>0) > 1)]
    tflu = tflu[keep]
    self.flu = self.flu.iloc[keep]

    nflu = tflu.shape[0]
    full = sum([ln for ln in self.alns.values() if ln.n>3])
    if full==0: full = sum(self.alns.values())

    sidx = [0]
    self._fmasks = []
    for jj in range(nflu):
      if jj < nflu-1:
        tl, tr = tflu[jj,1], tflu[jj+1,0]
        if 1.01*(tr-tl)/DataRefiner.DT < 3:
          sidx.append(np.argmax(self._ts >= 0.5*(tl+tr)))
        else:
          i0, i1 = np.argmax(self._ts>tl), np.argmax(self._ts>tr)-1
          imin = i0+np.argmin(full.int[i0:i1+1])

          sidx.append(imin if full.int[imin-1]<full.int[imin+1] else imin+1)
      else:
        sidx.append(len(self._ts))

      msk = np.zeros_like(self._ts, dtype=bool)
      msk[sidx[jj]:sidx[jj+1]] = True
      self._fmasks.append(msk)

    self._flu_splt = sidx

  # ----------------------------- Main operations ------------------------------

  def refine(self):
    nflu = len(self.flu)
    if nflu < 2: return

    aflu = {ii: MltLine.empty(self._ts) for ii in range(nflu)}

    # --- Find close lines ----------
    clids = self.lns.index.values
    cat = self.categorize_close_lines(clids)

    merge = []
    rem = []
    for ii,val in enumerate(cat):
      if val[2] == 'J':
        merge.append(val)
        rem.append(ii)
        continue

      if val[2] == '+': continue

      if nflu < 2:
        merge.append((val[0],val[1],'N'))
        rem.append(ii)
        continue

    for ii in rem[::-1]: del cat[ii]
    new_lids = self.merge_lines(merge)
    hold_merge = set()
    for cc in cat:
      add0 = cc[0] if not cc[0] in new_lids else new_lids[cc[0]]
      add1 = cc[1] if not cc[1] in new_lids else new_lids[cc[1]]
      hold_merge = set.union(hold_merge, [add0,add1])

    # --- First assignment for easy clusters into flow peaks ----------
    clids = [ll for ll in self.lns.index.values if not ll in hold_merge]
    cids, mdist = self.cluster_similar_lines({ll: self.alns[ll] for ll in clids}, max_dist=.25)
    acls = {cc: self.get_cluster(cids[cids==cc].index.values) for cc in cids[cids>=0].unique()}
    percs = self.flow_percentages(acls)
    fids = pd.Series(-1, index=pd.Index(clids,dtype=np.uint64))
    for cc,prc in zip(acls.keys(), percs):
      isort = np.argsort(prc)[::-1]
      if prc[isort[0]]-prc[isort[1]]>.5:
        _lids = cids[cids==cc].index.values
        fids[_lids] = isort[0]
        for ll in _lids: aflu[isort[0]] += self.alns[ll]

    # --- Find splittable lines ----------
    clids = fids[fids<0].index.values
    splt = self.check_splittable_lines(clids)
    hold_splt = set([ss[0] for ss in splt])
    fids.drop(hold_splt, inplace=True)

    # --- Second assignment into flow peaks ----------
    clids = [ll for ll in clids if not ll in hold_splt]
    percs = self.flow_percentages({ll: self.alns[ll] for ll in clids})
    for ii in np.argsort(np.max(percs,axis=1))[::-1]:
      idx = np.argmax(percs[ii])
      fids[clids[ii]] = idx
      aflu[idx] += self.alns[clids[ii]]

    # --- Check closeby lines ----------
    assign = {}
    for (ll0,ll1,mtype) in cat:
      if ll0 in new_lids: ll0 = new_lids[ll0]
      if ll1 in new_lids: ll1 = new_lids[ll1]

      aln0 = self.alns[ll0]
      aln1 = self.alns[ll1]
      _mdist = np.array([[_afl.distIH(_aln) for _afl in aflu.values()]
                         for _aln in (aln0,aln1,aln0+aln1)])

      (idx0,idx1,idxM) = np.argmin(_mdist,axis=1)
      if ((idx0 == idx1) and (idx1 == idxM)) or\
       ((idx0 != idx1) and (_mdist[2,idxM]<min(_mdist[0,idx0],_mdist[1,idx1]))):
        _mlids = self.merge_lines([(ll0,ll1,'M')])
        new_lids.update(_mlids)
        assign[new_lids[ll0]] = idxM
        for old,new in new_lids.items():
          if new in _mlids: new_lids[old] = new_lids[new]

        if (ll0 in assign) and (ll0 != new_lids[ll0]): del assign[ll0]
        if (ll1 in assign) and (ll1 != new_lids[ll1]): del assign[ll1]
      else:
        assign[ll0] = idx0
        assign[ll1] = idx1

    for ll,ii in assign.items(): fids[ll] = ii

    # --- Check splittable lines ----------
    assign = []
    for ss in splt:
      aln = self.alns[ss[0]]
      dst = np.array([af.distIH(aln) if not af.isempty else 1. for af in aflu.values()])
      (idx0,idx1) = np.argsort(dst)[:2]
      if (dst[idx1] == 1.) or (aflu[idx1].isempty):
        assign.append((ss[0], idx0))
        continue
      elif (aflu[idx0].isempty):
        continue

      idxL,idxH = min(idx0,idx1), max(idx0,idx1)
      half = self._flu_splt[idxH] if idxH-idxL==1 else\
             (self._flu_splt[idxL+1]+self._flu_splt[idxH])//2
      alnL = aln.mask(self._flu_splt[idxL],half)
      alnH = aln.mask(half,self._flu_splt[idxH+1])
      if np.all(~(alnL.E>0)):
        assign.append((ss[0],idxH))
        continue
      if np.all(~(alnH.E>0)):
        assign.append((ss[0],idxL))
        continue


      dst2 = [aflu[_idx].distIH(_aln) for _idx,_aln in zip((idxL,idxH),(alnL,alnH))]

      _pdiffL = np.max(np.diff(np.where(np.diff(np.hstack(([False],aflu[idxL].bool^aln.bool,[False]))))[0].reshape(-1,2), axis=1))\
                if np.sum(aflu[idxL].bool^aln.bool)>0 else 0
      _pdiffH = np.max(np.diff(np.where(np.diff(np.hstack(([False],aflu[idxH].bool^aln.bool,[False]))))[0].reshape(-1,2), axis=1))\
                if np.sum(aflu[idxH].bool^aln.bool)>0 else 0
      pdiff = min(_pdiffL,_pdiffH)

      s_imp = dst[idxL]-dst2[0] + dst[idxH]-dst2[1]
      s_dE = np.abs(np.mean(alnL.E[alnL.E>0])-np.mean(alnH.E[alnH.E>0]))
      s_pks = np.clip(0.25*(pdiff-2),-.25,1.)
      s_tot = (s_imp+s_dE+s_pks)/3

      s_n1 = .5 if (fids==idx1).sum() <= 2 else 0.
      s_fwhm = (np.max(aln.fwhmE[half-2:half+2])-np.mean(aln.fwhmE[(aln.fwhmE>.1)&(aln.fwhmE<np.max(aln.fwhmE))]))

      s_bonus = (s_fwhm>.25) or ((fids==idx1).sum() == 2)

      if (s_tot > .25) or ((s_tot > 0.) and s_bonus):
        new_lids = self.split_lines([(ss[0],self._ts[half])])
        for _lid,_idx in zip(new_lids[ss[0]], (idxL,idxH)): fids[_lid] = _idx
      else:
        assign.append((ss[0],idx0))

    for ss in assign: fids[ss[0]] = ss[1]

    if len(fids.unique()) > 1:
      omid = self.mid

      new_mlt = []
      ufids = np.sort(fids.unique())
      for ii,ff in enumerate(ufids):
        if ff < 0: logger.warning('Problem with fid < 0!!!')
        nmid = np.uint64(10+ii << 32) + (omid&IDX_MASK)
        clids = fids[fids==ff].index.values
        self.lns.loc[clids, 'mid'] = nmid

        new_mlt.append(pd.DataFrame(LaseAnalysis_Flow._multi_info(self.lns.loc[clids]),
                                    index=pd.Index([nmid], dtype=np.uint64)))

      self.mlt = pd.concat(new_mlt)
      self._changed = True

  def merge_lines(self, merge):
    self._do_copy()

    new_lids = {}
    cbase = np.uint64(10 << 32)
    for (lid0,lid1,mtype,*other) in merge:
      if lid0 in new_lids: lid0 = new_lids[lid0]
      if lid1 in new_lids: lid1 = new_lids[lid1]

      nlid = cbase+(min(lid0,lid1)&IDX_MASK)
      new_lids[lid0] = new_lids[lid1] = nlid

      cpks = self.pks[self.pks.lid.isin((lid0,lid1))]
      self.pks.loc[cpks.index.values,'lid'] = nlid
      self.lns.drop([lid0,lid1], inplace=True)

      new_lns = pd.DataFrame(LaseAnalysis_Flow._lines_info(cpks,self.fwhmE0),
                             index=pd.Index([nlid], dtype=np.uint64))
      new_lns['mid'] = self.mid
      self.lns = pd.concat([self.lns,new_lns])
      self.mlt.n -= 1

      self.alns[nlid] = self.alns[lid0]&self.alns[lid1]
      if mtype == 'J': self.alns[nlid].add_int(self.alns[other[0]])

    for old,new in new_lids.items():
      if new != old: del self.alns[old]

    self._changed = True
    return new_lids

  def split_lines(self, split):
    new_lids = {}
    for ss in split:
      olid = ss[0]
      nlidL = np.uint64(11 << 32) + (olid&IDX_MASK)
      nlidH = np.uint64(12 << 32) + (olid&IDX_MASK)

      pksL = self.pks.loc[lambda df: (df.lid==olid)&(df.t<ss[1])]
      pksH = self.pks.loc[lambda df: (df.lid==olid)&(df.t>=ss[1])]

      self.pks.loc[pksL.index, 'lid'] = nlidL
      self.pks.loc[pksH.index, 'lid'] = nlidH

      self.lns.drop(olid, inplace=True)
      lnsL = pd.DataFrame(LaseAnalysis_Flow._lines_info(pksL,self.fwhmE0),
                          index=pd.Index([nlidL], dtype=np.uint64))
      lnsL['mid'] = self.mid
      lnsH = pd.DataFrame(LaseAnalysis_Flow._lines_info(pksH,self.fwhmE0),
                          index=pd.Index([nlidH], dtype=np.uint64))
      lnsH['mid'] = self.mid
      self.lns = pd.concat([self.lns,lnsL,lnsH])

      self.mlt.n += 1

      idx_splt = np.argmax(self._ts >= ss[1])
      self.alns[nlidL] = self.alns[olid].mask(0,idx_splt)
      self.alns[nlidH] = self.alns[olid].mask(idx_splt,None)
      del self.alns[olid]

      self.changed = True
      new_lids[olid] = (nlidL, nlidH)

    return new_lids

  # ----------------------- Refinement support functions -----------------------

  def categorize_close_lines(self, lids, dE=1.5, jmp_dE=.5, jmp_int=(-70,15)):
    lns = self.lns.loc[lids]
    Es = lns.E.values
    isort = np.argsort(Es)
    Ediff = np.diff(Es[isort])
    close = np.where(Ediff<dE)[0]

    cat = []
    merged = []
    for idx in close:
      temp0 = lns.iloc[isort[idx]]
      temp1 = lns.iloc[isort[idx+1]]
      (ln0,ln1) = (temp0,temp1) if temp0.t < temp1.t else (temp1,temp0)
      E0 = 0.5*(ln0.E+ln1.E)

      if idx in merged:
        cat.append((ln0.name,ln1.name,'+'))
        merged.append(idx+1)
        continue

      if (ln0.t+1.01*ln0['dt']/2 >= ln1.t-1.01*ln1['dt']/2):
        cat.append((ln0.name,ln1.name,'T'))
        continue

      if (Ediff[idx]>jmp_dE):
        cat.append((ln0.name,ln1.name,'L'))
        continue

      other_lns = lns[(lns.E>E0+jmp_int[0])&(lns.E<E0+jmp_int[1])]
      if len(other_lns) == 0:
        cat.append((ln0.name,ln1.name,'N'))
        continue

      for olid,oln in other_lns.iterrows():
        if ((oln.t+oln['dt']/2) >= (ln1.t-1.01*ln1['dt']/2)) and\
           ((oln.t+oln['dt']/2) <= (ln1.t+1.01*ln1['dt']/2)) and\
           ((oln.t-oln['dt']/2) <= (ln0.t+1.01*ln0['dt']/2)) and\
           ((oln.t-oln['dt']/2) >= (ln0.t-1.01*ln0['dt']/2)):
          cat.append((ln0.name,ln1.name,'J',olid))
          merged.extend(([idx,idx+1]))
          break
      else:
        cat.append((ln0.name,ln1.name,'N'))

    return cat

  def check_splittable_lines(self, lids, min_fwhmE=1.3, max_high_fwhmE=.45,
                             min_size=3, min_dE=.5):
    splittable = []
    lens = {ll: np.sum(ln.bool) for ll,ln in self.alns.items()}
    lens_ord = np.sort(list(lens.values()))
    max_len = lens_ord[-2] if (lens_ord[-2]<lens_ord[-1]) else lens_ord[-3]
    for ll in lids:
      # --- Look for long lines ----------
      if lens[ll] < 6: continue

      if lens[ll] > max_len+1:
        splittable.append((ll, 'L', None))
        continue

      aln = self.alns[ll]

      # --- Look lines with holes ----------
      if not np.all(aln.bool[np.argmax(aln.bool):-np.argmax(aln.bool[::-1])]):
        splittable.append((ll, 'H', None))
        continue

      fwhm = aln.fwhmE[aln.bool]
      E = aln.E[aln.bool]
      imax = np.argmax(fwhm)
      n = len(fwhm)

      # --- Look lines with point at high FWHM ----------
      if (fwhm[imax] > min_fwhmE) and (imax >= min_size) and (imax < n-min_size) and\
         (np.sum(fwhm>min_fwhmE)/n < max_high_fwhmE):
        lab = 'F+' if np.abs(np.mean(E[:imax])-np.mean(E[imax+1:])) > min_dE else 'F-'
        splittable.append((ll, lab, np.where(aln.bool)[0][imax]))
        continue

    return splittable

  def cluster_similar_lines(self, alns, max_dist=.15):
    mdist = self.calcualte_lines_distances(alns)
    isort = np.argsort(mdist.data)
    cids = pd.Series(-1, index=list(alns.keys()))
    ic = 0
    for kk in isort:
      if mdist.data[kk] > max_dist: break
      ii,jj = mdist.coords(kk)

      if (cids.iloc[ii] == -1) and (cids.iloc[jj] == -1):
        cids.iloc[ii] = cids.iloc[jj] = ic
        ic += 1
      elif cids.iloc[ii] == -1:
        cids.iloc[ii] = cids.iloc[jj]
      elif cids.iloc[jj] == -1:
        cids.iloc[jj] = cids.iloc[ii]
      else:
        if cids.iloc[ii] != cids.iloc[jj]: cids.iloc[ii] = cids.iloc[jj] = -2

    cids[cids<0] = cids.max()+1+np.arange(len(cids[cids<0]))
    return cids, mdist

  def flow_percentages(self, alns):
    percs = np.zeros((len(alns),len(self._fmasks)))
    for ii, ln in enumerate(alns.values()):
      percs[ii,:] = [sum(ln.int*msk)/sum(ln.int) for msk in self._fmasks]

    return percs

  def calcualte_lines_distances(self, alns, func='fdistIH'):
    f = getattr(DataRefiner,func)

    dat = []
    for row, rln in enumerate(alns.values()):
      for col, cln in enumerate(alns.values()):
        if col>row: dat.append(f(rln.int,cln.int))

    return CondensedMatrix(np.array(dat))

  def get_cluster(self, lids):
    return sum([self.alns[ll] for ll in lids])

  # ---------------------------- Plotting functions ----------------------------

  def plot_peaks(self, ax, Elims=(760,1075)):
    tlims = (self.pks.t.min()-DataRefiner.DT, self.pks.t.max()+DataRefiner.DT)
    sct=ax.scatter(self.pks.E, self.pks.t, c=self.pks.fwhmE, s=np.clip(self.pks.a.values,100,4000)**0.5,
                   cmap='nipy_spectral', vmin=.9, vmax=1.3)
    sct.set_zorder(10)

    full = sum([ln for ln in self.alns.values() if ln.n>3])
    imax = np.max(np.array([ll.int for ll in self.alns.values()]), axis=0)

    bar=ax.bar(Elims[0]+10, bottom=tlims[0], height=tlims[1]-tlims[0], width=15., facecolor='white', zorder=15)
    sct=ax.scatter(np.full_like(self._ts,Elims[0]+5), self._ts, s=50, c=imax, marker='s', cmap='plasma', vmin=0.5, vmax=1)
    sct.set_zorder(20)
    sct=ax.scatter(np.full_like(self._ts,Elims[0]+15), self._ts, s=50, c=full.int, marker='s', cmap='plasma', vmin=0.5, vmax=1)
    sct.set_zorder(20)

    ax.set_xlim(*Elims)
    ax.set_ylim(*tlims)

  def plot_lines(self, ax, c=None, ok=True):
    cmap = matplotlib.cm.get_cmap('tab10', 10)
    iok = 0
    for ii,(lid,l) in enumerate(self.lns.iterrows()):
      if (not c is None) and (lid in c):
        txt = iok if ok else ii
        iok += 1
      else:
        txt = '-' if ok else ii
      if (not c is None) and (not lid in c): color = 'black'
      elif (c is None) or (c[lid]==-1):      color = 'lightgray'
      elif (c[lid]==-2):                     color = 'black'
      else:                                  color = cmap(c[lid]%10)

      bc = np.average(self._ts, weights=self.alns[lid].int)

      ax.bar(l['E'], bottom=l['t']-l['dt']/2, height=l['dt'], facecolor='none',
             edgecolor=color, width=4., alpha=.75, linewidth=3.)
      sct=ax.scatter(l['E'], bc, marker='_', s=50, linewidth=5, facecolor=color)
      sct.set_zorder(100)

      ax.text(l['E'],l['t']+l['dt']/2,txt,fontsize=15)


class DataRefiner:
  DT = 0.00010002

  def __init__(self, lase, fluo=None):
    self.lase = lase
    self.fluo = fluo

    self._blines = {}
    self._ilines = {}
    self._ilinesL = {}
    self._elines = {}

  @property
  def peaks(self):
    return self.lase.peaks

  @property
  def lines(self):
    return self.lase.lines

  @property
  def multi(self):
    return self.lase.multi

  # ------------------------- Lines distance functions -------------------------

  @staticmethod
  def fdistBH(b0,b1):
    return 1-np.sum(np.logical_and(b0,b1))/max(sum(b0),sum(b1))

  @staticmethod
  def fdistBM(b0,b1):
    return 1-2*np.sum(np.logical_and(b0,b1))/(sum(b0)+sum(b1))

  @staticmethod
  def fdistBL(b0,b1):
    return 1-np.sum(np.logical_and(b0,b1))/min(sum(b0),sum(b1))

  @staticmethod
  def fdistIH(i0,i1):
    return 1-np.sum(np.minimum(i0,i1))/max(np.sum(i0),np.sum(i1))

  @staticmethod
  def fdistIM(i0,i1):
    return 1-2*np.sum(np.minimum(i0,i1))/(np.sum(i0)+np.sum(i1))

  @staticmethod
  def fdistIL(i0,i1):
    return 1-np.sum(np.minimum(i0,i1))/min(np.sum(i0),np.sum(i1))

  # ------------------------- Line intensity functions -------------------------

  def _calc_lines(self, lids):
    lids = [ll for ll in lids if (not ll in self._ilines)]
    mids = self.lines.loc[lids, 'mid'].unique()

    for mid in mids:
      lns = self.lines[self.lines.mid == mid]
      pks = self.peaks[self.peaks.lid.isin(lns.index.values)]
      ts = DataRefiner._calc_ts(pks)

      for ii, (lid,ln) in enumerate(lns.iterrows()):
        cpks = pks[pks.lid==lid]
        idxt = np.searchsorted(ts, cpks.t.values)

        self._ilines[lid] = np.zeros(len(ts))
        self._ilines[lid][idxt] = np.log10(cpks.a.values)/np.log10(cpks.a.max())

        self._ilinesL[lid] = np.zeros(len(ts))
        self._ilinesL[lid][idxt] = cpks.a.values/cpks.a.max()

        self._blines[lid] = self._ilines[lid] > 0.

        self._elines[lid] = np.zeros(len(ts))
        self._elines[lid][idxt] = cpks.E.values

  @staticmethod
  def _calc_ts(pks):
    ts = np.sort(pks.t.unique())
    all_ts = [ts[0]]
    for t in ts[1:]:
      voids = int(np.round(((t-all_ts[-1])-DataRefiner.DT)/DataRefiner.DT))
      if voids>0: all_ts.extend([all_ts[-1]+(ii+1)*DataRefiner.DT for ii in range(voids)])
      all_ts.append(t)
    return np.array(all_ts)

  # ----------------------- Identify critical multiplets -----------------------

  def rate_consistency(self):
    out = {}
    for mid, lns in self.lines.groupby('mid'):
      if len(lns) < 3: continue
      lids3 = lns.index[lns.n >= 3]
      ilines = np.array([self._ilines[l] for l in lids3])
      out[mid] = 1-np.max(pdist(ilines,self.fdistIL))

    return pd.Series(out)

  def find_double_flow(self):
    flu_t0s = ((self.fluo.data.tr+self.fluo.data.tl)/2).values
    flu_dts = (self.fluo.data.tr-self.fluo.data.tl).values

    out = {}
    for mid, mlt in self.multi.iterrows():
      idxs = np.where(np.abs(flu_t0s-mlt['t']) < 0.5*np.maximum(flu_dts, mlt['dt']))[0]

      if len(idxs) > 1: out[np.uint64(mid)] = self.fluo.data.index[idxs].values

    return out

  # ------------------------- Line splitting functions -------------------------

  def check_splittings(self, lns, thr=-.025):
    lids = lns.index.values
    self._calc_lines(lids)

    pks = self.peaks[self.peaks.lid.isin(lids)]
    ts = self._calc_ts(pks)

    ilines = np.array([self._ilines[l] for l in lids])
    imax = np.max(ilines,axis=0)

    splits = self._find_splits(imax, thr)
    if len(splits) == 0: return {}

    split_lines = {}
    for lid in lids:
      spnt = self._split_points(self._ilines[lid],self._elines[lid],splits)
      if len(spnt) == 0: continue

      split_lines[lid] = ts[spnt]-self.DT/2

    return split_lines

  @staticmethod
  def _find_splits(iline, thr):
    tools = groupby(np.diff(iline), lambda x: x>=0)
    splt = []
    cnt = 0
    for tl in tools:
      vals = list(tl[1])
      cnt += len(vals)
      if (not tl[0]) and (np.min(vals)<thr) and not (cnt == len(iline)-1)\
      and not np.all(iline[cnt:]==0.): splt.append(cnt)
    return splt

  @staticmethod
  def _split_points(iline, eline, splits):
    npks = []
    ints = []

    diff = np.diff(iline)
    dos = []

    for ii, spl in enumerate(splits):
      idx0 = 0 if ii == 0 else splits[ii-1]
      lenPRE = sum(iline[idx0:spl]>0)
      lenPOST = sum(iline[spl+1:]>0)

      if (lenPRE <= 1) or (lenPOST <= 1): continue
      found = False

      j_order = [spl, spl-1, spl+1]
      # Check for holes
      for jj in j_order:
        if iline[jj] == 0:
          if not jj in dos: dos.append(jj)
          found = True
          break
      if found: continue

      # Check for minima
      if lenPRE >= 2: j_order.append(spl-2)
      if lenPOST >= 3: j_order.append(spl+2)
      for jj in j_order:
        if (diff[jj-1]<0) and (diff[jj]>0):
          if not jj in dos: dos.append(jj)
          found = True
          break
      if found: continue

      if min(lenPRE,lenPOST) > 3:
        if not spl in dos: dos.append(spl)

    sp_pnt = []
    idx0 = 0
    for spl in dos:
      lenPRE = sum(iline[idx0:spl] > 0)
      lenPOST = sum(iline[spl+1:] > 0)
      if (lenPRE < 2) or (lenPOST < 2): continue

      if iline[spl] == 0:
        act_spl = spl
      else:
        if (lenPRE == 2) and (lenPOST >= 3):   dd = 1
        elif (lenPRE >= 3) and (lenPOST == 2): dd = 0
        else:
          dd = 1 if np.abs(eline[spl-1]-eline[spl]) <= np.abs(eline[spl+1]-eline[spl])\
               else 0
        act_spl = spl+dd

      sp_pnt.append(act_spl)
      idx0 = act_spl

    return sp_pnt

  def split_line(self, lid, tspl):
    self._calc_lines([lid])
    mid = self.lines.mid[lid]

    bins = np.append(np.insert(tspl,0,-np.inf),np.inf)
    cpks = self.peaks[self.peaks.lid==lid]
    ts = self._calc_ts(self.peaks[self.peaks.lid.isin(self.lines.index[self.lines.mid==mid])])
    nlns = []

    for ii, (itv,df) in enumerate(cpks.groupby(pd.cut(cpks.t, bins=bins, precision=6))):
      new_lid = np.uint64((ii+1)<<32) + (lid&IDX_MASK)
      self.peaks.loc[df.index,'lid'] = new_lid
      nlns.append(self.lase._lns_info_df(df, self.lase._fwhm0, lid=new_lid, mid=mid))

      mask = (ts > itv.left) & (ts <= itv.right)

      self._blines[new_lid] = self._blines[lid]&mask
      self._ilines[new_lid] = self._ilines[lid]*mask
      self._ilinesL[new_lid] = self._ilinesL[lid]*mask
      self._elines[new_lid] = self._elines[lid]*mask

    nlns = pd.concat(nlns)
    self.lines.drop(lid, axis=0, inplace=True)
    self.lase.lines = pd.concat([self.lines, nlns])
    self.multi.loc[mid, 'n'] += len(tspl)

    del(self._ilines[lid])
    del(self._ilinesL[lid])
    del(self._blines[lid])
    del(self._elines[lid])

    return nlns.index.values

  # --------------------------- Re-cluster functions ---------------------------

  def re_cluster(self, mid):
    lns = self.lines[self.lines.mid==mid]

    split = {}
    if lns.n.mean() < 8:
      spl = self.check_splittings(lns, thr=-0.05)


    ilines = np.array([self._ilines[l] for l in lns.index.values])

    cids = self._similar_lines(ilines, self.fdistIH, .33)

  @staticmethod
  def _similar_lines(lines, fdist, thr):
    nlns = lines.shape[0]
    dst = pdist(lines,fdist)

    cnt = 0
    cids = np.full(nlns, -1)
    for idx in np.where(dst < thr)[0]:
      i0, i1 = condensed_to_square(idx,nlns)

      if (cids[i0] >= 0) and (cids[i1] >= 0):
        continue
      elif cids[i0] >= 0:
        cids[i1] = cids[i0]
      elif cids[i1] >= 0:
        cids[i0] = cids[i1]
      else:
        cids[i0] = cids[i1] = cnt
        cnt += 1

    return cids

  # ----------------------------- Refine clusters ------------------------------

  def refine(self, min_sample=2, do_print=False):
    doub = self.find_double_flow()

    for ii, (mid, fids) in enumerate(doub.items()):
      if do_print and (ii%50 == 0): print(f'Refining {ii}/{len(doub)}')
      flu = self.fluo.data.loc[fids]

      lns = self.lines[self.lines.mid==mid].loc[lambda df: df.n >= min_sample]
      lids = lns.index.values
      self._calc_lines(lids)
      pks = self.peaks[self.peaks.lid.isin(lids)]

      ilines = np.array([self._ilines[ll] for ll in lids])
      elines = np.array([self._elines[ll] for ll in lids])

      tmin = pks.t.min()-self.DT/2
      tmax = pks.t.max()+self.DT/2
      ts = self._calc_ts(pks)

      tflu = flu[['tl','tr']].values + self.DT
      for jj in range(tflu.shape[0]):
        tflu[jj,0] = tmin if (jj == 0) else tflu[jj-1,1]
        tflu[jj,1] = tmax if (jj == tflu.shape[0]-1)\
                     else 0.5*(tflu[jj,1]+tflu[jj+1,0])

      fmasks = [(ts>=tl)&(ts<tr) for tl,tr in tflu]
      flu_splt = np.array([np.argmin(np.abs(ts-tflu[jj,1]))
                           for jj in range(tflu.shape[0]-1)])

      # --- Calculate lines percentages into flow peaks ----------
      percs = np.zeros((len(lns),len(flu)))
      for jj, iln in enumerate(ilines):
        percs[jj,:] = [sum(iln*msk)/sum(iln) for msk in fmasks]

      # --- Assign lines to flow peaks ----------
      cids = pd.Series(-1, index=lids)
      for jj in range(len(lns)):
        isort = np.argsort(percs[jj])[::-1]
        if (percs[jj][isort[0]] > .5) and (percs[jj][isort[0]] > 3*percs[jj][isort[1]]):
          cids.iloc[jj] = isort[0]

      # --- Check lines to break ----------
      idx_out = np.where(cids == -1)[0]
      splits = {}
      for idx in idx_out:
        spnt = self._split_points(ilines[idx], elines[idx], flu_splt)
        _ln = lns.iloc[idx]
        if len(spnt) > 0: splits[_ln.name] = spnt

      for sp_lid, sp_pnt in splits.items():
        new_lids = self.split_line(sp_lid, ts[sp_pnt]-self.DT/2)

      # --- Update items ----------
      lns = self.lines[self.lines.mid==mid]
      lids = lns.index.values
      pks = self.peaks[self.peaks.lid.isin(lids)]

      ilines = np.array([self._ilines[ll] for ll in lids])
      ilinesL = np.array([self._ilinesL[ll] for ll in lids])

      # --- Calculate lines percentages into flow peaks ----------
      percs = np.zeros((len(lns),len(flu)))
      percsL = np.zeros_like(percs)
      for jj, (iln, ilnL) in enumerate(zip(ilines,ilinesL)):
        percs[jj,:] = [sum(iln*msk)/sum(iln) for msk in fmasks]
        percsL[jj,:] = [sum(ilnL*msk)/sum(ilnL) for msk in fmasks]

      # --- Assign lines to flow peaks ----------
      cids = np.full(len(lids), -1)
      for jj in range(len(lns)):
        isort = np.argsort(percs[jj])[::-1]
        if (percs[jj][isort[0]] > .5) and (percs[jj][isort[0]] > 3*percs[jj][isort[1]]):
          cids[jj] = isort[0]

      iclst = {}
      for cc in np.unique(cids[cids>=0]):
        iclst[cc] = np.sum(ilines[cids==cc,:],axis=0)
        iclst[cc] /= np.max(iclst[cc])

      # --- Check flow peaks with no lines ----------
      for iflu in range(len(flu)):
        if iflu in cids: continue
        idx_out = np.where(cids == -1)[0]
        if len(idx_out) == 0: continue

        for idx in idx_out:
          if percsL[idx,iflu] <= .33: continue
          if (len(iclst) > 0) and\
             (min([self.fdistIH(ilines[idx],iln) for iln in iclst.values()]) > .5):
            cids[idx] = iflu

      # --- Calculate clusters intensity profile ----------
      iclst = [None]*len(flu)
      for cc in range(len(flu)):
        isin = (cids == cc)
        if sum(isin) > 0:
          iclst[cc] = np.sum(ilines[isin,:],axis=0)
          iclst[cc] /= np.max(iclst[cc])
        else:
          iclst[cc] = None

      # --- Assign remaining lines ----------
      idx_out = np.where(cids == -1)[0]
      for idx in idx_out:
        dst = [self.fdistIH(ilines[idx], icl) if (not icl is None) else 1
               for icl in iclst ]
        cids[idx] = np.argmin(dst)

      if -1 in cids: raise RuntimeError('Refinement failed!')
      # --- Reindex multi and lines ----------
      if len(np.unique(cids)) == 1: continue

      new_mids = pd.Series((np.uint64(cids+1)<<32)+(mid&IDX_MASK),
                           index=lids)
      self.lines.loc[lids, 'mid'] = new_mids

      grps = self.lines.loc[lids].groupby('mid')
      out = [self._multi_info(lns) for _,lns in grps]
      new_mlt = pd.DataFrame(np.concatenate(out), index=list(grps.groups))\
                            .rename_axis('mid').astype(COLS.FMLT)

      self.multi.drop(mid, axis=0, inplace=True)
      self.lase.multi = pd.concat([self.lase.multi, new_mlt])

    self.lase._reindex_multi()
    self.lase._reindex_lines()

  # ------------------------------ Plot functions ------------------------------

  def plot_mlt_peaks(self, mid, ax):
    lns = self.lines[self.lines.mid==mid]
    pks = self.peaks[self.peaks.lid.isin(lns.index.values)]
    self._calc_lines(lns.index.values)


    ts = self._calc_ts(pks)
    imax = np.max(np.array([self._ilines[lid] for lid in lns.index.values]),axis=0)

    sct=ax.scatter(pks.E, pks.t, c=pks.fwhmE, s=np.clip(pks.a.values,100,4000)**0.5,
                   cmap='nipy_spectral', vmin=.9, vmax=1.3)
    # sct=ax.scatter(pks.wl, pks.t, c=np.log10(pks.a), s=20, cmap='plasma', vmin=1, vmax=3)
    sct.set_zorder(10)

    sct=ax.scatter(np.full_like(ts,750), ts, s=50, c=imax, marker='s', cmap='plasma', vmin=0, vmax=1)
    sct.set_zorder(20)

    ax.set_ylim(pks.t.min()-self.DT, pks.t.max()+self.DT)

  def plot_mlt_lines(self, mid, ax, nmin=3, c=None):
    lns = self.lines[self.lines.mid==mid].loc[lambda df: df.n >= nmin]

    cmap = matplotlib.cm.get_cmap('tab10', 10)
    for ii,(lid,l) in enumerate(lns.iterrows()):
      if (c is None) or (c[lid]==-1): color = 'lightgray'
      elif (c[lid]==-2):              color = 'black'
      else:                           color = cmap(c[lid]%10)

      ax.bar(l['E'], bottom=l['t']-l['dt']/2, height=l['dt'], facecolor='none',
             edgecolor=color, width=4., alpha=.75, linewidth=3.)
      ax.text(l['E'],l['t']+l['dt']/2,ii,fontsize=15)


# ------------------------------------------------------------------------------
# ------------------------------ OPTIONS CLASSES -------------------------------
# ------------------------------------------------------------------------------

class SynchOpts:
  OPTS = {'chs': ['FSC','SSC'], 'xlims': [-35000,-5000], 'plot': False}

  def __init__(self, **kwds):
    for opt,val in SynchOpts.OPTS.items():
      setattr(self, opt, kwds[opt] if (opt in kwds) else val)


class PeaksOpts:
  OPTS = {'pool': False, 'chunk': 100000, 'flip': True}

  def __init__(self, **kwds):
    for opt,val in PeaksOpts.OPTS.items():
      setattr(self, opt, kwds[opt] if (opt in kwds) else val)
    
    for opt,val in PeakFitOpts.OPTS.items():
      setattr(self, opt, kwds[opt] if (opt in kwds) else val)

  @property
  def fit_options(self):
    return PeakFitOpts(**{key: getattr(self, key) for key in PeakFitOpts.OPTS})


class FluoOpts:
  OPTS = {'trigger': ['FSC','SSC'], 'channels': ['FSC','SSC'], 'comp': False,
          'background_prc': 50, 'medfilt_wdw': None, 'medfilt_del': 1, 'smooth': 3,
          'prom_min': 0.1, 'prom_mlt': 0.5, 'wlen': 1000, 'width': 1., 'rel_height': .9,}

  def __init__(self, **kwds):
    for opt,val in FluoOpts.OPTS.items():
      if opt=='smooth':
        kwds[opt] = self._smooth_dict(kwds[opt] if opt in kwds else val, val)

      setattr(self, opt, kwds[opt] if (opt in kwds) else val)

  @staticmethod
  def _smooth_dict(opt, val):
    if isinstance(opt, dict):
      dd = defaultdict(lambda: opt['dflt'] if 'dflt' in opt else val)
      dd.update(opt)
    elif opt is None:
      dd = None
    else:
      dd = defaultdict(lambda: opt)

    return dd


class LinesOpts:
  FILTER_OPT = ['a', 'wl', 'E', 'fwhm', 'ph', 't']
  OPTS = {'a': None, 'wl': None, 'E': None, 'fwhm': None, 'ph': None, 't': None, 'bad': None,
          'eps': 3, 'min_samples': 1, 'scale': [1e-4, 1], 'max_ndt': 2.1, 'max_dE': 1., 'pool': False}

  def __init__(self, **kwds):
    for opt, dflt in self.OPTS.items():
      if opt=='scale': kwds[opt] = np.array(kwds[opt] if opt in kwds else dflt)
      setattr(self, opt, kwds[opt] if (opt in kwds) else dflt)

  def _filter_dict(self):
    return {key: self.__getattribute__(key) for key in self.FILTER_OPT}

  def filter(self):
    return {key: val for key, val in self._filter_dict().items()
            if not val is None}.items()


class MultiOpts:
  def __init__(self, **kwds):
    for opt, dflt in RefineOpts.MLT_OPTIONS['multiplets'].items():
      setattr(self, opt, kwds[opt] if (opt in kwds) else dflt)

  def __getitem__(self, key):
    return getattr(self, key)


class RefineOpts:
  FLU_OPTIONS = {'follow': {'dE_max': 1., 'dt_max': .1}}
  PRE_REFINEMENTS = {'small_closeby': {'n_max': 2, 'dt_max': 1.1e-3},
                     'dense_streaks': {'n_min': 3, 'ratio_max': [.1, 1.]},
                     'long_lines': {'dt_max': .5}}
  POST_REFINEMENTS = {'multi_broken_lines': {'dE_max': .5}}
  MLT_OPTIONS = {'multiplets': {'eps': .7, 'min_samples': 1, 'max_out': 10}}

  def __init__(self, **kwds):
    for DICT_OPT in (self.PRE_REFINEMENTS, self.POST_REFINEMENTS,
                 self.FLU_OPTIONS, self.MLT_OPTIONS):
      for ref, dflt in DICT_OPT.items():
        for opt, val in dflt.items():
          if (ref in kwds) and (kwds[ref] is None):
            setattr(self, ref, False)
          else:
            setattr(self, ref, True)
            setattr(self, ref+'_'+opt, kwds[ref][opt] if (ref in kwds)\
                                       and (opt in kwds[ref]) else val)

  def __getitem__(self, ref):
    for category in (self.PRE_REFINEMENTS, self.POST_REFINEMENTS):
      if ref in category:
        options = category[ref]
        break
    else:
      raise ValueError(f'Refinement {ref} not available!')

    return {opt: getattr(self, ref+'_'+opt) for opt in options}

  @property
  def flu_options(self):
    return {key: getattr(self, 'follow_'+key) for key in self.FLU_OPTIONS['follow']}

  @property
  def pre_refinements(self):
    return [ref for ref in self.PRE_REFINEMENTS if getattr(self,ref)]

  @property
  def mlt_options(self):
    return MultiOpts(**{key: getattr(self, 'multiplets_'+key) for key in self.MLT_OPTIONS['multiplets']})

  @property
  def post_refinements(self):
    return [ref for ref in self.POST_REFINEMENTS if getattr(self,ref)]
