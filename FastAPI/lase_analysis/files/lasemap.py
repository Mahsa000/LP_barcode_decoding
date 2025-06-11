import numpy as np
import pandas as pd

import warnings
from copy import copy, deepcopy

from .base import LaseFile
from ..analysis import LaseAnalysis_Map
from ..data import Spectrum, LaseData_Map_Confocal
from ..utils.constants import K_nm2meV as KE, COLS, LIDS
from ..utils.logging import logger


# --------------------------------- MAPS FILES ---------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class LaseFile_Full_Map(LaseFile):
  def __init__(self, *args, **kwds):
    super().__init__(*args, **kwds)
    if not self.ftype=='map':
      raise IOError(f'File extension not compatible with LaseFile_Full_Map {self.ftype}')
    
    self.load_info()

  def load_info(self):
    with self.read() as f:
      self.npixels = f['data']['spectra'].shape[1]
      self.nspectra = f['data']['spectra'].shape[0]

      self.info = {}
      for ig, info in enumerate(f['info']['map'].attrs['ginfo']):
        gmap = f['info']['map'].attrs[f'g{ig}_map']
        self.info[f'grp_{ig}'] = GroupInfo(ig, info, gmap)

      try:    self.start_time = f.attrs['start_time']
      except: self.start_time = self.file.t_modified

  def get_coordinates(self, idx=None):
    with self.read() as f:
      if idx is None: idx = np.arange(f['data']['coordinates'].shape[0])
      crds = f['data']['coordinates'][idx,:].astype(np.int16)

    return crds

  def get_spectra(self, idx=None):
    with self.read() as f:
      if idx is None: idx = np.arange(f['data']['spectra'].shape[0])
      ys = f['data']['spectra'][idx,:].astype(np.float64)

    return ys

  def get_wlaxis(self):
    with self.read() as f:
      x = f['data']['wl_axis'][:].astype(np.float64)
    return x

  def get_spectrum(self, ispt):
    with self.read() as f:
      x = f['data']['wl_axis'][:].astype(np.float64)
      y = f['data']['spectra'][ispt,:].flatten().astype(np.float64)

    return Spectrum(y=y, x=x, sid=ispt)

  def get_analysis(self, gname=None, analysis=None):
    if gname is None:
      if len(self.info) == 1: gname = list(self.info.keys())[0]
      else:                   raise ValueError(f'File has multiple groups. Must specify a name: {list(self.info.keys())}!')
    else:
      if not gname in self.info: raise KeyError(f'Group {gname} not present in the map file!')

    if analysis is None: return LaseAnalysis_Map(self, gname, None, None, None)

    with self.read() as f:
      if analysis=='base': return self._load_base(gname)
      else:                return self._load_diff(gname, analysis)

      # if not 'analysis' in f:        raise RuntimeError(f'File has not been analyzed yet, cannot load analysis "{analysis}"!')
      # if not gname in f['analysis']: raise RuntimeError(f'Group {gname} has not been analyzed yet!')
      # ghandle = f['analysis'][gname]
      # if not analysis in ghandle:    raise RuntimeError(f'Analysis {analysis} not present for group {gname}!')
      
      # pks = pd.DataFrame.from_records(ghandle['pks'][:]).set_index('pid')
      # pks.index = pks.index.astype(np.uint64)
      # pks['lid'] = ghandle[analysis]['lid'][:]
      # pks['E'] = KE/pks['wl']
      # pks['fwhmE'] = pks['fwhm']*pks['E']/pks['wl']
      # pks = pks.reindex(columns=COLS.MPKS).astype(COLS.MPKS)

      # lns = pd.DataFrame.from_records(ghandle[analysis]['lns'][:]).set_index('lid').astype(COLS.MLNS)\
      #       if 'lns' in ghandle[analysis] else None

      # mlt = pd.DataFrame.from_records(ghandle[analysis]['mlt'][:]).set_index('mid').astype(COLS.MMLT)\
      #       if 'mlt' in ghandle[analysis] else None

    # return LaseAnalysis_Map(self, gname, pks, lns, mlt)

  def _load_base(self, gname):
    with self.read() as f:
      if not 'analysis' in f: raise IOError('File does not have any analysis saved!')
      if not gname in f['analysis']: raise IOError(f'File does not have analysis for group "{gname}"!')
      gh = f['analysis'][gname]

      pks = pd.DataFrame.from_records(gh['pks'][:]).set_index('pid')
      pks['E'] = KE/pks['wl']
      pks['fwhmE'] = pks['fwhm']*pks['E']/pks['wl']
      pks = pks.reindex(columns=COLS.MPKS).astype(COLS.MPKS)

      lns = pd.DataFrame.from_records(gh['lns'][:]).set_index('lid').astype(COLS.MLNS)\
            if 'lns' in gh else None

      mlt = pd.DataFrame.from_records(gh['mlt'][:]).set_index('mid').astype(COLS.MMLT)\
            if 'mlt' in gh else None

    return LaseAnalysis_Map(self, gname, pks, lns, mlt)

  def _load_diff(self, gname, analysis):
    with self.read() as f:
      if not 'analysis' in f: raise IOError('File does not have any analysis saved!')
      if not gname in f['analysis']: raise IOError(f'File does not have analysis for group "{gname}"!')
      if not analysis in f['analysis'][gname]: raise IOError(f'Analysis {analysis} not present for group "{gname}"!')
      gh = f['analysis'][gname]
      ah = gh[analysis]

      # --- Peaks ----------
      idxB = ah['pks_base_idx'][:]
      pksB = pd.DataFrame.from_records(gh['pks'][idxB]).set_index('pid')
      pksB.index = ah['pks_base_pid'][:].astype(np.uint64)
      pksB.lid = ah['pks_base_lid'][:].astype(np.uint64)
      pksA = pd.DataFrame.from_records(ah['pks'][:]).set_index('pid')

      pks = pd.concat([pksB,pksA]).sort_index()
      pks['E'] = KE/pks['wl']
      pks['fwhmE'] = pks['fwhm']*pks['E']/pks['wl']
      pks = pks.reindex(columns=COLS.MPKS).astype(COLS.MPKS)

      # --- Lines ----------
      if 'lns' in ah:
        idxB = ah['lns_base_idx'][:]
        lnsB = pd.DataFrame.from_records(gh['lns'][idxB]).set_index('lid')
        lnsB.index = ah['lns_base_lid'][:].astype(np.uint64)
        lnsB.mid = ah['lns_base_mid'][:].astype(np.uint64)
        lnsA = pd.DataFrame.from_records(ah['lns'][:]).set_index('lid')

        lns = pd.concat([lnsB,lnsA]).sort_index().reindex(columns=COLS.MLNS).astype(COLS.MLNS)
      else:
        lns = None
    
      # --- Multi ----------
      mlt = pd.DataFrame.from_records(ah['mlt'][:]).set_index('mid').reindex(columns=COLS.MMLT).astype(COLS.MMLT)\
            if 'mlt' in ah else None


    return LaseAnalysis_Map(self, gname, pks, lns, mlt)

  def save_analysis(self, lma, analysis='base', overwrite=False):
    if lma.pks is None: raise RuntimeError('No analysis performed yet!')

    with self.write() as f:
      if not 'analysis' in f: f.create_group('analysis')

      if analysis=='base':
        if (lma.name in f['analysis']) and (not overwrite):
          raise RuntimeError(f'Base analysis already saved for group "{lma.name}". Set overwrite=True to overwrite!')
        elif (lma.name in f['analysis']) and overwrite:
          del f['analysis'][lma.name]

        self._save_base(lma, f)
      
      else:
        if (lma.name not in f['analysis']):
          raise RuntimeError(f'No base analysis present for group "{lma.name}". Cannot save differential analysis!')
        if analysis in f['analysis'][lma.name]: del f['analysis'][lma.name][analysis]
        
        self._save_diff(lma, f, analysis)

    logger.info(f'>>> Analysis "{analysis}" saved')

  def _save_base(self, lma, f):
    gh = f['analysis'].create_group(lma.name)

    pks = lma.pks.reindex(columns=list(COLS.MPKS_SAVE), copy=True).astype(COLS.MPKS_SAVE)
    gh.create_dataset('pks', data=pks.to_records())

    if not lma.lns is None:
      lns = lma.lns.reindex(columns=list(COLS.MLNS_SAVE), copy=True).astype(COLS.MLNS_SAVE)
      gh.create_dataset('lns', data=lns.to_records())

    if not lma.mlt is None:
      mlt = lma.mlt.reindex(columns=list(COLS.MMLT_SAVE), copy=True).astype(COLS.MMLT_SAVE)
      gh.create_dataset('mlt', data=mlt.to_records())
  
  def _save_diff(self, lma, f, analysis):
    sh = f['analysis'][lma.name].create_group(analysis)

    # --- Peaks ----------
    Bpks = pd.DataFrame.from_records(f['analysis'][lma.name]['pks'][:]).set_index('pid')
    hA = lma.pks_hash()
    hB = lma._pks_hash(Bpks)

    Bin = Bpks[np.isin(hB,hA)]
    Ain = lma.pks[np.isin(hA,hB)]

    Avls = Ain[['a','wl','fwhm','ph']].values
    Bvls = Bin[['a','wl','fwhm','ph']].values
    err = (200*np.abs(Avls-Bvls)/(Avls+Bvls)).max(axis=1)
    Ain = Ain.iloc[err<1.]

    sh.create_dataset('pks_base_idx', data=Bin.index[err<1.])
    sh.create_dataset('pks_base_pid', data=Ain.index.values)
    sh.create_dataset('pks_base_lid', data=Ain.lid.values)

    Aout = lma.pks.drop(index=Ain.index).reindex(columns=list(COLS.MPKS_SAVE), copy=True).astype(COLS.MPKS_SAVE)
    sh.create_dataset('pks', data=Aout.to_records())

    # --- Lines ----------
    if not lma.lns is None:
      Blns = pd.DataFrame.from_records(f['analysis'][lma.name]['lns'][:]).set_index('lid')
      hA = lma.lns_hash()
      hB = lma._lns_hash(Blns)

      Bin = Blns[np.isin(hB,hA)]
      Ain = lma.lns[np.isin(hA,hB)]

      Avls = Ain[['a','wl','dwl','ph']].values
      Bvls = Bin[['a','wl','dwl','ph']].values
      err = (200*np.abs(Avls-Bvls)/(Avls+Bvls)).max(axis=1)
      Ain = Ain.iloc[err<1.]

      sh.create_dataset('lns_base_idx', data=Bin.index[err<1.])
      sh.create_dataset('lns_base_lid', data=Ain.index.values)
      sh.create_dataset('lns_base_mid', data=Ain.mid.values)

      Aout = lma.lns.drop(index=Ain.index).reindex(columns=list(COLS.MLNS_SAVE), copy=True).astype(COLS.MLNS_SAVE)
      sh.create_dataset('lns', data=Aout.to_records())

    # --- Multi ----------
    if not lma.mlt is None:
      mlt = lma.mlt.reindex(columns=list(COLS.MMLT_SAVE), copy=True).astype(COLS.MMLT_SAVE)
      sh.create_dataset('mlt', data=mlt.to_records())
    
  def save_lite(self, analysis, fname=None, suff='', peaks=True, warn=True):
    if fname is None: fname = self.base
    # Check analysis of each group
    lmas = {}
    for gn in self.info:
      try:
        lmas[gn] = self.get_analysis(gn, analysis)
      except:
        if warn: warnings.warn(f'Group {gn} is not analyzed and will not be save!')
        else:    raise RuntimeError(f'Group {gn} is not analyzed!')
        continue
      
      if (lmas[gn].lns is None) or (lmas[gn].mlt is None): raise RuntimeError(f"Group {gn} hasn't been fully analyzed yet!")

    # Create new lase file
    file_lite = LaseFile(self.folder.joinpath(fname+suff+'.llm.lase'), mode='replace', version='3.0', ftype='llm')

    with file_lite.write() as flite:
      with self.read() as ffull:
        wl_axis = ffull['data']['wl_axis'][:].astype(np.float64)
        flite.attrs['matl'] = ffull.attrs['matl']
        flite.attrs['start_time'] = self.start_time

        ffull.copy('/info', flite)

        flite['info']['spectrometer'].attrs['min_wl'] = np.min(wl_axis)
        flite['info']['spectrometer'].attrs['max_wl'] = np.max(wl_axis)

        flite.create_group('analysis')
        flite['analysis'].attrs['gnames'] = list(self.info)
        for gname,lma in lmas.items():
          flite['analysis'].create_group(gname)
          if peaks:
            pks = lma.pks.reindex(columns=COLS.MPKS_SAVE).astype(COLS.MPKS_SAVE)
            flite['analysis'][gname].create_dataset('pks', data=pks.to_records())
            flite['analysis'][gname].create_dataset('lid', data=lma.pks.lid.values)
          flite['analysis'][gname].create_dataset('lns', data=lma.lns.to_records())
          flite['analysis'][gname].create_dataset('mlt', data=lma.mlt.to_records())

    logger.info(f'>>> Lite file saved: {fname}.llm.lase')

  def get_data(self, gname=None, analysis='base', peaks=True):
    lma = self.get_analysis(gname, analysis)
    if (lma.mlt is None) or (lma.lns is None):
      raise IOError('File does not contain lines and/or multi data, cannot return data class!')

    pks = lma.pks[lma.pks.lid.isin(lma.lns.index)] if peaks else None
    return LaseData_Map_Confocal(mlt=lma.mlt, lns=lma.lns, pks=pks, fts=None, name=lma.name, info=deepcopy(lma.info))


class LaseFile_Lite_Map(LaseFile):
  def __init__(self, *args, **kwds):
    super().__init__(*args, **kwds)
    if not self.ftype=='llm':
      raise IOError(f'File extension not compatible with LaseFile_Lite_Map {self.ftype}')
    
    self.load_info()

  def load_info(self):
    with self.read() as f:
      self.info = {}
      for ig, info in enumerate(f['info']['map'].attrs['ginfo']):
        gmap = f['info']['map'].attrs[f'g{ig}_map']
        self.info[f'grp_{ig}'] = GroupInfo(ig, info, gmap)
      
      self.start_time = f.attrs['start_time'] if 'start_time' in f.attrs else None

  def get_data(self, gname=None, peaks=True):
    if gname is None:
      if len(self.info) == 1: gname = list(self.info.keys())[0]
      else:                   raise ValueError(f'File has multiple groups. Must specify a name: {list(self.info.keys())}!')
    else:
      if not gname in self.info: raise KeyError(f'Group {gname} not present in the map file!')

    with self.read() as f:
      gh = f['analysis'][gname]
      if (not 'lns' in gh) or (not 'mlt' in gh):
        raise IOError('File does not contain lines and/or multi data, cannot return data class!')

      lns = pd.DataFrame.from_records(gh['lns'][:]).set_index('lid').astype(COLS.MLNS)
      mlt = pd.DataFrame.from_records(gh['mlt'][:]).set_index('mid').astype(COLS.MMLT)

      if peaks:
        pks = pd.DataFrame.from_records(gh['pks'][:]).set_index('pid')
        pks['E'] = KE/pks['wl']
        pks['fwhmE'] = pks['fwhm']*pks['E']/pks['wl']
        pks['lid'] = gh['lid'][:].astype(np.uint64)
        pks = pks.reindex(columns=COLS.MPKS).astype(COLS.MPKS)
        pks = pks[pks.lid.isin(lns.index.values)]
      else:
        pks = None
    
    return LaseData_Map_Confocal(mlt=mlt, lns=lns, pks=pks, fts=None, name=gname, info=deepcopy(self.info[gname]))


class LaseFile_LiteOld_Map(LaseFile):
  def __init__(self, *args, **kwds):
    super().__init__(*args, **kwds)
    if not self.ftype=='llm':
      raise IOError(f'File extension not compatible with LaseFile_Lite_Map {self.ftype}')
    
    self.load_info()

  def load_info(self):
    with self.read() as f:
      self.info = {}
      for ig, info in enumerate(f['info']['map'].attrs['ginfo']):
        gmap = f['info']['map'].attrs[f'g{ig}_map']
        self.info[f'grp_{ig}'] = GroupInfo(ig, info, gmap)
      
      self.start_time = f.attrs['start_time'] if 'start_time' in f.attrs else None

  def to_v3(self, suff='_v3', analysis=None):
    file_new = LaseFile(self.folder.joinpath(self.base+suff+'.llm.lase'),
                        mode='replace', version='3.0', ftype='llm')

    with file_new.write() as fnew:
      with self.read() as fold:
        if (not analysis is None) and (not analysis in fold): raise(f'Analysis "{analysis}" not present in file!')

        fnew.attrs['matl'] = fold.attrs['matl']
        if not self.start_time is None: fnew.attrs['start_time'] = self.start_time
        fold.copy('/info', fnew)

        fnew.create_group('analysis')
        fnew['analysis'].attrs['gnames'] = list(self.info)

        for gname in self.info:
          fnew['analysis'].create_group(gname)

          pks = pd.DataFrame.from_records(fold['data'][gname][:]).set_index('pid')\
                  .reindex(columns=COLS.MPKS_SAVE).astype(COLS.MPKS_SAVE)
          pks.index = pks.index.astype(np.uint64)
          pks['lid'] = fold[analysis][gname]['lid'][:].astype(np.uint64) if not analysis is None else LIDS.NON
          fnew['analysis'][gname].create_dataset('pks', data=pks.to_records())

          if analysis is None: continue
          
          gh = fold[analysis][gname]
          if 'lns' in gh:
            lns = pd.DataFrame.from_records(gh['lns'][:]).set_index('lid').astype(COLS.MLNS)
            fnew['analysis'][gname].create_dataset('lns', data=lns.to_records())
          if 'mlt' in gh:
            mlt = pd.DataFrame.from_records(gh['mlt'][:]).set_index('mid').astype(COLS.MMLT)
            fnew['analysis'][gname].create_dataset('mlt', data=mlt.to_records())

# ------------------------------ GROUP INFO CLASS ------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class GroupInfo:
  def __init__(self, id, ginfo, gmap):
    self.id = id
    self.gmap = gmap

    self.scan_area = (ginfo['width'],ginfo['height'])
    self.planes = ginfo['planes']
    self.areas = ginfo['areas']
    self.dwell = ginfo['dwell']

    self.width = (max(gmap[:,1])+1)*ginfo['width']
    self.height = (max(gmap[:,0])+1)*ginfo['height']

    self._i0 = 0
    self._j0 = 0
    self._k0 = 0

  def __repr__(self):
    return f'Group #{self.id}|{self.areas} areas|'+\
           f'{self.width}x{self.height}x{self.planes} pixels|'+\
           f'{self.dwell} ms dwell'

  # -------------------------------- Properties --------------------------------

  @property
  def shape(self):
    return (self.height, self.width, self.planes)

  @property
  def irange(self):
    return [self._i0, self._i0+self.width]

  @property
  def jrange(self):
    return [self._j0, self._j0+self.height]

  @property
  def krange(self):
    return [self._k0, self._k0+self.planes]

  @property
  def origin(self):
    return [self._i0, self._j0, self._k0]

  @origin.setter
  def origin(self, value):
    self._i0 = value[0]
    self._j0 = value[1]
    self._k0 = value[2]
