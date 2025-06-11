import numpy as np
import pandas as pd

from copy import copy, deepcopy
import matplotlib.pyplot as plt

from time import mktime,strptime
import warnings

from .base import LaseFile
from .flow import FileFCS
from ..analysis import LaseAnalysis_Flow
from ..data import LaseData_Flow_Guava, LaseData_Flow_LaseV3, Spectrum
from ..utils.constants import LIDS, FIDS, COLS, K_nm2meV as KE
from ..utils.logging import logger

# -------------------------------- GUAVA FILES ---------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class LaseFile_Guava_Flow(LaseFile):
  def __init__(self, *args, **kwds):
    super().__init__(*args, **kwds)
    if not self.ftype=='flw':
      raise IOError(f'File extension not compatible with LaseFile_Guava_Flow {self.ftype}')
    
    self.load_info()
  
  def load_info(self):
    with self.read() as f:
      self.info = {}
      self.start_time = mktime(strptime(f['Exp_0'].attrs['timestamp'],'%Y/%m/%d, %H:%M:%S'))
      for aname in f['Exp_0']:
        if aname=='wl_axis': continue
        ts = f['Exp_0'][aname]['cmdTimestamps'][:].flatten()
        cmds = [cc.decode('ASCII') for cc in f['Exp_0'][aname]['cmdCommands'][:].flatten()]
        cmds = [None if cc=='None' else cc for cc in cmds]
        mrks = [mm.decode('ASCII') for mm in f['Exp_0'][aname]['cmdMarkers'][:].flatten()]
        mrks = ['' if mm=='-' else mm for mm in mrks]
        daq = self._parse_daq_table(f['Exp_0'][aname].attrs['Channel Description'])

        self.info[aname] = AcqInfo(aname, cmds, mrks, ts, daq)
      
      self.nacq = len(self.info)

  @staticmethod
  def _parse_daq_table(chd):
    daq = {}
    for ss in chd[1:-1].split(','):
        a,b = ss.split(': ')
        if b=="'EIGHT'": b = "'YLW-V'"
        daq[b[1:-1]] = int(a)
    return daq

  def save_wlaxis(self, wl_axis):
    with self.write() as f:
      if 'wl_axis' in f['Exp_0']: f['Exp_0']['wl_axis'][:] = wl_axis
      else:                       f['Exp_0'].create_dataset('wl_axis', data=wl_axis)

  def get_spectrum(self, aname, ispt):
    with self.read() as f:
      x = f['Exp_0']['wl_axis'][:].astype(np.float64)
      y = f['Exp_0'][aname]['spectList'][ispt,:].flatten().astype(np.float64)

    return Spectrum(y=y, x=x, sid=ispt)

  def get_analysis(self, aname=None, analysis=None):
    if aname is None:
      if len(self.info) == 1: aname = list(self.info.keys())[0]
      else:                   raise ValueError(f'File has multiple groups. Must specify a name: {list(self.info.keys())}!')
    else:
      if not aname in self.info: raise KeyError(f'Group {aname} not present in the map file!')

    if analysis is None: return LaseAnalysis_Flow(self, aname, None, None, None, None)

    with self.read() as f:
      if analysis=='base': return self._load_base(aname)
      else:                return self._load_diff(aname, analysis)

  def _load_base(self, aname):
    with self.read() as f:
      if not 'analysis' in f: raise IOError('File does not have any analysis saved!')
      if not aname in f['analysis']: raise IOError(f'File does not have analysis for group "{aname}"!')
      gh = f['analysis'][aname]

      dt = gh.attrs['dt']

      pks = pd.DataFrame.from_records(gh['pks'][:]).set_index('pid')
      pks['E'] = KE/pks['wl']
      pks['fwhmE'] = pks['fwhm']*pks['E']/pks['wl']
      pks = pks.reindex(columns=COLS.FPKS).astype(COLS.FPKS)

      flu = pd.DataFrame.from_records(gh['flu'][:]).set_index('fid').astype(COLS.FFLU)\
            if 'flu' in gh else None

      lns = pd.DataFrame.from_records(gh['lns'][:]).set_index('lid').astype(COLS.FLNS)\
            if 'lns' in gh else None

      mlt = pd.DataFrame.from_records(gh['mlt'][:]).set_index('mid').astype(COLS.FMLT)\
            if 'mlt' in gh else None

      mtc = pd.Series(gh['mtc_fids'][:].astype(np.uint64), index=pd.Index(gh['mtc_mids'][:], dtype=np.uint64))\
            if 'mtc_mids' in gh else None

      comp = pd.DataFrame(gh['comp'][:], columns=gh['comp'].attrs['cols'], index=gh['comp'].attrs['rows'])\
            if 'comp' in gh else None

    obj = LaseAnalysis_Flow(self, aname, pks, lns, mlt, flu)
    obj.dt = dt
    obj._comp = comp
    obj.match_lf = mtc

    return obj

  def _load_diff(self, aname, analysis):
    with self.read() as f:
      if not 'analysis' in f: raise IOError('File does not have any analysis saved!')
      if not aname in f['analysis']: raise IOError(f'File does not have analysis for group "{aname}"!')
      if not analysis in f['analysis'][aname]: raise IOError(f'Analysis {analysis} not present for acquisition "{aname}"!')
      gh = f['analysis'][aname]
      ah = gh[analysis]
      dt = ah.attrs['dt']

      # --- Peaks ----------
      idxB = ah['pks_base_idx'][:]
      pksB = pd.DataFrame.from_records(gh['pks'][idxB]).set_index('pid')
      pksB.index = ah['pks_base_pid'][:].astype(np.uint64)
      pksB.lid = ah['pks_base_lid'][:].astype(np.uint64)
      pksA = pd.DataFrame.from_records(ah['pks'][:]).set_index('pid')

      pks = pd.concat([pksB,pksA]).sort_index()
      pks['E'] = KE/pks['wl']
      pks['fwhmE'] = pks['fwhm']*pks['E']/pks['wl']
      pks = pks.reindex(columns=COLS.FPKS).astype(COLS.FPKS)

      # --- Fluos ----------
      if 'flu' in ah:
        idxB = ah['flu_base_idx'][:]
        fluB = pd.DataFrame.from_records(gh['flu'][idxB]).set_index('fid')
        fluB.index = ah['flu_base_fid'][:].astype(np.uint64)
        fluA = pd.DataFrame.from_records(ah['flu'][:]).set_index('fid')

        flu = pd.concat([fluB,fluA]).sort_index().astype(COLS.FFLU)
      else:
        flu = None

      comp = pd.DataFrame(ah['comp'][:], columns=ah['comp'].attrs['cols'], index=ah['comp'].attrs['rows'])\
            if 'comp' in ah else None

      # --- Lines ----------
      if 'lns' in ah:
        idxB = ah['lns_base_idx'][:]
        lnsB = pd.DataFrame.from_records(gh['lns'][idxB]).set_index('lid')
        lnsB.index = ah['lns_base_lid'][:].astype(np.uint64)
        lnsB.mid = ah['lns_base_mid'][:].astype(np.uint64)
        lnsA = pd.DataFrame.from_records(ah['lns'][:]).set_index('lid')

        lns = pd.concat([lnsB,lnsA]).sort_index().reindex(columns=COLS.FLNS).astype(COLS.FLNS)
      else:
        lns = None
    
      # --- Multi ----------
      mlt = pd.DataFrame.from_records(ah['mlt'][:]).set_index('mid').reindex(columns=COLS.FMLT).astype(COLS.FMLT)\
            if 'mlt' in ah else None
      
      mtc = pd.Series(ah['mtc_fids'][:].astype(np.uint64), index=pd.Index(ah['mtc_mids'][:], dtype=np.uint64))\
            if 'mtc_mids' in ah else None

    obj = LaseAnalysis_Flow(self, aname, pks, lns, mlt, flu)
    obj.dt = dt
    obj._comp = comp
    obj.match_lf = mtc

    return obj

  def save_analysis(self, lfa, analysis='base', overwrite=False):
    if lfa.pks is None: raise RuntimeError('No analysis performed yet!')

    with self.write() as f:
      if not 'analysis' in f: f.create_group('analysis')

      if analysis=='base':
        if (lfa.name in f['analysis']) and (not overwrite):
          raise RuntimeError(f'Base analysis already saved for group "{lfa.name}". Set overwrite=True to overwrite!')
        elif (lfa.name in f['analysis']) and overwrite:
          del f['analysis'][lfa.name]

        self._save_base(lfa, f)
      
      else:
        if (lfa.name not in f['analysis']):
          raise RuntimeError(f'No base analysis present for group "{lfa.name}". Cannot save differential analysis!')
        if analysis in f['analysis'][lfa.name]: del f['analysis'][lfa.name][analysis]
        
        self._save_diff(lfa, f, analysis)

    logger.info(f'>>> Analysis "{analysis}" saved')

  def _save_base(self, lfa, f):
    gh = f['analysis'].create_group(lfa.name)
    gh.attrs['dt'] = lfa.dt

    pks = lfa.pks.reindex(columns=list(COLS.FPKS_SAVE), copy=True).astype(COLS.FPKS_SAVE)
    gh.create_dataset('pks', data=pks.to_records())

    if not lfa.flu is None:
      COLS_FLU = copy(COLS.FFLU_SAVE)
      COLS_FLU.update({ch: np.float32 for ch in lfa.flu.columns[3:]})
      sflu = lfa.flu.reindex(columns=list(COLS_FLU), copy=True).astype(COLS_FLU)
      gh.create_dataset('flu', data=sflu.to_records())
      if not lfa._comp is None:
        gh.create_dataset('comp', data=lfa._comp.values)
        gh['comp'].attrs['rows'] = list(lfa._comp.index.values)
        gh['comp'].attrs['cols'] = list(lfa._comp.columns.values)

    if not lfa.lns is None:
      lns = lfa.lns.reindex(columns=list(COLS.FLNS_SAVE), copy=True).astype(COLS.FLNS_SAVE)
      gh.create_dataset('lns', data=lns.to_records())

    if not lfa.mlt is None:
      mlt = lfa.mlt.reindex(columns=list(COLS.FMLT_SAVE), copy=True).astype(COLS.FMLT_SAVE)
      gh.create_dataset('mlt', data=mlt.to_records())
  
    if not lfa.match_lf is None:
      gh.create_dataset('mtc_mids', data=lfa.match_lf.index.values)
      gh.create_dataset('mtc_fids', data=lfa.match_lf.values)

  def _save_diff(self, lfa, f, analysis):
    sh = f['analysis'][lfa.name].create_group(analysis)
    sh.attrs['dt'] = lfa.dt

    # --- Peaks ----------
    Bpks = pd.DataFrame.from_records(f['analysis'][lfa.name]['pks'][:]).set_index('pid')
    hA = lfa.pks_hash()
    hB = lfa._pks_hash(Bpks)

    Bin = Bpks[np.isin(hB,hA)]
    Ain = lfa.pks[np.isin(hA,hB)]

    Avls = Ain[['a','wl','fwhm','ph']].values
    Bvls = Bin[['a','wl','fwhm','ph']].values
    err = (200*np.abs(Avls-Bvls)/(Avls+Bvls)).max(axis=1)
    Ain = Ain.iloc[err<1.]

    sh.create_dataset('pks_base_idx', data=Bin.index[err<1.])
    sh.create_dataset('pks_base_pid', data=Ain.index.values)
    sh.create_dataset('pks_base_lid', data=Ain.lid.values)

    Aout = lfa.pks.drop(index=Ain.index).reindex(columns=list(COLS.FPKS_SAVE), copy=True).astype(COLS.FPKS_SAVE)
    sh.create_dataset('pks', data=Aout.to_records())

    # --- Fluos ----------
    if not lfa.flu is None:
      COLS_FLU = copy(COLS.FFLU_SAVE)
      COLS_FLU.update({ch: np.float32 for ch in lfa.flu.columns[3:]})

      Bflu = pd.DataFrame.from_records(f['analysis'][lfa.name]['flu'][:]).set_index('fid')
      hA = lfa.flu_hash()
      hB = lfa._flu_hash(Bflu)

      Bin = Bflu[np.isin(hB,hA)]
      Ain = lfa.flu[np.isin(hA,hB)]

      if ('FSC-H' in Ain) and ('FSC-A' in Ain):
        Avls = Ain[['FSC-H','FSC-A']].values
        Bvls = Bin[['FSC-H','FSC-A']].values
        err = (200*np.abs(Avls-Bvls)/(Avls+Bvls)).max(axis=1)
      else:
        err = np.zeros(Ain.shape[0], dtype=float)

      Ain = Ain.iloc[err<1.]
      sh.create_dataset('flu_base_idx', data=Bin.index[err<1.])
      sh.create_dataset('flu_base_fid', data=Ain.index.values)

      Aout = lfa.flu.drop(index=Ain.index).reindex(columns=list(COLS_FLU), copy=True).astype(COLS_FLU)
      sh.create_dataset('flu', data=Aout.to_records())

      if not lfa._comp is None:
        sh.create_dataset('comp', data=lfa._comp.values)
        sh['comp'].attrs['rows'] = list(lfa._comp.index.values)
        sh['comp'].attrs['cols'] = list(lfa._comp.columns.values)


    # --- Lines ----------
    if not lfa.lns is None:
      Blns = pd.DataFrame.from_records(f['analysis'][lfa.name]['lns'][:]).set_index('lid')
      hA = lfa.lns_hash()
      hB = lfa._lns_hash(Blns)

      Bin = Blns[np.isin(hB,hA)]
      Ain = lfa.lns[np.isin(hA,hB)]

      Avls = Ain[['a','wl','dwl','ph']].values
      Bvls = Bin[['a','wl','dwl','ph']].values
      err = (200*np.abs(Avls-Bvls)/(Avls+Bvls)).max(axis=1)
      Ain = Ain.iloc[err<1.]

      sh.create_dataset('lns_base_idx', data=Bin.index[err<1.])
      sh.create_dataset('lns_base_lid', data=Ain.index.values)
      sh.create_dataset('lns_base_mid', data=Ain.mid.values)

      Aout = lfa.lns.drop(index=Ain.index).reindex(columns=list(COLS.FLNS_SAVE), copy=True).astype(COLS.FLNS_SAVE)
      sh.create_dataset('lns', data=Aout.to_records())

    # --- Multi ----------
    if not lfa.mlt is None:
      mlt = lfa.mlt.reindex(columns=list(COLS.FMLT_SAVE), copy=True).astype(COLS.FMLT_SAVE)
      sh.create_dataset('mlt', data=mlt.to_records())

    # --- Match ----------
    if not lfa.match_lf is None:
      sh.create_dataset('mtc_mids', data=lfa.match_lf.index.values)
      sh.create_dataset('mtc_fids', data=lfa.match_lf.values)

  def save_lite(self, analysis, fname=None, suff='', peaks=True, warn=True):
    if fname is None: fname = self.base
    # Check analysis of each group
    lfas = {}
    for an in self.info:
      try:
        lfas[an] = self.get_analysis(an, analysis)
      except:
        if warn: warnings.warn(f'Acquisition {an} is not analyzed and will not be save!')
        else:    raise RuntimeError(f'Acquisition {an} is not analyzed!')
        continue
      
      if (lfas[an].lns is None) or (lfas[an].mlt is None): raise RuntimeError(f"Acquisition {an} hasn't been fully analyzed yet!")

    # Create new lase file
    file_lite = LaseFile(self.folder.joinpath(fname+suff+'.llf.lase'), mode='replace', version='2.0', ftype='llf')

    with file_lite.write() as flite:
      with self.read() as ffull:
        wl_axis = ffull['Exp_0']['wl_axis'][:].astype(np.float64)

        flite.create_group('info')
        flite['info'].attrs['start_time'] = self.start_time
        flite['info'].attrs['min_wl'] = np.min(wl_axis)
        flite['info'].attrs['max_wl'] = np.max(wl_axis)

        flite.create_group('analysis')
        flite['analysis'].attrs['anames'] = list(self.info)
        for aname in self.info:
          try:
            lfa = self.get_analysis(aname, analysis)
          except Exception as err:
            if warn: warnings.warn(f'Group {aname} is not analyzed and will not be save!')
            else:    raise RuntimeError(f'Group {aname} is not analyzed!')
            continue

          flite['analysis'].create_group(aname)
          ah = flite['analysis'][aname]
          ah.create_dataset('cmdCommands', data=ffull['Exp_0'][aname]['cmdCommands'][:])
          ah.create_dataset('cmdMarkers', data=ffull['Exp_0'][aname]['cmdMarkers'][:])
          ah.create_dataset('cmdTimestamps', data=ffull['Exp_0'][aname]['cmdTimestamps'][:])
          ah.attrs['Channel Description'] = ffull['Exp_0'][aname].attrs['Channel Description']

          if peaks:
            pks = lfa.pks.reindex(columns=COLS.FPKS_SAVE).astype(COLS.FPKS_SAVE)
            ah.create_dataset('pks', data=pks.to_records())
            ah.create_dataset('lid', data=lfa.pks.lid.values)
          ah.create_dataset('lns', data=lfa.lns.to_records())
          ah.create_dataset('mlt', data=lfa.mlt.to_records())

          if not lfa.flu is None: ah.create_dataset('flu', data=lfa.flu.to_records())
          if not lfa._comp is None:
            ah.create_dataset('comp', data=lfa._comp.values)
            ah['comp'].attrs['rows'] = list(lfa._comp.index.values)
            ah['comp'].attrs['cols'] = list(lfa._comp.columns.values)
          if not lfa.match_lf is None:
            ah.create_dataset('mtc_mids', data=lfa.match_lf.index.values)
            ah.create_dataset('mtc_fids', data=lfa.match_lf.values)

    logger.info(f'>>> Lite file saved: {fname}.llf.lase')

  def get_data(self, aname=None, analysis='base', peaks=True):
    lfa = self.get_analysis(aname, analysis)
    if (lfa.mlt is None) or (lfa.lns is None):
      raise IOError('File does not contain lines and/or multi data, cannot return data class!')

    pks = lfa.pks[lfa.pks.lid.isin(lfa.lns.index)] if peaks else None
    obj = LaseData_Flow_Guava(mlt=lfa.mlt, lns=lfa.lns, pks=pks, fts=lfa.flu, name=lfa.name, info=deepcopy(lfa.info))
    obj.mid2fid = lfa.match_lf
    return obj


class LaseFile_Lite_Flow(LaseFile):
  def __init__(self, *args, **kwds):
    super().__init__(*args, **kwds)
    if not self.ftype=='llf':
      raise IOError(f'File extension not compatible with LaseFile_Lite_Flow {self.ftype}')
    
    self.load_info()

  def load_info(self):
    with self.read() as f:
      self.info = {}
      for aname in f['analysis'].attrs['anames']:
        ts = f['analysis'][aname]['cmdTimestamps'][:].flatten()
        cmds = [cc.decode('ASCII') for cc in f['analysis'][aname]['cmdCommands'][:].flatten()]
        cmds = [None if cc=='None' else cc for cc in cmds]
        mrks = [mm.decode('ASCII') for mm in f['analysis'][aname]['cmdMarkers'][:].flatten()]
        mrks = ['' if mm=='-' else mm for mm in mrks]
        daq = LaseFile_Guava_Flow._parse_daq_table(f['analysis'][aname].attrs['Channel Description'])\
              if 'Channel Description' in f['analysis'][aname].attrs else None

        self.info[aname] = AcqInfo(aname, cmds, mrks, ts, daq)
      
      self.start_time = f['info'].attrs['start_time']

  def get_data(self, aname=None, peaks=True):
    if aname is None:
      if len(self.info)==1: aname = list(self.info)[0]
      else:                 raise ValueError(f'File has multiple groups. Must specify a name: {list(self.info)}!')
    else:
      if not aname in self.info: raise KeyError(f'Acquisition {aname} not present in the flow file!')

    with self.read() as f:
      gh = f['analysis'][aname]
      if (not 'lns' in gh) or (not 'mlt' in gh):
        raise IOError('File does not contain lines and/or multi data, cannot return data class!')

      lns = pd.DataFrame.from_records(gh['lns'][:]).set_index('lid').astype(COLS.FLNS)
      mlt = pd.DataFrame.from_records(gh['mlt'][:]).set_index('mid').astype(COLS.FMLT)
      flu = pd.DataFrame.from_records(gh['flu'][:]).set_index('fid').astype(COLS.FFLU)\
            if 'flu' in gh else None

      if peaks:
        pks = pd.DataFrame.from_records(gh['pks'][:]).set_index('pid')
        pks['E'] = KE/pks['wl']
        pks['fwhmE'] = pks['fwhm']*pks['E']/pks['wl']
        pks['lid'] = gh['lid'][:].astype(np.uint64)
        pks = pks.reindex(columns=COLS.FPKS).astype(COLS.FPKS)
        pks = pks[pks.lid.isin(lns.index.values)]

      else:
        pks = None
   
      mtc = pd.Series(gh['mtc_fids'][:].astype(np.uint64), index=pd.Index(gh['mtc_mids'][:], dtype=np.uint64))\
            if 'mtc_mids' in gh else None

    obj = LaseData_Flow_Guava(mlt=mlt, lns=lns, pks=pks, fts=flu, name=aname, info=deepcopy(self.info[aname]))
    obj.mid2fid = mtc
    return obj

# ------------------------------- ACQ INFO CLASS -------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class AcqInfo:
  def __init__(self, name, cmds, mrks, ts, daq, dt=0):
    self.name = name
    self.cmds = cmds
    self.mrks = mrks
    self.ts = ts
    
    self._sidxs = [0,len(self.mrks)-1]
    self._tlims = [(ts[0]+dt,ts[-1]+dt,'DAT_0')]
    self.labels = ['DAT_0']
    self._sdt = dt

    self.daq_table = daq

  def __repr__(self):
    return f'{self.name}|#{len(self.ts)} cmds'
  
  def __len__(self):
    return len(self.cmds)

  @property
  def sidxs(self):
    return self._sidxs

  @property
  def tlims(self):
    return self._tlims

  @property
  def t0(self):
    return self.ts[0]+self._sdt

  @property
  def t1(self):
    return self.ts[-1]+self._sdt

  def get_cmd(self, icmd):
    return [self.ts[icmd], self.cmds[icmd], self.mrks[icmd]]

  def set_splits(self, idxs, labels=None, dt=0.):
    self._sidxs = idxs
    self.labels = [f'DAT_{ii}' for ii in range(len(idxs)-1)] if labels is None else labels
    self._tlims = [(self.ts[idxs[ii]]+dt,self.ts[idxs[ii+1]]+dt,self.labels[ii]) for ii in range(len(idxs)-1)]
    self._sdt = dt

  def split(self):
    outs = []
    for ii in range(len(self._sidxs)-1):
      cmds = self.cmds[self._sidxs[ii]:self._sidxs[ii+1]+1]
      mrks = self.mrks[self._sidxs[ii]:self._sidxs[ii+1]+1]
      ts = self.ts[self._sidxs[ii]:self._sidxs[ii+1]+1]
      outs.append(AcqInfo(self.labels[ii], cmds, mrks, ts, self.daq_table, dt=self._sdt))
    
    return outs

  def plot_splits(self, width=1.):
    _,ax = plt.subplots(figsize=(20,2))
    ax.bar(self.ts,1,width=width, facecolor='lightgray')
    ax.bar(self.ts[self._sidxs]+self._sdt,1,width=width, facecolor='magenta')
    plt.show()

# -------------------------------- LASEV3 FILES --------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class LaseFile_LASEV3_CSV:
  NCOLS = 30

  def __init__(self, flase=None, ffluo=None, ffluo_comp=None, wl_axis=None):
    if not flase is None:
      ext = flase.suffix[1:]
      if not ext=='csv':
        raise IOError(f'File extension "{ext}" not compatible with LaseFile_LASEV3_CSV!')

    self.flase = flase
    self.ffluo = ffluo
    self.ffluo_comp = ffluo_comp
    self.wl_axis = np.arange(2048) if wl_axis is None else wl_axis

    self.load_info()
  
  def load_info(self):
    pass

  def get_data(self, colname='name', rename=None):
    if self.ffluo is None:
      fts = None

    elif self.ffluo.suffix=='.csv':
      fts = pd.read_csv(self.ffluo)
      if not rename is None: fts = fts[list(rename)].rename(columns=rename)
      fts = fts[fts.ID1>=0]
      fts.ID = np.uint64(fts.ID+1000000*fts.ID1)
      fts = fts.drop(columns=['ID1']).set_index('ID').astype({'nLP': int})
      fts.index.rename('fid', inplace=True)

    elif self.ffluo.suffix=='.fcs':
      fcs = FileFCS(self.ffluo)
      fts = fcs.df(colname)
      fts = fts[fts.ID1>=0]
      fts['ID'] = np.uint64(fts['ID']+1000000*fts['ID1'])
      fts = fts.drop(columns=['ID1']).set_index('ID')

      if not self.ffluo_comp is None:
        fcsC = FileFCS(self.ffluo_comp)
        ftsC = fcsC.df(colname)
        ftsC = ftsC[ftsC.ID1>=0]
        ftsC['ID'] = np.uint64(ftsC['ID']+1e6*ftsC['ID1'])
        ftsC = ftsC.drop(columns=['ID1']).set_index('ID')

        fts = fts[['Time','LP_Count_Cutoff','LP_Count']]
        ftsC = ftsC[[lab for lab in ftsC.columns if not lab in fts.columns]]
        fts = pd.concat([fts,ftsC],axis=1)        

      fts = fts.drop(columns=['LP_Count']).rename(columns={'LP_Count_Cutoff': 'nLP'}).astype({'nLP': int})
      fts.index.rename('fid', inplace=True)
    else:
      raise IOError(f'File extension "{self.fluo.suffix}" not compatible with LaseFile_LASEV3_CSV!')

    if not self.flase is None:
      pks = None
      ldf = pd.read_csv(self.flase)
      ldf.ID = np.uint64(ldf.ID+1000000*ldf.ID1)
      mid_dat = ldf.ID.values
      n_dat = ldf.Count.values
      wl_dat = ldf[[f'H_{ii}' for ii in range(self.NCOLS)]].values
      a_dat = ldf[[f'X_{ii}' for ii in range(self.NCOLS)]].values

      dat_lns = []
      dat_mlt = []
      for mid,n,wls,As in zip(mid_dat,n_dat,wl_dat,a_dat):
        cn = min(n,self.NCOLS)
        dat_lns.extend([(wls[ii],As[ii],mid) for ii in range(cn)])
        dat_mlt.append((mid,cn))

      lns = pd.DataFrame(dat_lns, columns=['wl','a','mid'])
      mlt = pd.DataFrame(dat_mlt, columns=['mid','n']).astype({'mid': np.uint64, 'n': int}).set_index('mid')

      lns.wl = np.interp(lns.wl, np.arange(2048), self.wl_axis)
      lns['E'] = KE/lns.wl
      lns.index.rename('lid', inplace=True)
    else:
      if fts is None: raise RuntimeError('At least one of fluo or lase must be defined!')
      pks = None
      lns = pd.DataFrame([], columns=['wl','a','mid'])
      mlt = pd.DataFrame({'mid': fts.index.values, 'n': np.zeros_like(fts.index.values, dtype=np.int32)}).astype({'mid': np.uint64, 'n': int}).set_index('mid')

    return LaseData_Flow_LaseV3(mlt,lns,pks,fts,None)


