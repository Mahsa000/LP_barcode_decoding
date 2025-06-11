import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sklearn.metrics import r2_score

from copy import copy

from ..utils.lineshapes import f_gaussian1, f_gaussian2
from ..utils.constants import K_nm2meV as KE

import warnings

_SATURATION = 3850


class Spectrum:
  """
  Spectrum class handles single spectrum data and can be initialized either from
  actual (x, y) data arrays or by loading from a .spt.lase file.

  The class supports peak finding and fitting operation. Each peak is first identified
  by finding local maxima; each of these maxima is then fitted with a (saturated)
  gaussian function to extract peak information (a: amplitude, wl: central wavelength,
  fwhm: full-width at half-maximum, ph: #photons).

  Before peak analysis, it is suggested to remove any background from the y data,
  which can be due to camera baseline oscillations or broad fluorescence emission.
  """

  QE = np.poly1d([-3.715e-7, 0.001317, -0.263])
  GAIN = {'1x': 520, '1.3x': 390, '2x': 325, '4x': 270}
  FIT_COLUMNS = ['A', 'wl', 'fwhm', 'ph']

  def __init__(self, y, x=None, gain=None, sid=0):
    if y.ndim > 1:
      raise ValueError(f'y data should be a 1-dimensional array (got {y.ndim})')

    if x is None: x = np.arange(len(y))
    if y.shape != x.shape:
      raise ValueError(f'x array shape do not match with y ({x.shape} vs {y.shape})')

    self.x = x
    self.y = y

    self.ybg = None
    self.gain = gain

    self.pks = self.empty_peak
    self.id = sid

# --------------------------------- Properties ---------------------------------

  @property
  def dx(self):
    i0 = len(self.x)//2
    return self.x[i0]-self.x[i0-1]

  @property
  def empty_peak(self):
    return pd.DataFrame(columns=Spectrum.FIT_COLUMNS, dtype=np.float64)

  @property
  def peaks(self):
    return self.pks

# -------------------------- Peak finding and fitting --------------------------

  def remove_background(self, filter_window=101):
    """
    Remove background from the spectrum. The background is estimated by median filter
    of the data with a window much larger than and individual peak size (in pixels).
    The background is removed from the y data and saved as an internal parameter.

    Note that if the background has already been calculated, this function returns
    without doing anything.

    Args:
      filter_window: window of the median filter. Default is 101.
    """
    if not self.ybg is None: return

    self.y, self.ybg = self.spt_filter(self.y, filter_window)

  def find_maxima(self, threshold):
    """
    Finds local maxima in the spectrum. Local maxima are identified with a iterative
    procedure. At each step, it looks for the maximum in the y data. If its intensity
    is above a certain threshold, it saves its position as a local maximum, otherwise
    the procedure is stopped. After the local maximum has been found, all data around
    it above zeros is eliminated from the y array and the procedure is repeated.

    Args:
      thrshold: minimum intensity value for a local maximum to be retained.

    Returns:
      x_max: position (in units of x axis) of the local maxima found
    """
    return self.spt_maxima(self.y, threshold)

  def fit_peaks(self, opt=None, **kwds):
    """
    Find and fit peaks in the spectrum. Local maxima are first identified wth the
    'find_maxim' method, then each of them is fitted with a (saturated) gaussian
    function to extract the peak parameters (a, wl, fwhm,).

    By default, the function removes the backgroudn if not done yet. To avoid, pass
    'False' as med_filter in the fit options.

    After fitting, each peak is retained only if its FWHM is at least 1.5 times
    the resolution of the x axis. This is done to remove spurious noise peaks.
    Peaks are saved and returned as a pandas DataFrame.

    Args:
      opt: PeakFitOpts object to specify parameters of peak fitting. If None, create
      a new object.

      kwds: additional parameters to initialize the PeakFitOpts object (if opt is None);
      can be a combination of one of the following:
      - med_filter: window of the median filter for backgroudn removal. Default is 101.
      - threshold: threshold level for finding local maxima.
      - window: window (in units of the x axis) around the local maxima in which
        the data are fitted.w
      - bounds: bounds for the fitting parameters (a, wl, fwhm). Each bound is a
        tuple of 2 elements: [p_min, p_max]. To specifiy bounds for all parameters,
        a 3-element list of bounds should be passed; if only one bound is passed,
        it is applied to all parameters.
      - saturation: level of camera saturation for clipping gaussian fitting function.
        Default is 3850.
      - gain: gain level of the camera. If spectrum is loaded from .spt file, default
        is the value in the file info; otherwise, default is '4x'.

    Returns:
      peaks: pandas DataFrame with a row entry for each peak found and columns
      reporting the fit parameters (a, wl, fwhm) and photons (ph).
    """
    if not self.gain is None and not 'gain' in kwds: kwds['gain'] = self.gain
    if opt is None: opt = PeakFitOpts(**kwds)

    if opt.med_filter: self.remove_background(opt.med_filter)

    clst = int(1.5*opt.window/self.dx)
    midx = self.spt_maxima(self.y, opt.threshold) if opt.method==1 else\
           self.spt_maxima2(self.y, opt.threshold, opt.prominence, opt.distance, clst, opt.saturation)
           
    if len(midx)>0:
      ffit = self.spt_fit if opt.method==1 else self.spt_fit2
      out = ffit(self.x, self.y, self.id, midx, opt.window, opt.fwhmE0, opt.gain, opt.saturation)\
            if len(midx) > 0 else np.zeros((0,5))
      self.pks = pd.DataFrame(out[:,0:4], columns=Spectrum.FIT_COLUMNS).dropna()
    
    return self.pks

  def get_fit_curve(self, peak=0, window=5, x=None):
    if self.pks is None:   raise RuntimeError('Peaks not fitted yet!')
    if len(self.pks) == 0: return np.zeros_like(self.x)

    return sum([f_gaussian1(self.x, a, wl, fwhm, np.inf) for a,wl,fwhm in self.pks[['A','wl','fwhm']].values])

  @staticmethod
  def spt_background(y, flt):
    return median_filter(y, flt)

  @staticmethod
  def spt_filter(y, flt):
    bg = median_filter(y, flt)
    return y-bg, bg

  @staticmethod
  def spt_maxima(y, thr):
    idx_maxima = []
    y2 = y.copy()

    while True:
      imax = np.argmax(y2)
      if y2[imax] < thr: break

      idx_maxima.append(imax)
      right_0s = np.flatnonzero(y2[imax:-1] < 0)
      if right_0s.size == 0: i_right = y2.size
      else:                  i_right = imax+right_0s[0]

      left_0s = np.flatnonzero(y2[0:imax] < 0)
      if left_0s.size == 0: i_left = 0
      else:                 i_left = left_0s[-1]

      y2[i_left:i_right] = 0

    idx_maxima = np.array(idx_maxima)
    idx_maxima = idx_maxima[(idx_maxima>1)&(idx_maxima<y.shape[-1]-2)]

    return idx_maxima

  @staticmethod
  def spt_maxima2(y, thr, prm, dst, clst, sat):
    pks = find_peaks(y ,height=thr, distance=dst, prominence=prm)[0]
    pks = pks[(pks>1)&(pks<y.shape[-1]-2)]

    if len(pks)>1:
      out = []
      for pps in np.split(pks, np.where(np.diff(pks)>clst)[0]+1):
        tmp = list(pps)
        # Check for multiple peaks in saturated regions
        while True:
          for ii in range(len(tmp)-1):
            if y[tmp[ii]:tmp[ii+1]].min() > 0.95*sat:
              tmp[ii] = (pps[ii]+pps[ii+1])//2
              del tmp[ii+1]
              break
          else:
            break
        out.append(np.array(tmp))      
      return out

    else:
      return [pks]

  @staticmethod
  def spt_fit(x, y, ispt, midx, wdw, fwhmE0, gain, sat=_SATURATION):
    pinfo = []
    for mm in midx:
      dx = x[mm+1]-x[mm]
      pwdw = int(np.ceil(wdw/dx))

      xfit = x[max(0,mm-pwdw):min(x.shape[-1],mm+pwdw)]
      yfit = y[max(0,mm-pwdw):min(y.shape[-1],mm+pwdw)]

      max_int = np.max(yfit)
      area = np.abs(yfit.sum())
      par0 = [max_int, x[mm], 0.939*area*dx/max_int]
      try:
        fun = lambda x, *par: f_gaussian1(x, *par, sat)
        pfit, _ = curve_fit(fun, xfit, yfit, p0=par0, bounds=[0,np.inf])
        if area==0.: area = 1.0649*pfit[2]*pfit[0]/dx
        ph = area*Spectrum.GAIN[gain]/Spectrum.QE(pfit[1])
        if pfit[2] > 1.5*dx: pinfo.append(np.append(pfit, [ph, ispt]))
          # if opt.R2:
          #   r2 = r2_score(yfit,fun(xfit,*par))
          #   pinfo[-1] = np.append(pinfo[-1],[r2])
      except Exception as err:
        pass

    # if opt.R2:
    #   if len(pinfo) > 0: return np.stack(pinfo).reshape(-1,6)
    #   else:              return np.array([[np.nan, np.nan, np.nan, np.nan, ispt, np.nan]])
    # else:
    if len(pinfo) > 0: return np.stack(pinfo).reshape(-1,5)
    else:              return np.array([[np.nan, np.nan, np.nan, np.nan, ispt]])

  @staticmethod
  def spt_fit2(x, y, ispt, midx, wdw, fwhmE0, gain, sat=_SATURATION):
    pinfo = []
    for mms in midx:
      if len(mms)==0: continue
      dx = x[mms[0]+1]-x[mms[0]]
      pwdw = int(np.ceil(wdw/dx))

      Nm = len(mms)
      if Nm==0: continue
      xfit = x[max(0,mms[0]-pwdw):min(x.shape[-1],mms[-1]+pwdw)]
      yfit = y[max(0,mms[0]-pwdw):min(x.shape[-1],mms[-1]+pwdw)]
      p0s = sum([[y[mm],x[mm],x[mm]*x[mm]*fwhmE0/KE] for mm in mms],[])

      if Nm==1:   ffit = lambda x,*par: f_gaussian1(x, *par, sat)
      elif Nm==2: ffit = lambda x,*par: f_gaussian2(x, *par, sat)
      else:       ffit = lambda x,*par: sum([f_gaussian1(x, *par[3*ii:3*(ii+1)], sat) for ii in range(Nm)])
      try:
        pfit,_ = curve_fit(ffit, xfit, yfit, p0=p0s, bounds=[0,np.inf])
      except Exception as err:
        # warnings.warn(f'Cannot fit, spt #{ispt}')
        continue

      if Nm>1:
        pfit = pfit.reshape(-1,3)
        splt = np.where(np.diff(np.sort(pfit[:,1]))>2*dx)[0]+1
        # In case two peak fits are collapsed on the same wavelength
        if len(splt)!=len(mms)-1:
          # warnings.warn(f'Two peaks fit at the same wavelength (spt {ispt})!')
          p0s = sum([[sub[:,0].sum(), sub[:,1].mean(), sub[:,2].mean()] for sub in np.split(pfit,splt)],[])
          Nm = len(splt)+1
          if Nm==1:   ffit = lambda x,*par: f_gaussian1(x, *par, sat)
          elif Nm==2: ffit = lambda x,*par: f_gaussian2(x, *par, sat)
          else:       ffit = lambda x,*par: sum([f_gaussian1(x, *par[3*ii:3*(ii+1)], sat) for ii in range(Nm)])
          try:    pfit,_ = curve_fit(ffit, xfit, yfit, p0=p0s, bounds=[0,np.inf])
          except: pfit = np.array(p0s)
          pfit = pfit.reshape(-1,3)
        
        out = np.zeros((pfit.shape[0],5), dtype=float)
        out[:,:3] = pfit.reshape(-1,3)
        out[:,3] = [1.0649*row[2]*row[0]/dx * Spectrum.GAIN[gain]/Spectrum.QE(row[1])
                    for row in pfit]
        out[:,4] = ispt
      
      else:
        area = np.abs(yfit.sum())
        if area==0.: area = 1.0649*pfit[2]*pfit[0]/dx
        out = np.zeros((1,5), dtype=float)
        out[0,:3] = pfit
        out[0,3] = area*Spectrum.GAIN[gain]/Spectrum.QE(pfit[1])
        out[0,4] = ispt
    
      pinfo.extend([row for row in out if row[2]>1.5*dx])

    if len(pinfo) > 0: return np.array(pinfo)
    else:              return np.array([[np.nan, np.nan, np.nan, np.nan, ispt]])

# ------------------------------ Window functions ------------------------------

  def window(self, x0, Dx):
    """
    Return a window of the x and y data around a specific point.

    Args:
      x0: center position of the window (in units of the x axis).

      Dx: size of the window around the central position (in units of the x axis).

    Returns:
      x: x axis in the window

      y: y data in the window.
    """
    i0 = np.argmax(self.x >= x0-Dx/2)
    i1 = np.argmax(self.x >= x0+Dx/2)
    if i1 == 0: i1 = self.x.size

    return self.x[i0:i1], self.y[i0:i1]

  def window_spectrum(self, ctr, wdw):
    x, y = self.window(ctr, wdw)
    return Spectrum(x=x, y=y, id=self.id)

# ------------------------------- OPTIONS CLASS --------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class PeakFitOpts:
  OPTS = {'med_filter': 101, 'threshold': 18, 'prominence': 18, 'distance': 3,
          'window': 2.5, 'bounds': [0,np.inf], 'fwhmE0': .75, 'saturation': 3850,
          'gain': '4x', 'R2': False, 'method': 2}

  def __init__(self, **kwds):
    for opt, val in PeakFitOpts.OPTS.items():
      setattr(self, opt, kwds[opt] if (opt in kwds) else val)