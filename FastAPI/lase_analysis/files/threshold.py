import numpy as np
import pandas as pd

from .base import LaseFile
from ..data import Spectrum
from ..utils.constants import K_nm2meV as KE, COLS, LIDS
from ..utils.logging import logger

# ------------------------------- THRESHOLD FILE -------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class LaseFile_Threshold(LaseFile):
  def __init__(self, *args, **kwds):
    super().__init__(*args, **kwds)
    if not self.ftype=='thr':
      raise IOError(f'File extension not compatible with LaseFile_Threshold {self.ftype}')

  @property
  def powers(self):
    with self.read() as f:
      return f['data']['powers'][:].astype(np.float64)

  @property
  def exposure(self):
    with self.read() as f:
      return f['info']['camera'].attrs['exposure']

  @property
  def data(self):
    with self.read() as f:
      return f['data']['spectra'][:,:].astype(np.float64)

  @property
  def wl_axis(self):
    with self.read() as f:
      return f['data']['wl_axis'][:].astype(np.float64)

      
