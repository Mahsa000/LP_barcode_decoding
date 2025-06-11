import pandas as pd
import numpy as np
import array

# ------------------------------------------------------------------------------
# ---------------------------------- FCS FILE ----------------------------------
# ------------------------------------------------------------------------------

class FileFCS:
  def __init__(self, file):
    self.file = file
    self._load_info()
    self._load_params()
    self._load_data()

  def _read(self, i0, n):
    with open(self.file, 'rb') as f:
      f.seek(i0)
      ret = f.read(n)
    return ret

  def _load_info(self):
    i0 = int(self._read(10,8))
    i1 = int(self._read(18,8))
    head = self._read(i0, i1-i0)

    dlt = head[0:1].decode()

    temp = head.decode().split(dlt)[1:]
    self.info = {temp[2*ii]: temp[2*ii+1] for ii in range(len(temp)//2)}

  def _load_params(self):
    self.pars = []
    for ip in range(self.npar):
      self.pars.append({'nbit': int(self.info[f'$P{ip+1}B']), 'range': int(self.info[f'$P{ip+1}R']),
                         'name': self.info[f'$P{ip+1}N'], 'ampl': self.info[f'$P{ip+1}E']})
      if f'$P{ip+1}S' in self.info:  self.pars[-1]['short'] = self.info[f'$P{ip+1}S']
      if f'$P{ip+1}G' in self.info:  self.pars[-1]['gain'] = float(self.info[f'$P{ip+1}G'])

  def _load_data(self):
    i0 = int(self._read(26,8))
    i1 = int(self._read(34,8))
    fmt = self.info['$DATATYPE'].lower()
    if self.info['$BYTEORD'] == '1,2,3,4':   swap = False
    elif self.info['$BYTEORD'] == '4,3,2,1': swap = True
    else:
      raise ValueError(f"ByteOrder not recognized ({self.info['$BYTEORD']})")

    arr = array.array(fmt, self._read(i0, i1-i0+1))
    if swap: arr.byteswap()
    self.data = np.array(arr).reshape(-1,self.npar)

  @property
  def n(self):
    return int(self.info['$TOT'])

  @property
  def npar(self):
    return int(self.info['$PAR'])

  def df(self, colname='name'):
    return pd.DataFrame(self.data, columns=[pp[colname] for pp in self.pars])
