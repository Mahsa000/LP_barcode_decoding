import numpy as np
from .functions import condensed_to_square, square_to_condensed
from scipy.spatial.distance import squareform

class SplitArray:
  def __init__(self, vals, cums):
    self._vals = vals
    self._cums = cums

  def __getitem__(self, idx):
    return self._vals[self._cums[idx]:self._cums[idx+1]]

  def __len__(self):
    return len(self._vals)

  @property
  def all(self):
    return self._vals

  @property
  def list(self):
    return np.split(self._vals, self._cums[1:])[:-1]

  @property
  def min(self):
    return np.min(self._vals)

  @property
  def max(self):
    return np.max(self._vals)

  def apply(self, func):
    return SplitArray(func(self._vals), self._cums)


class CondensedMatrix:
  def __init__(self, data):
    N = len(data)
    M = (1+(1+8*N)**0.5)/2
    if np.abs(M-np.round(M)) > 1e-6:
      raise ValueError('Number of element in data array not compatible with condensed matrix!')

    self._M = int(M)
    self._N = N
    self.data = np.array(data)

  def __len__(self):
    return self._N

  def __str__(self):
    return str(self.data)

  def __repr__(self):
    return repr(self.data)

  def __getitem__(self, idx):
    if isinstance(idx, tuple):
      if len(idx) != 2:
        raise IndexError(f'Too many indices for array (2-dimensional)')
      return self.data[square_to_condensed(idx[0],idx[1],self._N)]
    elif isinstance(idx, (list,int)):
      return self.data[idx]
    elif isinstance(idx, np.ndarray):
      if idx.ndim == 1:
        return self.data[idx]
      elif idx.ndim == 2:
        return np.array([self.data[square_to_condensed(i,j,self._N)]
                         for i,j in idx])
      else:
        raise IndexError(f'Too many indices for array (2-dimensional)')
    else:
      raise IndexError(f'Cannot understand indexing!')

  def coords(self, k):
    if k >= self._N: raise IndexError('Index k too large!')

    return condensed_to_square(k, self._M)

  def index(self, i, j):
    if i >= self._M: raise IndexError('Index i too large!')
    if j >= self._M: raise IndexError('Index j too large!')

    return square_to_condensed(i, j, self._M)

  def square(self):
    return squareform(self.data)


  @property
  def shape(self):
    return (self._M, self._M)
