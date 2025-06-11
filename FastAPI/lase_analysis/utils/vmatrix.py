import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, spmatrix

from numbers import Number


class vmatrix_sps:
  def __init__(self, mat, values):
    self.mat = mat
    self.values = values
    for key,val in self.values.items(): setattr(self, key, _indexer(self.mat, val))

  @property
  def indptr(self):
    return self.mat.indptr
  
  @property
  def indices(self):
    return self.mat.indices

  @property
  def shape(self):
    return self.mat.shape

  @property
  def T(self):
    new_mat = self.mat.T
    if isinstance(new_mat, csr_matrix): new_mat = new_mat.tocsc()
    if isinstance(new_mat, csc_matrix): new_mat = new_mat.tocsr()

    return self.__class__(new_mat.indptr, new_mat.indices, new_mat.shape,
                          {key: self.values[key][new_mat.data-1] for key in self.values})

  def __add__(self, other):
    if type(self) != type(other):
      raise TypeError('Matrices must be of the same type!')
    for key in self.values:
       if not key in other.values: raise KeyError(f'Second matrix does not have value {key}')
    for key in other.values:
       if not key in self.values: raise KeyError(f'First matrix does not have value {key}')

    return self.__class__(np.concatenate([self.indptr, other.indptr[1:]+self.indptr[-1]]),
                          np.concatenate([self.indices, other.indices+self.shape[1]]),
                          (self.shape[0]+other.shape[0], self.shape[1]+other.shape[1]),
                          {key: np.concatenate([self.values[key], other.values[key]]) for key in self.values})

  def __radd__(self, other) :
    return self.__add__(other)
  
  def __or__(self, other):
    for key in self.values:
       if not key in other.values: raise KeyError(f'Second matrix does not have value {key}')
    for key in other.values:
       if not key in self.values: raise KeyError(f'First matrix does not have value {key}')

    if (self.mat>0).multiply(other.mat>0).nnz > 0: raise ValueError('The two scores are not orthogonal!')

    tmp = other.mat.copy()
    tmp.data += self.mat.data.max()
    new_mat = self.mat + tmp
    del tmp

    return self.__class__(new_mat.indptr, new_mat.indices, new_mat.shape,
                          {key: np.concatenate([self.values[key],other.values[key]])[new_mat.data-1] for key in self.values})
  
  def __ror__(self, other):
    return self.__or__(other)
  
  def sparse(self, key):
    ret = self.mat.copy()
    ret.data = self.values[key].copy()
    return ret


class vmatrix_csr(vmatrix_sps):
  def __init__(self, indptr, indices, shape, values):
    super().__init__(csr_matrix((np.arange(len(indices))+1, indices, indptr), shape), values)
  
  @classmethod
  def from_csr(cls, mat, name='data'):
    return cls(mat.indptr.copy(), mat.indices.copy(), mat.shape, {name: mat.data.copy()})

  @classmethod
  def from_array(cls, mat, name='data'):
    mat = csr_matrix(mat)
    return cls(mat.indptr, mat.indices, mat.shape, {name: mat.data})


class vmatrix_csc(vmatrix_sps):
  def __init__(self, indptr, indices, shape, values):
    super().__init__(csc_matrix((np.arange(len(indices))+1, indices, indptr), shape), values)

  @classmethod
  def from_csc(cls, mat, name='data'):
    return cls(mat.indptr, mat.indices, mat.shape, {name: mat.data})

  @classmethod
  def from_array(cls, mat, name='data'):
    mat = csc_matrix(mat)
    return cls(mat.indptr, mat.indices, mat.shape, {name: mat.data})


class _indexer:
  def __init__(self, mat, val):
    self.mat = mat
    self.val = val

    for met in ('__eq__','__ne__','__lt__','__gt__','__le__','__ge__'):
      setattr(self, met, lambda val: getattr(self.val,met)(val))
  
  @property
  def dtype(self):
    return self.val.dtype
  
  @property
  def data(self):
    return self.val

  def __getitem__(self, key):
    idxs = self.mat[key]

    if isinstance(idxs, np.ndarray) and (idxs.dtype==bool):
      return self.val[idxs]

    if isinstance(idxs, Number):
      return self.val[idxs-1] if idxs>0 else self.dtype.type(0)

    if isinstance(idxs, spmatrix):
      idxs = idxs.data
    elif isinstance(idxs, np.matrix):
      idxs = idxs.A1
    else:
      raise TypeError(f'Indexer not recognized: {type(ret)}')

    ret = np.zeros_like(idxs, dtype=self.dtype)
    ret[idxs>0] = self.val[idxs[idxs>0]-1]
    return ret

  def __eq__(self, val):
    return self.val==val
  
  def __ne__(self, val):
    return self.val!=val
  
  def __gt__(self, val):
    return self.val>val
  
  def __lt__(self, val):
    return self.val<val
  
  def __ge__(self, val):
    return self.val>=val
  
  def __le__(self, val):
    return self.val<=val