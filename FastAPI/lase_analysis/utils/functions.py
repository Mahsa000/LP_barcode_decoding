import numpy as np
import struct


def _calc_row_idx(k, n):
  return int(np.ceil((1/2.)*(-(-8*k+4*n**2-4*n-7)**0.5+2*n-1)-1))

def _elem_in_i_rows(i, n):
  return i*(n-1-i)+(i*(i+1))//2

def _calc_col_idx(k, i, n):
  return int(n-_elem_in_i_rows(i+1,n)+k)

def condensed_to_square(k, n):
  i = _calc_row_idx(k, n)
  j = _calc_col_idx(k, i, n)
  return i, j

def square_to_condensed(i, j, n):
  assert i != j, "no diagonal elements in condensed matrix"
  if i < j: i, j = j, i
  return n*j - j*(j+1)//2 + i - 1 - j


def unravel(idx, shape):
  i = idx%shape[2]
  temp = int((idx-i)//shape[2])
  j = temp%shape[1]
  k = int(temp//shape[1])

  return k,j,i

def ravel(crd, shape):
  return crd[2]+crd[1]*shape[2]+crd[0]*shape[2]*shape[1]

MSK32 = np.uint64(2**32-1)
MSK16 = np.uint64(2**16-1)
MSK8 = np.uint64(2**8-1)
FSCRS = {'ana': lambda x: ((x&(MSK16<<np.uint64(48)))>>48).astype(np.float32)/65535.,
         'dig': lambda x: ((x&(MSK16<<np.uint64(32)))>>32).astype(np.float32)/65535.,
         'tot': lambda x: (((x&(MSK16<<np.uint64(48)))>>48).astype(np.float32)/65535.)*\
                          (((x&(MSK16<<np.uint64(32)))>>32).astype(np.float32)/65535.),
         'mH': lambda x: ((x&(MSK8<<np.uint64(16)))>>16).astype(np.uint8),
         'mL': lambda x: ((x&(MSK8<<np.uint64(8)))>>8).astype(np.uint8),
         'nx': lambda x: (x&MSK8).astype(np.uint8)}

def convert_score(vals, what=None):
  if what is None: what = list(FSCRS.keys())

  if isinstance(what, str): return FSCRS[what](vals)
  else:                     return {kk: FSCRS[kk](vals) for kk in what}

FPRBS = {'p0': lambda x: np.frombuffer((x>>np.uint64(32)).astype(np.uint32).tobytes(), dtype=np.float32),
         'mH': lambda x: ((x&(MSK8<<np.uint64(16)))>>16).astype(np.uint8),
         'mL': lambda x: ((x&(MSK8<<np.uint64(8)))>>8).astype(np.uint8),
         'nx': lambda x: (x&MSK8).astype(np.uint8)}

def convert_prob(vals, what=None):
  if what is None: what = list(FPRBS.keys())

  if isinstance(what, str): return FPRBS[what](vals)
  else:                     return {kk: FPRBS[kk](vals) for kk in what}


