cimport cython

import numpy as np
cimport numpy as np

CINT32 = np.int32
CUINT8 = np.uint8
ctypedef np.int32_t CINT32_t
ctypedef np.uint8_t CUINT8_t

cdef extern from "c_dict.h":
  struct item:
    CINT32_t key
    CINT32_t value

  void set_item(item** dictionary, CINT32_t key, CINT32_t value)
  CINT32_t get_item(item** dictionary, CINT32_t key, CINT32_t *value)
  CINT32_t contains_key(item** dictionary, CINT32_t key)
  void delete_all(item** dictionary)

cdef item* cdict=NULL

def dset(CINT32_t key, CINT32_t val):
  set_item(&cdict, key, val)
  return 0

def dget(CINT32_t key):
  cdef CINT32_t ret, val
  ret = get_item(&cdict, key, &val)
  if ret<0: raise RuntimeError(f'Error {ret}')
  return val

def dhas(CINT32_t key):
  cdef CINT32_t ret, val
  return bool(contains_key(&cdict, key))

def dclc():
  delete_all(&cdict)
  return 0