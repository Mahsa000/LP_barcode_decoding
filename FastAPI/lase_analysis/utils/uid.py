import numpy as np
from numbers import Number

class UID:
  MASK_UID = np.uint64(np.uint32(2**32-1) << 32)
  MASK_IDX = np.uint64(2**32-1)

  def __init__(self, val=None):
    if isinstance(val,str): val = np.uint64(int(val+'00000000',16))

    if val is None:               self.val = np.random.randint(0,2**32-1, dtype=np.uint64)
    elif isinstance(val, Number): self.val = np.uint64((val&UID.MASK_UID) >> np.uint64(32))

  def __str__(self):
    return self.hex

  def __repr__(self):
    return self.hex

  def __index__(self):
    return int(self.val)


  @property
  def reserved(self):
    return self.val < 100

  @property
  def hex(self):
    ss = hex(self)[2:].upper()
    return '0'*(8-len(ss)) + ss

  @property
  def full(self):
    return np.uint64(self.val << np.uint64(32))


  def __eq__(self, other):
    return self.val == other.val

  def __ne__(self, other):
    return self.val != other.val

  def __lt__(self, other):
    return self.val < other.val

  def __le__(self, other):
    return self.val <= other.val

  def __gt__(self, other):
    return self.val > other.val

  def __ge__(self, other):
    return self.val >= other.val
