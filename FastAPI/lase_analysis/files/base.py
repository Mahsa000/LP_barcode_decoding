import h5py as h5
from pathlib import Path

MODES_TABLE = {'r': 'read', 'r+': 'write', 'x': 'create', 'w': 'replace'}
TYPES_TABLE = {'spectrum': 'spt', 'threshold': 'thr', 'lasemap': 'map',
               'litemap': 'llm', 'laseflow': 'flw', 'liteflow': 'llf'}

class LaseFile:
  EXTS = ['spt', 'thr', 'map', 'llm', 'flw', 'llf']

  def __init__(self, file, mode='read', version=None, ftype=None, update=False, **kwds):
    self.file = Path(file)
    if self.extension!='lase': raise IOError('File extension not of a LASE file!')
    
    if mode in MODES_TABLE: mode = MODES_TABLE[mode]


    if mode in ('read','write'):
      with self.read() as f: pass

    elif mode in ('create','replace'):
      if (version is None) or (ftype is None): raise ValueError('New file must declare version and type!')
      if not ftype in LaseFile.EXTS:           raise ValueError(f'Filetype "{ftype}" not recognized!')
      with getattr(self, mode)() as f:
        f.attrs['type'] = ftype
        f.attrs['version'] = version
    
    else:
      raise ValueError(f'Mode "{mode}" not recongized!')

    if update and (mode=='write'):
      if version is None and ftype is None:
        raise ValueError('Version and ftype are both None, cannot update!')
      with self.write() as f:
        if not version is None: f.attrs['version'] = version
        if not ftype is None:   f.attrs['type'] = ftype

    elif update:
      raise ValueError('Cannot update if mode is not write!')


    with self.read() as f:
      self.version = f.attrs['version']
      ftype = f.attrs['type']
      if ftype in TYPES_TABLE: ftype = TYPES_TABLE[ftype]
    
    if ftype != self.ftype: raise IOError('File extension does not match with file type!')


  @property
  def folder(self):
    return self.file.parent

  @property
  def name(self):
    return self.file.name

  @property
  def base(self):
    return '.'.join(self.name.split('.')[:-2])

  @property
  def ftype(self):
    return self.name.split('.')[-2]

  @property
  def extension(self):
    return self.name.split('.')[-1]

  @property
  def exist(self):
    return self.file.is_file()


  def read(self):
    return h5.File(self.file, 'r')

  def write(self):
    return h5.File(self.file, 'r+')

  def create(self):
    return h5.File(self.file, 'w-')

  def replace(self):
    return h5.File(self.file, 'w')

  def write_create(self):
    return h5.File(self.file, 'a')
