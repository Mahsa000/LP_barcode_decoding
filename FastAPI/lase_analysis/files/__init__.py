from .base import LaseFile
from .lasemap import LaseFile_Full_Map, LaseFile_Lite_Map, LaseFile_LiteOld_Map
from .laseflow import LaseFile_Guava_Flow, LaseFile_Lite_Flow, LaseFile_LASEV3_CSV
from .threshold import LaseFile_Threshold
# from .flow import FileFCS

def read_lasefile(fpath, ltype=None, **kwds):
  ext = str(fpath).split('.')[-1]
  if ext=='lase':
    file = LaseFile(fpath,**kwds)
    if   file.ftype == 'map':
      return LaseFile_Full_Map(fpath,**kwds)
    
    elif file.ftype == 'llm':
      vA = int(file.version.split('.')[0])
      if vA < 3: return LaseFile_LiteOld_Map(fpath,**kwds)
      else:      return LaseFile_Lite_Map(fpath,**kwds)

    elif file.ftype == 'flw':
      return LaseFile_Guava_Flow(fpath,**kwds)
    
    elif file.ftype == 'llf':
      vA = int(file.version.split('.')[0])
      if vA < 2: raise NotImplementedError('Support for old llf files not implemented yet!')
      return LaseFile_Lite_Flow(fpath, **kwds)

    elif file.ftype == 'thr':
      return LaseFile_Threshold(fpath, **kwds)

    else:
      raise IOError('File type unknown!!')
    
  elif ext=='csv':
    if ltype=='LASEV3':
      return LaseFile_LASEV3_CSV(flase=fpath, **kwds)

    elif ltype=='FLUOV3':
      return LaseFile_LASEV3_CSV(ffluo=fpath, **kwds)

    else:
      raise IOError(f'File type {ltype} unknown!!')

  else:
    raise IOError(f'File extension {ext} not recognized!!')