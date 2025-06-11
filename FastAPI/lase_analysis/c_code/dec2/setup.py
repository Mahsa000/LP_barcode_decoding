from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy


numpy_include = numpy.get_include()


ext = Extension('functions', ['functions.pyx','../c_utils/c_dict/c_dict.c','../c_utils/c_argsort/c_argsort.c'],
                include_dirs=[numpy_include,'../c_utils/c_dict','../c_utils/c_argsort'],
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                language='c')

setup(
    ext_modules = cythonize(ext, annotate=True, compiler_directives={'language_level': '3'}),
)
