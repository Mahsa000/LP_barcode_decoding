from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy


numpy_include = numpy.get_include()


ext = Extension('functions', ['functions.pyx'],
                include_dirs=[numpy_include],
                language='c')

setup(
    ext_modules = cythonize(ext, annotate=True, compiler_directives={'language_level': '3'}),
)