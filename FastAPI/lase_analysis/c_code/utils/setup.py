# # from distutils.core import setup
# # from distutils.extension import Extension
# # from Cython.Build import cythonize
# # import numpy


# # numpy_include = numpy.get_include()


# # ext = Extension('functions', ['functions.pyx'],
# #                 include_dirs=[numpy_include],
# #                 define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
# #                 language='c')

# # setup(
# #     ext_modules = cythonize(ext, annotate=True, compiler_directives={'language_level': '3'}),
# # )

# from distutils.core import setup
# from distutils.extension import Extension
# from Cython.Build import cythonize
# import numpy

# numpy_include = numpy.get_include()

# ext = Extension(
#     "functions",
#     ["functions.pyx"],
#     include_dirs=[numpy_include],
#     define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
#     extra_compile_args=["-O2"],  # Add architecture flag
#     language="c",
# )

# setup(
#     name="functions",
#     ext_modules=cythonize(
#         ext, annotate=True, compiler_directives={"language_level": "3"}
#     ),
# )


from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy


numpy_include = numpy.get_include()


ext = Extension(
    "functions",
    ["functions.pyx"],
    include_dirs=[numpy_include],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    language="c",
)

setup(
    ext_modules=cythonize(
        ext, annotate=True, compiler_directives={"language_level": "3"}
    ),
)
