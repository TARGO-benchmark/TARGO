"""Build the Cython extension used by the ConvONets mesh evaluation utilities.

Run from the repository root:

    python scripts/convonet_setup.py build_ext --inplace

This compiles ``triangle_hash`` (efficient triangle/point intersection), which
``vgn.ConvONets.eval.MeshEvaluator`` uses to compute IoU between meshes.
"""
import numpy
from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension

triangle_hash_module = Extension(
    "src.vgn.ConvONets.utils.libmesh.triangle_hash",
    sources=["src/vgn/ConvONets/utils/libmesh/triangle_hash.pyx"],
    language="c++",
    libraries=["m"],  # Unix-like specific
    include_dirs=[numpy.get_include()],
)

setup(ext_modules=cythonize([triangle_hash_module]))
