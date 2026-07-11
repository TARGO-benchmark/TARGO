try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import numpy


# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

# Extensions
# pykdtree (kd tree)
pykdtree = Extension(
    'src.vgn.ConvONets.utils.libkdtree.pykdtree.kdtree',
    sources=[
        'src/vgn/ConvONets/utils/libkdtree/pykdtree/kdtree.c',
        'src/vgn/ConvONets/utils/libkdtree/pykdtree/_kdtree_core.c'
    ],
    language='c',
    extra_compile_args=['-std=c99', '-O3', '-fopenmp'],
    extra_link_args=['-lgomp'],
    include_dirs=[numpy_include_dir]
)

# mcubes (marching cubes algorithm)
mcubes_module = Extension(
    'src.vgn.ConvONets.utils.libmcubes.mcubes',
    sources=[
        'src/vgn/ConvONets/utils/libmcubes/mcubes.pyx',
        'src/vgn/ConvONets/utils/libmcubes/mcubes.cpp'
    ],
    language='c++',
    extra_compile_args=['-std=c++11'],
    include_dirs=[numpy_include_dir]
)

# triangle hash (efficient mesh intersection)
triangle_hash_module = Extension(
    'src.vgn.ConvONets.utils.libmesh.triangle_hash',
    sources=[
        'src/vgn/ConvONets/utils/libmesh/triangle_hash.pyx'
    ],
    libraries=['m'],  # Unix-like specific
    include_dirs=[numpy_include_dir]
)

# mise (efficient mesh extraction)
mise_module = Extension(
    'src.vgn.ConvONets.utils.libmise.mise',
    sources=[
        'src/vgn/ConvONets/utils/libmise/mise.pyx'
    ],
)

# simplify (efficient mesh simplification)
simplify_mesh_module = Extension(
    'src.vgn.ConvONets.utils.libsimplify.simplify_mesh',
    sources=[
        'src/vgn/ConvONets/utils/libsimplify/simplify_mesh.pyx'
    ],
    include_dirs=[numpy_include_dir]
)

# voxelization (efficient mesh voxelization)
voxelize_module = Extension(
    'src.vgn.ConvONets.utils.libvoxelize.voxelize',
    sources=[
        'src/vgn/ConvONets/utils/libvoxelize/voxelize.pyx'
    ],
    libraries=['m']  # Unix-like specific
)

# Gather all extension modules
ext_modules = [
    #pykdtree,
    #mcubes_module,  # Optional mesh utility; prebuilt py3.8/py3.9 binaries are bundled when available.
    triangle_hash_module,
    #mise_module,  # Optional mesh utility.
    #simplify_mesh_module,  # Optional mesh utility.
    #voxelize_module,  # Optional mesh utility.
]

setup(
    ext_modules=cythonize(ext_modules),
    cmdclass={
        'build_ext': BuildExtension
    }
)
