from setuptools import setup, Extension

from codecs import open
from os import path

from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

extensions = cythonize(
    list(Extension(
        f"pyezzi.{f}",
        [f"pyezzi/{f}.pyx"],
        extra_compile_args=['-fopenmp', '-O3'],
        extra_link_args=['-fopenmp']
    ) for f in ('laplace', 'yezzi')))

cmdclass = {"build_ext": build_ext}

setup(
    name="pyezzi",
    version="0.5.2",
    description="Thickness calculation on binary 3D images",
    long_description=long_description,
    ext_modules=extensions,
    license="CeCILL-C",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='medical image processing',
    packages=["pyezzi"],
    install_requires=['numpy', 'cython'],
    cmdclass=cmdclass,
    url="https://gitlab.inria.fr/ncedilni/pyezzi",
    author="Nicolas Cedilnik",
    author_email="nicoco@nicoco.fr",
    include_dirs=[np.get_include()]
)
