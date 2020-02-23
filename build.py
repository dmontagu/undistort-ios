"""
Adapted from https://github.com/pybind/cmake_example
"""
from typing import Any, Dict, List

from Cython.Build import cythonize
from Cython.Distutils import Extension
from numpy import get_include as get_numpy_include
from setuptools_rust import Binding, RustExtension, build_ext as RustExtensionBuilder

from setuptools_cpp import ExtensionBuilder as CppExtensionBuilder, Pybind11Extension


class ExtensionBuilder(CppExtensionBuilder, RustExtensionBuilder):
    pass


def get_pybind_modules() -> List[Pybind11Extension]:
    return [
        Pybind11Extension(
            "undistort_ios.cpp.compiled",
            ["src/cpp/src/undistorter.cpp"],
            include_dirs=["src/cpp/include"]
        )
    ]


def get_cython_modules() -> List[Extension]:
    return cythonize(
        [
            Extension(
                "undistort_ios.cython.compiled",
                sources=["src/cython/undistort_cy.pyx"],
                include_dirs=[get_numpy_include(), "."],
            )
        ]
    )


def get_rust_modules() -> List[RustExtension]:
    return [RustExtension("undistort_ios.rust.compiled", binding=Binding.PyO3, debug=False)]


def build(setup_kwargs: Dict[str, Any]) -> None:
    ext_modules: List[Extension] = [
        *get_pybind_modules(),
        *get_cython_modules()
    ]
    setup_kwargs.update(
        {
            "rust_extensions": get_rust_modules(),
            "ext_modules": ext_modules,
            "cmdclass": dict(build_ext=ExtensionBuilder),
            "zip_safe": False,
        }
    )
