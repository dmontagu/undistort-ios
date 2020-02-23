from typing import Type

import numpy as np
import pytest

from tests.conftest import UndistortionArgs
from undistort_ios.benchmark import UndistorterProtocol, benchmark
from undistort_ios.cpp import Undistorter as CppUndistorter
from undistort_ios.cython import Undistorter as CythonUndistorter
from undistort_ios.numpy_impl import Undistorter as NumpyUndistorter
from undistort_ios.python_impl import Undistorter as PythonUndistorter
from undistort_ios.rust import Undistorter as RustUndistorter

undistorters = [
    (CppUndistorter, "cpp"),
    (CythonUndistorter, "cython"),
    (RustUndistorter, "rust"),
    (NumpyUndistorter, "numpy"),
    (PythonUndistorter, "python"),
]


@pytest.mark.parametrize("cls,kind", undistorters)
def test_implementations(
    undistortion_args: UndistortionArgs, undistortion_image: np.ndarray, cls: Type[UndistorterProtocol], kind: str
) -> None:
    undistorter = undistortion_args.get_undistorter(cls)
    input_image = undistortion_image
    output_image = np.zeros_like(input_image)
    undistorter.undistort_image(input_image, output_image)


def test_benchmark() -> None:
    benchmark()
