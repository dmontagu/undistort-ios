import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Protocol, Tuple, Type, TypeVar

import numpy as np
from imageio import imread

from undistort_ios.cpp import Undistorter as CppUndistorter
from undistort_ios.cython import Undistorter as CythonUndistorter
from undistort_ios.numpy_impl import Undistorter as NumpyUndistorter
from undistort_ios.python_impl import Undistorter as PythonUndistorter
from undistort_ios.rust import Undistorter as RustUndistorter

DATA_PATH = Path(__file__).parent.parent / "tests" / "data"


class UndistorterProtocol(Protocol):
    def __init__(
        self, n_rows: int, n_cols: int, disto_center_row: float, disto_center_col: float, lookup_table: List[float]
    ):
        """ Initializer """

    def undistort_image(self, image: np.ndarray, output: np.ndarray) -> None:
        """ Undistort the provided image using a return argument """


T = TypeVar("T", bound=UndistorterProtocol)


@dataclass
class UndistortionArgs:
    n_rows: int
    n_cols: int
    disto_center_row: float
    disto_center_col: float
    lookup_table: List[float]

    def get_undistorter(self, undistorter_type: Type[T]) -> T:
        return undistorter_type(**asdict(self))


def undistortion_data() -> Dict[str, Any]:
    undistortion_data_json = (DATA_PATH / "test_calibration_data.json").read_text()
    return json.loads(undistortion_data_json)


def get_undistortion_args() -> UndistortionArgs:
    return UndistortionArgs(**undistortion_data())


def get_undistortion_image() -> np.ndarray:
    image_array = imread(DATA_PATH / "diamonds-1.jpg")
    image_array = np.swapaxes(image_array, 0, 1)
    return image_array


undistorters: List[Tuple[Type[UndistorterProtocol], str]] = [
    (CppUndistorter, "cpp"),
    (CythonUndistorter, "cython"),
    (RustUndistorter, "rust"),
    (NumpyUndistorter, "numpy"),
    (PythonUndistorter, "python"),
]


def benchmark() -> None:
    args = get_undistortion_args()
    image = get_undistortion_image()

    for cls, kind in undistorters:
        undistorter = args.get_undistorter(cls)
        input_image = image
        output_image = np.zeros_like(input_image)

        t0 = time.time()
        repeats = 1 if kind == "python" else 5
        for _ in range(repeats):
            undistorter.undistort_image(input_image, output_image)
        t1 = time.time()
        print(kind, (t1 - t0) / repeats)


if __name__ == "__main__":
    benchmark()
