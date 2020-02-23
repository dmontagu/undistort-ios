from typing import List

import numpy as np

class Undistorter:
    def __init__(
        self, n_rows: int, n_cols: int, disto_center_row: float, disto_center_col: float, lookup_table: List[float]
    ): ...
    def undistort_image(self, image: np.ndarray, output: np.ndarray) -> None: ...
    def undistort_points(self, image: np.ndarray, output: np.ndarray) -> None: ...
