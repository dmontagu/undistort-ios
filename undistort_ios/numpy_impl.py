from typing import List

import numpy as np


class Undistorter:
    def __init__(
        self, n_rows: int, n_cols: int, disto_center_row: float, disto_center_col: float, lookup_table: List[float]
    ):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.disto_center_row = disto_center_row
        self.disto_center_col = disto_center_col
        self.lookup_table = lookup_table

        delta_ocx_max = max(disto_center_col, n_cols - disto_center_col)
        delta_ocy_max = max(disto_center_row, n_rows - disto_center_row)
        self.r_max = np.sqrt(delta_ocx_max * delta_ocx_max + delta_ocy_max * delta_ocy_max)
        self.n_magnifications = len(lookup_table)
        # repeat the final magnification to simplify index lookup logic
        self.magnification_factors = np.array(lookup_table + [lookup_table[-1]]) + 1

    def undistort_image(self, image: np.ndarray, output: np.ndarray) -> None:
        cols, rows = np.meshgrid(np.arange(self.n_cols), np.arange(self.n_rows))
        rows_cols = np.dstack([rows, cols])
        center = np.array([[[self.disto_center_row, self.disto_center_col]]])
        deltas_to_center = rows_cols - center
        radii = np.sqrt(np.sum(np.power(deltas_to_center, 2), axis=2))
        proportions, indices = np.modf(self.n_magnifications * (radii / self.r_max))
        indices = indices.astype(int)
        magnifications = (1 - proportions) * self.magnification_factors[
            indices
        ] + proportions * self.magnification_factors[indices + 1]
        distorted_deltas_to_center = deltas_to_center * magnifications[..., np.newaxis]
        distorted_rows_cols = (distorted_deltas_to_center + center).astype(int)
        distorted_r = distorted_rows_cols[..., 0]
        distorted_c = distorted_rows_cols[..., 1]
        visible = (distorted_r >= 0) & (distorted_r < self.n_rows) & (distorted_c >= 0) & (distorted_c < self.n_cols)
        distorted_r = np.clip(distorted_r, 0, self.n_rows - 1)
        distorted_c = np.clip(distorted_c, 0, self.n_cols - 1)
        visible_rc = distorted_r[visible], distorted_c[visible]
        output[visible] = image[visible_rc]
