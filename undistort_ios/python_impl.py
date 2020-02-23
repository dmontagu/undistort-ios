from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class ImagePoint:
    x: float
    y: float


@dataclass
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
        self.lookup_array_size = len(lookup_table)
        lookup_table.append(lookup_table[-1])  # Makes indexing simpler
        self.lookup_array = np.array(lookup_table)

    def undistort_image(self, image: np.ndarray, output: np.ndarray) -> None:
        assert image.shape[1] == self.n_cols and image.shape[0] == self.n_rows, (
            f"Expected shape: {(self.n_rows, self.n_cols)}; " f"received {(image.shape[0], image.shape[1])}"
        )
        for i in range(3):
            assert image.shape[i] == output.shape[i], "Input and output arrays didn't have the same shape"

        distortion_cx = self.disto_center_col
        distortion_cy = self.disto_center_row
        r_max = self.r_max
        lookup_array = self.lookup_array
        lookup_array_size = self.lookup_array_size

        depth = output.shape[2]
        height = self.n_rows
        width = self.n_cols
        undistorted_pixel = ImagePoint(x=0, y=0)
        for x in range(width):
            for y in range(height):
                _distort_point(
                    x, y, distortion_cx, distortion_cy, r_max, lookup_array, lookup_array_size, undistorted_pixel
                )
                undistorted_x = int(undistorted_pixel.x)
                undistorted_y = int(undistorted_pixel.y)
                if _is_pixel_visible(undistorted_x, undistorted_y, width, height):
                    for z in range(depth):
                        output[y, x, z] = image[undistorted_y, undistorted_x, z]


def _is_pixel_visible(pixel_x: int, pixel_y: int, image_width: int, image_height: int) -> bool:
    return 0 <= pixel_x < image_width and 0 <= pixel_y < image_height


def _distort_point(
    x: float,
    y: float,
    distortion_cx: float,
    distortion_cy: float,
    r_max: float,
    lookup_array: np.ndarray,
    lookup_array_size: int,
    output: ImagePoint,
) -> None:
    v_point_x = x - distortion_cx
    v_point_y = y - distortion_cy
    r_point = np.sqrt(v_point_x * v_point_x + v_point_y * v_point_y)

    value = r_point * (lookup_array_size - 1) / r_max
    index = int(value)
    proportion = value - index
    magnification = (1 - proportion) * lookup_array[index] + proportion * lookup_array[index + 1]
    new_v_point_x = v_point_x + magnification * v_point_x
    new_v_point_y = v_point_y + magnification * v_point_y
    output.x = new_v_point_x + distortion_cx
    output.y = new_v_point_y + distortion_cy
