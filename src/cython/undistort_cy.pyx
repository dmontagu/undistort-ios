# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False
import numpy as np
cimport numpy as np
# from scipy.optimize import root_scalar

FLOAT = np.float
UINT8 = np.uint8
ctypedef np.float_t FLOAT_t
ctypedef np.uint8_t UINT8_t

cdef extern from "math.h":
    double sqrt(double m)


cdef class ImagePoint:
    cdef public double x
    cdef public double y

    def __init__(self, double x, double y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"ImagePoint(x={self.x}, y={self.y})"


cdef class Undistorter:
    cdef int width, height
    cdef double distortion_cx, distortion_cy, r_max
    cdef FLOAT_t[:] lookup_array
    cdef int lookup_array_size

    def __init__(
        self,
        int n_rows,
        int n_cols,
        double disto_center_row,
        double disto_center_col,
        list lookup_table
    ):
        self.width = n_cols
        self.height = n_rows
        self.distortion_cx = disto_center_col
        self.distortion_cy = disto_center_row
        cdef double delta_ocx_max = max(disto_center_col, n_cols - disto_center_col)
        cdef double delta_ocy_max = max(disto_center_row, n_rows - disto_center_row)
        self.r_max = sqrt(delta_ocx_max * delta_ocx_max + delta_ocy_max * delta_ocy_max)

        self.lookup_array = 1 + np.array(lookup_table)
        self.lookup_array_size = len(lookup_table)

    cpdef undistort_image(self, UINT8_t[:, :, :] image, UINT8_t[:, :, :] output):
        if image.shape[1] != self.width or image.shape[0] != self.height:
            raise ValueError(
                f"Expected shape: {(self.height, self.width)}; "
                f"received {(image.shape[0], image.shape[1])}"
            )
        for i in range(3):
            if image.shape[i] != output.shape[i]:
                raise ValueError("Input and output arrays didn't have the same shape")

        cdef double distortion_cx = self.distortion_cx
        cdef double distortion_cy = self.distortion_cy
        cdef double r_max = self.r_max
        cdef FLOAT_t[:] lookup_array = self.lookup_array
        cdef int lookup_array_size = self.lookup_array_size

        # cdef UINT8_t[:, :, :] output = np.zeros_like(image)
        cdef int depth = output.shape[2], height = self.height, width = self.width
        cdef int x, y, z, undistorted_x, undistorted_y
        cdef ImagePoint undistorted_pixel = ImagePoint(x=0, y=0)

        for x in range(width):
            for y in range(height):
                _distort_point(
                    x, y, distortion_cx, distortion_cy, r_max, lookup_array, lookup_array_size, undistorted_pixel
                )
                undistorted_x = int(undistorted_pixel.x)
                undistorted_y = int(undistorted_pixel.y)
                if _is_pixel_visible(undistorted_x, undistorted_y, width, height):
                    for z in range(depth):
                        output[y, x, z] =  image[undistorted_y, undistorted_x, z]

    cpdef ImagePoint distort_point(self, ImagePoint point):
        cdef ImagePoint output = ImagePoint(0, 0)
        _distort_point(
            point.x,
            point.y,
            self.distortion_cx,
            self.distortion_cy,
            self.r_max,
            self.lookup_array,
            self.lookup_array_size,
            output
        )
        return output



cdef bint _is_pixel_visible(int pixel_x, int pixel_y, int image_width, int image_height):
    return not (pixel_x < 0 or pixel_x >= image_width or pixel_y < 0 or pixel_y >= image_height)


cdef _distort_point(
    double x,
    double y,
    double distortion_cx,
    double distortion_cy,
    double r_max,
    FLOAT_t[:] lookup_array,
    int lookup_array_size,
    ImagePoint output
):
    cdef double value, proportion, magnification
    cdef int index

    cdef double v_point_x = x - distortion_cx
    cdef double v_point_y = y - distortion_cy
    cdef double r_point = sqrt(v_point_x * v_point_x + v_point_y * v_point_y)

    if r_point < r_max:
        value = r_point * (lookup_array_size - 1) / r_max
        index = int(value)
        proportion = value - index
        magnification = (1 - proportion) * lookup_array[index] + proportion * lookup_array[index + 1]
    else:
        magnification = lookup_array[lookup_array_size - 1]
    cdef double new_v_point_x = magnification * v_point_x
    cdef double new_v_point_y = magnification * v_point_y
    output.x = new_v_point_x + distortion_cx
    output.y = new_v_point_y + distortion_cy


# def _undistort_point(
#     x: float,
#     y: float,
#     distortion_cx: float,
#     distortion_cy: float,
#     r_max: float,
#     lookup_array: FLOAT_t[:],
#     lookup_array_size: int,
#     output: ImagePoint
# ) -> None:
#     cdef double value, proportion, magnification
#     cdef int index
#
#     cdef double v_point_x = x - distortion_cx
#     cdef double v_point_y = y - distortion_cy
#     cdef double r_point = sqrt(v_point_x * v_point_x + v_point_y * v_point_y)
#
#     def magnified_r(r: float) -> float:
#         if r < r_max:
#             value = r * (lookup_array_size - 1) / r_max
#             index = int(value)
#             proportion = value - index
#             magnification = (1 - proportion) * lookup_array[index] + proportion * lookup_array[index + 1]
#         else:
#             magnification = lookup_array[lookup_array_size - 1]
#         return (magnification + 1) * r
#
#     def r_point_inverse_root(input_r: float) -> float:
#         return magnified_r(input_r) - r_point
#
#     inverse_magnification = root_scalar(r_point_inverse_root, bracket=[r_point / 2, r_point * 2], method='brentq').root
#     scaling_factor = inverse_magnification / max(r_point, 1e-10)
#     cdef double new_v_point_x = v_point_x * scaling_factor
#     cdef double new_v_point_y = v_point_y * scaling_factor
#     output.x = distortion_cx + new_v_point_x
#     output.y = distortion_cy + new_v_point_y
