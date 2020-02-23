import numpy as np
import pytest

from undistort_ios.benchmark import UndistortionArgs, get_undistortion_args, get_undistortion_image


@pytest.fixture()
def undistortion_args() -> UndistortionArgs:
    return get_undistortion_args()


@pytest.fixture()
def undistortion_image() -> np.ndarray:
    return get_undistortion_image()
