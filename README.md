# Undistort iOS

![Distorted vs. undistorted](https://raw.githubusercontent.com/dmontagu/undistort-ios/master/animation.gif)

Various implementations of image undistortion for images captured with calibration data on an iOS device.  

This package is mostly intended as a comparison of various approaches to writing high-performance
native python extensions, but is indeed usable out of the box for undistorting iOS images if that's what you need.

If you find any changes that could improve performance for any of the implementations, please share them!  

Performance:
------------
Extension | Execution Time (s) | Factor
--------- | ------------------ | ------
Rust/PyO3 (1 line unsafe) | 0.0462 | 1.0
Cython | 0.0529 | 1.145
C++/Pybind11 | 0.0610 |  1.320
Rust (safe) | 0.1123 | 2.431
Numpy | 0.5520 | 11.95
Python | 15.36 | 332.5
