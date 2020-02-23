#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;

// Second template parameter 0 means don't copy data
// Can also use py::array::c_style | py::array::forcecast
typedef py::array_t<double, 0> np_array_double;
typedef py::array_t<uint8_t, 0> np_array_uint8;


namespace undistort_ios {
    struct ImagePoint {
        ImagePoint(double row, double col) : row(row), col(col) {};
        double row;
        double col;

        [[nodiscard]] std::string toString() const;
    };

    struct ImageSize {
        ImageSize(size_t height, size_t width) : height(height), width(width) {};
        size_t height;
        size_t width;

        [[nodiscard]] std::string toString() const;
    };

    class Undistorter {
    private:
        ImageSize size_;
        ImagePoint distortion_center_;
        std::vector<double> lookup_table_;
        double max_radius_;
        double max_magnification_;

    public:
        Undistorter(ImageSize image_size, ImagePoint distortion_center, std::vector<double> lookup_table);

        static std::unique_ptr<Undistorter> build(size_t n_rows, size_t n_cols, double disto_center_row, double disto_center_col, std::vector<double> lookup_table);

        void undistortImage(const np_array_uint8 &image, np_array_uint8 &output) const;

        void undistortPoints(const np_array_double &points, np_array_double &output) const;

        void distortPoint(const ImagePoint &point, ImagePoint &output) const;

    private:
        [[nodiscard]] bool isPixelVisible(int row, int col) const {
            return row >= 0 && row < size_.height && col >= 0 && col < size_.width;
        }
    };
}
