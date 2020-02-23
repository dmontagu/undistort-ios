#include "undistorter.hpp"


namespace undistort_ios {
    using undistort_ios::ImagePoint;
    using undistort_ios::ImageSize;
    using undistort_ios::Undistorter;

    std::string ImagePoint::toString() const {
        std::stringstream ss;
        ss << "ImagePoint(row=" << row << ", col=" << col << ")";
        return ss.str();
    }

    std::string ImageSize::toString() const {
        std::stringstream ss;
        ss << "ImageSize(height=" << height << ", width=" << width << ")";
        return ss.str();
    }

    std::unique_ptr<Undistorter>
    Undistorter::build(size_t n_rows, size_t n_cols, double disto_center_row, double disto_center_col,
                       std::vector<double> lookup_table) {
        auto image_size = ImageSize{n_rows, n_cols};
        auto distortion_center = ImagePoint{disto_center_row, disto_center_col};
        return std::make_unique<Undistorter>(image_size, distortion_center, std::move(lookup_table));
    }

    Undistorter::Undistorter(ImageSize image_size, ImagePoint distortion_center, std::vector<double> lookup_table) :
            size_(image_size),
            distortion_center_(distortion_center),
            lookup_table_(std::move(lookup_table)) {
        if (lookup_table_.empty()) {
            throw std::runtime_error("lookup_table must be nonempty");
        }
        // Lookup table contains percentage changes,
        // but the arithmetic is a little faster if we store multiplication factors
        for (auto &magnification : lookup_table_) {
           magnification += 1.0;
        }
        auto col_delta_max = std::max(distortion_center_.col, size_.width - distortion_center_.col);
        auto row_delta_max = std::max(distortion_center_.row, size_.height - distortion_center_.row);
        max_radius_ = sqrt(col_delta_max * col_delta_max + row_delta_max * row_delta_max);
        max_magnification_ = lookup_table_[lookup_table_.size() - 1];
    }

    void Undistorter::undistortImage(const np_array_uint8 &image, np_array_uint8 &output) const {
        // TODO: Need to check that both inputs have at least 3 dimensions
        auto distorted = image.unchecked<3>();
        auto undistorted = output.mutable_unchecked<3>();
        for (int dim = 0; dim < 3; ++dim) {
            if (distorted.shape(dim) != output.shape(dim)) {
                throw std::runtime_error("Input and output arrays had different shapes");
            }
        }

        auto n_rows = distorted.shape(0);
        auto n_cols = distorted.shape(1);
        auto n_channels = distorted.shape(2);
        if (n_rows != size_.height) {
            throw std::runtime_error("Input image had wrong number of rows");
        }
        if (n_cols != size_.width) {
            throw std::runtime_error("Input image had wrong number of columns");
        }

        // Idea: to get value of pixel at undistorted point (r, c), use
        // value of distorted pixel at (distorted(r), distorted(c))
        ImagePoint output_point{0.0, 0.0};
        for (size_t row = 0; row < n_rows; ++row) {
            for (size_t col = 0; col < n_cols; ++col) {
                ImagePoint input_point{(double) row, (double) col};
                distortPoint(input_point, output_point);
                auto distorted_row = (int) output_point.row;
                auto distorted_col = (int) output_point.col;
                if (isPixelVisible(distorted_row, distorted_col)) {
                    for (size_t channel = 0; channel < n_channels; ++channel) {
                        undistorted(row, col, channel) = distorted(distorted_row, distorted_col, channel);
                    }
                }
            }
        }
    }

    void Undistorter::undistortPoints(const np_array_double &points, np_array_double &output) const {
        auto points_buffer = points.unchecked<2>();
        auto output_buffer = output.mutable_unchecked<2>();
        if (points_buffer.shape(1) != 2) {
            throw std::runtime_error("Points array must have precisely two columns");
        }
        for (int dim = 0; dim < 2; ++dim) {
            if (points_buffer.shape(dim) != output_buffer.shape(dim)) {
                throw std::runtime_error("Input and output arrays had different shapes");
            }
        }
        ImagePoint output_point{0.0, 0.0};
        size_t n_points = points_buffer.shape(0);
        for (size_t point_idx = 0; point_idx < n_points; ++point_idx) {
            ImagePoint input_point{points_buffer(point_idx, 0), points_buffer(point_idx, 0)};
            distortPoint(input_point, output_point);
            output_buffer(point_idx, 0) = output_point.row;
            output_buffer(point_idx, 1) = output_point.col;
        }
    }

    void Undistorter::distortPoint(const ImagePoint &point, ImagePoint &output) const {
        auto v_col = point.col - distortion_center_.col;
        auto v_row = point.row - distortion_center_.row;
        auto input_radius = sqrt(v_col * v_col + v_row * v_row);
        double magnification;
        if (input_radius < max_radius_) {
            auto distance_steps = (input_radius / max_radius_) * double(lookup_table_.size() - 1);
            auto index = (size_t) distance_steps;
            double intpart;
            auto proportion = modf(distance_steps, &intpart);
            magnification = (1.0 - proportion) * lookup_table_[index] + proportion * lookup_table_[index + 1];
        } else {
            magnification = max_magnification_;
        }
        // Note: the magnification was already converted from a percentage delta into a multiplication factor
        output.col = v_col * magnification + distortion_center_.col;
        output.row = v_row * magnification + distortion_center_.row;
    }
}

void pybind_undistorter(py::module &m) {
//    using undistort_ios::ImagePoint;
//    using undistort_ios::ImageSize;
    using undistort_ios::Undistorter;

//    py::class_<ImagePoint> image_point(m, "ImagePoint");
//    image_point
//            .def(py::init([](double height, double width) {
//                return std::make_unique<ImagePoint>(height, width);
//            }), "row"_a, "col"_a)
//            .def_readwrite("row", &ImagePoint::row)
//            .def_readwrite("col", &ImagePoint::col)
//            .def("__repr__", &ImagePoint::toString);
//
//    py::class_<ImageSize> image_size(m, "ImageSize");
//    image_size
//            .def(py::init([](size_t height, size_t width) {
//                return std::make_unique<ImageSize>(height, width);
//            }), "height"_a, "width"_a)
//            .def_readwrite("height", &ImageSize::height)
//            .def_readwrite("width", &ImageSize::width)
//            .def("__repr__", &ImageSize::toString);

    py::class_<Undistorter> undistorter(m, "Undistorter");
    undistorter
            .def(py::init([](
                    size_t n_rows, size_t n_cols, double disto_center_row, double disto_center_col,
                    std::vector<double> lookup_table) {
                return Undistorter::build(n_rows, n_cols, disto_center_row, disto_center_col, std::move(lookup_table));
            }), "n_rows"_a, "n_cols"_a, "disto_center_row"_a, "disto_center_col"_a, "lookup_table"_a)
            .def("undistort_image", &Undistorter::undistortImage, "image"_a, "output"_a)
            .def("undistort_points", &Undistorter::undistortPoints, "points"_a, "output"_a);
}

PYBIND11_MODULE(compiled, m) {
    m.doc() = "Undistort calibrated iOS images";

    pybind_undistorter(m);
}
