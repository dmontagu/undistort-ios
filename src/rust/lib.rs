#![cfg_attr(all(test, feature = "unstable"), feature(test))]

use core::fmt;

use ndarray::{
    ArrayBase, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Dimension, RawData, Zip,
};
use numpy::{PyArray2, PyArray3};
use pyo3::{exceptions, prelude::*, PyResult};

#[pyclass(name = ImagePoint)]
#[derive(Debug)]
struct ImagePoint {
    row: f64,
    col: f64,
}

#[pymethods]
impl ImagePoint {
    #[new]
    fn new_py(obj: &PyRawObject, row: f64, col: f64) {
        obj.init(ImagePoint { row, col });
    }
}

impl fmt::Display for ImagePoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ImagePoint(row={}, col={})", self.row, self.col)
    }
}

#[pyclass(name = ImageSize)]
#[derive(Debug)]
struct ImageSize {
    height: usize,
    width: usize,
}

#[pymethods]
impl ImageSize {
    #[new]
    fn new_py(obj: &PyRawObject, height: usize, width: usize) {
        obj.init(ImageSize { height, width });
    }
}

impl fmt::Display for ImageSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ImageSize(height={}, width={})", self.height, self.width)
    }
}

#[pyclass(name = Undistorter)]
#[derive(Debug)]
pub(crate) struct Undistorter {
    size: ImageSize,
    distortion_center: ImagePoint,
    lookup_table: Vec<f64>,

    max_radius: f64,
    max_magnification: f64,
}

impl Undistorter {
    fn new(size: ImageSize, distortion_center: ImagePoint, lookup_table: Vec<f64>) -> Undistorter {
        let col_delta_max = distortion_center
            .col
            .max(size.width as f64 - distortion_center.col);
        let row_delta_max = distortion_center
            .row
            .max(size.height as f64 - distortion_center.row);
        let max_radius = (col_delta_max * col_delta_max + row_delta_max * row_delta_max).sqrt();
        let max_magnification = match lookup_table.last() {
            Some(x) => *x,
            None => 0.0,
        };
        Undistorter {
            size,
            distortion_center,
            lookup_table,
            max_radius,
            max_magnification,
        }
    }
}

#[pymethods]
impl Undistorter {
    #[new]
    fn new_py(
        obj: &PyRawObject,
        n_rows: usize,
        n_cols: usize,
        disto_center_row: f64,
        disto_center_col: f64,
        lookup_table: Vec<f64>,
    ) -> PyResult<()> {
        if lookup_table.is_empty() {
            return Err(PyErr::new::<exceptions::ValueError, _>(
                "lookup_table must be nonempty",
            ));
        }
        obj.init(Undistorter::new(
            ImageSize {
                height: n_rows,
                width: n_cols,
            },
            ImagePoint {
                row: disto_center_row,
                col: disto_center_col,
            },
            lookup_table,
        ));
        Ok(())
    }

    fn undistort_image(&self, image: &PyArray3<u8>, output: &mut PyArray3<u8>) -> PyResult<()> {
        let in_view = image.as_array();
        let mut out_view = output.as_array_mut();
        check_same_shape(&in_view, &out_view)?;
        if in_view.shape()[0] != self.size.height || in_view.shape()[1] != self.size.width {
            return Err(PyErr::new::<exceptions::ValueError, _>(format!(
                "image shape must be {}",
                self.size
            )));
        }

        let mut output_point = ImagePoint { row: 0.0, col: 0.0 };
        for (row, mut row_data) in out_view.outer_iter_mut().enumerate() {
            for (col, mut channels_data) in row_data.outer_iter_mut().enumerate() {
                self.distort_point(
                    &ImagePoint {
                        row: row as f64,
                        col: col as f64,
                    },
                    &mut output_point,
                );
                let distorted_row = output_point.row as i64;
                let distorted_col = output_point.col as i64;
                if is_pixel_visible(&self.size, distorted_row, distorted_col) {
                    for i in 0..3 {
                        unsafe {
                            *channels_data.uget_mut(i) =
                                *in_view.uget((distorted_row as usize, distorted_col as usize, i))
                        }
                    }
                    // Could use the following safe API at ~2.5x overhead due to bounds checks
                    //channels_data.assign(&in_view.slice(s![
                    //    distorted_row as usize,
                    //    distorted_col as usize,
                    //    ..
                    //]));
                }
            }
        }
        Ok(())
    }

    fn undistort_points(
        &self,
        points_rc: &PyArray2<f64>,
        output: &mut PyArray2<f64>,
    ) -> PyResult<()> {
        let in_view = points_rc.as_array();
        let mut out_view = output.as_array_mut();
        check_same_shape(&in_view, &out_view)?;
        if in_view.ncols() != 2 {
            return Err(PyErr::new::<exceptions::ValueError, _>(
                "Input array must have two columns",
            ));
        }
        let mut output_point = ImagePoint { row: 0.0, col: 0.0 };
        let (in_rows, in_cols) = get_array_rows_cols(&in_view);
        let (out_rows, out_cols) = get_array_rows_cols_mut(&mut out_view);
        Zip::from(in_rows)
            .and(in_cols)
            .and(out_rows)
            .and(out_cols)
            .apply(|in_row, in_col, out_row, out_col| {
                self.distort_point(
                    &ImagePoint {
                        row: *in_row,
                        col: *in_col,
                    },
                    &mut output_point,
                );
                *out_row = output_point.row;
                *out_col = output_point.col;
            });
        Ok(())
    }

    fn distort_point(&self, input: &ImagePoint, output: &mut ImagePoint) {
        let v_col = input.col - self.distortion_center.col;
        let v_row = input.row - self.distortion_center.row;
        let input_radius = (v_col * v_col + v_row * v_row).sqrt();

        let magnification = if input_radius < self.max_radius {
            let distance_steps =
                input_radius / self.max_radius * (self.lookup_table.len() - 1) as f64;
            let index = distance_steps as usize;
            let proportion = distance_steps.fract();
            (1.0 - proportion) * self.lookup_table[index]
                + proportion * self.lookup_table[index + 1]
        } else {
            self.max_magnification
        };

        output.col = v_col * (1.0 + magnification) + self.distortion_center.col;
        output.row = v_row * (1.0 + magnification) + self.distortion_center.row;
    }
}

// Internals
fn is_pixel_visible(size: &ImageSize, row: i64, col: i64) -> bool {
    row >= 0 && row < size.height as i64 && col >= 0_i64 && col < size.width as i64
}

// Array utilities
trait Shaped {
    fn shape(&self) -> &[usize];
}

impl<S, D> Shaped for ArrayBase<S, D>
where
    S: RawData,
    D: Dimension,
{
    fn shape(&self) -> &[usize] {
        ArrayBase::shape(self)
    }
}

fn check_same_shape<T1: Shaped, T2: Shaped>(array1: &T1, array2: &T2) -> PyResult<()> {
    if array1.shape() != array2.shape() {
        return Err(PyErr::new::<exceptions::ValueError, _>(
            "Input and output arrays must have matching shape",
        ));
    }
    Ok(())
}

fn get_array_rows_cols<'a>(
    array_view: &'a ArrayView2<'a, f64>,
) -> (ArrayView1<'a, f64>, ArrayView1<'a, f64>) {
    let mut rows: Option<_> = None;
    let mut cols: Option<_> = None;
    for (index, column) in array_view.gencolumns().into_iter().enumerate() {
        if index == 0 {
            rows = Some(column);
        } else if index == 1 {
            cols = Some(column);
        }
    }
    (rows.unwrap(), cols.unwrap())
}

fn get_array_rows_cols_mut<'a>(
    array_view: &'a mut ArrayViewMut2<'a, f64>,
) -> (ArrayViewMut1<'a, f64>, ArrayViewMut1<'a, f64>) {
    let mut rows: Option<_> = None;
    let mut cols: Option<_> = None;
    for (index, column) in array_view.gencolumns_mut().into_iter().enumerate() {
        if index == 0 {
            rows = Some(column);
        } else if index == 1 {
            cols = Some(column);
        }
    }
    (rows.unwrap(), cols.unwrap())
}

// For the parallel implementation of lookup_array below
//use ndarray::parallel::prelude::*;
//use ndarray::{ArrayView2, Zip};

#[pymodule]
fn compiled(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Undistorter>()?;
    Ok(())
}
