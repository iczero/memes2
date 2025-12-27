// note: this module's name must match the .so that python ends up importing
#[pyo3::pymodule(gil_used = false)]
mod native {
    use numpy::{PyReadwriteArrayDyn};

    #[pyo3::pyfunction]
    fn add_one<'py>(mut x: PyReadwriteArrayDyn<'py, f64>) {
        let mut x = x.as_array_mut();
        x += 1.0;
    }
}

