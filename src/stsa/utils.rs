use ndarray::{Array1, ArrayBase, Data, Ix2};
use num::{Complex, Float};


pub fn noise_power_est<T, D>(spectrum: &ArrayBase<D, Ix2>) -> Array1<T>
where
    T: Float + std::ops::AddAssign + ndarray::ScalarOperand,
    D: Data<Elem = Complex<T>>,
{
    let mut noise_pwr = Array1::zeros(spectrum.shape()[1]);
    let norm = T::from(spectrum.shape()[0]).unwrap();

    for ((_, i), v) in spectrum.indexed_iter() {
        noise_pwr[i] += v.norm_sqr() / norm;
    }

    noise_pwr
}
