use crate::window::WindowFunction;

use ndarray::{s, Array1, ArrayView1, ArrayViewMut1, Axis};
use num::traits::{Float, Zero};


pub fn magnitude_to_db<F: Float>(m: F) -> F {
    F::from(20.0).unwrap() * F::log10(m)
}

pub fn fftshift<T: Copy + Zero>(input: &ArrayView1<T>) -> Array1<T> {
    let mut output = Array1::zeros(input.len());
    fftshift_into(&input.view(), &mut output.view_mut());
    output
}

pub fn fftshift_into<T: Copy>(input: &ArrayView1<T>, output: &mut ArrayViewMut1<T>) {
    output.slice_mut(s![0..input.len()/2]).assign(&input.slice(s![(input.len()+1)/2..]));
    output.slice_mut(s![input.len()/2..]).assign(&input.slice(s![..(input.len()+1)/2]));
}

pub fn ifftshift<T: Copy + Zero>(input: &ArrayView1<T>) -> Array1<T> {
    let mut output = Array1::zeros(input.len());
    ifftshift_into(&input.view(), &mut output.view_mut());
    output
}

pub fn ifftshift_into<T: Copy>(input: &ArrayView1<T>, output: &mut ArrayViewMut1<T>) {
    output.slice_mut(s![0..(input.len()+1)/2]).assign(&input.slice(s![input.len()/2..]));
    output.slice_mut(s![(input.len()+1)/2..]).assign(&input.slice(s![..input.len()/2]));
}

pub fn fftfreq<F: Float>(n: usize, sample_rate: F) -> Array1<F> {
    let scale = sample_rate / F::from(n).unwrap();
    let center = (n - 1) / 2 + 1;

    let mut result = Array1::zeros(n);
    result.slice_mut(s![..center]).assign(&Array1::range(F::zero(), F::from(center).unwrap(), F::one()));
    result.slice_mut(s![center..]).assign(&Array1::range(-F::from(n / 2).unwrap(), F::zero(), F::one()));
    result.mapv_into(|v| v * scale)
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InversionMethod {
    Basic,      // overlap-add
    Weighted,   // weighted overlap-add
}

// check for non-zero overlap add compliance
pub fn check_nola<W, F>(window: &W, overlap: usize, eps: F) -> bool
where
    W: WindowFunction<F>,
    F: Float + std::ops::AddAssign,
{
    let window: Array1<F> = window.to_array();
    let step = window.len() - overlap;

    let mut bins: Array1<F> = Array1::zeros(step);
    for i in 0..(window.len() / step) {
        for b in 0..bins.len() {
            bins[b] += window[i * step + b].powi(2);
        }
    }

    if window.len() % step != 0 {
        for b in 0..(window.len() % step) {
            bins[b] += window[(window.len() / step) * step + b].powi(2);
        }
    }

    let min = bins.fold(F::infinity(), |a, b| F::min(a, *b));
    min > eps
}

// check for weak COLA compliance
pub fn check_cola<W, F>(window: &W, overlap: usize, method: InversionMethod, eps: F) -> bool
where
    W: WindowFunction<F>,
    F: Float + std::ops::AddAssign + std::ops::SubAssign,
{
    let window: Array1<F> = window.to_array();
    let step = window.len() - overlap;

    let a = match method {
        InversionMethod::Basic    => 1,
        InversionMethod::Weighted => 2,
    };

    let mut bins: Array1<F> = Array1::zeros(step);
    for i in 0..(window.len() / step) {
        for b in 0..bins.len() {
            bins[b] += window[i * step + b].powi(a);
        }
    }

    if window.len() % step != 0 {
        for b in 0..(window.len() % step) {
            bins[b] += window[(window.len() / step) * step + b].powi(a);
        }
    }

    bins -= &bins.mean_axis(Axis(0));
    bins.fold(F::zero(), |a, b| F::max(a, F::abs(*b))) < eps
}
