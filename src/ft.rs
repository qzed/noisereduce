// see https://ccrma.stanford.edu/~jos/sasp/Summary_STFT_Computation_Using.html

use crate::window::WindowFunction;

use std::sync::Arc;

use num::Complex;
use num::traits::{Float, Zero, NumAssign};
use ndarray::{s, ArrayBase, Array1, Array2, Axis, Ix1, Ix2, Data, DataMut};
use rustfft::{FFT, FFTnum, FFTplanner};


pub fn magnitude_to_db<F: Float>(m: F) -> F {
    F::from(20.0).unwrap() * F::log10(m)
}

pub fn fftshift<D, T>(input: &ArrayBase<D, Ix1>) -> Array1<T>
where
    D: Data<Elem = T>,
    T: Clone + Zero,
{
    let mut output = Array1::zeros(input.len());
    fftshift_into(input, &mut output);
    output
}

pub fn fftshift_into<D, M, T>(input: &ArrayBase<D, Ix1>, output: &mut ArrayBase<M, Ix1>)
where
    D: Data<Elem = T>,
    M: DataMut<Elem = T>,
    T: Clone,
{
    output.slice_mut(s![0..input.len()/2]).assign(&input.slice(s![(input.len()+1)/2..]));
    output.slice_mut(s![input.len()/2..]).assign(&input.slice(s![..(input.len()+1)/2]));
}

pub fn ifftshift<D, T>(input: &ArrayBase<D, Ix1>) -> Array1<T>
where
    D: Data<Elem = T>,
    T: Clone + Zero,
{
    let mut output = Array1::zeros(input.len());
    ifftshift_into(&input.view(), &mut output.view_mut());
    output
}

pub fn ifftshift_into<D, M, T>(input: &ArrayBase<D, Ix1>, output: &mut ArrayBase<M, Ix1>)
where
    D: Data<Elem = T>,
    M: DataMut<Elem = T>,
    T: Clone,
{
    output.slice_mut(s![0..(input.len()+1)/2]).assign(&input.slice(s![input.len()/2..]));
    output.slice_mut(s![(input.len()+1)/2..]).assign(&input.slice(s![..input.len()/2]));
}

pub fn fftfreq<F: Float>(n: usize, sample_rate: F) -> Array1<F> {
    let mut result = Array1::zeros(n);
    fftfreq_into(n, sample_rate, &mut result);
    result
}

pub fn fftfreq_into<D, F>(n: usize, sample_rate: F, output: &mut ArrayBase<D, Ix1>)
where
    D: DataMut<Elem = F>,
    F: Float,
{
    let center = (n - 1) / 2 + 1;
    let scale = sample_rate / F::from(n).unwrap();

    output.slice_mut(s![..center]).assign(&Array1::range(F::zero(), F::from(center).unwrap(), F::one()));
    output.slice_mut(s![center..]).assign(&Array1::range(-F::from(n / 2).unwrap(), F::zero(), F::one()));
    output.mapv_inplace(|v| v * scale);
}

pub fn stft_times<F>(input_len: usize, segment_len: usize, overlap: usize, padded: bool, sample_rate: F)
    -> Array1<F>
where
    F: Float
{
    let step = segment_len - overlap;
    let len = (input_len - overlap) / step;

    let mut output = Array1::zeros(len);
    stft_times_into(input_len, segment_len, overlap, padded, sample_rate, &mut output);
    output
}

pub fn stft_times_into<D, F>(input_len: usize, segment_len: usize, overlap: usize, padded: bool,
                             sample_rate: F, output: &mut ArrayBase<D, Ix1>)
where
    D: DataMut<Elem = F>,
    F: Float,
{
    let step = segment_len - overlap;
    let len = (input_len - overlap) / step;

    assert!(output.len() >= len);

    let offs = if padded {
        0
    } else {
        (segment_len - 1) / 2
    };

    for (i, v) in output.slice_mut(s![0..len]).indexed_iter_mut() {
        *v = F::from(i + offs).unwrap() * F::from(step).unwrap() / sample_rate;
    }
}

pub fn sample_times<F>(num_samples: usize, sample_rate: F) -> Array1<F>
where
    F: Float,
{
    Array1::range(F::zero(), F::from(num_samples).unwrap(), F::one()).mapv(|v| v / sample_rate)
}

pub fn sample_times_into<D, F>(sample_rate: F, output: &mut ArrayBase<D, Ix1>)
where
    D: DataMut<Elem = F>,
    F: Float,
{
    for (i, v) in output.indexed_iter_mut() {
        *v = F::from(i).unwrap() / sample_rate
    }
}


pub fn extend_zero<D, F>(input: &ArrayBase<D, Ix1>, n: usize) -> Array1<F>
where
    D: Data<Elem = F>,
    F: Clone + Zero,
{
    let mut result = Array1::zeros(input.len() + 2 * n);
    extend_zero_into(n, input, &mut result);
    result
}

pub fn extend_zero_into<D, M, F>(n: usize, input: &ArrayBase<D, Ix1>, output: &mut ArrayBase<M, Ix1>)
where
    D: Data<Elem = F>,
    M: DataMut<Elem = F>,
    F: Clone + Zero,
{
    output.slice_mut(s![0..n]).fill(F::zero());
    output.slice_mut(s![n..n+input.len()]).assign(&input);
    output.slice_mut(s![n+input.len()..]).fill(F::zero());
}

pub fn extend_const<D, F>(input: &ArrayBase<D, Ix1>, n: usize) -> Array1<F>
where
    D: Data<Elem = F>,
    F: Clone + Zero,
{
    let mut result = Array1::zeros(input.len() + 2 * n);
    extend_const_into(n, input, &mut result);
    result
}

pub fn extend_const_into<D, M, F>(n: usize, input: &ArrayBase<D, Ix1>, output: &mut ArrayBase<M, Ix1>)
where
    D: Data<Elem = F>,
    M: DataMut<Elem = F>,
    F: Clone,
{
    let a = input[0].clone();
    let b = input[input.len()-1].clone();

    output.slice_mut(s![0..n]).fill(a);
    output.slice_mut(s![n..n+input.len()]).assign(&input);
    output.slice_mut(s![n+input.len()..]).fill(b);
}

pub fn extend_even<D, F>(input: &ArrayBase<D, Ix1>, n: usize) -> Array1<F>
where
    D: Data<Elem = F>,
    F: Clone + Zero,
{
    let mut result = Array1::zeros(input.len() + 2 * n);
    extend_even_into(n, input, &mut result);
    result
}

pub fn extend_even_into<D, M, F>(n: usize, input: &ArrayBase<D, Ix1>, output: &mut ArrayBase<M, Ix1>)
where
    D: Data<Elem = F>,
    M: DataMut<Elem = F>,
    F: Clone,
{
    output.slice_mut(s![0..n]).assign(&input.slice(s![1..n+1; -1]));
    output.slice_mut(s![n..n+input.len()]).assign(&input);
    output.slice_mut(s![n+input.len()..]).assign(&input.slice(s![input.len()-n-1..input.len()-1; -1]));
}

pub fn extend_odd<D, F>(input: &ArrayBase<D, Ix1>, n: usize) -> Array1<F>
where
    D: Data<Elem = F>,
    F: Clone + Zero + std::ops::Add<Output=F> + std::ops::SubAssign,
{
    let mut result = Array1::zeros(input.len() + 2 * n);
    extend_odd_into(n, input, &mut result);
    result
}

pub fn extend_odd_into<D, M, F>(n: usize, input: &ArrayBase<D, Ix1>, output: &mut ArrayBase<M, Ix1>)
where
    D: Data<Elem = F>,
    M: DataMut<Elem = F>,
    F: Clone + std::ops::Add<Output=F> + std::ops::SubAssign,
{
    let a = input[0].clone() + input[0].clone();
    let b = input[input.len()-1].clone() + input[input.len()-1].clone();

    let mut ext_a = output.slice_mut(s![0..n]);
    ext_a.fill(a);
    ext_a -= &input.slice(s![1..n+1; -1]);

    output.slice_mut(s![n..n+input.len()]).assign(&input);

    let mut ext_b = output.slice_mut(s![n+input.len()..]);
    ext_b.fill(b);
    ext_b -= &input.slice(s![input.len()-n-1..input.len()-1; -1]);
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


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Padding {
    None,
    Zero,
    Even,
    Odd,
    Const,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InversionMethod {
    Basic,      // overlap-add
    Weighted,   // weighted overlap-add
}


pub struct StftBuilder<T> {
    fft:     Arc<dyn FFT<T>>,
    padding: Padding,
    window:  Array1<T>,
    overlap: Option<usize>,
    shifted: bool,
    inverse: InversionMethod,
}

impl<T> StftBuilder<T>
where
    T: FFTnum,
{
    pub fn new<W>(window: &W) -> Self
    where
        W: WindowFunction<T>
    {
        Self::with_fft(window, FFTplanner::new(false).plan_fft(window.len()))
    }

    pub fn with_len<W>(window: &W, fft_len: usize) -> Self
    where
        W: WindowFunction<T>
    {
        Self::with_fft(window, FFTplanner::new(false).plan_fft(fft_len))
    }

    pub fn with_fft<W>(window: &W, fft: Arc<dyn FFT<T>>) -> Self
    where
        W: WindowFunction<T>
    {
        StftBuilder {
            fft,
            padding: Padding::Zero,
            window:  window.to_array(),
            overlap: None,
            shifted: false,
            inverse: InversionMethod::Weighted,
        }
    }

    pub fn overlap(mut self, overlap: usize) -> Self {
        self.overlap = Some(overlap);
        self
    }

    pub fn padding(mut self, padding: Padding) -> Self {
        self.padding = padding;
        self
    }

    pub fn shifted(mut self, shifted: bool) -> Self {
        self.shifted = shifted;
        self
    }

    pub fn inversion_method(mut self, inverse: InversionMethod) -> Self {
        self.inverse = inverse;
        self
    }

    pub fn build(self) -> Stft<T> {
        let len_fft = self.fft.len();
        let len_segment = self.window.len();
        let len_overlap = self.overlap.unwrap_or_else(|| (len_segment / 4) * 3);

        assert!(len_overlap < len_segment);
        assert!(len_fft >= len_segment);

        Stft {
            fft:     self.fft,
            window:  self.window,
            padding: self.padding,
            shifted: self.shifted,
            inverse: self.inverse,
            buf_in:  Array1::zeros(len_fft),
            buf_out: Array1::zeros(len_fft),
            buf_pad: Vec::new(),
            len_fft, len_segment, len_overlap
        }
    }
}

pub struct Stft<T> {
    fft:         Arc<dyn FFT<T>>,
    len_fft:     usize,
    len_overlap: usize,
    len_segment: usize,
    shifted:     bool,
    inverse:     InversionMethod,
    window:      Array1<T>,
    padding:     Padding,
    buf_in:      Array1<Complex<T>>,
    buf_out:     Array1<Complex<T>>,
    buf_pad:     Vec<Complex<T>>,
}

impl<T> Stft<T>
where
    T: FFTnum + NumAssign,
{
    pub fn len_segment(&self) -> usize {
        self.len_segment
    }

    pub fn len_fft(&self) -> usize {
        self.len_fft
    }

    pub fn len_overlap(&self) -> usize {
        self.len_overlap
    }

    pub fn len_output_forward(&self, input_len: usize) -> usize {
        let input_len = if self.padding != Padding::None {
            input_len + 2 * (self.len_segment / 2)
        } else {
            input_len
        };

        (input_len - self.len_overlap) / (self.len_segment - self.len_overlap)
    }

    pub fn forward_times<F: Float>(&self, input_len: usize, sample_rate: F) -> Array1<F> {
        stft_times(input_len, self.len_segment, self.len_overlap, self.padding != Padding::None, sample_rate)
    }

    pub fn forward_freqs<F: Float>(&self, sample_rate: F) -> Array1<F> {
        if self.shifted {
            fftshift(&fftfreq(self.len_fft, sample_rate))
        } else {
            fftfreq(self.len_fft, sample_rate)
        }
    }

    pub fn forward<D>(&mut self, input: &ArrayBase<D, Ix1>) -> Array2<Complex<T>>
    where
        D: Data<Elem = Complex<T>>,
    {
        let mut output = Array2::zeros((self.len_output_forward(input.len()), self.len_fft));
        self.forward_into(input, &mut output);
        output
    }

    pub fn forward_into<D, M>(&mut self, input: &ArrayBase<D, Ix1>, output: &mut ArrayBase<M, Ix2>)
    where
        D: Data<Elem = Complex<T>>,
        M: DataMut<Elem = Complex<T>>,
    {
        let step = self.len_segment - self.len_overlap;
        let n_seg = self.len_output_forward(input.len());

        assert!(output.shape()[0] >= n_seg);
        assert!(output.shape()[1] >= self.len_fft);

        // pad input signal
        let input_padded = if self.padding == Padding::None {
            input.view()
        } else {
            self.buf_pad.resize(input.len() + 2 * (self.len_segment / 2), Complex::zero());
            let mut padded = ndarray::aview_mut1(&mut self.buf_pad[..]);

            match self.padding {
                Padding::None  => unreachable!(),
                Padding::Zero  => extend_zero_into(self.len_segment / 2, input, &mut padded),
                Padding::Even  => extend_even_into(self.len_segment / 2, input, &mut padded),
                Padding::Odd   => extend_odd_into(self.len_segment / 2, input, &mut padded),
                Padding::Const => extend_const_into(self.len_segment / 2, input, &mut padded),
            };

            ndarray::aview1(&self.buf_pad[..])
        };

        // compute FFTs
        for i in 0..n_seg {
            let k = i * step;

            // zero padding at start and end + windowing
            let s = (self.len_fft - self.len_segment) / 2;
            for j in 0..s {
                self.buf_in[j] = Complex::zero();
            }
            for j in 0..self.len_segment {
                self.buf_in[s + j] = input_padded[k + j] * self.window[j];
            }
            for j in s+self.len_segment..self.len_fft {
                self.buf_in[j] = Complex::zero();
            }

            // fft
            self.fft.process(self.buf_in.as_slice_mut().unwrap(), self.buf_out.as_slice_mut().unwrap());

            // copy to output segment
            if !self.shifted {
                output.slice_mut(s![i, 0..self.len_fft]).assign(&self.buf_out);
            } else {
                fftshift_into(&self.buf_out, &mut output.slice_mut(s![i, 0..self.len_fft]));
            }
        }
    }
}
