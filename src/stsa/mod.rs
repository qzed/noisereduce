pub mod gain;
pub mod noise;
pub mod snr;
pub mod utils;


use crate::proc::Processor;
use crate::vad::b::SpeechProbabilityEstimator;

use ndarray::{azip, Array1, ArrayView1, ArrayViewMut1};
use num::{Complex, Float};


pub trait Gain<T> {
    fn update(
        &mut self,
        spectrum: ArrayView1<Complex<T>>,
        snr_pre: ArrayView1<T>,
        snr_post: ArrayView1<T>,
        gain: ArrayViewMut1<T>,
    );
}

impl<T, G> Gain<T> for Box<G>
where
    G: Gain<T>,
{
    fn update(
        &mut self,
        spectrum: ArrayView1<Complex<T>>,
        snr_pre: ArrayView1<T>,
        snr_post: ArrayView1<T>,
        gain: ArrayViewMut1<T>,
    ) {
        self.as_mut().update(spectrum, snr_pre, snr_post, gain)
    }
}

impl<T> Gain<T> for Box<dyn Gain<T>> {
    fn update(
        &mut self,
        spectrum: ArrayView1<Complex<T>>,
        snr_pre: ArrayView1<T>,
        snr_post: ArrayView1<T>,
        gain: ArrayViewMut1<T>,
    ) {
        self.as_mut().update(spectrum, snr_pre, snr_post, gain)
    }
}

impl<T, G> Gain<T> for &mut G
where
    G: Gain<T>,
{
    fn update(
        &mut self,
        spectrum: ArrayView1<Complex<T>>,
        snr_pre: ArrayView1<T>,
        snr_post: ArrayView1<T>,
        gain: ArrayViewMut1<T>,
    ) {
        (**self).update(spectrum, snr_pre, snr_post, gain)
    }
}

impl<T> Gain<T> for &mut dyn Gain<T> {
    fn update(
        &mut self,
        spectrum: ArrayView1<Complex<T>>,
        snr_pre: ArrayView1<T>,
        snr_post: ArrayView1<T>,
        gain: ArrayViewMut1<T>,
    ) {
        (**self).update(spectrum, snr_pre, snr_post, gain)
    }
}


pub trait NoiseTracker<T> {
    fn update(&mut self, spectrum: ArrayView1<Complex<T>>, noise_est: ArrayViewMut1<T>);
}

impl<T, N> NoiseTracker<T> for Box<N>
where
    N: NoiseTracker<T>,
{
    fn update(&mut self, spectrum: ArrayView1<Complex<T>>, noise_est: ArrayViewMut1<T>) {
        self.as_mut().update(spectrum, noise_est)
    }
}

impl<T> NoiseTracker<T> for Box<dyn NoiseTracker<T>> {
    fn update(&mut self, spectrum: ArrayView1<Complex<T>>, noise_est: ArrayViewMut1<T>) {
        self.as_mut().update(spectrum, noise_est)
    }
}

impl<T, N> NoiseTracker<T> for &mut N
where
    N: NoiseTracker<T>,
{
    fn update(&mut self, spectrum: ArrayView1<Complex<T>>, noise_est: ArrayViewMut1<T>) {
        (**self).update(spectrum, noise_est)
    }
}

impl<T> NoiseTracker<T> for &mut dyn NoiseTracker<T> {
    fn update(&mut self, spectrum: ArrayView1<Complex<T>>, noise_est: ArrayViewMut1<T>) {
        (**self).update(spectrum, noise_est)
    }
}


pub trait SnrEstimator<T> {
    fn update(
        &mut self,
        spectrum: ArrayView1<Complex<T>>,
        noise_power: ArrayView1<T>,
        gain: ArrayView1<T>,
        snr_pre: ArrayViewMut1<T>,
        snr_post: ArrayViewMut1<T>,
    );
}

impl<T, S> SnrEstimator<T> for Box<S>
where
    S: SnrEstimator<T>,
{
    fn update(
        &mut self,
        spectrum: ArrayView1<Complex<T>>,
        noise_power: ArrayView1<T>,
        gain: ArrayView1<T>,
        snr_pre: ArrayViewMut1<T>,
        snr_post: ArrayViewMut1<T>,
    ) {
        self.as_mut().update(spectrum, noise_power, gain, snr_pre, snr_post)
    }
}

impl<T> SnrEstimator<T> for Box<dyn SnrEstimator<T>> {
    fn update(
        &mut self,
        spectrum: ArrayView1<Complex<T>>,
        noise_power: ArrayView1<T>,
        gain: ArrayView1<T>,
        snr_pre: ArrayViewMut1<T>,
        snr_post: ArrayViewMut1<T>,
    ) {
        self.as_mut().update(spectrum, noise_power, gain, snr_pre, snr_post)
    }
}

impl<T, S> SnrEstimator<T> for &mut S
where
    S: SnrEstimator<T>,
{
    fn update(
        &mut self,
        spectrum: ArrayView1<Complex<T>>,
        noise_power: ArrayView1<T>,
        gain: ArrayView1<T>,
        snr_pre: ArrayViewMut1<T>,
        snr_post: ArrayViewMut1<T>,
    ) {
        (**self).update(spectrum, noise_power, gain, snr_pre, snr_post)
    }
}

impl<T> SnrEstimator<T> for &mut dyn SnrEstimator<T> {
    fn update(
        &mut self,
        spectrum: ArrayView1<Complex<T>>,
        noise_power: ArrayView1<T>,
        gain: ArrayView1<T>,
        snr_pre: ArrayViewMut1<T>,
        snr_post: ArrayViewMut1<T>,
    ) {
        (**self).update(spectrum, noise_power, gain, snr_pre, snr_post)
    }
}


pub trait SetNoiseEstimate<T> {
    fn set_noise_estimate(&mut self, noise: ArrayView1<T>);
}


pub trait NoiseReductionProcessor<T>: Processor<T> + SetNoiseEstimate<T> {}

impl<T, P> NoiseReductionProcessor<T> for P
where
    P: Processor<T> + SetNoiseEstimate<T>,
{}


pub struct Stsa<T, G, S, N> {
    block_size: usize,

    gain_fn: G,
    snr_est: S,
    noise_est: N,

    noise_pwr: Array1<T>,
    snr_pre: Array1<T>,
    snr_post: Array1<T>,
    gain: Array1<T>,
}

impl<T, G, S, N> Stsa<T, G, S, N>
where
    T: Float,
    G: Gain<T>,
    N: NoiseTracker<T>,
    S: SnrEstimator<T>,
{
    pub fn new(block_size: usize, gain: G, snr_est: S, noise_est: N) -> Self {
        Stsa {
            block_size,

            gain_fn: gain,
            snr_est,
            noise_est,

            noise_pwr: Array1::zeros(block_size),
            snr_pre: Array1::from_elem(block_size, T::one()),
            snr_post: Array1::from_elem(block_size, T::one()),
            gain: Array1::from_elem(block_size, T::one()),
        }
    }
}

impl<T, G, S, N> SetNoiseEstimate<T> for Stsa<T, G, S, N>
where
    T: Float,
{
    fn set_noise_estimate(&mut self, noise: ArrayView1<T>) {
        self.noise_pwr.assign(&noise);
    }
}

impl<T, G, S, N> Processor<T> for Stsa<T, G, S, N>
where
    T: Float,
    G: Gain<T>,
    N: NoiseTracker<T>,
    S: SnrEstimator<T>,
{
    fn block_size(&self) -> usize {
        self.block_size
    }

    fn process(
        &mut self,
        spectrum_in: ArrayView1<Complex<T>>,
        mut spectrum_out: ArrayViewMut1<Complex<T>>,
    ) {
        // update noise estimate
        self.noise_est
            .update(spectrum_in.view(), self.noise_pwr.view_mut());

        // update a priori and a posteriori snr
        self.snr_est.update(
            spectrum_in.view(),
            self.noise_pwr.view(),
            self.gain.view(),
            self.snr_pre.view_mut(),
            self.snr_post.view_mut(),
        );

        // calculate gain function
        self.gain_fn.update(
            spectrum_in.view(),
            self.snr_pre.view(),
            self.snr_post.view(),
            self.gain.view_mut(),
        );

        // apply gain
        azip!(mut spectrum_out, spectrum_in, gain (&self.gain) in {
            *spectrum_out = spectrum_in * gain.min(T::one())
        });
    }
}


// Modified (Short-Time) Spectral Amplitude, i.e. for Optimally-Modified
// Log-Spectral Amplitude (OM-LSA) implementation.
pub struct ModStsa<T, N, P, S, G> {
    block_size: usize,
    snr_pre: Array1<T>,
    snr_post: Array1<T>,
    noise_power: Array1<T>,
    gain_h1: Array1<T>,
    p_gain: Array1<T>,
    gain_min: T,

    noise_est: N,
    p_est: P,
    snr_est: S,
    gain_fn: G,
}

impl<T, N, P, S, G> ModStsa<T, N, P, S, G>
where
    T: Float,
{
    pub fn new(block_size: usize, noise_est: N, p_est: P, snr_est: S, gain_h1: G, gain_min: T) -> Self {
        ModStsa {
            block_size,
            snr_pre: Array1::zeros(block_size),
            snr_post: Array1::zeros(block_size),
            noise_power: Array1::zeros(block_size),
            gain_h1: Array1::zeros(block_size),
            p_gain: Array1::zeros(block_size),
            gain_min,
            noise_est,
            snr_est,
            p_est,
            gain_fn: gain_h1,
        }
    }
}

impl<T, N, P, S, G> SetNoiseEstimate<T> for ModStsa<T, N, P, S, G>
where
    T: Float,
{
    fn set_noise_estimate(&mut self, noise: ArrayView1<T>) {
        self.noise_power.assign(&noise);
    }
}

impl<T, N, P, S, G> Processor<T> for ModStsa<T, N, P, S, G>
where
    T: Float,
    N: NoiseTracker<T>,
    S: SnrEstimator<T>,
    P: SpeechProbabilityEstimator<T>,
    G: Gain<T>,
{
    fn block_size(&self) -> usize {
        self.block_size
    }

    fn process(&mut self, spec_in: ArrayView1<Complex<T>>, spec_out: ArrayViewMut1<Complex<T>>) {
        // noise spectrum estimation
        self.noise_est.update(spec_in, self.noise_power.view_mut());

        // update a priori and a posteriori SNR
        self.snr_est.update(
            spec_in,
            self.noise_power.view(),
            self.gain_h1.view(),
            self.snr_pre.view_mut(),
            self.snr_post.view_mut(),
        );

        // compute speech probability for gain (p_gain)
        self.p_est.update(
            spec_in,
            self.snr_pre.view(),
            self.snr_post.view(),
            self.p_gain.view_mut(),
        );

        // compute gain for speech presence
        self.gain_fn.update(
            spec_in,
            self.snr_pre.view(),
            self.snr_post.view(),
            self.gain_h1.view_mut(),
        );

        // apply gain
        azip!(mut spec_out (spec_out), spec_in (spec_in), gain_h (&self.gain_h1), p (&self.p_gain) in {
            *spec_out = spec_in * gain_h.powf(p) * self.gain_min.powf(T::one() - p);
        });
    }
}


pub struct Subtraction<T> {
    block_size: usize,
    factor: T,
    post_gain: T,
    noise_est: Array1<T>,
}

impl<T> Subtraction<T>
where
    T: Float,
{
    pub fn new(block_size: usize, factor: T, post_gain: T) -> Self {
        Subtraction {
            block_size,
            factor,
            post_gain,
            noise_est: Array1::zeros(block_size),
        }
    }
}

impl<T> SetNoiseEstimate<T> for Subtraction<T>
where
    T: Float,
{
    fn set_noise_estimate(&mut self, noise: ArrayView1<T>) {
        self.noise_est.assign(&noise);
    }
}

impl<T> Processor<T> for Subtraction<T>
where
    T: Float,
{
    fn block_size(&self) -> usize {
        self.block_size
    }

    fn process(&mut self, spec_in: ArrayView1<Complex<T>>, mut spec_out: ArrayViewMut1<Complex<T>>) {
        azip!(mut spec_out, spec_in, noise_est (&self.noise_est) in {
            let r = spec_in.norm() - self.factor * noise_est;
            let t = spec_in.arg();

            *spec_out = Complex::from_polar(&T::zero().max(r), &t) * self.post_gain;
        });
    }
}
