pub mod gain;
pub mod noise;
pub mod snr;
pub mod utils;


use ndarray::{azip, Array1, ArrayBase, ArrayView1, ArrayViewMut1, Data, Ix1};
use num::{Complex, Float};


pub trait Processor<T> {
    fn block_size(&self) -> usize;
    fn process(&mut self, spectrum_in: ArrayView1<Complex<T>>, spectrum_out: ArrayViewMut1<Complex<T>>);
}


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

    pub fn set_noise_power_estimate<D>(&mut self, noise_pwr: &ArrayBase<D, Ix1>)
    where
        D: Data<Elem = T>,
    {
        self.noise_pwr.assign(noise_pwr)
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
