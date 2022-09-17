use super::NoiseTracker;
use crate::vad::b::{VoiceActivityDetector, VoicePresenceDetector};

use ndarray::{azip, Array1, ArrayView1, ArrayViewMut1};
use num::{Complex, Float};


pub struct NoUpdate<T> {
    _p: std::marker::PhantomData<*const T>,
}

impl<T> NoUpdate<T> {
    pub fn new() -> Self {
        NoUpdate {
            _p: std::marker::PhantomData,
        }
    }
}

impl<T> Default for NoUpdate<T> {
    fn default() -> Self {
        Self { _p: Default::default() }
    }
}

impl<T> NoiseTracker<T> for NoUpdate<T>
where
    T: Float,
{
    fn update(&mut self, _spectrum: ArrayView1<Complex<T>>, _noise_est: ArrayViewMut1<T>) {
        // do nothing
    }
}


pub struct ExpTimeAvg<T, V> {
    voiced: Array1<bool>,
    alpha: T,
    vad: V,
}

impl<T, V> ExpTimeAvg<T, V>
where
    T: Float,
{
    pub fn new(block_size: usize, alpha: T, vad: V) -> Self {
        ExpTimeAvg {
            voiced: Array1::from_elem(block_size, false),
            alpha,
            vad,
        }
    }
}

impl<T, V> NoiseTracker<T> for ExpTimeAvg<T, V>
where
    T: Float,
    V: VoiceActivityDetector<T>,
{
    fn update(&mut self, spectrum: ArrayView1<Complex<T>>, noise_est: ArrayViewMut1<T>) {
        self.vad.detect_into(&spectrum, &mut self.voiced);
        azip!((noise in noise_est, voiced in &self.voiced, spectrum in spectrum) {
            if !voiced {
                *noise = self.alpha * *noise + (T::one() - self.alpha) * spectrum.norm_sqr();
            }
        });
    }
}


// time-variant exponential averaging via speech probabilities
pub struct ProbabilisticExpTimeAvg<T, V> {
    p_voice: Array1<T>,
    alpha: T,
    vad: V,
}

impl<T, V> ProbabilisticExpTimeAvg<T, V>
where
    T: Float,
{
    pub fn new(block_size: usize, alpha: T, vad: V) -> Self {
        ProbabilisticExpTimeAvg {
            p_voice: Array1::zeros(block_size),
            alpha,
            vad,
        }
    }
}

impl<T, V> NoiseTracker<T> for ProbabilisticExpTimeAvg<T, V>
where
    T: Float,
    V: VoicePresenceDetector<T>,
{
    fn update(&mut self, spectrum: ArrayView1<Complex<T>>, noise_est: ArrayViewMut1<T>) {
        self.vad.detect_into(&spectrum, &mut self.p_voice);

        let alpha_d = self.alpha;
        azip!((noise in noise_est, p in &self.p_voice, spectrum in spectrum) {
            let alpha = alpha_d + (T::one() - alpha_d) * *p;
            *noise = alpha * *noise + (T::one() - alpha) * spectrum.norm_sqr();
        });
    }
}
