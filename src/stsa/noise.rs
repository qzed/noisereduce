use super::NoiseTracker;
use crate::vad::b::VoiceActivityDetector;

use ndarray::{azip, Array1, ArrayView1, ArrayViewMut1};
use num::{Complex, Float};


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
        ExpTimeAvg{
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
        azip!(mut noise (noise_est), voiced (&self.voiced), spectrum (spectrum) in {
            if !voiced {
                *noise = self.alpha * *noise + (T::one() - self.alpha) * spectrum.norm_sqr();
            }
        });
    }
}
