use super::SnrEstimator;

use ndarray::{azip, ArrayView1, ArrayViewMut1};
use num::{Complex, Float};


pub struct DecisionDirected<T> {
    alpha: T,
    snr_pre_max: T,
    snr_post_max: T,
}

impl<T: Float> DecisionDirected<T> {
    pub fn new(alpha: T) -> Self {
        DecisionDirected {
            alpha,
            snr_pre_max: T::from(1e3).unwrap(),
            snr_post_max: T::from(1e3).unwrap(),
        }
    }
}

impl<T> SnrEstimator<T> for DecisionDirected<T>
where
    T: Float,
{
    fn update(
        &mut self,
        spectrum: ArrayView1<Complex<T>>,
        noise_power: ArrayView1<T>,
        gain: ArrayView1<T>,
        snr_pre: ArrayViewMut1<T>,
        snr_post: ArrayViewMut1<T>,
    ) {
        azip!(
            mut snr_pre (snr_pre),
            mut snr_post (snr_post),
            spectrum (spectrum),
            noise_power (noise_power),
            gain (gain)
        in {
            let snr_post_ = spectrum.norm_sqr() / noise_power;
            let snr_pre_ = self.alpha * gain.powi(2) * *snr_post
                + (T::one() - self.alpha) * (snr_post_ - T::one()).max(T::zero());

            *snr_pre = snr_pre_.min(self.snr_pre_max);
            *snr_post = snr_post_.min(self.snr_post_max);
        });
    }
}
