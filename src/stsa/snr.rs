use super::SnrEstimator;

use ndarray::{azip, Array1, ArrayView1, ArrayViewMut1};
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
            snr_pre_max: T::from(1e30).unwrap(),
            snr_post_max: T::from(1e30).unwrap(),
        }
    }

    pub fn with_parameters(alpha: T, snr_pre_max: T, snr_post_max: T) -> Self {
        DecisionDirected { alpha, snr_pre_max, snr_post_max }
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


pub struct MaximumLikelihood<T> {
    snr_avg: Array1<T>,
    alpha: T,
    beta:  T,
}

impl<T: Float> MaximumLikelihood<T> {
    pub fn new(block_size: usize, alpha: T, beta: T) -> Self {
        MaximumLikelihood {
            snr_avg: Array1::from_elem(block_size, T::one()),
            alpha,
            beta,
        }
    }
}

impl<T> SnrEstimator<T> for MaximumLikelihood<T>
where
    T: Float,
{
    fn update(
        &mut self,
        spectrum: ArrayView1<Complex<T>>,
        noise_power: ArrayView1<T>,
        _gain: ArrayView1<T>,
        snr_pre: ArrayViewMut1<T>,
        snr_post: ArrayViewMut1<T>,
    ) {
        let snr_avg = &mut self.snr_avg;
        let a = self.alpha;
        let b = (T::one() - self.alpha) / self.beta;

        azip!(
            mut snr_pre (snr_pre),
            mut snr_post (snr_post),
            mut snr_avg (snr_avg),
            spectrum (spectrum),
            noise_power (noise_power),
        in {
            *snr_post = spectrum.norm_sqr() / noise_power;
            *snr_avg = a * *snr_avg + b * *snr_post;
            *snr_pre = (*snr_avg - T::one()).max(T::zero());
        });
    }
}
