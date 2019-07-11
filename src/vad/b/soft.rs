use super::SpeechProbabilityEstimator;
use crate::ft::extend_zero_into;
use crate::window::WindowFunction;

use ndarray::{azip, Array1, ArrayView1, ArrayViewMut1};
use num::traits::NumAssign;
use num::{Complex, Float};


pub struct SoftDecisionProbabilityEstimator<T> {
    block_size: usize,
    snr_pre_avg: Array1<T>,
    snr_pre_avg_padded: Array1<T>,
    p_global: Array1<T>,
    p_local: Array1<T>,
    snr_pre_avg_frame: T,
    snr_pre_peak: T,
    beta: T,
    h_local: Array1<T>,
    h_global: Array1<T>,
    snr_pre_min: T,
    snr_pre_max: T,
    snr_pre_peak_min: T,
    snr_pre_peak_max: T,
    q_max: T,
}

impl<T> SoftDecisionProbabilityEstimator<T>
where
    T: Float,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new<W1, W2>(
        block_size: usize,
        beta: T,
        h_local: &W1,
        h_global: &W2,
        snr_pre_min: T,
        snr_pre_max: T,
        snr_pre_peak_min: T,
        snr_pre_peak_max: T,
        q_max: T,
    ) -> Self
    where
        W1: WindowFunction<T> + ?Sized,
        W2: WindowFunction<T> + ?Sized,
    {
        assert!(h_local.len() % 2 == 1, "local window function must have odd size");
        assert!(h_global.len() % 2 == 1, "global window function must have odd size");
        assert!(h_global.len() >= h_local.len(), "global window must be larger than local window");

        let h_local = h_local.to_array();
        let h_global = h_global.to_array();

        SoftDecisionProbabilityEstimator {
            block_size,
            snr_pre_avg: Array1::zeros(block_size),
            snr_pre_avg_padded: Array1::zeros(block_size + h_global.len() - 1),
            p_global: Array1::zeros(block_size),
            p_local: Array1::zeros(block_size),
            snr_pre_avg_frame: T::zero(),
            snr_pre_peak: T::zero(),
            beta,
            h_local,
            h_global,
            snr_pre_min,
            snr_pre_max,
            snr_pre_peak_min,
            snr_pre_peak_max,
            q_max,
        }
    }
}

impl<T> SpeechProbabilityEstimator<T> for SoftDecisionProbabilityEstimator<T>
where
    T: Float + NumAssign,
{
    fn update(
        &mut self,
        _s: ArrayView1<Complex<T>>,
        snr_pre: ArrayView1<T>,
        snr_post: ArrayView1<T>,
        p: ArrayViewMut1<T>,
    ) {
        // average a priori SNR over time
        let beta = self.beta;
        azip!(mut snr_pre_avg (&mut self.snr_pre_avg), snr_pre (snr_pre) in {
            *snr_pre_avg = beta * *snr_pre_avg + (T::one() - beta) * snr_pre;
        });

        // average a priori SNR over frequencies and compute probabilities
        let w_local = (self.h_local.len() - 1) / 2;
        let w_global = (self.h_global.len() - 1) / 2;

        extend_zero_into(w_global, &self.snr_pre_avg, &mut self.snr_pre_avg_padded);

        for k in 0..self.block_size {
            // compute frequency average for local window
            self.p_local[k] = T::zero();
            for i in -(w_local as isize)..=(w_local as isize) {
                let idx_window = (i + w_local as isize) as usize;
                let idx_spectr = (i + k as isize + w_global as isize) as usize;

                self.p_local[k] += self.snr_pre_avg_padded[idx_spectr] * self.h_local[idx_window];
            }

            // compute actual p_local
            self.p_local[k] = if self.p_local[k] <= self.snr_pre_min {
                T::zero()
            } else if self.p_local[k] >= self.snr_pre_max {
                T::one()
            } else {
                T::ln(self.p_local[k] / self.snr_pre_min)
                    / T::ln(self.snr_pre_max / self.snr_pre_min)
            };

            // compute frequency average for global window
            self.p_global[k] = T::zero();
            for i in -(w_global as isize)..=(w_global as isize) {
                let idx_window = (i + w_global as isize) as usize;
                let idx_spectr = (i + k as isize + w_global as isize) as usize;

                self.p_global[k] += self.snr_pre_avg_padded[idx_spectr] * self.h_global[idx_window];
            }

            // compute actual p_global
            self.p_global[k] = if self.p_global[k] <= self.snr_pre_min {
                T::zero()
            } else if self.p_global[k] >= self.snr_pre_max {
                T::one()
            } else {
                T::ln(self.p_global[k] / self.snr_pre_min)
                    / T::ln(self.snr_pre_max / self.snr_pre_min)
            };
        }

        let norm = T::one() / T::from(self.snr_pre_avg.len()).unwrap();
        let snr_pre_avg_frame = self.snr_pre_avg.fold(T::zero(), |a, b| a + *b * norm);

        let p_frame = if snr_pre_avg_frame <= self.snr_pre_min {
            T::zero()
        } else if snr_pre_avg_frame > self.snr_pre_avg_frame {
            self.snr_pre_peak = snr_pre_avg_frame
                .max(self.snr_pre_peak_min)
                .min(self.snr_pre_peak_max);

            T::one()
        } else if snr_pre_avg_frame <= self.snr_pre_peak * self.snr_pre_min {
            T::zero()
        } else if snr_pre_avg_frame >= self.snr_pre_peak * self.snr_pre_max {
            T::one()
        } else {
            T::ln(snr_pre_avg_frame / self.snr_pre_peak / self.snr_pre_min)
                / T::ln(self.snr_pre_max / self.snr_pre_min)
        };

        self.snr_pre_avg_frame = snr_pre_avg_frame;

        // speech absence probability and speech presence probability
        azip!(
            mut p (p),
            snr_pre (snr_pre),
            snr_post (snr_post),
            p_local (&self.p_local),
            p_global (&self.p_global),
        in {
            // compute speech absence probability estimation
            let q = T::one() - p_local * p_global * p_frame;
            let q = q.min(self.q_max);

            // compute conditional speech presence probability
            let nu = (snr_pre / (T::one() + snr_pre)) * snr_post;

            let p_ = T::one() + (q / (T::one() - q)) * (T::one() + snr_pre) * T::exp(-nu);
            let p_ = T::one() / p_;
            *p = p_;
        });
    }
}
