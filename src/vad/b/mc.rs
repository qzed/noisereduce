//! Minima Controlled Voice Activity Detection.
//! 
//! For noise estimation via Minima Controlled Recursive Averaging (MCRA) as
//! proposed in https://doi.org/10.1016/S0165-1684(01)00128-1.

use super::VoicePresenceDetector;
use crate::window::WindowFunction;

use ndarray::{s, azip, Array1, ArrayBase, Data, DataMut, Ix1};
use num::{Complex, Float};
use num::traits::NumAssign;


pub struct MinimaControlledVad<T> {
    block_size: usize,
    spectrum_padded: Array1<T>,
    spectrum_f: Array1<T>,          // S_f(k, l)
    spectrum_avg: Array1<T>,        // S(k, l)
    spectrum_min: Array1<T>,        // S_min(k, l)
    spectrum_tmp: Array1<T>,        // S_tmp(k, l)
    window: Array1<T>,              // b(i)
    alpha_s: T,                     // alpha for recursive time-averaging of spectrum
    alpha_p: T,                     // alpha for recursive time-averaging of speech probability
    delta: T,                       // delta, discriminator threshold for speech activity
    tracking_len: usize,            // L
    tracking_frame: usize,
}

impl<T> MinimaControlledVad<T>
where
    T: Float,
{
    pub fn new<W>(block_size: usize, window: &W, alpha_s: T, alpha_p: T, delta: T, tracking_len: usize)
        -> Self
    where
        W: WindowFunction<T>,
    {
        assert!(window.len() % 2 != 0, "window must have odd size");

        MinimaControlledVad {
            block_size,
            spectrum_padded: Array1::zeros(block_size + window.len() - 1),
            spectrum_f: Array1::zeros(block_size),
            spectrum_avg: Array1::zeros(block_size),
            spectrum_min: Array1::zeros(block_size),
            spectrum_tmp: Array1::zeros(block_size),
            window: window.to_array(),
            alpha_s,
            alpha_p,
            delta,
            tracking_len,
            tracking_frame: 0,
        }
    }
}

impl<T> VoicePresenceDetector<T> for MinimaControlledVad<T>
where
    T: Float + NumAssign,
{
    fn detect_into<D, E>(&mut self, spectrum: &ArrayBase<D, Ix1>, prob: &mut ArrayBase<E, Ix1>)
    where
        D: Data<Elem = Complex<T>>,
        E: DataMut<Elem = T>,
    {
        let w = (self.window.len() - 1) / 2;

        // pad and compute power/variance
        self.spectrum_padded.slice_mut(s![..w]).fill(T::zero());
        azip!(mut p (&mut self.spectrum_padded.slice_mut(s![w..w+self.block_size])), s (spectrum) in {
            *p = s.norm_sqr();
        });
        self.spectrum_padded.slice_mut(s![w+self.block_size..]).fill(T::zero());

        // average spectrum over frequency (S_f)
        self.spectrum_f.fill(T::zero());
        for k in 0..self.block_size {
            for i in -(w as isize)..=(w as isize) {
                let idx_window = (i + w as isize) as usize;
                let idx_spectr = (i + k as isize + w as isize) as usize;

                self.spectrum_f[k] += self.spectrum_padded[idx_spectr] * self.window[idx_window];
            }
        }

        // average spectrum recursively over time (S)
        let alpha_s = self.alpha_s;
        azip!(mut s (&mut self.spectrum_avg), sf (&self.spectrum_f) in {
            *s = alpha_s * *s + (T::one() - alpha_s) * sf;
        });

        // minima tracking (S_min, S_tmp)
        azip!(mut s_min (&mut self.spectrum_min), mut s_tmp (&mut self.spectrum_tmp), s (&self.spectrum_avg) in {
            *s_min = s_min.min(s);
            *s_tmp = s_tmp.min(s);
        });

        self.tracking_frame += 1;
        if self.tracking_frame == self.tracking_len {
            self.spectrum_min.assign(&self.spectrum_tmp);
            self.spectrum_tmp.assign(&self.spectrum_avg);
            self.tracking_frame = 0;
        }

        // compute speech presence probability (p')
        let alpha_p = self.alpha_p;
        let delta = self.delta;
        azip!(mut p (prob), s (&self.spectrum_avg), s_min (&self.spectrum_min) in {
            let i = if s / s_min > delta { T::one() } else { T::zero() };
            *p = alpha_p * *p + (T::one() - alpha_p) * i;
        });
    }
}
