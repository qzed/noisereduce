use super::Gain;
use crate::math::{bessel, expint, NumCastUnchecked};

use ndarray::{azip, ArrayView1, ArrayViewMut1};
use num::traits::FloatConst;
use num::{Complex, Float};


pub struct Mmse<T> {
    nu_min: T,
    nu_max: T,
}

impl<T: Float> Mmse<T> {
    pub fn new(nu_min: T, nu_max: T) -> Self {
        Mmse { nu_min, nu_max }
    }
}

impl<T: Float> Default for Mmse<T> {
    fn default() -> Self {
        Mmse {
            nu_min: T::from(1e-50).unwrap(),
            nu_max: T::from(500.0).unwrap(),
        }
    }
}

impl<T> Gain<T> for Mmse<T>
where
    T: Float + FloatConst + NumCastUnchecked,
{
    fn update(
        &mut self,
        spectrum: ArrayView1<Complex<T>>,
        snr_pre: ArrayView1<T>,
        snr_post: ArrayView1<T>,
        gain: ArrayViewMut1<T>,
    ) {
        let half = T::from(0.5).unwrap();
        let fspi2 = T::PI().sqrt() * half;

        azip!((gain in gain, spectrum in spectrum, snr_pre in snr_pre, snr_post in snr_post) {
            let nu = *snr_pre / (T::one() + *snr_pre) * *snr_post;
            let nu = nu.max(self.nu_min).min(self.nu_max);          // prevent over/underflows

            *gain = fspi2
                * (T::sqrt(nu) * T::exp(-nu * half) / *snr_post)
                * ((T::one() + nu) * bessel::I0(nu * half) + nu * bessel::I1(nu * half))
                * spectrum.norm();
        });
    }
}


pub struct LogMmse<T> {
    nu_min: T,
    nu_max: T,
}

impl<T: Float> LogMmse<T> {
    pub fn new(nu_min: T, nu_max: T) -> Self {
        LogMmse { nu_min, nu_max }
    }
}

impl<T: Float> Default for LogMmse<T> {
    fn default() -> Self {
        LogMmse {
            nu_min: T::from(1e-50).unwrap(),
            nu_max: T::from(500.0).unwrap(),
        }
    }
}

impl<T> Gain<T> for LogMmse<T>
where
    T: Float + FloatConst + NumCastUnchecked,
{
    fn update(
        &mut self,
        _spectrum: ArrayView1<Complex<T>>,
        snr_pre: ArrayView1<T>,
        snr_post: ArrayView1<T>,
        gain: ArrayViewMut1<T>,
    ) {
        let half = T::from(0.5).unwrap();

        azip!((gain in gain, snr_pre in snr_pre, snr_post in snr_post) {
            let nu = *snr_pre / (T::one() + *snr_pre) * *snr_post;
            let nu = nu.max(self.nu_min).min(self.nu_max);          // prevent over/underflows

            *gain = (*snr_pre / (T::one() + *snr_pre)) * T::exp(-half * expint::Ei(-nu));
        });
    }
}
