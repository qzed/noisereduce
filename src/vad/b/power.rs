use super::{VoiceActivityDetector, VoicePresenceDetector};

use ndarray::{azip, Array1, ArrayBase, Data, DataMut, Ix1, Ix2};
use num::{Complex, Float};

pub struct PowerThresholdVad<T> {
    floor: Array1<T>,
    ratio: T,
}

impl<T> PowerThresholdVad<T> {
    pub fn new(floor: Array1<T>, ratio: T) -> Self {
        PowerThresholdVad { floor, ratio }
    }
}

impl<T> VoiceActivityDetector<T> for PowerThresholdVad<T>
where
    T: Float,
{
    fn detect_into<D, B>(&mut self, spectrum: &ArrayBase<D, Ix1>, decision: &mut ArrayBase<B, Ix1>)
    where
        D: Data<Elem = Complex<T>>,
        B: DataMut<Elem = bool>,
    {
        azip!((decision in decision, spectrum in spectrum, floor in &self.floor) {
            *decision = spectrum.norm_sqr() > *floor * self.ratio;
        });
    }
}

impl<T> VoicePresenceDetector<T> for PowerThresholdVad<T>
where
    T: Float,
{
    fn detect_into<D, E>(&mut self, spectrum: &ArrayBase<D, Ix1>, decision: &mut ArrayBase<E, Ix1>)
    where
        D: Data<Elem = Complex<T>>,
        E: DataMut<Elem = T>,
    {
        azip!((decision in decision, spectrum in spectrum, floor in &self.floor) {
            *decision = if spectrum.norm_sqr() > *floor * self.ratio {
                T::one()
            } else {
                T::zero()
            }
        });
    }
}

pub fn noise_floor_est<T, D>(spectrum: &ArrayBase<D, Ix2>) -> Array1<T>
where
    T: Float + std::ops::AddAssign,
    D: Data<Elem = Complex<T>>,
{
    let mut out = Array1::zeros(spectrum.shape()[1]);
    noise_floor_est_into(spectrum, &mut out);
    out
}

pub fn noise_floor_est_into<T, D, M>(spectrum: &ArrayBase<D, Ix2>, out: &mut ArrayBase<M, Ix1>)
where
    T: Float + std::ops::AddAssign,
    D: Data<Elem = Complex<T>>,
    M: DataMut<Elem = T>,
{
    let norm = T::from(spectrum.shape()[0]).unwrap();

    out.fill(T::zero());
    for ((_, i), v) in spectrum.indexed_iter() {
        out[i] += norm * v.norm_sqr();
    }
}
