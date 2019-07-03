use super::VoiceActivityDetector;

use ndarray::{ArrayBase, Axis, Data, Ix1, Ix2};
use num::{Complex, Float};


pub struct EnergyThresholdVad<T> {
    floor: T,
    ratio: T,
}

impl<T> EnergyThresholdVad<T> {
    pub fn new(floor: T, ratio: T) -> Self {
        EnergyThresholdVad { floor, ratio }
    }
}

impl<T> VoiceActivityDetector<T> for EnergyThresholdVad<T>
where
    T: Float,
{
    fn detect<D>(&self, spectrum: &ArrayBase<D, Ix1>) -> bool
    where
        D: Data<Elem=Complex<T>>
    {
        let energy = spectrum.fold(T::zero(), |e, v| e + v.norm_sqr());
        let energy = energy / T::from(spectrum.len()).unwrap();

        energy > self.floor * self.ratio
    }
}


pub fn noise_floor<T, D>(spectrum: &ArrayBase<D, Ix2>) -> T
where
    T: Float,
    D: Data<Elem=Complex<T>>,
{
    let norm = T::one() / T::from(spectrum.shape()[1]).unwrap() / T::from(spectrum.shape()[0]).unwrap();
    spectrum.fold(T::zero(), |e, v| e + v.norm_sqr() * norm)
}
