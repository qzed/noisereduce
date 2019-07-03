use super::VoiceActivityDetector;

use ndarray::{ArrayBase, Ix1, Data};
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
