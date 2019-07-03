//! Energy based voice activity detectors.

use super::VoiceActivityDetector;

use ndarray::{ArrayBase, Data, Ix1, Ix2};
use num::{Complex, Float};


/// Voice activity detector based on average spectrum energy.
/// 
/// Uses a noise-floor estimate (average energy) and a ratio to discriminate
/// noise from a signal. If the average energy `e` of the given spectrum is
/// larger than the threshold computed via `floor * ratio` the spectrum is
/// classified as speech, otherwise it is classified as noise, i.e. the value
/// returned by this detector is `e > floor * ratio`.
pub struct EnergyThresholdVad<T> {
    floor: T,
    ratio: T,
}

impl<T> EnergyThresholdVad<T> {
    /// Creates a new average-energy threshold voice activity detector.
    /// 
    /// - `floor` represents the noise-floor of the signal as average energy over
    /// the spectrum.
    /// - `ratio` is a ratio describing how much the average signal
    /// energy may exceed the noise-floor before it is classified as speech.
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


/// Compute an average-energy noise-floor estimation based on the given spectrum
/// segments.
/// 
/// The spectrum segments are given as 2D array or slice with dimension `[n, m]`
/// where `n` is the number of spectras to average over and `m` is the length of
/// a single spectrum.
pub fn noise_floor_est<T, D>(spectrum: &ArrayBase<D, Ix2>) -> T
where
    T: Float,
    D: Data<Elem=Complex<T>>,
{
    let norm = T::one() / T::from(spectrum.shape()[1]).unwrap() / T::from(spectrum.shape()[0]).unwrap();
    spectrum.fold(T::zero(), |e, v| e + v.norm_sqr() * norm)
}
