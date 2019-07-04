//! Full-Spectrum Voice activity detection.

pub mod energy;

use crate::vad;

use ndarray::{ArrayBase, Data, DataMut, Ix1};
use num::{Complex, Float};


/// Voice activity detector with boolean output.
pub trait VoiceActivityDetector<T> {
    /// Returns `true` if voice activity has been detected in the given
    /// spectrum, `false` otherwise.
    fn detect<D>(&self, spectrum: &ArrayBase<D, Ix1>) -> bool
    where
        D: Data<Elem = Complex<T>>;

    fn per_band(self) -> PerBand<Self>
    where
        Self: Sized,
    {
        PerBand { base: self }
    }
}

/// Voice activity detector with probabilisitc output.
pub trait VoicePresenceDetector<T>
where
    T: Float,
{
    /// Returns a probability-like measure describing the presence (`1.0`) or
    /// absence (`0.0`) of voice in the given spectrum.
    fn detect<D>(&self, spectrum: &ArrayBase<D, Ix1>) -> T
    where
        D: Data<Elem = Complex<T>>;

    /// Transforms this detector into a binary/boolean detector where any voice
    /// presence probability returned by the original detector that is above the
    /// given threshold is interpreted as present (`true`).
    fn binary(self, threshold: T) -> Binary<Self, T>
    where
        Self: Sized,
    {
        Binary { base: self, threshold }
    }

    fn per_band(self) -> PerBand<Self>
    where
        Self: Sized,
    {
        PerBand { base: self }
    }
}


impl<V, T> VoicePresenceDetector<T> for V
where
    V: VoiceActivityDetector<T>,
    T: Float,
{
    fn detect<D>(&self, spectrum: &ArrayBase<D, Ix1>) -> T
    where
        D: Data<Elem = Complex<T>>,
    {
        if VoiceActivityDetector::detect(self, spectrum) {
            T::one()
        } else {
            T::zero()
        }
    }
}


/// The voice activity detector returned by the
/// [`binary`](VoicePresenceDetector::binary) method of a
/// [`VoicePresenceDetector`](VoicePresenceDetector).
pub struct Binary<B, T> {
    base: B,
    threshold: T,
}

impl<B, T> VoiceActivityDetector<T> for Binary<B, T>
where
    B: VoicePresenceDetector<T>,
    T: Float,
{
    fn detect<D>(&self, spectrum: &ArrayBase<D, Ix1>) -> bool
    where
        D: Data<Elem = Complex<T>>,
    {
        self.base.detect(spectrum) > self.threshold
    }
}


pub struct PerBand<B> {
    base: B,
}

impl<B, T> vad::b::VoiceActivityDetector<T> for PerBand<B>
where
    B: VoiceActivityDetector<T>,
    T: Float,
{
    fn detect_into<D, E>(&self, spectrum: &ArrayBase<D, Ix1>, decision: &mut ArrayBase<E, Ix1>)
    where
        D: Data<Elem = Complex<T>>,
        E: DataMut<Elem = bool>,
    {
        decision.fill(self.base.detect(spectrum))
    }
}

impl<B, T> vad::b::VoicePresenceDetector<T> for PerBand<B>
where
    B: VoicePresenceDetector<T>,
    T: Float,
{
    fn detect_into<D, E>(&self, spectrum: &ArrayBase<D, Ix1>, decision: &mut ArrayBase<E, Ix1>)
    where
        D: Data<Elem = Complex<T>>,
        E: DataMut<Elem = T>,
    {
        decision.fill(self.base.detect(spectrum))
    }
}
