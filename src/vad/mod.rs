//! Voice activity detection.

use ndarray::{ArrayBase, Ix1, Data};
use num::{Complex, Float};

pub mod energy;


/// Voice activity detector with boolean output.
pub trait VoiceActivityDetector<T> {
    /// Returns `true` if voice activity has been detected in the given
    /// spectrum, `false` otherwise.
    fn detect<D>(&self, spectrum: &ArrayBase<D, Ix1>) -> bool
    where
        D: Data<Elem=Complex<T>>;
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
        D: Data<Elem=Complex<T>>;

    /// Transforms this detector into a binary/boolean detector where any voice
    /// presence probability returned by the original detector that is above the
    /// given threshold is interpreted as present (`true`).
    fn binary(self, threshold: T) -> Binary<Self, T>
    where
        Self: Sized
    {
        Binary { base: self, threshold }
    }
}


impl<V, T> VoicePresenceDetector<T> for V
where
    V: VoiceActivityDetector<T>,
    T: Float,
{
    fn detect<D>(&self, spectrum: &ArrayBase<D, Ix1>) -> T
    where
        D: Data<Elem=Complex<T>>
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
        D: Data<Elem=Complex<T>>,
    {
        self.base.detect(spectrum) > self.threshold
    }
}
