use ndarray::{ArrayBase, Ix1, Data};
use num::{Complex, Float};

pub mod energy;


/// Voice activity detector with boolean output.
pub trait VoiceActivityDetector<T> {
    fn detect<D>(&self, spectrum: &ArrayBase<D, Ix1>) -> bool
    where
        D: Data<Elem=Complex<T>>;
}

pub trait VoicePresenceDetector<T> {
    fn detect<D>(&self, spectrum: &ArrayBase<D, Ix1>) -> T
    where
        D: Data<Elem=Complex<T>>;

    fn binary(self, threshold: T) -> Binary<Self, T>
    where
        Self: Sized
    {
        Binary { base: self, threshold }
    }
}


/// Voice activity detector with probabilisitc output.
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
