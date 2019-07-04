//! Per Frequency-Band Voice Activity Detection.

use ndarray::{ArrayBase, Array1, Data, DataMut, Ix1};
use num::Complex;

pub mod power;


pub trait VoiceActivityDetector<T> {
    fn detect_into<D, B>(&self, spectrum: &ArrayBase<D, Ix1>, decision: &mut ArrayBase<B, Ix1>)
    where
        D: Data<Elem=Complex<T>>,
        B: DataMut<Elem=bool>;

    fn detect<D>(&self, spectrum: &ArrayBase<D, Ix1>) -> Array1<bool>
    where
        D: Data<Elem=Complex<T>>,
    {
        let mut out = Array1::from_elem(spectrum.len(), false);
        self.detect_into(spectrum, &mut out);
        out
    }
}

pub trait VoicePresenceDetector<T> {
    fn detect_into<D, E>(&self, spectrum: &ArrayBase<D, Ix1>, decision: &mut ArrayBase<E, Ix1>)
    where
        D: Data<Elem=Complex<T>>,
        E: DataMut<Elem=T>;

    fn detect<D>(&self, spectrum: &ArrayBase<D, Ix1>) -> Array1<T>
    where
        D: Data<Elem=Complex<T>>,
        T: Clone + num::Zero,
    {
        let mut out = Array1::zeros(spectrum.len());
        self.detect_into(spectrum, &mut out);
        out
    }
}
