pub mod utils;


use ndarray::{ArrayView1, ArrayViewMut1};
use num::Complex;


pub trait Processor<T> {
    fn block_size(&self) -> usize;
    fn process(&mut self, spectrum_in: ArrayView1<Complex<T>>, spectrum_out: ArrayViewMut1<Complex<T>>);
}

impl<T, P> Processor<T> for Box<P>
where
    P: Processor<T>,
{
    fn block_size(&self) -> usize {
        self.as_ref().block_size()
    }

    fn process(&mut self, spectrum_in: ArrayView1<Complex<T>>, spectrum_out: ArrayViewMut1<Complex<T>>) {
        self.as_mut().process(spectrum_in, spectrum_out)
    }
}

impl<T> Processor<T> for Box<dyn Processor<T>> {
    fn block_size(&self) -> usize {
        self.as_ref().block_size()
    }

    fn process(&mut self, spectrum_in: ArrayView1<Complex<T>>, spectrum_out: ArrayViewMut1<Complex<T>>) {
        self.as_mut().process(spectrum_in, spectrum_out)
    }
}

impl<T, P> Processor<T> for &mut P
where
    P: Processor<T>,
{
    fn block_size(&self) -> usize {
        (**self).block_size()
    }

    fn process(&mut self, spectrum_in: ArrayView1<Complex<T>>, spectrum_out: ArrayViewMut1<Complex<T>>) {
        (**self).process(spectrum_in, spectrum_out)
    }
}

impl<T> Processor<T> for &mut dyn Processor<T> {
    fn block_size(&self) -> usize {
        (**self).block_size()
    }

    fn process(&mut self, spectrum_in: ArrayView1<Complex<T>>, spectrum_out: ArrayViewMut1<Complex<T>>) {
        (**self).process(spectrum_in, spectrum_out)
    }
}
