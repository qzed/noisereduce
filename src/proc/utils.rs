use super::Processor;

use ndarray::{Array2, ArrayBase, Axis, Data, DataMut, Ix2};
use num::{Complex, Num};


pub fn process_spectrum_into<T, P, D1, D2>(processor: &mut P, input: &ArrayBase<D1, Ix2>, output: &mut ArrayBase<D2, Ix2>)
where
    P: Processor<T> + ?Sized,
    D1: Data<Elem = Complex<T>>,
    D2: DataMut<Elem = Complex<T>>,
{
    let n = input.shape()[0];
    for i in 0..n {
        let y_in = input.index_axis(Axis(0), i);
        let y_out = output.index_axis_mut(Axis(0), i);

        processor.process(y_in, y_out);
    }
}

pub fn process_spectrum<T, P, D>(processor: &mut P, input: &ArrayBase<D, Ix2>) -> Array2<Complex<T>>
where
    T: Num + Clone,
    P: Processor<T> + ?Sized,
    D: Data<Elem = Complex<T>>,
{
    let mut output = Array2::zeros(input.raw_dim());
    process_spectrum_into(processor, input, &mut output);
    output
}
