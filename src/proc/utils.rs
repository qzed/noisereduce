use super::Processor;

use ndarray::{ArrayBase, Axis, Data, DataMut, Ix2};
use num::Complex;


pub fn process<T, P, D1, D2>(processor: P, input: &ArrayBase<D1, Ix2>, output: &mut ArrayBase<D2, Ix2>)
where
    P: Processor<T>,
    D1: Data<Elem = Complex<T>>,
    D2: DataMut<Elem = Complex<T>>,
{
    let mut processor = processor;

    let n = input.shape()[0];
    for i in 0..n {
        let y_in = input.index_axis(Axis(0), i);
        let y_out = output.index_axis_mut(Axis(0), i);

        processor.process(y_in, y_out);
    }
}
