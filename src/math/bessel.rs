use super::NumCastUnchecked;

use num::Float;


#[allow(non_snake_case)]
pub fn I0<T: Float + NumCastUnchecked>(x: T) -> T {
    T::from_unchecked(rgsl::bessel::I0(x.to_f64_unchecked()))
}

#[allow(non_snake_case)]
pub fn I1<T: Float + NumCastUnchecked>(x: T) -> T {
    T::from_unchecked(rgsl::bessel::I1(x.to_f64_unchecked()))
}
