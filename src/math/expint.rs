use super::NumCastUnchecked;

use num::Float;


#[allow(non_snake_case)]
pub fn Ei<T: Float + NumCastUnchecked>(x: T) -> T {
    T::from_unchecked(rgsl::exponential_integrals::Ei(x.to_f64_unchecked()))
}
