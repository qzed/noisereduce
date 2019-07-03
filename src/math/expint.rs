use num::Float;


#[allow(non_snake_case)]
pub fn Ei<T: Float>(x: T) -> T {
    T::from(rgsl::exponential_integrals::Ei(x.to_f64().unwrap())).unwrap()
}
