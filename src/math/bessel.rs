use num::Float;


#[allow(non_snake_case)]
pub fn I0<T: Float>(x: T) -> T {
    T::from(rgsl::bessel::I0(x.to_f64().unwrap())).unwrap()
}

#[allow(non_snake_case)]
pub fn I1<T: Float>(x: T) -> T {
    T::from(rgsl::bessel::I1(x.to_f64().unwrap())).unwrap()
}
