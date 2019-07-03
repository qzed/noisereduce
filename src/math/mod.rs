pub mod bessel;
pub mod expint;


pub trait ToPrimitiveUnchecked {
    fn to_i64_unchecked(&self) -> i64;
    fn to_u64_unchecked(&self) -> u64;
    fn to_isize_unchecked(&self) -> isize;
    fn to_i8_unchecked(&self) -> i8;
    fn to_i16_unchecked(&self) -> i16;
    fn to_i32_unchecked(&self) -> i32;
    fn to_usize_unchecked(&self) -> usize;
    fn to_u8_unchecked(&self) -> u8;
    fn to_u16_unchecked(&self) -> u16;
    fn to_u32_unchecked(&self) -> u32;
    fn to_f32_unchecked(&self) -> f32;
    fn to_f64_unchecked(&self) -> f64;
}

pub trait NumCastUnchecked: Sized + ToPrimitiveUnchecked {
    fn from_unchecked<T: ToPrimitiveUnchecked>(n: T) -> Self;
}

impl ToPrimitiveUnchecked for f32 {
    fn to_i64_unchecked(&self) -> i64 { *self as _ }
    fn to_u64_unchecked(&self) -> u64 { *self as _ }
    fn to_isize_unchecked(&self) -> isize { *self as _ }
    fn to_i8_unchecked(&self) -> i8 { *self as _ }
    fn to_i16_unchecked(&self) -> i16 { *self as _ }
    fn to_i32_unchecked(&self) -> i32 { *self as _ }
    fn to_usize_unchecked(&self) -> usize { *self as _ }
    fn to_u8_unchecked(&self) -> u8 { *self as _ }
    fn to_u16_unchecked(&self) -> u16 { *self as _ }
    fn to_u32_unchecked(&self) -> u32 { *self as _ }
    fn to_f32_unchecked(&self) -> f32 { *self as _ }
    fn to_f64_unchecked(&self) -> f64 { *self as _ }
}

impl NumCastUnchecked for f32 {
    fn from_unchecked<T: ToPrimitiveUnchecked>(n: T) -> Self {
        n.to_f32_unchecked()
    }
}

impl ToPrimitiveUnchecked for f64 {
    fn to_i64_unchecked(&self) -> i64 { *self as _ }
    fn to_u64_unchecked(&self) -> u64 { *self as _ }
    fn to_isize_unchecked(&self) -> isize { *self as _ }
    fn to_i8_unchecked(&self) -> i8 { *self as _ }
    fn to_i16_unchecked(&self) -> i16 { *self as _ }
    fn to_i32_unchecked(&self) -> i32 { *self as _ }
    fn to_usize_unchecked(&self) -> usize { *self as _ }
    fn to_u8_unchecked(&self) -> u8 { *self as _ }
    fn to_u16_unchecked(&self) -> u16 { *self as _ }
    fn to_u32_unchecked(&self) -> u32 { *self as _ }
    fn to_f32_unchecked(&self) -> f32 { *self as _ }
    fn to_f64_unchecked(&self) -> f64 { *self as _ }
}

impl NumCastUnchecked for f64 {
    fn from_unchecked<T: ToPrimitiveUnchecked>(n: T) -> Self {
        n.to_f64_unchecked()
    }
}
