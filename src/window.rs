use num::{Float, traits::FloatConst};


pub trait WindowFunction<T> {
    fn len(&self) -> usize;
    fn coef(&self, index: usize) -> T;

    fn iter(&self) -> WindowFunctionIter<Self, T>
        where Self: Sized
    {
        WindowFunctionIter {
            function: self,
            _phantom: std::marker::PhantomData,
            start: 0,
            end: self.len(),
        }
    }
}

pub struct WindowFunctionIter<'a, T, F> {
    function: &'a T,
    _phantom: std::marker::PhantomData<*const F>,
    start: usize,
    end: usize,
}

impl<'a, T, F> Iterator for WindowFunctionIter<'a, T, F>
where
    T: WindowFunction<F>,
{
    type Item = F;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start < self.end {
            let value = self.function.coef(self.start);
            self.start += 1;
            Some(value)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.end - self.start;
        (len, Some(len))
    }

    fn count(self) -> usize {
        self.end - self.start
    }

    fn last(self) -> Option<Self::Item> {
        if self.start < self.end {
            Some(self.function.coef(self.end - 1))
        } else {
            None
        }
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if self.start + n < self.end {
            let val = self.function.coef(self.start + n);
            self.start = self.start + n + 1;
            Some(val)
        } else {
            None
        }
    }
}

impl<'a, T, F> DoubleEndedIterator for WindowFunctionIter<'a, T, F>
where
    T: WindowFunction<F>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start < self.end {
            self.end = self.end - 1;
            Some(self.function.coef(self.end))
        } else {
            None
        }
    }
}

impl<'a, T, F> ExactSizeIterator for WindowFunctionIter<'a, T, F>
where
    T: WindowFunction<F>,
{
    fn len(&self) -> usize {
        self.end - self.start
    }
}


pub struct Rectangular {
    len: usize,
}

impl Rectangular {
    pub fn new(len: usize) -> Self {
        Rectangular { len }
    }
}

impl<T: Float> WindowFunction<T> for Rectangular {
    fn len(&self) -> usize {
        self.len
    }

    fn coef(&self, _index: usize) -> T {
        T::one()
    }
}


pub struct Triangular {
    len: usize,
    l: usize,
}

impl Triangular {
    pub fn new(len: usize, l: usize) -> Self {
        Triangular { len, l }
    }
}

impl<T: Float> WindowFunction<T> for Triangular {
    fn len(&self) -> usize {
        self.len
    }

    fn coef(&self, index: usize) -> T {
        let i = T::from(index).unwrap();
        let n = T::from(self.len - 1).unwrap();
        let l = T::from(self.l).unwrap();
        let two = T::from(2.0).unwrap();

        T::one() - T::abs((i - (n / two)) / (l / two))
    }
}


pub struct GenericHann<T> {
    len: usize,
    a0: T,
    a1: T,
}

impl<T> GenericHann<T> {
    pub fn new(len: usize, a0: T, a1: T) -> Self {
        GenericHann { len, a0, a1 }
    }
}

impl<T: Float + FloatConst> WindowFunction<T> for GenericHann<T> {
    fn len(&self) -> usize {
        self.len
    }

    fn coef(&self, index: usize) -> T {
        let two_pi = T::from(2.0).unwrap() * T::PI();
        let n = T::from(self.len - 1).unwrap();
        let i = T::from(index).unwrap();

        self.a0 - self.a1 * T::cos(two_pi * i / n)
    }
}


pub struct Blackman<T> {
    len: usize,
    a0: T,
    a1: T,
    a2: T,
}

impl<T> Blackman<T> {
    pub fn new(len: usize, a0: T, a1: T, a2: T) -> Self {
        Blackman { len, a0, a1, a2 }
    }
}

impl<T: Float + FloatConst> WindowFunction<T> for Blackman<T> {
    fn len(&self) -> usize {
        self.len
    }

    fn coef(&self, index: usize) -> T {
        let two_pi = T::from(2.0).unwrap() * T::PI();
        let four_pi = T::from(4.0).unwrap() * T::PI();
        let n = T::from(self.len - 1).unwrap();
        let i = T::from(index).unwrap();

        self.a0 - self.a1 * T::cos(two_pi * i / n) + self.a2 * T::cos(four_pi * i / n)
    }
}


pub struct Nuttall<T> {
    len: usize,
    a0: T,
    a1: T,
    a2: T,
    a3: T,
}

impl<T> Nuttall<T> {
    pub fn new(len: usize, a0: T, a1: T, a2: T, a3: T) -> Self {
        Nuttall { len, a0, a1, a2, a3 }
    }
}

impl<T: Float + FloatConst> WindowFunction<T> for Nuttall<T> {
    fn len(&self) -> usize {
        self.len
    }

    fn coef(&self, index: usize) -> T {
        let two_pi = T::from(2.0).unwrap() * T::PI();
        let four_pi = T::from(4.0).unwrap() * T::PI();
        let six_pi = T::from(6.0).unwrap() * T::PI();

        let n = T::from(self.len - 1).unwrap();
        let i = T::from(index).unwrap();

        self.a0
            - self.a1 * T::cos(two_pi * i / n)
            + self.a2 * T::cos(four_pi * i / n)
            - self.a3 * T::cos(six_pi * i / n)
    }
}


pub struct FlatTop<T> {
    len: usize,
    a0: T,
    a1: T,
    a2: T,
    a3: T,
    a4: T,
}

impl<T> FlatTop<T> {
    pub fn new(len: usize, a0: T, a1: T, a2: T, a3: T, a4: T) -> Self {
        FlatTop { len, a0, a1, a2, a3, a4 }
    }
}

impl<T: Float + FloatConst> WindowFunction<T> for FlatTop<T> {
    fn len(&self) -> usize {
        self.len
    }

    fn coef(&self, index: usize) -> T {
        let two_pi = T::from(2.0).unwrap() * T::PI();
        let four_pi = T::from(4.0).unwrap() * T::PI();
        let six_pi = T::from(6.0).unwrap() * T::PI();
        let eight_pi = T::from(8.0).unwrap() * T::PI();

        let n = T::from(self.len - 1).unwrap();
        let i = T::from(index).unwrap();

        self.a0
            - self.a1 * T::cos(two_pi * i / n)
            + self.a2 * T::cos(four_pi * i / n)
            - self.a3 * T::cos(six_pi * i / n)
            + self.a4 * T::cos(eight_pi * i / n)
    }
}


pub struct Gaussian<T> {
    len: usize,
    sigma: T,
}

impl<T> Gaussian<T> {
    pub fn new(len: usize, sigma: T) -> Self {
        Gaussian { len, sigma }
    }
}

impl<T: Float> WindowFunction<T> for Gaussian<T> {
    fn len(&self) -> usize {
        self.len
    }

    fn coef(&self, index: usize) -> T {
        let two = T::from(2.0).unwrap();
        let n_two = T::from(self.len - 1).unwrap() / two;
        let i = T::from(index).unwrap();

        let a = (i - n_two) / (self.sigma * n_two);
        T::exp(-(a*a) / two)
    }
}


pub struct GaussianConfined<T> {
    len: usize,
    sigma_t: T,
}

impl<T> GaussianConfined<T> {
    pub fn new(len: usize, sigma_t: T) -> Self {
        GaussianConfined { len, sigma_t }
    }
}

impl<T: Float> WindowFunction<T> for GaussianConfined<T> {
    fn len(&self) -> usize {
        self.len
    }

    fn coef(&self, index: usize) -> T {
        fn g<T: Float>(x: T, n: T, sigma_t: T) -> T {
            let two = T::from(2.0).unwrap();
            let a = (x - n / two) / (two * (n + T::one()) * sigma_t);
            T::exp(-a * a)
        }

        let a = T::from(0.5).unwrap();
        let b = T::from(3.0).unwrap() / T::from(2.0).unwrap();

        let i = T::from(index).unwrap();
        let n = T::from(self.len - 1).unwrap();

        let f0 = g(i,                n, self.sigma_t);
        let f1 = g(-a,               n, self.sigma_t);
        let f2 = g(i + n + T::one(), n, self.sigma_t);
        let f3 = g(i - n - T::one(), n, self.sigma_t);
        let f4 = g(a + n,            n, self.sigma_t);
        let f5 = g(-b + n,           n, self.sigma_t);

        f0 - (f1 * (f2 + f3)) / (f4 + f5)
    }
}


pub struct GeneralizedNormal<T> {
    len: usize,
    sigma: T,
    p: T,
}

impl<T> GeneralizedNormal<T> {
    pub fn new(len: usize, sigma: T, p: T) -> Self {
        GeneralizedNormal { len, sigma, p }
    }
}

impl<T: Float> WindowFunction<T> for GeneralizedNormal<T> {
    fn len(&self) -> usize {
        self.len
    }

    fn coef(&self, index: usize) -> T {
        let i = T::from(index).unwrap();
        let n_two = T::from(self.len - 1).unwrap() / T::from(2.0).unwrap();
        T::exp(- T::powf((i - n_two) / (self.sigma * n_two), self.p))
    }
}


pub struct Tukey<T> {
    len: usize,
    alpha: T,
}

impl<T> Tukey<T> {
    pub fn new(len: usize, alpha: T) -> Self {
        Tukey { len, alpha }
    }
}

impl<T: Float + FloatConst> WindowFunction<T> for Tukey<T> {
    fn len(&self) -> usize {
        self.len
    }

    fn coef(&self, index: usize) -> T {
        let two = T::from(2.0).unwrap();
        let n = T::from(self.len - 1).unwrap();
        let i = T::from(index).unwrap();

        if i < (self.alpha * n) / two {
            (
                T::one() + T::cos(
                    T::PI() * ((two * i) / (self.alpha * n) - T::one())
                )
            ) / two
        } else if i <= n * (T::one() - self.alpha / two) {
            T::one()
        } else {
            (
                T::one() + T::cos(
                    T::PI() * (
                        (two * i) / (self.alpha * n) - (two / self.alpha) + T::one()
                    )
                )
            ) / two
        }
    }
}


pub fn rectangular(len: usize) -> Rectangular {
    Rectangular::new(len)
}

pub fn triangular(len: usize, l: usize) -> Triangular {
    Triangular::new(len, l)
}

pub fn bartlett(len: usize) -> Triangular {
    Triangular::new(len, len - 1)
}

pub fn hann<T: Float>(len: usize) -> GenericHann<T> {
    let a0 = T::from(0.5).unwrap();
    let a1 = T::one() - a0;

    GenericHann::new(len, a0, a1)
}

pub fn hamming<T: Float>(len: usize) -> GenericHann<T> {
    let a0 = T::from(25.0 / 46.0).unwrap();
    let a1 = T::one() - a0;

    GenericHann::new(len, a0, a1)
}

pub fn blackman<T: Float>(len: usize) -> Blackman<T> {
    let a0 = T::from(0.42).unwrap();
    let a1 = T::from(0.50).unwrap();
    let a2 = T::from(0.08).unwrap();

    Blackman::new(len, a0, a1, a2)
}

pub fn blackman_exact<T: Float>(len: usize) -> Blackman<T> {
    let a0 = T::from(7938.0).unwrap() / T::from(18608.0).unwrap();
    let a1 = T::from(9240.0).unwrap() / T::from(18608.0).unwrap();
    let a2 = T::from(1430.0).unwrap() / T::from(18608.0).unwrap();

    Blackman::new(len, a0, a1, a2)
}

pub fn blackman_alpha<T: Float>(len: usize, alpha: T) -> Blackman<T> {
    let a0 = T::from(0.5).unwrap() * (T::one() - alpha);
    let a1 = T::from(0.5).unwrap();
    let a2 = T::from(0.5).unwrap() * alpha;

    Blackman::new(len, a0, a1, a2)
}

pub fn nuttall<T: Float>(len: usize) -> Nuttall<T> {
    let a0 = T::from(0.355768).unwrap();
    let a1 = T::from(0.487396).unwrap();
    let a2 = T::from(0.144232).unwrap();
    let a3 = T::from(0.012604).unwrap();

    Nuttall::new(len, a0, a1, a2, a3)
}

pub fn blackman_nuttall<T: Float>(len: usize) -> Nuttall<T> {
    let a0 = T::from(0.3635819).unwrap();
    let a1 = T::from(0.4891775).unwrap();
    let a2 = T::from(0.1365995).unwrap();
    let a3 = T::from(0.0106411).unwrap();

    Nuttall::new(len, a0, a1, a2, a3)
}

pub fn blackman_harris<T: Float>(len: usize) -> Nuttall<T> {
    let a0 = T::from(0.35875).unwrap();
    let a1 = T::from(0.48829).unwrap();
    let a2 = T::from(0.14128).unwrap();
    let a3 = T::from(0.01168).unwrap();

    Nuttall::new(len, a0, a1, a2, a3)
}

pub fn flat_top<T: Float>(len: usize) -> FlatTop<T> {
    let a0 = T::from(0.21557895).unwrap();
    let a1 = T::from(0.41663158).unwrap();
    let a2 = T::from(0.277263158).unwrap();
    let a3 = T::from(0.083578947).unwrap();
    let a4 = T::from(0.006947368).unwrap();

    FlatTop::new(len, a0, a1, a2, a3, a4)
}

pub fn gaussian<T: Float>(len: usize, sigma: T) -> Gaussian<T> {
    Gaussian::new(len, sigma)
}

pub fn gaussian_confined<T: Float>(len: usize, sigma_t: T) -> GaussianConfined<T> {
    GaussianConfined::new(len, sigma_t)
}

pub fn generalized_normal<T: Float>(len: usize, sigma: T, p: T) -> GeneralizedNormal<T> {
    GeneralizedNormal::new(len, sigma, p)
}

pub fn tukey<T: Float>(len: usize, alpha: T) -> Tukey<T> {
    Tukey::new(len, alpha)
}
