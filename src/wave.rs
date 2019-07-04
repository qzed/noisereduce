use std::io::{Read, Error as IoError, ErrorKind as IoErrorKind};

use hound::{WavReader, WavSamples, WavSpec, Sample as InputSample, Error, SampleFormat};
use ndarray::Array2;
use num::traits::Zero;

use sample::{Sample, FromSample, I24};


pub trait WavReaderExt<R> {
    fn samples_convert_dyn<T>(&mut self) -> ConvertDyn<R, T>
    where
        T: Sample + FromSignedSample + FromSample<f32>;

    fn samples_convert<S, T>(&mut self) -> Convert<R, S, T>
    where
        T: Sample + FromSample<S>,
        S: RawSample,
        S::Raw: InputSample;

    fn collect<S>(self) -> Result<(Array2<S>, WavSpec), Error>
    where
        S: InputSample + Zero + Clone;

    fn collect_convert<S, T>(self) -> Result<(Array2<T>, WavSpec), Error>
    where
        T: Sample + FromSample<S> + Zero,
        S: RawSample,
        S::Raw: InputSample;

    fn collect_convert_dyn<T>(self) -> Result<(Array2<T>, WavSpec), Error>
    where
        T: Sample + FromSignedSample + FromSample<f32> + Zero;
}

impl<R> WavReaderExt<R> for WavReader<R>
where
    R: Read,
{
    fn collect<T: InputSample + Zero + Clone>(self) -> Result<(Array2<T>, WavSpec), Error>
    where
        T: InputSample + Zero + Clone,
    {
        let spec = self.spec();
        let samples = self.duration() as usize;
        let channels = spec.channels as usize;
        let mut iter = self.into_samples::<T>();

        let mut data = Array2::zeros((samples, channels));

        for i in 0..samples {
            for c in 0..channels {
                data[(i, c)] = iter.next().ok_or_else(|| {
                    Error::IoError(IoError::new(IoErrorKind::InvalidData, "not enough data"))
                })??;
            }
        }

        Ok((data, spec))
    }

    fn samples_convert_dyn<T>(&mut self) -> ConvertDyn<R, T>
    where
        T: Sample + FromSignedSample + FromSample<f32>,
    {
        let spec = self.spec();
        let inner = match spec.sample_format {
            hound::SampleFormat::Int => {
                ConvertDynInner::Int(self.samples::<i32>())
            },
            hound::SampleFormat::Float => {
                ConvertDynInner::Float(self.samples::<f32>())
            },
        };

        ConvertDyn {
            inner,
            bits: spec.bits_per_sample,
            _p: std::marker::PhantomData,
        }
    }

    fn samples_convert<S, T>(&mut self) -> Convert<R, S, T>
    where
        T: Sample + FromSample<S>,
        S: RawSample,
        S::Raw: InputSample,
    {
        Convert {
            iter: self.samples(),
            _p: std::marker::PhantomData,
        }
    }

    fn collect_convert<S, T>(mut self) -> Result<(Array2<T>, WavSpec), Error>
    where
        T: Sample + FromSample<S> + Zero,
        S: RawSample,
        S::Raw: InputSample,
    {
        let spec = self.spec();
        let samples = self.duration() as usize;
        let channels = spec.channels as usize;
        let mut iter = self.samples_convert::<S, T>();

        let mut data = Array2::zeros((samples, channels));

        for i in 0..samples {
            for c in 0..channels {
                data[(i, c)] = iter.next().ok_or_else(|| {
                    Error::IoError(IoError::new(IoErrorKind::InvalidData, "not enough data"))
                })??;
            }
        }

        Ok((data, spec))
    }

    fn collect_convert_dyn<T>(self) -> Result<(Array2<T>, WavSpec), Error>
    where
        T: Sample + FromSignedSample + FromSample<f32> + Zero,
    {
        let spec = self.spec();

        match (spec.sample_format, spec.bits_per_sample) {
            (SampleFormat::Float, 32) => self.collect_convert::<f32, T>(),
            (SampleFormat::Int,    8) => self.collect_convert::<i8, T>(),
            (SampleFormat::Int,   16) => self.collect_convert::<i16, T>(),
            (SampleFormat::Int,   24) => self.collect_convert::<I24, T>(),
            (SampleFormat::Int,   32) => self.collect_convert::<i32, T>(),
            _ => unimplemented!(),
        }
    }
}


pub trait RawSample: Sample {
    type Raw: Copy;

    fn from_raw(raw: Self::Raw) -> Self;
    fn to_raw(&self) -> Self::Raw;
}

impl RawSample for i8 {
    type Raw = i8;

    fn from_raw(raw: Self::Raw) -> Self { raw }
    fn to_raw(&self) -> Self::Raw { *self }
}

impl RawSample for i16 {
    type Raw = i16;

    fn from_raw(raw: Self::Raw) -> Self { raw }
    fn to_raw(&self) -> Self::Raw { *self }
}

impl RawSample for I24 {
    type Raw = i32;

    fn from_raw(raw: Self::Raw) -> Self { I24::new_unchecked(raw) }
    fn to_raw(&self) -> Self::Raw { self.inner() }
}

impl RawSample for i32 {
    type Raw = i32;

    fn from_raw(raw: Self::Raw) -> Self { raw }
    fn to_raw(&self) -> Self::Raw { *self }
}

impl RawSample for f32 {
    type Raw = f32;

    fn from_raw(raw: Self::Raw) -> Self { raw }
    fn to_raw(&self) -> Self::Raw { *self }
}


pub trait FromSignedSample: FromSample<i8> + FromSample<i16> + FromSample<I24> + FromSample<i32> {}

impl<T> FromSignedSample for T
where
    T: FromSample<i8> + FromSample<i16> + FromSample<I24> + FromSample<i32>
{}



pub struct Convert<'r, R, S, T>
where
    S: RawSample,
    S::Raw: InputSample,
{
    iter: WavSamples<'r, R, S::Raw>,
    _p: std::marker::PhantomData<*const T>,
}

impl<'wr, R, S, T> Iterator for Convert<'wr, R, S, T>
where
    R: Read,
    S: RawSample,
    S::Raw: InputSample,
    T: Sample + FromSample<S>,
{
    type Item = Result<T, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|s| s.map(|s: S::Raw| T::from_sample(S::from_raw(s))))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    fn count(self) -> usize {
        self.iter.count()
    }

    fn last(self) -> Option<Self::Item> {
        self.iter.last().map(|s| s.map(|s: S::Raw| T::from_sample(S::from_raw(s))))
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.iter.nth(n).map(|s| s.map(|s: S::Raw| T::from_sample(S::from_raw(s))))
    }
}

impl<'wr, R, S, T> ExactSizeIterator for Convert<'wr, R, S, T>
where
    R: Read,
    S: RawSample,
    S::Raw: InputSample,
    T: Sample + FromSample<S>,
{
    fn len(&self) -> usize {
        self.iter.len()
    }
}


pub struct ConvertDyn<'wr, R, T> {
    inner: ConvertDynInner<'wr, R>,
    bits: u16,
    _p: std::marker::PhantomData<*const T>,
}

enum ConvertDynInner<'wr, R> {
    Int(WavSamples<'wr, R, i32>),
    Float(WavSamples<'wr, R, f32>),
}

impl<'wr, R, T> Iterator for ConvertDyn<'wr, R, T>
where
    R: Read,
    T: Sample + FromSignedSample + FromSample<f32>,
{
    type Item = Result<T, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            ConvertDynInner::Int(iter)   => iter.next().map(|s| s.map(|s| convert_int(self.bits, s))),
            ConvertDynInner::Float(iter) => iter.next().map(|s| s.map(|s| convert_float(self.bits, s))),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.inner {
            ConvertDynInner::Int(iter)   => iter.size_hint(),
            ConvertDynInner::Float(iter) => iter.size_hint(),
        }
    }

    fn count(self) -> usize {
        match self.inner {
            ConvertDynInner::Int(iter)   => iter.count(),
            ConvertDynInner::Float(iter) => iter.count(),
        }
    }

    fn last(self) -> Option<Self::Item> {
        let bits = self.bits;

        match self.inner {
            ConvertDynInner::Int(iter)   => iter.last().map(|s| s.map(|s| convert_int(bits, s))),
            ConvertDynInner::Float(iter) => iter.last().map(|s| s.map(|s| convert_float(bits, s))),
        }
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        match &mut self.inner {
            ConvertDynInner::Int(iter)   => iter.nth(n).map(|s| s.map(|s| convert_int(self.bits, s))),
            ConvertDynInner::Float(iter) => iter.nth(n).map(|s| s.map(|s| convert_float(self.bits, s))),
        }
    }
}

impl<'wr, R, T> ExactSizeIterator for ConvertDyn<'wr, R, T>
where
    R: Read,
    T: Sample + FromSignedSample + FromSample<f32>,
{
    fn len(&self) -> usize {
        match &self.inner {
            ConvertDynInner::Int(iter)   => iter.len(),
            ConvertDynInner::Float(iter) => iter.len(),
        }
    }
}

fn convert_int<T: Sample + FromSignedSample>(bits: u16, sample: i32) -> T {
    match bits {
        8  => T::from_sample(sample as i8),
        16 => T::from_sample(sample as i16),
        24 => T::from_sample(I24::new_unchecked(sample)),
        32 => T::from_sample(sample as i32),
        _  => unimplemented!(),
    }
}

fn convert_float<T: Sample + FromSample<f32>>(bits: u16, sample: f32) -> T {
    match bits {
        32 => T::from_sample(sample),
        _  => unimplemented!(),
    }
}
