use std::io::{Read, Error as IoError, ErrorKind as IoErrorKind};

use hound::{WavReader, WavSamples, WavSpec, Sample, Error};
use ndarray::Array2;
use num::traits::Zero;


pub trait WavReaderExt<R> {
    fn samples_f32<'wr>(&'wr mut self) -> WavSamplesF32<'wr, R>;
    fn into_array_f32(self) -> Result<(Array2<f32>, WavSpec), Error>;
    fn into_array<T: Sample + Zero + Clone>(self) -> Result<(Array2<T>, WavSpec), Error>;
}

impl<R> WavReaderExt<R> for WavReader<R>
where
    R: Read,
{
    fn samples_f32<'wr>(&'wr mut self) -> WavSamplesF32<'wr, R> {
        let spec = self.spec();
        match spec.sample_format {
            hound::SampleFormat::Int => {
                WavSamplesF32::I32(self.samples::<i32>(), ((spec.bits_per_sample - 1) as f32).exp2())
            },
            hound::SampleFormat::Float => {
                WavSamplesF32::F32(self.samples::<f32>())
            },
        }
    }

    fn into_array_f32(mut self) -> Result<(Array2<f32>, WavSpec), Error> {
        let spec = self.spec();
        let samples = self.duration() as usize;
        let channels = spec.channels as usize;
        let iter = self.samples_f32();

        Ok((wave_to_array(samples, channels, iter)?, spec))
    }

    fn into_array<T: Sample + Zero + Clone>(self) -> Result<(Array2<T>, WavSpec), Error> {
        let spec = self.spec();
        let samples = self.duration() as usize;
        let channels = spec.channels as usize;
        let iter = self.into_samples::<T>();

        Ok((wave_to_array(samples, channels, iter)?, spec))
    }
}


pub enum WavSamplesF32<'wr, R> {
    I32(WavSamples<'wr, R, i32>, f32),
    F32(WavSamples<'wr, R, f32>),
}

impl<'wr, R> Iterator for WavSamplesF32<'wr, R>
where
    R: Read,
{
    type Item = Result<f32, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            WavSamplesF32::I32(iter, norm) => iter.next().map(|s| s.map(|s| s as f32 / *norm)),
            WavSamplesF32::F32(iter)       => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            WavSamplesF32::I32(iter, _norm) => iter.size_hint(),
            WavSamplesF32::F32(iter)        => iter.size_hint(),
        }
    }

    fn count(self) -> usize {
        match self {
            WavSamplesF32::I32(iter, _norm) => iter.count(),
            WavSamplesF32::F32(iter)        => iter.count(),
        }
    }

    fn last(self) -> Option<Self::Item> {
        match self {
            WavSamplesF32::I32(iter, norm) => iter.last().map(|s| s.map(|s| s as f32 / norm)),
            WavSamplesF32::F32(iter)       => iter.last(),
        }
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        match self {
            WavSamplesF32::I32(iter, norm) => iter.nth(n).map(|s| s.map(|s| s as f32 / *norm)),
            WavSamplesF32::F32(iter)       => iter.nth(n),
        }
    }
}

impl<'wr, R> ExactSizeIterator for WavSamplesF32<'wr, R>
where
    R: Read,
{
    fn len(&self) -> usize {
        match self {
            WavSamplesF32::I32(iter, _norm) => iter.len(),
            WavSamplesF32::F32(iter)        => iter.len(),
        }
    }
}


fn wave_to_array<I, T>(samples: usize, channels: usize, iter: I) -> Result<Array2<T>, Error>
where
    I: Iterator<Item=Result<T, Error>>,
    T: Clone + Zero,
{
    let mut iter = iter;
    let mut data = Array2::zeros((samples, channels));

    for i in 0..samples {
        for c in 0..channels {
            data[(i, c)] = iter.next().ok_or_else(|| {
                Error::IoError(IoError::new(IoErrorKind::InvalidData, "not enough data"))
            })??;
        }
    }

    Ok(data)
}
