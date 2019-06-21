use std::io::Read;

use hound::{WavReader, WavSamples};


pub trait WavReaderExt<R> {
    fn samples_f32<'wr>(&'wr mut self) -> WavSamplesF32<'wr, R>;
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
}


pub enum WavSamplesF32<'wr, R> {
    I32(WavSamples<'wr, R, i32>, f32),
    F32(WavSamples<'wr, R, f32>),
}

impl<'wr, R> Iterator for WavSamplesF32<'wr, R>
where
    R: Read,
{
    type Item = Result<f32, hound::Error>;

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
