use sspse::wave::WavReaderExt;
use sspse::window::{self as W, WindowFunction};
use sspse::ft;

use hound::{WavReader, Error};
use num::Complex;
use ndarray::{Axis, Array2};


fn main() -> Result<(), Error> {
    let path_in = std::env::args_os().nth(1).expect("missing input file argument");
    let path_out = std::env::args_os().nth(2).expect("missing output file argument");

    // load first channel of wave file
    let (samples, samples_spec) = WavReader::open(path_in)?.collect_convert_dyn::<f32>()?;
    let samples = samples.index_axis_move(Axis(1), 0);

    // convert real to complex
    let samples_c = samples.mapv(|v| Complex { re: v, im: 0.0 });

    // TODO: algorithm to find perfect window sizes

    // build window for fft
    let window_a = W::periodic(W::sqrt(W::hamming(512)));
    let window_s = W::periodic(W::sqrt(W::hamming(328)));

    let ratio = window_s.len() as f32 / window_a.len() as f32;
    let new_sample_rate = samples_spec.sample_rate as f32 * ratio;

    // build STFT and compute complex spectrum
    let mut stft = ft::StftBuilder::new(&window_a)
        .overlap(window_a.len() / 4 * 3)
        .padding(ft::Padding::Zero)
        .build();

    let mut istft = ft::IstftBuilder::new(&window_s)
        .overlap(window_s.len() / 4 * 3)
        .method(ft::InversionMethod::Weighted)
        .remove_padding(true)
        .build();

    // perform stft to compute spectrum
    let spectrum = stft.process(&samples_c);

    // adapt spectrum
    let mut new_spectrum = Array2::zeros((spectrum.shape()[0], window_s.len()));
    for i in 0..spectrum.shape()[0] {
        ft::spectrum_resize_into(&spectrum.index_axis(Axis(0), i), &mut new_spectrum.index_axis_mut(Axis(0), i));
    }

    // perform istft to restore signal
    let out = istft.process(&new_spectrum);

    // drop imaginary part and scale
    let out = out.mapv(|v| v.re * ratio);

    // write
    let out_spec = hound::WavSpec {
        channels: 1,
        sample_rate: new_sample_rate as _,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer = hound::WavWriter::create(path_out, out_spec)?;
    for x in out.iter() {
        writer.write_sample(*x)?;
    }
    writer.finalize()?;

    Ok(())
}