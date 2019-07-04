use sspse::wave::WavReaderExt;
use sspse::window::{self as W, WindowFunction};
use sspse::ft;

use hound::{WavReader, Error};
use num::Complex;
use ndarray::{Axis, Array2};
use clap::{value_t_or_exit, App, Arg};


fn app() -> App<'static, 'static> {
    App::new("Example: Re-sample Audio Signal using Short-Time Fourier Transform")
        .author(clap::crate_authors!())
        .arg(Arg::with_name("input")
            .help("The input file to use (wav)")
            .value_name("INPUT")
            .required(true))
        .arg(Arg::with_name("output")
            .help("The file to write the result to (wav)")
            .value_name("OUTPUT")
            .required(true))
        .arg(Arg::with_name("samplerate")
            .help("The target sample-rate")
            .short("r")
            .long("sample-rate")
            .takes_value(true)
            .default_value("24000"))
}

fn main() -> Result<(), Error> {
    let matches = app().get_matches();

    let path_in = matches.value_of_os("input").unwrap();
    let path_out = matches.value_of_os("output").unwrap();
    let new_sample_rate = value_t_or_exit!(matches, "samplerate", u32);

    // load first channel of wave file
    let (samples, samples_spec) = WavReader::open(path_in)?.collect_convert_dyn::<f32>()?;
    let samples = samples.index_axis_move(Axis(1), 0);

    // convert real to complex
    let samples_c = samples.mapv(|v| Complex { re: v, im: 0.0 });

    // compute new window sizes by fixing window length in time
    let window_secs = 0.02;
    let window_a_len = (samples_spec.sample_rate as f32 * window_secs) as usize;
    let window_s_len = (new_sample_rate as f32 * window_secs) as usize;

    // build window for fft
    let window_a = W::periodic(W::sqrt(W::kaiser(window_a_len, 8.0)));
    let window_s = W::periodic(W::sqrt(W::kaiser(window_s_len, 8.0)));

    // build STFT and compute complex spectrum
    let mut stft = ft::StftBuilder::new(&window_a)
        .overlap(window_a.len() / 4 * 3)                    // choose 3/4 overlap
        .padding(ft::Padding::Zero)
        .build();

    let mut istft = ft::IstftBuilder::new(&window_s)
        .overlap(window_s.len() / 4 * 3)                    // choose 3/4 overlap
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
    let norm = window_s.len() as f32 / window_a.len() as f32;
    let out = out.mapv(|v| v.re * norm);

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
