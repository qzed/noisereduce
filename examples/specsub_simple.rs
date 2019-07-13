use sspse::ft;
use sspse::proc;
use sspse::stsa::{self, Subtraction, SetNoiseEstimate};
use sspse::utils;
use sspse::wave::WavReaderExt;
use sspse::window as W;

use clap::{App, Arg};
use hound::{Error, WavReader};
use ndarray::{s, Axis};
use num::Complex;


fn app() -> App<'static, 'static> {
    App::new("Example: Noise Reduction via Spectral Substraction")
        .author(clap::crate_authors!())
        .arg(Arg::with_name("input")
                .help("The input file to use (wav)")
                .value_name("INPUT")
                .required(true))
        .arg(Arg::with_name("output")
                .help("The file to write the result to (wav)")
                .value_name("OUTPUT")
                .required(false))
        .arg(Arg::with_name("plot")
                .help("Wheter to plot the results or not")
                .short("p")
                .long("plot"))
}

fn main() -> Result<(), Error> {
    let matches = app().get_matches();

    let path_in = matches.value_of_os("input").unwrap();
    let path_out = matches.value_of_os("output");
    let plot = matches.is_present("plot");

    // load first channel of wave file
    let (samples, samples_spec) = WavReader::open(path_in)?.collect_convert_dyn::<f64>()?;
    let samples = samples.index_axis_move(Axis(1), 0);

    // convert real to complex
    let samples_c = samples.mapv(|v| Complex { re: v, im: 0.0 });

    // fft parameters
    let segment_len = (samples_spec.sample_rate as f64 * 0.02) as usize;
    let overlap = segment_len / 2;

    // build window for fft
    let window = W::periodic(W::sqrt(W::hann(segment_len)));

    // build STFT and compute complex spectrum
    let mut stft = ft::StftBuilder::new(&window)
        .overlap(overlap)
        .padding(ft::Padding::Zero)
        .build();

    let mut istft = ft::IstftBuilder::new(&window)
        .overlap(overlap)
        .method(ft::InversionMethod::Weighted)
        .remove_padding(true)
        .build();

    // compute spectrum
    let spectrum_in = stft.process(&samples_c);

    // spectral substraction
    let factor = 1.0;
    let post_gain = 1.5;

    let noise_est = stsa::noise::NoUpdate::new();
    let mut p = Subtraction::new(segment_len, 1.0, factor, post_gain, noise_est);
    p.set_noise_estimate(stsa::utils::noise_power_est(&spectrum_in.slice(s![..3, ..])).view());

    let spectrum_out = proc::utils::process_spectrum(&mut p, &spectrum_in);

    // compute signal from spectrum
    let out = istft.process(&spectrum_out);

    // drop imaginary part
    let out = out.mapv(|v| v.re * post_gain);

    // write
    if let Some(path_out) = path_out {
        utils::write_wav(path_out, &out, samples_spec.sample_rate)?;
    }

    // plot
    if plot {
        utils::plot_spectras(&spectrum_in, &spectrum_out, &stft, samples.len(), samples_spec.sample_rate);
    }

    Ok(())
}
