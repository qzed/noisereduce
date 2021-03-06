use sspse::ft;
use sspse::proc;
use sspse::stsa::gain::{LogMmse, Mmse};
use sspse::stsa::noise::ExpTimeAvg;
use sspse::stsa::snr::DecisionDirected;
use sspse::stsa::{self, Gain, Stsa, SetNoiseEstimate};
use sspse::utils;
use sspse::vad::b::power::{self, PowerThresholdVad};
// use sspse::vad::f::energy::{self, EnergyThresholdVad};
use sspse::wave::WavReaderExt;
use sspse::window as W;

use clap::{value_t_or_exit, App, Arg};
use hound::{Error, WavReader};
use ndarray::{s, Axis};
use num::Complex;


fn app() -> App<'static, 'static> {
    App::new("Example: Noise reduction via MMSE/log-MMSE STSA Method")
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
        .arg(Arg::with_name("log_mmse")
                .help("Use log-MMSE instead of plain MMSE")
                .short("l")
                .long("log-mmse"))
        .arg(Arg::with_name("snr_alpha")
                .help("Alpha value for SNR estimator (exp. avg.)")
                .long("snr-alpha")
                .takes_value(true)
                .default_value("0.98"))
        .arg(Arg::with_name("noise_alpha")
                .help("Alpha value for noise tracker (exp. avg.)")
                .long("noise-alpha")
                .takes_value(true)
                .default_value("0.8"))
        .arg(Arg::with_name("vad_ratio")
                .help("Ratio for VAD voice classification")
                .long("vad-ratio")
                .takes_value(true)
                .default_value("1.3"))
}

fn main() -> Result<(), Error> {
    let matches = app().get_matches();

    let path_in  = matches.value_of_os("input").unwrap();
    let path_out = matches.value_of_os("output");
    let plot     = matches.is_present("plot");
    let logmmse  = matches.is_present("log_mmse");

    let snr_alpha   = value_t_or_exit!(matches, "snr_alpha",   f64);
    let noise_alpha = value_t_or_exit!(matches, "noise_alpha", f64);
    let vad_ratio   = value_t_or_exit!(matches, "vad_ratio",   f64);

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

    // make sure our window satisfies the constraint for reconstruction
    assert!(ft::check_cola(&window, overlap, ft::InversionMethod::Weighted, 1e-6));

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

    // perform stft
    let spectrum = stft.process(&samples_c);

    // set-up voice-activity detector
    let noise_floor = power::noise_floor_est(&spectrum.slice(s![..3, ..]));
    let vad = PowerThresholdVad::new(noise_floor, vad_ratio);

    // let noise_floor = energy::noise_floor_est(&spectrum.slice(s![..3, ..]));
    // let vad = EnergyThresholdVad::new(noise_floor, vad_ratio).per_band();

    // set-up algorithm parts
    let noise_tracker = ExpTimeAvg::new(segment_len, noise_alpha, vad);
    let snr_est = DecisionDirected::new(snr_alpha);
    // let snr_est = MaximumLikelihood::new(segment_len, 0.725, 2.0);

    let gain_fn: Box<dyn Gain<_>> = if logmmse {
        Box::new(LogMmse::default())
    } else {
        Box::new(Mmse::default())
    };

    let mut stsa = Stsa::new(segment_len, gain_fn, snr_est, noise_tracker);
    stsa.set_noise_estimate(stsa::utils::noise_power_est(&spectrum.slice(s![..3, ..])).view());

    // main algorithm loop over spectrum frames
    let spectrum_out = proc::utils::process_spectrum(&mut stsa, &spectrum);

    // perform istft
    let out = istft.process(&spectrum_out);
    let out = out.mapv(|v| v.re as f32);

    // write
    if let Some(path_out) = path_out {
        utils::write_wav(path_out, &out, samples_spec.sample_rate)?;
    }

    // plot
    if plot {
        utils::plot_spectras(&spectrum, &spectrum_out, &stft, samples.len(), samples_spec.sample_rate);
    }

    Ok(())
}
