use sspse::ft;
use sspse::proc;
use sspse::stsa::{self, Subtraction, SetNoiseEstimate};
use sspse::wave::WavReaderExt;
use sspse::window as W;

use clap::{App, Arg};
use gnuplot::{AutoOption, AxesCommon, Figure};
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
    let mut spectrum = stft.process(&samples_c);
    let spectrum_orig = spectrum.clone();

    // spectral substraction
    let factor = 1.0;
    let post_gain = 1.5;

    let mut p = Subtraction::new(segment_len, factor, post_gain);
    p.set_noise_estimate(stsa::utils::noise_amplitude_est(&spectrum_orig.slice(s![..3, ..])).view());

    proc::utils::process_spectrum(p, &spectrum_orig, &mut spectrum);

    // compute signal from spectrum
    let out = istft.process(&spectrum);

    // drop imaginary part
    let out = out.mapv(|v| v.re * post_gain);

    // write
    if let Some(path_out) = path_out {
        let out_spec = hound::WavSpec {
            channels: 1,
            sample_rate: samples_spec.sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };

        let mut writer = hound::WavWriter::create(path_out, out_spec)?;
        for x in out.iter() {
            writer.write_sample(*x as f32)?;
        }
        writer.finalize()?;
    }

    // plot
    if plot {
        let fftfreq = ft::fftshift(&stft.spectrum_freqs(samples_spec.sample_rate as f64));
        let f0 = fftfreq[0];
        let f1 = fftfreq[fftfreq.len() - 1];

        let times = stft.spectrum_times(samples.len(), samples_spec.sample_rate as f64);
        let t0 = times[0];
        let t1 = times[times.len() - 1];

        // plot original spectrum
        let visual = ft::spectrum_to_visual(&spectrum_orig, -1e2, 1e2);

        let mut fig = Figure::new();
        let ax = fig.axes2d();
        ax.set_palette(gnuplot::HELIX);
        ax.set_x_range(AutoOption::Fix(t0), AutoOption::Fix(t1));
        ax.set_y_range(AutoOption::Fix(f0), AutoOption::Fix(f1));
        ax.image(visual.t().iter(), visual.shape()[1], visual.shape()[0], Some((t0, f0, t1, f1)), &[]);
        fig.show();

        // plot modified spectrum
        let visual = ft::spectrum_to_visual(&spectrum, -1e2, 1e2);

        let mut fig = Figure::new();
        let ax = fig.axes2d();
        ax.set_palette(gnuplot::HELIX);
        ax.set_x_range(AutoOption::Fix(t0), AutoOption::Fix(t1));
        ax.set_y_range(AutoOption::Fix(f0), AutoOption::Fix(f1));
        ax.image(visual.t().iter(), visual.shape()[1], visual.shape()[0], Some((t0, f0, t1, f1)), &[]);
        fig.show();
    }

    Ok(())
}
