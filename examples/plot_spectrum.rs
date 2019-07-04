use sspse::ft;
use sspse::wave::WavReaderExt;

use clap::{App, Arg};
use gnuplot::{AutoOption, AxesCommon, Figure};
use hound::{Error, WavReader};
use ndarray::Axis;
use num::Complex;


fn app() -> App<'static, 'static> {
    App::new("Example: Compute and Plot Short-Time Spectral Amplitude")
        .author(clap::crate_authors!())
        .arg(Arg::with_name("input")
                .help("The input file to use (wav)")
                .value_name("INPUT")
                .required(true))
}

fn main() -> Result<(), Error> {
    let matches = app().get_matches();
    let path_in = matches.value_of_os("input").unwrap();

    // load first channel of wave file
    let (samples, samples_spec) = WavReader::open(path_in)?.collect_convert_dyn::<f32>()?;
    let samples = samples.index_axis_move(Axis(1), 0);

    // fft parameters
    let fftlen = 256;
    let seglen = 256;
    let overlap = 192;

    // build window for fft
    let window = sspse::window::hamming(seglen);
    let window = sspse::window::sqrt(window);
    let window = sspse::window::periodic(window);

    let mut stft = ft::StftBuilder::with_len(&window, fftlen)
        .overlap(overlap)
        .padding(ft::Padding::Zero)
        .shifted(true)
        .build();

    let samples = samples.mapv(|v| Complex { re: v, im: 0.0 });

    let out = stft.process(&samples);
    let out = out.mapv(|v| ft::magnitude_to_db(v.norm()));

    // plot
    let fftfreq = stft.spectrum_freqs(samples_spec.sample_rate as f64);
    let f0 = fftfreq[0];
    let f1 = fftfreq[fftfreq.len() - 1];

    let times = stft.spectrum_times(samples.len(), samples_spec.sample_rate as f64);
    let t0 = times[0];
    let t1 = times[times.len() - 1];

    let mut fig = Figure::new();
    let ax = fig.axes2d();
    ax.set_palette(gnuplot::HELIX);
    ax.set_x_range(AutoOption::Fix(t0), AutoOption::Fix(t1));
    ax.set_y_range(AutoOption::Fix(f0), AutoOption::Fix(f1));
    ax.image(out.t().iter(), out.shape()[1], out.shape()[0], Some((t0, f0, t1, f1)), &[]);
    fig.show();

    Ok(())
}
