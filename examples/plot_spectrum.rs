use sspse::wave::WavReaderExt;
use sspse::window::WindowFunction;
use sspse::ft;

use hound::{WavReader, Error};
use num::Complex;
use ndarray::{Axis};
use gnuplot::{Figure, AxesCommon, AutoOption};


fn main() -> Result<(), Error> {
    let path = std::env::args_os().nth(1).expect("missing file argument");

    // load first channel of wave file
    let (samples, samples_spec) = WavReader::open(path)?.into_array_f32()?;
    let samples = samples.index_axis_move(Axis(1), 0);

    // fft parameters
    let fftlen = 512;
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

    let window_norm = 2.0 / window.to_array().scalar_sum();

    let samples = samples.mapv(|v| Complex { re: v, im: 0.0 });

    let out = stft.perform(&samples);
    let out = out.mapv(|v| ft::magnitude_to_db(v.norm() * window_norm));

    // plot
    let fftfreq = ft::fftshift(&ft::fftfreq(fftlen, samples_spec.sample_rate as f64));
    let f0 = fftfreq[0];
    let f1 = fftfreq[fftfreq.len() - 1];

    let times = stft.row_times(samples.len(), samples_spec.sample_rate as f64);
    let t0 = times[0];
    let t1 = times[times.len() - 1];

    let mut fig = Figure::new();
    let ax = fig.axes2d();
    ax.set_palette(gnuplot::HELIX);
    ax.set_x_range(AutoOption::Fix(t0), AutoOption::Fix(t1));
    ax.set_y_range(AutoOption::Fix(fftfreq[0]), AutoOption::Fix(fftfreq[fftfreq.len()-1]));
    ax.image(out.t().iter(), out.shape()[1], out.shape()[0], Some((t0, f0, t1, f1)), &[]);
    fig.show();

    Ok(())
}
