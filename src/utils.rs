use crate::math::NumCastUnchecked;
use crate::ft::{self, Stft};

use std::path::Path;

use ndarray::{ArrayBase, Data, Ix1, Ix2};
use num::{Complex, Float, traits::NumAssign};
use rustfft::FFTnum;


pub fn write_wav<T, D, P: AsRef<Path>>(path: P, data: &ArrayBase<D, Ix1>, sample_rate: u32)
    -> Result<(), hound::Error>
where
    T: Float + NumCastUnchecked,
    D: Data<Elem = T>,
{
    use hound::{WavSpec, WavWriter};

    let out_spec = WavSpec {
        channels: 1,
        sample_rate: sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer = WavWriter::create(path, out_spec)?;
    for x in data.iter() {
        writer.write_sample(x.to_f32_unchecked())?;
    }

    writer.finalize()
}

pub fn plot_spectras<T, D1, D2>(spectrum_in: &ArrayBase<D1, Ix2>, spectrum_out: &ArrayBase<D2, Ix2>,
                            stft: &Stft<T>, num_samples: usize, sample_rate: u32)
where
    T: FFTnum + Float + NumAssign + NumCastUnchecked,
    for<'a> &'a T: gnuplot::DataType,
    D1: Data<Elem = Complex<T>>,
    D2: Data<Elem = Complex<T>>,
{
    use gnuplot::{AutoOption, AxesCommon, Figure};

    let fftfreq = ft::fftshift(&stft.spectrum_freqs(sample_rate as f64));
    let f0 = fftfreq[0];
    let f1 = fftfreq[fftfreq.len() - 1];

    let times = stft.spectrum_times(num_samples, sample_rate as f64);
    let t0 = times[0];
    let t1 = times[times.len() - 1];

    // plot original spectrum
    let visual = ft::spectrum_to_visual(&spectrum_in, T::from_unchecked(-1e2), T::from_unchecked(1e2));

    let mut fig = Figure::new();
    let ax = fig.axes2d();
    ax.set_palette(gnuplot::HELIX);
    ax.set_x_range(AutoOption::Fix(t0), AutoOption::Fix(t1));
    ax.set_y_range(AutoOption::Fix(f0), AutoOption::Fix(f1));
    ax.image(visual.t().iter(), visual.shape()[1], visual.shape()[0], Some((t0, f0, t1, f1)), &[]);
    fig.show();

    // plot modified spectrum
    let visual = ft::spectrum_to_visual(&spectrum_out, T::from_unchecked(-1e2), T::from_unchecked(1e2));

    let mut fig = Figure::new();
    let ax = fig.axes2d();
    ax.set_palette(gnuplot::HELIX);
    ax.set_x_range(AutoOption::Fix(t0), AutoOption::Fix(t1));
    ax.set_y_range(AutoOption::Fix(f0), AutoOption::Fix(f1));
    ax.image(visual.t().iter(), visual.shape()[1], visual.shape()[0], Some((t0, f0, t1, f1)), &[]);
    fig.show();
}
