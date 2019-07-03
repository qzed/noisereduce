use sspse::wave::WavReaderExt;
use sspse::window as W;
use sspse::ft;
use sspse::vad::{self, VoiceActivityDetector, energy::EnergyThresholdVad};

use hound::{WavReader, Error};
use num::{Complex, Float};
use ndarray::{s, Array1, Axis, ArrayBase, Data, Ix2};
use gnuplot::{Figure, AxesCommon, AutoOption};


pub fn noise_power_est<T, D>(spectrum: &ArrayBase<D, Ix2>) -> Array1<T>
where
    T: Float + std::ops::AddAssign + ndarray::ScalarOperand,
    D: Data<Elem=Complex<T>>,
{
    let mut lambda_d = Array1::zeros(spectrum.shape()[1]);
    let norm = T::from(spectrum.shape()[0]).unwrap();

    for ((_, i), v) in spectrum.indexed_iter() {
        lambda_d[i] += v.norm_sqr() / norm;
    }

    lambda_d
}


fn main() -> Result<(), Error> {
    let path_in = std::env::args_os().nth(1).expect("missing input file argument");
    let path_out = std::env::args_os().nth(2).expect("missing output file argument");

    // load first channel of wave file
    let (samples, samples_spec) = WavReader::open(path_in)?.collect_convert_dyn::<f32>()?;
    let samples = samples.index_axis_move(Axis(1), 0);

    // convert real to complex
    let samples_c = samples.mapv(|v| Complex { re: v, im: 0.0 });

    // fft parameters
    let segment_len = (samples_spec.sample_rate as f64 * 0.02) as usize;
    let overlap     = segment_len / 2;

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
    let mut spectrum = stft.process(&samples_c);
    let spectrum_orig = spectrum.clone();

    // initial noise estimate
    let mut lambda_d = noise_power_est(&spectrum.slice(s![..3, ..])).mapv(f64::from);

    let noise_floor = vad::energy::noise_floor(&spectrum.slice(s![..3, ..]));
    let vad = EnergyThresholdVad::new(noise_floor, 1.3);

    // set parameters
    let alpha = 0.98;

    // initial a priori and a posteriori snr
    let mut gain = Array1::zeros(segment_len);
    let mut gamma = &spectrum.index_axis(Axis(0), 0).mapv(|v| v.norm_sqr() as f64) / &lambda_d;
    let mut gamma_new = Array1::zeros(segment_len);
    let mut xi = alpha + (1.0 - alpha) * gamma.mapv(|v| f64::max(v - 1.0, 0.0));

    // main algorithm loop over spectrum frames
    let num_frames = spectrum.shape()[0];
    for i in 0..num_frames {
        let mut yk = spectrum.index_axis_mut(Axis(0), i);

        // update noise estimate
        if !vad.detect(&yk) {
            let a = 0.5;
            lambda_d = a * lambda_d + (1.0 - a) * yk.mapv(|v| v.norm_sqr() as f64);
        } else {
        }

        // calculate gain function
        let g = std::f64::consts::PI.sqrt() / 2.0;

        let nu = xi.mapv(|v| v / (1.0 + v)) * &gamma;
        let nu = nu.mapv_into(|v| f64::max(f64::min(v, 200.0), 1e-20));     // prevent over-/underflow

        let t1 = nu.mapv(|v| v.sqrt() * f64::exp(-v / 2.0)) / &gamma;
        let t2 = nu.mapv(|v| (1.0 + v) * rgsl::bessel::I0(v / 2.0) + v * rgsl::bessel::I1(v / 2.0));

        gain.assign(&(g * t1 * t2 * yk.mapv(|v| v.norm() as f64)));

        // update a priori and a posteriori snr (decision directed)
        if i < num_frames - 1 {
            gamma_new.assign(&(yk.mapv(|v| v.norm_sqr() as f64) / &lambda_d));

            let g2 = gain.mapv(|v| v.powi(2));
            xi = alpha * &g2 * &gamma + (1.0 - alpha) * gamma_new.mapv(|v| f64::max(v - 1.0, 0.0));

            gamma.assign(&gamma_new);
        }

        // apply gain
        for (j, v) in yk.indexed_iter_mut() {
            *v *= f32::min(gain[j] as f32, 1.0);
            // *v *= gain[j] as f32 * 0.05;
        }
    }

    // perform istft
    let out = istft.process(&spectrum);
    let out = out.mapv(|v| v.re);

    // write
    let out_spec = hound::WavSpec {
        channels: 1,
        sample_rate: samples_spec.sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer = hound::WavWriter::create(path_out, out_spec)?;
    for x in out.iter() {
        writer.write_sample(*x)?;
    }
    writer.finalize()?;

    // plot
    let fftfreq = ft::fftshift(&stft.spectrum_freqs(samples_spec.sample_rate as f64));
    let f0 = fftfreq[0];
    let f1 = fftfreq[fftfreq.len() - 1];

    let times = stft.spectrum_times(samples.len(), samples_spec.sample_rate as f64);
    let t0 = times[0];
    let t1 = times[times.len() - 1];

    // plot original spectrum
    let visual = ft::spectrum_to_visual(&spectrum_orig);

    let mut fig = Figure::new();
    let ax = fig.axes2d();
    ax.set_palette(gnuplot::HELIX);
    ax.set_x_range(AutoOption::Fix(t0), AutoOption::Fix(t1));
    ax.set_y_range(AutoOption::Fix(f0), AutoOption::Fix(f1));
    ax.image(visual.t().iter(), visual.shape()[1], visual.shape()[0], Some((t0, f0, t1, f1)), &[]);
    fig.show();

    // plot modified spectrum
    let visual = ft::spectrum_to_visual(&spectrum);

    let mut fig = Figure::new();
    let ax = fig.axes2d();
    ax.set_palette(gnuplot::HELIX);
    ax.set_x_range(AutoOption::Fix(t0), AutoOption::Fix(t1));
    ax.set_y_range(AutoOption::Fix(f0), AutoOption::Fix(f1));
    ax.image(visual.t().iter(), visual.shape()[1], visual.shape()[0], Some((t0, f0, t1, f1)), &[]);
    fig.show();

    Ok(())
}
