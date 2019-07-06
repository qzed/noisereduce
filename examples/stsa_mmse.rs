use sspse::ft;
use sspse::vad::b::{power, power::PowerThresholdVad, VoiceActivityDetector};
// use sspse::vad::f::{energy, VoiceActivityDetector, energy::EnergyThresholdVad};
use sspse::wave::WavReaderExt;
use sspse::window as W;
use sspse::math::{bessel, expint, NumCastUnchecked};

use clap::{App, Arg};
use gnuplot::{AutoOption, AxesCommon, Figure};
use hound::{Error, WavReader};
use ndarray::{azip, s, Array1, ArrayBase, Axis, Data, DataMut, Ix1, Ix2};
use num::{traits::FloatConst, Complex, Float};


// SNR estimation:
// - ml:
//   - fixed: alpha, beta
//   - store: gamma-avg
//   - param: gamma(n)
//
// - dd
//   - fixed: alpha
//   - store: -
//   - param: gain(n-1), gamma(n-1), gamma(n)

// Gain computation
// - MMSE
//   - param: xi_k, gamma_k, Y_k
//
// - log-MMSE
//   - param: xi_k, gamma_k


fn gain_mmse<T, D1, D2, D3, Do>(
    y_spec: &ArrayBase<D1, Ix1>,
    xi: &ArrayBase<D2, Ix1>,
    gamma: &ArrayBase<D3, Ix1>,
    gain: &mut ArrayBase<Do, Ix1>,
) where
    T: Float + FloatConst + NumCastUnchecked,
    D1: Data<Elem = Complex<T>>,
    D2: Data<Elem = T>,
    D3: Data<Elem = T>,
    Do: DataMut<Elem = T>,
{
    let nu_min = T::from(1e-50).unwrap();
    let nu_max = T::from(500.0).unwrap();

    let half = T::from(0.5).unwrap();
    let fspi2 = T::PI().sqrt() * half;

    azip!(mut gain (gain), yk (y_spec), xi (xi), gamma (gamma) in {
        let nu = xi / (T::one() + xi) * gamma;
        let nu = nu.max(nu_min).min(nu_max);        // prevent over/underflows

        *gain = fspi2
            * (T::sqrt(nu) * T::exp(-nu * half) / gamma)
            * ((T::one() + nu) * bessel::I0(nu * half) + nu * bessel::I1(nu * half))
            * yk.norm();
    });
}

fn gain_logmmse<T, D1, D2, Do>(
    xi: &ArrayBase<D1, Ix1>,
    gamma: &ArrayBase<D2, Ix1>,
    gain: &mut ArrayBase<Do, Ix1>,
) where
    T: Float + FloatConst + NumCastUnchecked,
    D1: Data<Elem = T>,
    D2: Data<Elem = T>,
    Do: DataMut<Elem = T>,
{
    let nu_min = T::from(1e-50).unwrap();
    let nu_max = T::from(500.0).unwrap();

    let half = T::from(0.5).unwrap();

    azip!(mut gain (gain), xi (xi), gamma (gamma) in {
        let nu = xi / (T::one() + xi) * gamma;
        let nu = nu.max(nu_min).min(nu_max);        // prevent over/underflows

        *gain = (xi / (T::one() + xi)) * T::exp(-half * expint::Ei(-nu));
    });
}

pub fn noise_power_est<T, D>(spectrum: &ArrayBase<D, Ix2>) -> Array1<T>
where
    T: Float + std::ops::AddAssign + ndarray::ScalarOperand,
    D: Data<Elem = Complex<T>>,
{
    let mut lambda_d = Array1::zeros(spectrum.shape()[1]);
    let norm = T::from(spectrum.shape()[0]).unwrap();

    for ((_, i), v) in spectrum.indexed_iter() {
        lambda_d[i] += v.norm_sqr() / norm;
    }

    lambda_d
}

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
}

fn main() -> Result<(), Error> {
    let matches = app().get_matches();

    let path_in = matches.value_of_os("input").unwrap();
    let path_out = matches.value_of_os("output");
    let plot = matches.is_present("plot");
    let logmmse = matches.is_present("log_mmse");

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
    let mut spectrum = stft.process(&samples_c);
    let spectrum_orig = spectrum.clone();

    // initial noise estimate
    let mut lambda_d = noise_power_est(&spectrum.slice(s![..3, ..]));

    let noise_floor = power::noise_floor_est(&spectrum.slice(s![..3, ..]));
    let vad = PowerThresholdVad::new(noise_floor, 1.3);

    // let noise_floor = energy::noise_floor_est(&spectrum.slice(s![..3, ..]));
    // let vad = EnergyThresholdVad::new(noise_floor, 1.3).per_band();

    // set parameters
    let alpha = 0.98;

    // initial a priori and a posteriori snr
    let mut gain = Array1::from_elem(segment_len, 1.0);
    let mut gamma = &spectrum.index_axis(Axis(0), 0).mapv(|v| v.norm_sqr()) / &lambda_d;
    let mut xi = alpha + (1.0 - alpha) * gamma.mapv(|v| (v - 1.0).max(0.0));
    let mut voice_activity = Array1::from_elem(segment_len, false);

    // main algorithm loop over spectrum frames
    let num_frames = spectrum.shape()[0];
    for i in 0..num_frames {
        let mut yk = spectrum.index_axis_mut(Axis(0), i);

        // update noise estimate
        vad.detect_into(&yk, &mut voice_activity);
        azip!(mut lambda_d, voice_activity, yk in {
            if !voice_activity {
                let a = 0.5;
                *lambda_d = a * *lambda_d + (1.0 - a) * yk.norm_sqr();
            }
        });

        // calculate gain function
        if logmmse {
            gain_logmmse(&xi, &gamma, &mut gain);
        } else {
            gain_mmse(&yk, &xi, &gamma, &mut gain);
        }

        // update a priori and a posteriori snr (decision directed)
        if i < num_frames - 1 {
            azip!(mut xi, mut gamma, yk, lambda_d, gain in {
                let gamma_ = yk.norm_sqr() / lambda_d;
                let xi_ = alpha * gain.powi(2) * *gamma + (1.0 - alpha) * (gamma_ - 1.0).max(0.0);

                *xi = xi_.min(1e3);
                *gamma = gamma_.min(1e3);
            });
        }

        // apply gain
        azip!(mut yk, gain in { *yk *= gain.min(1.0) });
    }

    // perform istft
    let out = istft.process(&spectrum);
    let out = out.mapv(|v| v.re as f32);

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
            writer.write_sample(*x)?;
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
