use sspse::ft;
use sspse::vad::b::{power, power::PowerThresholdVad, VoiceActivityDetector};
// use sspse::vad::f::{energy, VoiceActivityDetector, energy::EnergyThresholdVad};
use sspse::math::{bessel, expint, NumCastUnchecked};
use sspse::wave::WavReaderExt;
use sspse::window as W;

use clap::{App, Arg};
use gnuplot::{AutoOption, AxesCommon, Figure};
use hound::{Error, WavReader};
use ndarray::{azip, s, Array1, ArrayView1, ArrayViewMut1, ArrayBase, Axis, Data, Ix2};
use num::{traits::FloatConst, Complex, Float};


pub trait Gain<T> {
    fn update(
        &mut self,
        spectrum: ArrayView1<Complex<T>>,
        snr_pre: ArrayView1<T>,
        snr_post: ArrayView1<T>,
        gain: ArrayViewMut1<T>,
    );
}


impl<T, G> Gain<T> for Box<G>
where
    G: Gain<T>,
{
    fn update(
        &mut self,
        spectrum: ArrayView1<Complex<T>>,
        snr_pre: ArrayView1<T>,
        snr_post: ArrayView1<T>,
        gain: ArrayViewMut1<T>,
    ) {
        self.as_mut().update(spectrum, snr_pre, snr_post, gain)
    }
}


pub struct Mmse<T> {
    nu_min: T,
    nu_max: T,
}

impl<T: Float> Mmse<T> {
    pub fn new() -> Self {
        Mmse {
            nu_min: T::from(1e-50).unwrap(),
            nu_max: T::from(500.0).unwrap(),
        }
    }
}

impl<T> Gain<T> for Mmse<T>
where
    T: Float + FloatConst + NumCastUnchecked,
{
    fn update(
        &mut self,
        spectrum: ArrayView1<Complex<T>>,
        snr_pre: ArrayView1<T>,
        snr_post: ArrayView1<T>,
        gain: ArrayViewMut1<T>,
    ) {
        let half = T::from(0.5).unwrap();
        let fspi2 = T::PI().sqrt() * half;

        azip!(mut gain (gain), spectrum (spectrum), snr_pre (snr_pre), snr_post (snr_post) in {
            let nu = snr_pre / (T::one() + snr_pre) * snr_post;
            let nu = nu.max(self.nu_min).min(self.nu_max);          // prevent over/underflows

            *gain = fspi2
                * (T::sqrt(nu) * T::exp(-nu * half) / snr_post)
                * ((T::one() + nu) * bessel::I0(nu * half) + nu * bessel::I1(nu * half))
                * spectrum.norm();
        });
    }
}


pub struct LogMmse<T> {
    nu_min: T,
    nu_max: T,
}

impl<T: Float> LogMmse<T> {
    pub fn new() -> Self {
        LogMmse {
            nu_min: T::from(1e-50).unwrap(),
            nu_max: T::from(500.0).unwrap(),
        }
    }
}

impl<T> Gain<T> for LogMmse<T>
where
    T: Float + FloatConst + NumCastUnchecked,
{
    fn update(
        &mut self,
        _spectrum: ArrayView1<Complex<T>>,
        snr_pre: ArrayView1<T>,
        snr_post: ArrayView1<T>,
        gain: ArrayViewMut1<T>,
    ) {
        let half = T::from(0.5).unwrap();

        azip!(mut gain (gain), snr_pre (snr_pre), snr_post (snr_post) in {
            let nu = snr_pre / (T::one() + snr_pre) * snr_post;
            let nu = nu.max(self.nu_min).min(self.nu_max);          // prevent over/underflows

            *gain = (snr_pre / (T::one() + snr_pre)) * T::exp(-half * expint::Ei(-nu));
        });
    }
}


pub trait NoiseTracker<T> {
    fn update(&mut self, spectrum: ArrayView1<Complex<T>>, noise_est: ArrayViewMut1<T>);
}


impl<T, N> NoiseTracker<T> for Box<N>
where
    N: NoiseTracker<T>,
{
    fn update(&mut self, spectrum: ArrayView1<Complex<T>>, noise_est: ArrayViewMut1<T>) {
        self.as_mut().update(spectrum, noise_est)
    }
}


pub struct ExpTimeAvgNoise<T, V> {
    voiced: Array1<bool>,
    alpha: T,
    vad: V,
}

impl<T, V> ExpTimeAvgNoise<T, V>
where
    T: Float,
{
    pub fn new(block_size: usize, alpha: T, vad: V) -> Self {
        ExpTimeAvgNoise {
            voiced: Array1::from_elem(block_size, false),
            alpha,
            vad,
        }
    }
}

impl<T, V> NoiseTracker<T> for ExpTimeAvgNoise<T, V>
where
    T: Float,
    V: VoiceActivityDetector<T>,
{
    fn update(&mut self, spectrum: ArrayView1<Complex<T>>, noise_est: ArrayViewMut1<T>) {
        self.vad.detect_into(&spectrum, &mut self.voiced);
        azip!(mut noise (noise_est), voiced (&self.voiced), spectrum (spectrum) in {
            if !voiced {
                *noise = self.alpha * *noise + (T::one() - self.alpha) * spectrum.norm_sqr();
            }
        });
    }
}


pub trait SnrEstimator<T> {
    fn update(
        &mut self,
        spectrum: ArrayView1<Complex<T>>,
        noise_power: ArrayView1<T>,
        gain: ArrayView1<T>,
        snr_pre: ArrayViewMut1<T>,
        snr_post: ArrayViewMut1<T>,
    );
}

pub struct DecisionDirected<T> {
    alpha: T,
    snr_pre_max: T,
    snr_post_max: T,
}

impl<T: Float> DecisionDirected<T> {
    pub fn new(alpha: T) -> Self {
        DecisionDirected {
            alpha,
            snr_pre_max: T::from(1e3).unwrap(),
            snr_post_max: T::from(1e3).unwrap(),
        }
    }
}

impl<T> SnrEstimator<T>  for DecisionDirected<T>
where
    T: Float,
{
    fn update(
        &mut self,
        spectrum: ArrayView1<Complex<T>>,
        noise_power: ArrayView1<T>,
        gain: ArrayView1<T>,
        snr_pre: ArrayViewMut1<T>,
        snr_post: ArrayViewMut1<T>,
    ) {
        azip!(
            mut snr_pre (snr_pre),
            mut snr_post (snr_post),
            spectrum (spectrum),
            noise_power (noise_power),
            gain (gain)
        in {
            let snr_post_ = spectrum.norm_sqr() / noise_power;
            let snr_pre_ = self.alpha * gain.powi(2) * *snr_post
                + (T::one() - self.alpha) * (snr_post_ - T::one()).max(T::zero());

            *snr_pre = snr_pre_.min(self.snr_pre_max);
            *snr_post = snr_post_.min(self.snr_post_max);
        });
    }
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
        .arg(
            Arg::with_name("input")
                .help("The input file to use (wav)")
                .value_name("INPUT")
                .required(true),
        )
        .arg(
            Arg::with_name("output")
                .help("The file to write the result to (wav)")
                .value_name("OUTPUT")
                .required(false),
        )
        .arg(
            Arg::with_name("plot")
                .help("Wheter to plot the results or not")
                .short("p")
                .long("plot"),
        )
        .arg(
            Arg::with_name("log_mmse")
                .help("Use log-MMSE instead of plain MMSE")
                .short("l")
                .long("log-mmse"),
        )
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

    let mut noise_tracker = ExpTimeAvgNoise::new(segment_len, 0.5, vad);
    let mut snr_est = DecisionDirected::new(alpha);

    let mut gain_fn: Box<dyn Gain<_>> = if logmmse {
        Box::new(LogMmse::new())
    } else {
        Box::new(Mmse::new())
    };

    // main algorithm loop over spectrum frames
    let num_frames = spectrum.shape()[0];
    for i in 0..num_frames {
        let mut yk = spectrum.index_axis_mut(Axis(0), i);

        // update noise estimate
        noise_tracker.update(yk.view(), lambda_d.view_mut());

        // calculate gain function
        gain_fn.update(yk.view(), xi.view(), gamma.view(), gain.view_mut());

        // update a priori and a posteriori snr (decision directed)
        if i < num_frames - 1 {
            snr_est.update(yk.view(), lambda_d.view(), gain.view(), xi.view_mut(), gamma.view_mut());
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
