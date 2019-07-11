use sspse::ft;
use sspse::math::NumCastUnchecked;
use sspse::proc::{self, Processor};
use sspse::stsa::gain::LogMmse;
use sspse::stsa::noise::ProbabilisticExpTimeAvg;
use sspse::stsa::snr::DecisionDirected;
use sspse::stsa::{Gain, NoiseTracker, SetNoiseEstimate, SnrEstimator};
use sspse::vad::b::mc::MinimaControlledVad;
use sspse::vad::b::soft::SoftDecisionProbabilityEstimator;
use sspse::vad::b::SpeechProbabilityEstimator;
use sspse::wave::WavReaderExt;
use sspse::window as W;

use clap::{App, Arg};
use gnuplot::{AutoOption, AxesCommon, Figure};
use hound::{Error, WavReader};
use ndarray::{azip, s, Array1, ArrayView1, ArrayViewMut1, Axis};
use num::{traits::FloatConst, traits::NumAssign, Complex, Float};


fn app() -> App<'static, 'static> {
    App::new("Example: Noise reduction via OM-LSA+MCRA Method")
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
    let mut spectrum_out = spectrum.clone();

    // main algorithm loop over spectrum frames
    let mut p = setup_proc(segment_len);
    p.set_noise_estimate(sspse::stsa::utils::noise_power_est(&spectrum.slice(s![..3, ..])).view());

    proc::utils::process(p, &spectrum, &mut spectrum_out);

    // perform istft
    let out = istft.process(&spectrum_out);
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
        let visual = ft::spectrum_to_visual(&spectrum, -1e2, 1e2);

        let mut fig = Figure::new();
        let ax = fig.axes2d();
        ax.set_palette(gnuplot::HELIX);
        ax.set_x_range(AutoOption::Fix(t0), AutoOption::Fix(t1));
        ax.set_y_range(AutoOption::Fix(f0), AutoOption::Fix(f1));
        ax.image(visual.t().iter(), visual.shape()[1], visual.shape()[0], Some((t0, f0, t1, f1)), &[]);
        fig.show();

        // plot modified spectrum
        let visual = ft::spectrum_to_visual(&spectrum_out, -1e2, 1e2);

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

fn setup_proc<T>(block_size: usize) -> impl Processor<T> + SetNoiseEstimate<T>
where
    T: Float + FloatConst + NumCastUnchecked + NumAssign,
{
    // parameters for noise spectrum estimation
    let w = 1;
    let b = W::hamming::<T>(w * 2 + 1);
    let alpha_s = T::from(0.8).unwrap();
    let alpha_p = T::from(0.2).unwrap();
    let alpha_d = T::from(0.95).unwrap();
    let delta = T::from(5.0).unwrap();
    let vad = MinimaControlledVad::new(block_size, &b, alpha_s, alpha_p, delta, 125);
    let noise_est = ProbabilisticExpTimeAvg::new(block_size, alpha_d, vad);

    // parameters for SNR estimation
    let alpha = T::from(0.92).unwrap();
    let snr_est = DecisionDirected::new(alpha);

    // parameters for speech probability estimationa
    let beta = T::from(0.7).unwrap();
    let w_local = 1;
    let h_local = W::hamming::<T>(w_local * 2 + 1);
    let w_global = 15;
    let h_global = W::hamming::<T>(w_global * 2 + 1);
    let snr_pre_min = T::from(1e-3).unwrap();
    let snr_pre_max = T::from(1e3).unwrap();
    let snr_pre_peak_min = T::from(1.0).unwrap();
    let snr_pre_peak_max = T::from(1e5).unwrap();
    let q_max = T::from(0.95).unwrap();
    let p_est = SoftDecisionProbabilityEstimator::new(
        block_size,
        beta,
        &h_local,
        &h_global,
        snr_pre_min,
        snr_pre_max,
        snr_pre_peak_min,
        snr_pre_peak_max,
        q_max,
    );

    // parameters for gain computation
    let gain_fn = LogMmse::default();
    let gain_min = T::from(0.001).unwrap();

    // create
    Proc::new(block_size, noise_est, p_est, snr_est, gain_fn, gain_min)
}


struct Proc<T, N, P, S, G> {
    block_size: usize,
    snr_pre: Array1<T>,
    snr_post: Array1<T>,
    noise_power: Array1<T>,
    gain_h1: Array1<T>,
    p_gain: Array1<T>,
    gain_min: T,

    noise_est: N,
    p_est: P,
    snr_est: S,
    gain_fn: G,
}

impl<T, N, P, S, G> Proc<T, N, P, S, G>
where
    T: Float,
{
    pub fn new(block_size: usize, noise_est: N, p_est: P, snr_est: S, gain_h1: G, gain_min: T) -> Self {
        Proc {
            block_size,
            snr_pre: Array1::zeros(block_size),
            snr_post: Array1::zeros(block_size),
            noise_power: Array1::zeros(block_size),
            gain_h1: Array1::zeros(block_size),
            p_gain: Array1::zeros(block_size),
            gain_min,
            noise_est,
            snr_est,
            p_est,
            gain_fn: gain_h1,
        }
    }
}

impl<T, N, P, S, G> SetNoiseEstimate<T> for Proc<T, N, P, S, G>
where
    T: Float,
{
    fn set_noise_estimate(&mut self, noise: ArrayView1<T>) {
        self.noise_power.assign(&noise);
    }
}

impl<T, N, P, S, G> Processor<T> for Proc<T, N, P, S, G>
where
    T: Float,
    N: NoiseTracker<T>,
    S: SnrEstimator<T>,
    P: SpeechProbabilityEstimator<T>,
    G: Gain<T>,
{
    fn block_size(&self) -> usize {
        self.block_size
    }

    fn process(&mut self, spec_in: ArrayView1<Complex<T>>, spec_out: ArrayViewMut1<Complex<T>>) {
        // noise spectrum estimation
        self.noise_est.update(spec_in, self.noise_power.view_mut());

        // update a priori and a posteriori SNR
        self.snr_est.update(
            spec_in,
            self.noise_power.view(),
            self.gain_h1.view(),
            self.snr_pre.view_mut(),
            self.snr_post.view_mut(),
        );

        // compute speech probability for gain (p_gain)
        self.p_est.update(
            spec_in,
            self.snr_pre.view(),
            self.snr_post.view(),
            self.p_gain.view_mut(),
        );

        // compute gain for speech presence
        self.gain_fn.update(
            spec_in,
            self.snr_pre.view(),
            self.snr_post.view(),
            self.gain_h1.view_mut(),
        );

        // apply gain
        azip!(mut spec_out (spec_out), spec_in (spec_in), gain_h (&self.gain_h1), p (&self.p_gain) in {
            *spec_out = spec_in * gain_h.powf(p) * self.gain_min.powf(T::one() - p);
        });
    }
}
