use sspse::ft;
use sspse::math::NumCastUnchecked;
use sspse::proc::{self, Processor};
use sspse::wave::WavReaderExt;
use sspse::window::{self as W, WindowFunction};

use sspse::stsa::{Gain, SnrEstimator, NoiseTracker};
use sspse::stsa::gain::LogMmse;
use sspse::stsa::snr::DecisionDirected;
use sspse::stsa::noise::ProbabilisticExpTimeAvg;
use sspse::vad::b::mc::MinimaControlledVad;

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
    let mut p = Proc::new(segment_len);
    p.noise_power = sspse::stsa::utils::noise_power_est(&spectrum.slice(s![..3, ..]));

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


struct Proc<T> {
    block_size: usize,
    frame: usize,
    snr_pre: Array1<T>,
    snr_pre_avg: Array1<T>,
    snr_pre_avg_beta: T,
    snr_post: Array1<T>,
    noise_power: Array1<T>,
    gain_h1: Array1<T>,
    p_gain: Array1<T>,
    gain_min: T,
    snr_pre_avg_frame: T,
    snr_pre_avg_peak: T,

    gain_fn: LogMmse<T>,
    snr_est: DecisionDirected<T>,
    noise_est: ProbabilisticExpTimeAvg<T, MinimaControlledVad<T>>,
}

impl<T> Proc<T>
where
    T: Float + FloatConst,
{
    pub fn new(block_size: usize) -> Self {
        // parameters for noise spectrum estimation
        let w = 1;
        let b = sspse::window::hamming::<T>(w * 2 + 1);
        let alpha_s = T::from(0.8).unwrap();
        let alpha_p = T::from(0.2).unwrap();
        let alpha_d = T::from(0.95).unwrap();
        let delta = T::from(5.0).unwrap();
        let vad = MinimaControlledVad::new(block_size, &b, alpha_s, alpha_p, delta, 125);
        let noise_est = ProbabilisticExpTimeAvg::new(block_size, alpha_d, vad);

        // parameters for SNR estimation
        let alpha = T::from(0.92).unwrap();
        let snr_est = DecisionDirected::new(alpha);

        // parameters for gain computation
        let gain_fn = LogMmse::default();

        Proc {
            block_size,
            frame: 0,
            snr_pre: Array1::from_elem(block_size, T::one()),
            snr_pre_avg: Array1::from_elem(block_size, T::zero()),
            snr_pre_avg_beta: T::from(0.7).unwrap(),
            snr_post: Array1::from_elem(block_size, T::one()),
            noise_power: Array1::from_elem(block_size, T::zero()),
            gain_h1: Array1::from_elem(block_size, T::one()),
            p_gain: Array1::from_elem(block_size, T::one()),
            gain_min: T::from(0.001).unwrap(),
            snr_pre_avg_frame: T::from(1.0).unwrap(),
            snr_pre_avg_peak: T::from(1.0).unwrap(),

            gain_fn,
            snr_est,
            noise_est,
        }
    }
}

impl<T> Processor<T> for Proc<T>
where
    T: Float + FloatConst + NumAssign + NumCastUnchecked,
{
    fn block_size(&self) -> usize {
        self.block_size
    }

    fn process(&mut self, spec_in: ArrayView1<Complex<T>>, spec_out: ArrayViewMut1<Complex<T>>) {
        // update a priori and a posteriori SNR
        self.snr_est.update(spec_in, self.noise_power.view(), self.gain_h1.view(), self.snr_pre.view_mut(), self.snr_post.view_mut());

        {   // compute speech probability for gain (p_gain)
            // a priori SNR averaging over time
            let snr_pre_avg = &mut self.snr_pre_avg;
            let snr_pre = &self.snr_pre;
            let beta = self.snr_pre_avg_beta;

            azip!(mut snr_pre_avg (snr_pre_avg), snr_pre (snr_pre) in {
                *snr_pre_avg = beta * *snr_pre_avg + (T::one() - beta) * snr_pre;
            });

            // a priori SNR averaging over frequencies
            let w_local = 1;
            let w_global = 15;

            let h_local = sspse::window::hamming::<T>(w_local * 2 + 1).to_array();
            let h_global = sspse::window::hamming::<T>(w_global * 2 + 1).to_array();

            let snr_pre_avg_padded = sspse::ft::extend_zero(&self.snr_pre_avg, w_global);
            let mut snr_pre_avg_local = Array1::zeros(self.block_size);
            let mut snr_pre_avg_global = Array1::zeros(self.block_size);

            for k in 0..self.block_size {
                for i in -(w_local as isize)..=(w_local as isize) {
                    let idx_window = (i + w_local as isize) as usize;
                    let idx_spectr = (i + k as isize + w_global as isize) as usize;

                    snr_pre_avg_local[k] += snr_pre_avg_padded[idx_spectr] * h_local[idx_window];
                }

                for i in -(w_global as isize)..=(w_global as isize) {
                    let idx_window = (i + w_global as isize) as usize;
                    let idx_spectr = (i + k as isize + w_global as isize) as usize;

                    snr_pre_avg_global[k] += snr_pre_avg_padded[idx_spectr] * h_global[idx_window];
                }
            }

            let norm = T::one() / T::from(self.snr_pre_avg.len()).unwrap();
            let snr_pre_avg_frame = self.snr_pre_avg.fold(T::zero(), |a, b| a + *b * norm);

            // compute probabilities
            let snr_pre_avg_min = T::from(1e-3).unwrap();
            let snr_pre_avg_max = T::from(1e3).unwrap();
            let snr_pre_avg_peak_min = T::from(1.0).unwrap();
            let snr_pre_avg_peak_max = T::from(1e5).unwrap();

            let p_local = snr_pre_avg_local.mapv_into(|v| {
                if v <= snr_pre_avg_min {
                    T::zero()
                } else if v >= snr_pre_avg_max {
                    T::one()
                } else {
                    T::ln(v / snr_pre_avg_min) / T::ln(snr_pre_avg_max / snr_pre_avg_min)
                }
            });

            let p_global = snr_pre_avg_global.mapv_into(|v| {
                if v <= snr_pre_avg_min {
                    T::zero()
                } else if v >= snr_pre_avg_max {
                    T::one()
                } else {
                    T::ln(v / snr_pre_avg_min) / T::ln(snr_pre_avg_max / snr_pre_avg_min)
                }
            });

            let p_frame = if snr_pre_avg_frame <= snr_pre_avg_min {
                T::zero()
            } else if snr_pre_avg_frame > self.snr_pre_avg_frame {
                self.snr_pre_avg_peak = snr_pre_avg_frame
                    .max(snr_pre_avg_peak_min)
                    .min(snr_pre_avg_peak_max);

                T::one()
            } else {
                if snr_pre_avg_frame <= self.snr_pre_avg_peak * snr_pre_avg_min {
                    T::zero()
                } else if snr_pre_avg_frame >= self.snr_pre_avg_peak * snr_pre_avg_max {
                    T::one()
                } else {
                    T::ln(snr_pre_avg_frame / self.snr_pre_avg_peak / snr_pre_avg_min)
                        / T::ln(snr_pre_avg_max / snr_pre_avg_min)
                }
            };

            self.snr_pre_avg_frame = snr_pre_avg_frame;

            // speech absence probability and speech presence probability
            let p = &mut self.p_gain;
            let snr_pre = &self.snr_pre;
            let snr_post = &self.snr_post;

            azip!(mut p (p), snr_pre (snr_pre), snr_post (snr_post), p_local, p_global, in {
                // compute speech absence probability estimation
                let q = T::one() - p_local * p_global * p_frame;
                let q = q.min(T::from(0.95).unwrap());

                // compute conditional speech presence probability
                let nu = (snr_pre / (T::one() + snr_pre)) * snr_post;

                let p_ = T::one() + (q / (T::one() - q)) * (T::one() + snr_pre) * T::exp(-nu);
                let p_ = T::one() / p_;
                *p = p_;
            });
        }

        // compute gain for speech presence
        self.gain_fn.update(spec_in, self.snr_pre.view(), self.snr_post.view(), self.gain_h1.view_mut());

        // noise spectrum estimation
        self.noise_est.update(spec_in, self.noise_power.view_mut());

        // apply gain
        azip!(mut spec_out (spec_out), spec_in (spec_in), gain_h (&self.gain_h1), p (&self.p_gain) in {
            *spec_out = spec_in * gain_h.powf(p) * self.gain_min.powf(T::one() - p);
        });

        self.frame += 1;
    }
}
