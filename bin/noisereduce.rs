use sspse::ft::{Istft, IstftBuilder, Stft, StftBuilder};
use sspse::math::NumCastUnchecked;
use sspse::proc;
use sspse::stsa::{self, NoiseReductionProcessor, Gain, SnrEstimator, NoiseTracker};
use sspse::stsa::{Subtraction, Stsa, ModStsa};
use sspse::stsa::gain::{Mmse, LogMmse};
use sspse::stsa::noise::{ExpTimeAvg, ProbabilisticExpTimeAvg};
use sspse::stsa::snr::{DecisionDirected, MaximumLikelihood};
use sspse::utils;
use sspse::vad::b::SpeechProbabilityEstimator as PEstimator;
use sspse::vad::b::power::{self, PowerThresholdVad};
use sspse::vad::b::mc::MinimaControlledVad;
use sspse::vad::b::soft::SoftDecisionProbabilityEstimator;
use sspse::vad::f::energy::{self, EnergyThresholdVad};
use sspse::wave::WavReaderExt;
use sspse::window::{self, WindowFunction};


use clap::{App, Arg};
use ndarray::{s, Axis, ArrayBase, Array1, Data, Ix2};
use num::{Complex, Float, traits::FloatConst, traits::NumAssign};
use rustfft::FFTnum;
use hound::WavReader;
use serde::{Deserialize, Serialize};


#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Parameters {
    #[serde(default)]
    stft: StftParameters,

    algorithm: Algorithm,

    #[serde(default)]
    noise: NoiseInit,
}


#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StftParameters {
    #[serde(default = "param_defaults::stft_window")]
    window: Window,

    #[serde(default = "param_defaults::block_length")]
    block_length: f64,          // in seconds

    #[serde(default)]
    overlap: Option<f64>,

    #[serde(default)]
    fft_length: Option<usize>,
}

impl Default for StftParameters {
    fn default() -> Self {
        StftParameters {
            window: Window::default(),
            block_length: param_defaults::block_length(),
            overlap: None,
            fft_length: None,
        }
    }
}


#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub struct Window {
    #[serde(default, flatten)]
    ty: WindowType,

    #[serde(default)]
    periodic: bool,

    #[serde(default)]
    sqrt: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "kebab-case", tag = "type")]
pub enum WindowType {
    Rectangular,
    Triangular { l: usize },
    Bartlett,
    Hann,
    Hamming,
    Blackman,
    BlackmanExact,
    BlackmanAlpha { alpha: f64 },
    Nuttall,
    BlackmanNuttall,
    BlackmanHarris,
    FlatTop,
    Gaussian { sigma: f64 },
    GaussianConfined { sigma_t: f64 },
    GeneralizedNormal { sigma: f64, p: f64 },
    Tukey { alpha: f64 },
    Kaiser { pa: f64 },
}

impl Default for WindowType {
    fn default() -> Self {
        WindowType::Hamming
    }
}


#[allow(clippy::large_enum_variant)]
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "kebab-case", tag = "type")]
pub enum Algorithm {
    SpectralSubtraction {
        #[serde(flatten, default)]
        params: SpectralSubtractionParams,
    },

    Mmse {
        #[serde(flatten, default)]
        params: MmseParams,
    },

    OmLsa {
        #[serde(flatten, default)]
        params: OmLsaParams,
    }
}


#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SpectralSubtractionParams {
    #[serde(default = "param_defaults::spectral_subtraction_factor")]
    factor: f64,

    #[serde(default = "param_defaults::spectral_subtraction_post_gain")]
    post_gain: f64,
}

impl Default for SpectralSubtractionParams {
    fn default() -> Self {
        SpectralSubtractionParams {
            factor: 1.0,
            post_gain: 1.0,
        }
    }
}


#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct MmseParams {
    #[serde(default)]
    gain: GainFn,

    #[serde(default)]
    snr_estimator: SnrEstimatorParams,

    #[serde(default)]
    noise_estimator: NoiseEstimator,
}


#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "kebab-case", tag = "type")]
pub enum NoiseEstimator {
    ExpTimeAvg {
        #[serde(default)]
        vad: Vad,

        #[serde(default = "param_defaults::mmse_noise_est_alpha")]
        alpha: f64,
    },

    ProbabilisticExpTimeAvg {
        #[serde(default)]
        vad: ProbabilisticVad,

        #[serde(default = "param_defaults::omlsa_noise_est_alpha_d")]
        alpha_d: f64,
    }
}

impl Default for NoiseEstimator {
    fn default() -> Self {
        NoiseEstimator::ExpTimeAvg {
            vad: Vad::default(),
            alpha: param_defaults::mmse_noise_est_alpha(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "kebab-case", tag = "type")]
pub enum Vad {
    PowerThreshold {
        #[serde(default = "param_defaults::mmse_vad_decision_ratio")]
        decision_ratio: f64,

        #[serde(default)]
        noise: NoiseInit,
    },
    EnergyThreshold {
        #[serde(default = "param_defaults::mmse_vad_decision_ratio")]
        decision_ratio: f64,

        #[serde(default)]
        noise: NoiseInit,
    },
}

impl Default for Vad {
    fn default() -> Self {
        Vad::PowerThreshold {
            decision_ratio: param_defaults::mmse_vad_decision_ratio(),
            noise: NoiseInit::default(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "kebab-case", tag = "type")]
pub enum ProbabilisticVad {
    MinimaControlled {
        #[serde(default = "param_defaults::minima_controlled_b")]
        b: Window,

        #[serde(default = "param_defaults::minima_controlled_w")]
        w: usize,

        #[serde(default = "param_defaults::minima_controlled_alpha_s")]
        alpha_s: f64,

        #[serde(default = "param_defaults::minima_controlled_alpha_p")]
        alpha_p: f64,

        #[serde(default = "param_defaults::minima_controlled_delta")]
        delta: f64,

        #[serde(default = "param_defaults::minima_controlled_l")]
        l: usize,
    }
}

impl Default for ProbabilisticVad {
    fn default() -> Self {
        ProbabilisticVad::MinimaControlled {
            b: param_defaults::minima_controlled_b(),
            w: param_defaults::minima_controlled_w(),
            alpha_s: param_defaults::minima_controlled_alpha_s(),
            alpha_p: param_defaults::minima_controlled_alpha_p(),
            delta: param_defaults::minima_controlled_delta(),
            l: param_defaults::minima_controlled_l(),
        }
    }
}


#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "kebab-case", tag = "type")]
pub enum SnrEstimatorParams {
    DecisionDirected { alpha: f64 },
    MaximumLikelihood { alpha: f64, beta: f64 },
}

impl Default for SnrEstimatorParams {
    fn default() -> Self {
        SnrEstimatorParams::DecisionDirected { alpha: 0.97 }
    }
}


#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all="kebab-case", tag = "type")]
pub enum GainFn {
    Mmse {
        #[serde(default = "param_defaults::mmse_nu_min")]
        nu_min: f64,

        #[serde(default = "param_defaults::mmse_nu_max")]
        nu_max: f64,
    },
    LogMmse {
        #[serde(default = "param_defaults::mmse_nu_min")]
        nu_min: f64,

        #[serde(default = "param_defaults::mmse_nu_max")]
        nu_max: f64,
    },
}

impl Default for GainFn {
    fn default() -> Self {
        GainFn::Mmse {
            nu_min: param_defaults::mmse_nu_min(),
            nu_max: param_defaults::mmse_nu_max(),
        }
    }
}


#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OmLsaParams {
    #[serde(default = "param_defaults::omlsa_gain")]
    gain: GainFn,

    #[serde(default = "param_defaults::omlsa_gain_min")]
    gain_min: f64,

    #[serde(default = "param_defaults::omlsa_snr_estimator")]
    snr_estimator: SnrEstimatorParams,

    #[serde(default = "param_defaults::omlsa_noise_estimator")]
    noise_estimator: NoiseEstimator,

    #[serde(default = "param_defaults::omlsa_p_estimator")]
    p_estimator: SpeechProbabilityEstimator,
}

impl Default for OmLsaParams {
    fn default() -> Self {
        OmLsaParams {
            gain: param_defaults::omlsa_gain(),
            gain_min: param_defaults::omlsa_gain_min(),
            snr_estimator: param_defaults::omlsa_snr_estimator(),
            noise_estimator: param_defaults::omlsa_noise_estimator(),
            p_estimator: param_defaults::omlsa_p_estimator(),
        }
    }
}


#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "kebab-case", tag = "type")]
pub enum SpeechProbabilityEstimator {
    SoftDecision {
        #[serde(default = "param_defaults::soft_decision_beta")]
        beta: f64,

        #[serde(default = "param_defaults::soft_decision_w_local")]
        w_local: usize,

        #[serde(default = "param_defaults::soft_decision_h_local")]
        h_local: Window,

        #[serde(default = "param_defaults::soft_decision_w_global")]
        w_global: usize,

        #[serde(default = "param_defaults::soft_decision_h_global")]
        h_global: Window,

        #[serde(default = "param_defaults::soft_decision_zeta_min")]
        zeta_min: f64,

        #[serde(default = "param_defaults::soft_decision_zeta_max")]
        zeta_max: f64,

        #[serde(default = "param_defaults::soft_decision_zeta_peak_min")]
        zeta_peak_min: f64,

        #[serde(default = "param_defaults::soft_decision_zeta_peak_max")]
        zeta_peak_max: f64,

        #[serde(default = "param_defaults::soft_decision_q_max")]
        q_max: f64,
    }
}

impl Default for SpeechProbabilityEstimator {
    fn default() -> Self {
        SpeechProbabilityEstimator::SoftDecision {
            beta: param_defaults::soft_decision_beta(),
            w_local: param_defaults::soft_decision_w_local(),
            h_local: param_defaults::soft_decision_h_local(),
            w_global: param_defaults::soft_decision_w_global(),
            h_global: param_defaults::soft_decision_h_global(),
            zeta_min: param_defaults::soft_decision_zeta_min(),
            zeta_max: param_defaults::soft_decision_zeta_max(),
            zeta_peak_min: param_defaults::soft_decision_zeta_peak_min(),
            zeta_peak_max: param_defaults::soft_decision_zeta_peak_max(),
            q_max: param_defaults::soft_decision_q_max(),
        }
    }
}


#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "kebab-case", tag = "type")]
pub enum NoiseInit {
    Zero,
    Frames { range: (usize, usize) },
}

impl Default for NoiseInit {
    fn default() -> Self {
        NoiseInit::Zero
    }
}


mod param_defaults {
    use super::*;

    pub fn stft_window() -> Window {
        Window {
            ty: WindowType::Hamming,
            periodic: true,
            sqrt: true,
        }
    }

    pub fn block_length() -> f64 {
        0.020       // 20 ms
    }

    pub fn spectral_subtraction_factor() -> f64 {
        1.0
    }

    pub fn spectral_subtraction_post_gain() -> f64 {
        1.0
    }

    pub fn mmse_nu_min() -> f64 {
        1e-50
    }

    pub fn mmse_nu_max() -> f64 {
        500.0
    }

    pub fn mmse_vad_decision_ratio() -> f64 {
        1.3
    }

    pub fn mmse_noise_est_alpha() -> f64 {
        0.8
    }

    pub fn omlsa_noise_est_alpha_d() -> f64 {
        0.95
    }

    pub fn minima_controlled_b() -> Window {
        Window {
            ty: WindowType::Hamming,
            periodic: false,
            sqrt: false,
        }
    }

    pub fn minima_controlled_w() -> usize {
        1
    }

    pub fn minima_controlled_alpha_s() -> f64 {
        0.8
    }

    pub fn minima_controlled_alpha_p() -> f64 {
        0.2
    }

    pub fn minima_controlled_delta() -> f64 {
        5.0
    }

    pub fn minima_controlled_l() -> usize {
        125
    }

    pub fn omlsa_gain() -> GainFn {
        GainFn::LogMmse {
            nu_min: mmse_nu_min(),
            nu_max: mmse_nu_max(),
        }
    }

    pub fn omlsa_gain_min() -> f64 {
        0.001
    }

    pub fn omlsa_snr_estimator() -> SnrEstimatorParams {
        SnrEstimatorParams::DecisionDirected {
            alpha: 0.92
        }
    }

    pub fn omlsa_noise_estimator() -> NoiseEstimator {
        NoiseEstimator::ProbabilisticExpTimeAvg {
            vad: ProbabilisticVad::default(),
            alpha_d: 0.95,
        }
    }

    pub fn omlsa_p_estimator() -> SpeechProbabilityEstimator {
        SpeechProbabilityEstimator::default()
    }

    pub fn soft_decision_beta() -> f64 {
        0.7
    }

    pub fn soft_decision_w_local() -> usize {
        1
    }

    pub fn soft_decision_h_local() -> Window {
        Window::default()
    }

    pub fn soft_decision_w_global() -> usize {
        15
    }

    pub fn soft_decision_h_global() -> Window {
        Window::default()
    }

    pub fn soft_decision_zeta_min() -> f64 {
        1e-3
    }

    pub fn soft_decision_zeta_max() -> f64 {
        1e3
    }

    pub fn soft_decision_zeta_peak_min() -> f64 {
        1.0
    }

    pub fn soft_decision_zeta_peak_max() -> f64 {
        1e5
    }

    pub fn soft_decision_q_max() -> f64 {
        0.95
    }
}



fn create_window<'a, T>(window: &Window, len: usize) -> Box<dyn WindowFunction<T> + 'a>
where
    T: Float + FloatConst + NumCastUnchecked + 'a,
{
    match (window.sqrt, window.periodic) {
        (false, false) => create_window_ty(&window.ty, len),
        (true, false)  => create_window_ty_sq(&window.ty, len),
        (false, true)  => create_window_ty_p(&window.ty, len),
        (true, true)   => create_window_ty_sqp(&window.ty, len),
    }
}

fn create_window_ty<'a, T>(ty: &WindowType, len: usize) -> Box<dyn WindowFunction<T> + 'a>
where
    T: Float + FloatConst + NumCastUnchecked + 'a,
{
    match ty {
        WindowType::Rectangular                     => Box::new(window::rectangular(len)),
        WindowType::Triangular { l }                => Box::new(window::triangular(len, *l)),
        WindowType::Bartlett                        => Box::new(window::bartlett(len)),
        WindowType::Hann                            => Box::new(window::hann(len)),
        WindowType::Hamming                         => Box::new(window::hamming(len)),
        WindowType::Blackman                        => Box::new(window::blackman(len)),
        WindowType::BlackmanExact                   => Box::new(window::blackman_exact(len)),
        WindowType::BlackmanAlpha { alpha }         => Box::new(window::blackman_alpha(len, T::from_unchecked(*alpha))),
        WindowType::Nuttall                         => Box::new(window::nuttall(len)),
        WindowType::BlackmanNuttall                 => Box::new(window::blackman_nuttall(len)),
        WindowType::BlackmanHarris                  => Box::new(window::blackman_harris(len)),
        WindowType::FlatTop                         => Box::new(window::flat_top(len)),
        WindowType::Gaussian { sigma }              => Box::new(window::gaussian(len, T::from_unchecked(*sigma))),
        WindowType::GaussianConfined { sigma_t }    => Box::new(window::gaussian_confined(len, T::from_unchecked(*sigma_t))),
        WindowType::GeneralizedNormal { sigma, p }  => Box::new(window::generalized_normal(len, T::from_unchecked(*sigma), T::from_unchecked(*p))),
        WindowType::Tukey { alpha }                 => Box::new(window::tukey(len, T::from_unchecked(*alpha))),
        WindowType::Kaiser { pa }                   => Box::new(window::kaiser(len, T::from_unchecked(*pa))),
    }
}

fn create_window_ty_sqp<'a, T>(ty: &WindowType, len: usize) -> Box<dyn WindowFunction<T> + 'a>
where
    T: Float + FloatConst + NumCastUnchecked + 'a,
{
    match ty {
        WindowType::Rectangular                     => Box::new(window::periodic(window::sqrt(window::rectangular(len)))),
        WindowType::Triangular { l }                => Box::new(window::periodic(window::sqrt(window::triangular(len, *l)))),
        WindowType::Bartlett                        => Box::new(window::periodic(window::sqrt(window::bartlett(len)))),
        WindowType::Hann                            => Box::new(window::periodic(window::sqrt(window::hann(len)))),
        WindowType::Hamming                         => Box::new(window::periodic(window::sqrt(window::hamming(len)))),
        WindowType::Blackman                        => Box::new(window::periodic(window::sqrt(window::blackman(len)))),
        WindowType::BlackmanExact                   => Box::new(window::periodic(window::sqrt(window::blackman_exact(len)))),
        WindowType::BlackmanAlpha { alpha }         => Box::new(window::periodic(window::sqrt(window::blackman_alpha(len, T::from_unchecked(*alpha))))),
        WindowType::Nuttall                         => Box::new(window::periodic(window::sqrt(window::nuttall(len)))),
        WindowType::BlackmanNuttall                 => Box::new(window::periodic(window::sqrt(window::blackman_nuttall(len)))),
        WindowType::BlackmanHarris                  => Box::new(window::periodic(window::sqrt(window::blackman_harris(len)))),
        WindowType::FlatTop                         => Box::new(window::periodic(window::sqrt(window::flat_top(len)))),
        WindowType::Gaussian { sigma }              => Box::new(window::periodic(window::sqrt(window::gaussian(len, T::from_unchecked(*sigma))))),
        WindowType::GaussianConfined { sigma_t }    => Box::new(window::periodic(window::sqrt(window::gaussian_confined(len, T::from_unchecked(*sigma_t))))),
        WindowType::GeneralizedNormal { sigma, p }  => Box::new(window::periodic(window::sqrt(window::generalized_normal(len, T::from_unchecked(*sigma), T::from_unchecked(*p))))),
        WindowType::Tukey { alpha }                 => Box::new(window::periodic(window::sqrt(window::tukey(len, T::from_unchecked(*alpha))))),
        WindowType::Kaiser { pa }                   => Box::new(window::periodic(window::sqrt(window::kaiser(len, T::from_unchecked(*pa))))),
    }
}

fn create_window_ty_p<'a, T>(ty: &WindowType, len: usize) -> Box<dyn WindowFunction<T> + 'a>
where
    T: Float + FloatConst + NumCastUnchecked + 'a,
{
    match ty {
        WindowType::Rectangular                     => Box::new(window::periodic(window::rectangular(len))),
        WindowType::Triangular { l }                => Box::new(window::periodic(window::triangular(len, *l))),
        WindowType::Bartlett                        => Box::new(window::periodic(window::bartlett(len))),
        WindowType::Hann                            => Box::new(window::periodic(window::hann(len))),
        WindowType::Hamming                         => Box::new(window::periodic(window::hamming(len))),
        WindowType::Blackman                        => Box::new(window::periodic(window::blackman(len))),
        WindowType::BlackmanExact                   => Box::new(window::periodic(window::blackman_exact(len))),
        WindowType::BlackmanAlpha { alpha }         => Box::new(window::periodic(window::blackman_alpha(len, T::from_unchecked(*alpha)))),
        WindowType::Nuttall                         => Box::new(window::periodic(window::nuttall(len))),
        WindowType::BlackmanNuttall                 => Box::new(window::periodic(window::blackman_nuttall(len))),
        WindowType::BlackmanHarris                  => Box::new(window::periodic(window::blackman_harris(len))),
        WindowType::FlatTop                         => Box::new(window::periodic(window::flat_top(len))),
        WindowType::Gaussian { sigma }              => Box::new(window::periodic(window::gaussian(len, T::from_unchecked(*sigma)))),
        WindowType::GaussianConfined { sigma_t }    => Box::new(window::periodic(window::gaussian_confined(len, T::from_unchecked(*sigma_t)))),
        WindowType::GeneralizedNormal { sigma, p }  => Box::new(window::periodic(window::generalized_normal(len, T::from_unchecked(*sigma), T::from_unchecked(*p)))),
        WindowType::Tukey { alpha }                 => Box::new(window::periodic(window::tukey(len, T::from_unchecked(*alpha)))),
        WindowType::Kaiser { pa }                   => Box::new(window::periodic(window::kaiser(len, T::from_unchecked(*pa)))),
    }
}

fn create_window_ty_sq<'a, T>(ty: &WindowType, len: usize) -> Box<dyn WindowFunction<T> + 'a>
where
    T: Float + FloatConst + NumCastUnchecked + 'a,
{
    match ty {
        WindowType::Rectangular                     => Box::new(window::sqrt(window::rectangular(len))),
        WindowType::Triangular { l }                => Box::new(window::sqrt(window::triangular(len, *l))),
        WindowType::Bartlett                        => Box::new(window::sqrt(window::bartlett(len))),
        WindowType::Hann                            => Box::new(window::sqrt(window::hann(len))),
        WindowType::Hamming                         => Box::new(window::sqrt(window::hamming(len))),
        WindowType::Blackman                        => Box::new(window::sqrt(window::blackman(len))),
        WindowType::BlackmanExact                   => Box::new(window::sqrt(window::blackman_exact(len))),
        WindowType::BlackmanAlpha { alpha }         => Box::new(window::sqrt(window::blackman_alpha(len, T::from_unchecked(*alpha)))),
        WindowType::Nuttall                         => Box::new(window::sqrt(window::nuttall(len))),
        WindowType::BlackmanNuttall                 => Box::new(window::sqrt(window::blackman_nuttall(len))),
        WindowType::BlackmanHarris                  => Box::new(window::sqrt(window::blackman_harris(len))),
        WindowType::FlatTop                         => Box::new(window::sqrt(window::flat_top(len))),
        WindowType::Gaussian { sigma }              => Box::new(window::sqrt(window::gaussian(len, T::from_unchecked(*sigma)))),
        WindowType::GaussianConfined { sigma_t }    => Box::new(window::sqrt(window::gaussian_confined(len, T::from_unchecked(*sigma_t)))),
        WindowType::GeneralizedNormal { sigma, p }  => Box::new(window::sqrt(window::generalized_normal(len, T::from_unchecked(*sigma), T::from_unchecked(*p)))),
        WindowType::Tukey { alpha }                 => Box::new(window::sqrt(window::tukey(len, T::from_unchecked(*alpha)))),
        WindowType::Kaiser { pa }                   => Box::new(window::sqrt(window::kaiser(len, T::from_unchecked(*pa)))),
    }
}

#[allow(clippy::cast_lossless)]
fn build_stft<T>(params: &StftParameters, sample_rate: u32) -> Stft<T>
where
    T: Float + FloatConst + NumCastUnchecked + FFTnum,
{
    let block_length = (params.block_length * sample_rate as f64) as usize;
    let fft_length = params.fft_length.unwrap_or(block_length);
    let overlap = (params.overlap.unwrap_or(0.5) * block_length as f64) as usize;
    let window = create_window(&params.window, block_length);

    StftBuilder::with_len(window.as_ref(), fft_length)
        .overlap(overlap)
        .padding(sspse::ft::Padding::Zero)
        .build()
}

#[allow(clippy::cast_lossless)]
fn build_istft<T>(params: &StftParameters, sample_rate: u32) -> Istft<T>
where
    T: Float + FloatConst + NumCastUnchecked + FFTnum,
{
    let block_length = (params.block_length * sample_rate as f64) as usize;
    let fft_length = params.fft_length.unwrap_or(block_length);
    let overlap = (params.overlap.unwrap_or(0.5) * block_length as f64) as usize;
    let window = create_window(&params.window, block_length);

    IstftBuilder::with_len(window.as_ref(), fft_length)
        .overlap(overlap)
        .method(sspse::ft::InversionMethod::Weighted)
        .remove_padding(true)
        .build()
}

fn build_spectral_subtraction<T, D>(params: &SpectralSubtractionParams, spectrum: &ArrayBase<D, Ix2>)
    -> Box<dyn NoiseReductionProcessor<T>>
where
    T: Float + NumCastUnchecked + 'static,
    D: Data<Elem = Complex<T>>,
{
    let factor = T::from_unchecked(params.factor);
    let post_gain = T::from_unchecked(params.post_gain);

    Box::new(Subtraction::new(spectrum.dim().1, factor, post_gain))
}

fn build_mmse<T, D>(params: &MmseParams, spectrum: &ArrayBase<D, Ix2>)
    -> Box<dyn NoiseReductionProcessor<T>>
where
    T: Float + FloatConst + NumCastUnchecked + NumAssign + 'static,
    D: Data<Elem = Complex<T>>,
{
    let block_size = spectrum.dim().1;

    let gain = build_gain(&params.gain);
    let snr_est = build_snr_estimator(&params.snr_estimator, block_size);
    let noise_est = build_noise_estimator(&params.noise_estimator, spectrum);

    Box::new(Stsa::new(block_size, gain, snr_est, noise_est))
}

fn build_omlsa<T, D>(params: &OmLsaParams, spectrum: &ArrayBase<D, Ix2>)
    -> Box<dyn NoiseReductionProcessor<T>>
where
    T: Float + FloatConst + NumCastUnchecked + NumAssign + 'static,
    D: Data<Elem = Complex<T>>,
{
    let block_size = spectrum.dim().1;

    let gain = build_gain(&params.gain);
    let gain_min = T::from_unchecked(params.gain_min);
    let snr_est = build_snr_estimator(&params.snr_estimator, block_size);
    let noise_est = build_noise_estimator(&params.noise_estimator, spectrum);
    let p_est = build_p_estimator(&params.p_estimator, block_size);

    Box::new(ModStsa::new(block_size, noise_est, p_est, snr_est, gain, gain_min))
}

fn build_processor<T, D>(params: &Algorithm, spectrum: &ArrayBase<D, Ix2>)
    -> Box<dyn NoiseReductionProcessor<T>>
where
    T: Float + FloatConst + NumAssign + NumCastUnchecked + 'static,
    D: Data<Elem = Complex<T>>,
{
    match params {
        Algorithm::SpectralSubtraction { params } => build_spectral_subtraction(params, spectrum),
        Algorithm::Mmse { params }                => build_mmse(params, spectrum),
        Algorithm::OmLsa { params }               => build_omlsa(params, spectrum),
    }
}

fn build_gain<T>(params: &GainFn) -> Box<dyn Gain<T>>
where
    T: Float + FloatConst + NumCastUnchecked + 'static,
{
    match params {
        GainFn::Mmse { nu_min, nu_max } => {
            let nu_min = T::from_unchecked(*nu_min);
            let nu_max = T::from_unchecked(*nu_max);

            Box::new(Mmse::new(nu_min, nu_max))
        },
        GainFn::LogMmse { nu_min, nu_max } => {
            let nu_min = T::from_unchecked(*nu_min);
            let nu_max = T::from_unchecked(*nu_max);

            Box::new(LogMmse::new(nu_min, nu_max))
        },
    }
}

fn build_snr_estimator<T>(params: &SnrEstimatorParams, block_size: usize) -> Box<dyn SnrEstimator<T>>
where
    T: Float + FloatConst + NumCastUnchecked + 'static,
{
    match params {
        SnrEstimatorParams::DecisionDirected { alpha } => {
            let alpha = T::from_unchecked(*alpha);

            Box::new(DecisionDirected::new(alpha))
        },
        SnrEstimatorParams::MaximumLikelihood { alpha, beta } => {
            let alpha = T::from_unchecked(*alpha);
            let beta = T::from_unchecked(*beta);

            Box::new(MaximumLikelihood::new(block_size, alpha, beta))
        },
    }
}

#[allow(clippy::deref_addrof)]
fn build_noise_estimator<T, D>(params: &NoiseEstimator, spectrum: &ArrayBase<D, Ix2>)
    -> Box<dyn NoiseTracker<T>>
where
    T: Float + FloatConst + NumCastUnchecked + NumAssign + 'static,
    D: Data<Elem = Complex<T>>,
{
    let block_size = spectrum.dim().1;

    match params {
        NoiseEstimator::ExpTimeAvg { vad, alpha } => {
            let alpha = T::from_unchecked(*alpha);

            match vad {
                Vad::PowerThreshold { decision_ratio, noise } => {
                    let noise_floor = match noise {
                        NoiseInit::Zero => Array1::zeros(block_size),
                        NoiseInit::Frames { range } => {
                            power::noise_floor_est(&spectrum.slice(s![range.0..range.1, ..]))
                        }
                    };

                    let decision_ratio = T::from_unchecked(*decision_ratio);
                    let vad = PowerThresholdVad::new(noise_floor, decision_ratio);

                    Box::new(ExpTimeAvg::new(block_size, alpha, vad))
                },
                Vad::EnergyThreshold { decision_ratio, noise } => {
                    use sspse::vad::f::VoiceActivityDetector;

                    let noise_floor = match noise {
                        NoiseInit::Zero => T::zero(),
                        NoiseInit::Frames { range } => {
                            energy::noise_floor_est(&spectrum.slice(s![range.0..range.1, ..]))
                        }
                    };

                    let decision_ratio = T::from_unchecked(*decision_ratio);
                    let vad = EnergyThresholdVad::new(noise_floor, decision_ratio).per_band();

                    Box::new(ExpTimeAvg::new(block_size, alpha, vad))
                },
            }
        },
        NoiseEstimator::ProbabilisticExpTimeAvg { vad, alpha_d } => {
            let alpha_d = T::from_unchecked(*alpha_d);

            let vad: MinimaControlledVad<T> = match vad {
                ProbabilisticVad::MinimaControlled { b, w, alpha_s, alpha_p, delta, l } => {
                    let b = create_window(b, 2 * w + 1);
                    let alpha_s = T::from_unchecked(*alpha_s);
                    let alpha_p = T::from_unchecked(*alpha_p);
                    let delta = T::from_unchecked(*delta);

                    MinimaControlledVad::new(block_size, b.as_ref(), alpha_s, alpha_p, delta, *l)
                }
            };

            Box::new(ProbabilisticExpTimeAvg::new(block_size, alpha_d, vad))
        },
    }
}

fn build_p_estimator<T>(params: &SpeechProbabilityEstimator, block_size: usize) -> impl PEstimator<T>
where
    T: Float + FloatConst + NumAssign + NumCastUnchecked + 'static,
{
    match params {
        SpeechProbabilityEstimator::SoftDecision {
            beta, w_local, h_local, w_global, h_global, zeta_min, zeta_max,
            zeta_peak_min, zeta_peak_max, q_max
        } => {
            let beta = T::from_unchecked(*beta);
            let zeta_min = T::from_unchecked(*zeta_min);
            let zeta_max = T::from_unchecked(*zeta_max);
            let zeta_peak_min = T::from_unchecked(*zeta_peak_min);
            let zeta_peak_max = T::from_unchecked(*zeta_peak_max);
            let q_max = T::from_unchecked(*q_max);

            let h_local = create_window(h_local, w_local * 2 + 1);
            let h_global = create_window(h_global, w_global * 2 + 1);

            SoftDecisionProbabilityEstimator::new(
                block_size,
                beta,
                h_local.as_ref(),
                h_global.as_ref(),
                zeta_min,
                zeta_max,
                zeta_peak_min,
                zeta_peak_max,
                q_max,
            )
        }
    }
}

#[allow(clippy::deref_addrof)]
fn compute_noise_estimate<T, D>(params: &Parameters, spectrum: &ArrayBase<D, Ix2>) -> Array1<T>
where
    T: Float + std::ops::AddAssign + ndarray::ScalarOperand,
    D: Data<Elem = Complex<T>>,
{
    match params.noise {
        NoiseInit::Zero => {
            Array1::zeros(spectrum.dim().1)
        },
        NoiseInit::Frames { range } => {
            let frames = spectrum.slice(s![range.0..range.1, ..]);

            match params.algorithm {
                Algorithm::SpectralSubtraction { .. } => {
                    stsa::utils::noise_amplitude_est(&frames)
                },
                Algorithm::OmLsa { .. } | Algorithm::Mmse { .. } => {
                    stsa::utils::noise_power_est(&frames)
                },
            }
        }
    }

}


fn app() -> App<'static, 'static> {
    App::new("Example: Noise reduction via MMSE/log-MMSE STSA Method")
        .author(clap::crate_authors!())
        .version(clap::crate_version!())
        .arg(Arg::with_name("input")
                .help("The input file to use (wav)")
                .value_name("INPUT")
                .required(true))
        .arg(Arg::with_name("output")
                .help("The file to write the result to (wav)")
                .value_name("OUTPUT")
                .required(false))
        .arg(Arg::with_name("params")
                .help("The parameters to use (as yaml file)")
                .value_name("PARAMS")
                .short("p")
                .long("params")
                .required(true))
        .arg(Arg::with_name("show")
                .help("Wheter to plot the spectras or not")
                .short("s")
                .long("show"))
}

fn main() {
    let matches = app().get_matches();
    let path_in     = matches.value_of_os("input").unwrap();
    let path_out    = matches.value_of_os("output");
    let path_params = matches.value_of_os("params").unwrap();
    let show        = matches.is_present("show");

    // load parameters
    let params = std::fs::File::open(path_params).unwrap();
    let params: Parameters = serde_yaml::from_reader(params).unwrap();

    // load first channel of wave file
    let (samples, samples_spec) = WavReader::open(path_in).unwrap().collect_convert_dyn::<f64>().unwrap();
    let samples = samples.index_axis_move(Axis(1), 0);
    let samples_c = samples.mapv(|v| Complex { re: v, im: 0.0 });

    // build STFT/ISTFT
    let mut stft = build_stft(&params.stft, samples_spec.sample_rate);
    let mut istft = build_istft(&params.stft, samples_spec.sample_rate);

    // compute spectrum
    let spectrum_in = stft.process(&samples_c);

    // set-up algorithm
    let mut processor = build_processor(&params.algorithm, &spectrum_in);

    // compute and set initial noise estimate
    let noise_est = compute_noise_estimate(&params, &spectrum_in);
    processor.set_noise_estimate(noise_est.view());

    // run algorithm
    let spectrum_out = proc::utils::process_spectrum(processor.as_mut(), &spectrum_in);

    // compute signal
    let out = istft.process(&spectrum_out);
    let out = out.mapv(|v| v.re);

    // write
    if let Some(path_out) = path_out {
        utils::write_wav(path_out, &out, samples_spec.sample_rate).unwrap();
    }

    // plot
    if show {
        utils::plot_spectras(&spectrum_in, &spectrum_out, &stft, samples.len(), samples_spec.sample_rate);
    }
}

// TODO: csv/json data-dump?
