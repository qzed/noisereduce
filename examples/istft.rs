use sspse::wave::WavReaderExt;
use sspse::window::{self as W, WindowFunction};
use sspse::ft;

use hound::{WavReader, Error};
use num::{Complex, traits::Zero};
use ndarray::{s, Axis, Array1};
use rustfft::{FFTplanner};
use gnuplot::{Figure, AxesCommon, AutoOption};


fn main() -> Result<(), Error> {
    let path = std::env::args_os().nth(1).expect("missing file argument");

    // load first channel of wave file
    let (samples, samples_spec) = WavReader::open(path)?.into_array_f32()?;
    let samples = samples.index_axis_move(Axis(1), 0);

    // convert real to complex
    let samples_c = samples.mapv(|v| Complex { re: v, im: 0.0 });

    // fft parameters
    let fft_len     = 512;
    let segment_len = 256;
    let overlap     = 192;

    let method = ft::InversionMethod::Weighted;
    let padded = true;

    // build window for fft
    let window = W::periodic(W::sqrt(W::hann(segment_len)));

    // make sure our window satisfies the constraint for reconstruction
    assert!(ft::check_cola(&window, overlap, ft::InversionMethod::Weighted, 1e-6));

    // build STFT and compute complex spectrum
    let mut stft = ft::StftBuilder::with_len(&window, fft_len)
        .overlap(overlap)
        .padding(ft::Padding::Zero)
        .build();

    let spectrum = stft.forward(&samples_c);

    // compute inverse STFT
    let window = window.to_array();

    let a = match method {
        ft::InversionMethod::Weighted => 1,
        ft::InversionMethod::Basic    => 0,
    };

    let hop_ratio = (segment_len - 166) as f32 / ((segment_len - 192) as f32);
    let overlap = 166;
    let new_sample_rate = (samples_spec.sample_rate as f32 * hop_ratio) as u32;

    let num_segments = spectrum.shape()[0];
    let step_len = segment_len - overlap;
    let len = segment_len + (num_segments - 1) * step_len;

    let mut out: Array1<Complex<f32>> = Array1::zeros(len);
    let mut norm: Array1<f32> = Array1::zeros(len);

    let mut buf_in: Array1<Complex<f32>>  = Array1::zeros(fft_len);
    let mut buf_out: Array1<Complex<f32>> = Array1::zeros(fft_len);

    // reset arrays
    out.fill(Complex::zero());
    norm.fill(0.0);

    let ifft = FFTplanner::new(true).plan_fft(fft_len);
    let fftnorm = 1.0 / fft_len as f32;

    // overlap and add
    for i in 0..num_segments {
        buf_in.assign(&spectrum.slice(s![i, ..]));

        ifft.process(buf_in.as_slice_mut().unwrap(), buf_out.as_slice_mut().unwrap());

        // handle fft and segment length difference (remove padding)
        let s = (fft_len - segment_len) / 2;
        let out_unpadded = buf_out.slice(s![s..s+segment_len]);

        // (weighted) overlap+add (+ fft normalization)
        for (j, v) in out_unpadded.indexed_iter() {
            out[i * step_len + j]  += v * fftnorm * window[j].powi(a);
            norm[i * step_len + j] += window[j].powi(a + 1);
        }
    }

    // remove data introduced by padding in forward STFT
    if padded {
        out.slice_collapse(s![segment_len/2..out.len()-segment_len/2]);
        norm.slice_collapse(s![segment_len/2..norm.len()-segment_len/2]);
    }

    // normalization
    for (i, v) in out.indexed_iter_mut() {
        if norm[i] > 1e-10 { *v = *v / norm[i]; }
    }

    // drop imaginary part
    let out = out.mapv(|v| v.re * hop_ratio);

    // plot
    let tx_s = ft::sample_times(samples.len(), samples_spec.sample_rate as f64);
    let tx_o = ft::sample_times(out.len(), new_sample_rate as f64);

    let mut fig = Figure::new();
    let ax = fig.axes2d();
    ax.lines(tx_s.iter(), samples.iter(), &[]);
    ax.lines(tx_o.iter(), out.iter(), &[]);
    fig.show();

    // write
    let out_spec = hound::WavSpec {
        channels: 1,
        sample_rate: new_sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer = hound::WavWriter::create("out.wav", out_spec)?;
    for x in out.iter() {
        writer.write_sample(*x)?;
    }
    writer.finalize()?;

    Ok(())
}
