use sspse::ft;
use sspse::wave::WavReaderExt;
use sspse::window as W;

use clap::{App, Arg};
use gnuplot::Figure;
use hound::{Error, WavReader};
use ndarray::Axis;
use num::Complex;


fn app() -> App<'static, 'static> {
    App::new("Example: Forward and Inverse Short-Time Fourier Transform")
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
    let (samples, samples_spec) = WavReader::open(path_in)?.collect_convert_dyn::<f32>()?;
    let samples = samples.index_axis_move(Axis(1), 0);

    // convert real to complex
    let samples_c = samples.mapv(|v| Complex { re: v, im: 0.0 });

    // fft parameters
    let fft_len     = 512;
    let segment_len = 256;
    let overlap     = 192;

    // build window for fft
    let window = W::periodic(W::sqrt(W::hann(segment_len)));

    // make sure our window satisfies the constraint for reconstruction
    assert!(ft::check_cola(&window, overlap, ft::InversionMethod::Weighted, 1e-6));

    // build STFT and compute complex spectrum
    let mut stft = ft::StftBuilder::with_len(&window, fft_len)
        .overlap(overlap)
        .padding(ft::Padding::Zero)
        .build();

    let mut istft = ft::IstftBuilder::with_len(&window, fft_len)
        .overlap(overlap)
        .method(ft::InversionMethod::Weighted)
        .remove_padding(true)
        .build();

    // perform stft and istft
    let out = istft.process(&stft.process(&samples_c));

    // drop imaginary part
    let out = out.mapv(|v| v.re);

    // plot
    if plot {
        let tx_s = ft::sample_times(samples.len(), samples_spec.sample_rate as f64);
        let tx_o = ft::sample_times(out.len(), samples_spec.sample_rate as f64);

        let mut fig = Figure::new();
        let ax = fig.axes2d();
        ax.lines(tx_s.iter(), samples.iter(), &[]);
        ax.lines(tx_o.iter(), out.iter(), &[]);
        fig.show().unwrap();
    }

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

    Ok(())
}
