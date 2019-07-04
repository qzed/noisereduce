// based on https://www.mathworks.com/help/signal/ref/istft.html
// implements synchronous overlap and add signal streching
// see also https://dspace.cvut.cz/bitstream/handle/10467/77279/F8-BP-2018-Onderka-Jan-thesis.pdf?sequence=-1&isAllowed=y

use sspse::wave::WavReaderExt;
use sspse::window as W;
use sspse::ft;

use hound::{WavReader, Error};
use num::Complex;
use ndarray::Axis;
use gnuplot::Figure;
use clap::{App, Arg};


fn app() -> App<'static, 'static> {
    App::new("Example: Pitch modification via Short-Time Fourier Transform")
        .author(clap::crate_authors!())
        .arg(Arg::with_name("input")
            .help("The input file to use (wav)")
            .value_name("INPUT")
            .required(true))
        .arg(Arg::with_name("output")
            .value_name("OUTPUT")
            .help("The file to write the result to (wav)")
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
    let fft_len     = 256;
    let segment_len = 256;
    let overlap_f   = 192;
    let overlap_s   = 166;

    let hop_ratio = (segment_len - overlap_s) as f32 / ((segment_len - overlap_f) as f32);
    let new_sample_rate = (samples_spec.sample_rate as f32 * hop_ratio) as u32;

    // build window for fft
    let window = W::periodic(W::sqrt(W::hann(segment_len)));

    // build STFT and compute complex spectrum
    let mut stft = ft::StftBuilder::with_len(&window, fft_len)
        .overlap(overlap_f)
        .padding(ft::Padding::Zero)
        .build();

    let mut istft = ft::IstftBuilder::with_len(&window, fft_len)
        .overlap(overlap_s)
        .method(ft::InversionMethod::Weighted)
        .remove_padding(true)
        .build();

    let out = istft.process(&stft.process(&samples_c));

    // drop imaginary part, scale for overlap-difference
    let out = out.mapv(|v| v.re * hop_ratio);

    // plot
    if plot {
        let tx_s = ft::sample_times(samples.len(), samples_spec.sample_rate as f64);
        let tx_o = ft::sample_times(out.len(), new_sample_rate as f64);

        let mut fig = Figure::new();
        let ax = fig.axes2d();
        ax.lines(tx_s.iter(), samples.iter(), &[]);
        ax.lines(tx_o.iter(), out.iter(), &[]);
        fig.show();
    }

    // write
    if let Some(path_out) = path_out {
        let out_spec = hound::WavSpec {
            channels: 1,
            sample_rate: new_sample_rate,
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
