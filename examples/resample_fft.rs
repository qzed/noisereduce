use rustfft::FftDirection;
use sspse::ft;
use sspse::wave::WavReaderExt;

use clap::{value_t_or_exit, Arg, Command};
use hound::{Error, WavReader};
use ndarray::Axis;
use num::Complex;


fn app() -> Command<'static> {
    Command::new("Example: Re-sample Audio Signal using Fast Fourier Transform")
        .author(clap::crate_authors!())
        .arg(Arg::with_name("input")
                .help("The input file to use (wav)")
                .value_name("INPUT")
                .required(true)
                .allow_invalid_utf8(true))
        .arg(Arg::with_name("output")
                .help("The file to write the result to (wav)")
                .value_name("OUTPUT")
                .required(true)
                .allow_invalid_utf8(true))
        .arg(Arg::with_name("samplerate")
                .help("The target sample-rate")
                .short('r')
                .long("sample-rate")
                .takes_value(true)
                .default_value("24000"))
}

fn main() -> Result<(), Error> {
    let matches = app().get_matches();

    let path_in = matches.value_of_os("input").unwrap();
    let path_out = matches.value_of_os("output").unwrap();
    let sample_rate = value_t_or_exit!(matches, "samplerate", u32);

    // load first channel of wave file
    let (samples, samples_spec) = WavReader::open(path_in)?.collect_convert_dyn::<f32>()?;
    let samples = samples.index_axis_move(Axis(1), 0);

    // convert real to complex
    let mut buf = samples.mapv(|v| Complex { re: v, im: 0.0 });

    // compute parameters based on new sample rate
    let num_input_samples = buf.len();
    let num_output_samples =
        (num_input_samples as f64 * sample_rate as f64 / samples_spec.sample_rate as f64) as usize;

    // compute fft
    let fft = rustfft::FftPlanner::new().plan_fft(num_input_samples, FftDirection::Forward);
    fft.process(buf.as_slice_mut().unwrap());

    // adapt spectrum
    let mut buf = ft::spectrum_resize(num_output_samples, &buf);

    // compute inverse fft
    let ifft = rustfft::FftPlanner::new().plan_fft(num_output_samples, FftDirection::Inverse);
    ifft.process(buf.as_slice_mut().unwrap());

    // drop imaginary part and normalize
    let norm = num_output_samples as f32 / num_input_samples as f32;
    let norm = norm / ((num_output_samples as f32).sqrt() * (num_output_samples as f32).sqrt());
    let out = buf.mapv(|v| v.re * norm);

    // write
    let out_spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer = hound::WavWriter::create(path_out, out_spec)?;
    for x in out.iter() {
        writer.write_sample(*x)?;
    }
    writer.finalize()?;

    Ok(())
}
