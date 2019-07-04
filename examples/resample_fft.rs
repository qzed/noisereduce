use sspse::ft;
use sspse::wave::WavReaderExt;

use clap::{value_t_or_exit, App, Arg};
use hound::{Error, WavReader};
use ndarray::{Array1, Axis};
use num::Complex;


fn app() -> App<'static, 'static> {
    App::new("Example: Re-sample Audio Signal using Fast Fourier Transform")
        .author(clap::crate_authors!())
        .arg(Arg::with_name("input")
                .help("The input file to use (wav)")
                .value_name("INPUT")
                .required(true))
        .arg(Arg::with_name("output")
                .help("The file to write the result to (wav)")
                .value_name("OUTPUT")
                .required(true))
        .arg(Arg::with_name("samplerate")
                .help("The target sample-rate")
                .short("r")
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
    let mut input = samples.mapv(|v| Complex { re: v, im: 0.0 });

    // compute parameters based on new sample rate
    let num_input_samples = input.len();
    let num_output_samples =
        (num_input_samples as f64 * sample_rate as f64 / samples_spec.sample_rate as f64) as usize;

    // compute fft
    let mut spectrum = Array1::zeros(num_input_samples);
    let mut output = Array1::zeros(num_output_samples);

    let fft = rustfft::FFTplanner::new(false).plan_fft(num_input_samples);
    fft.process(input.as_slice_mut().unwrap(), spectrum.as_slice_mut().unwrap());

    // adapt spectrum
    let mut spectrum = ft::spectrum_resize(num_output_samples, &spectrum);

    // compute inverse fft
    let ifft = rustfft::FFTplanner::new(true).plan_fft(num_output_samples);
    ifft.process(spectrum.as_slice_mut().unwrap(), output.as_slice_mut().unwrap());

    // drop imaginary part and normalize
    let norm = num_output_samples as f32 / num_input_samples as f32;
    let norm = norm / ((num_output_samples as f32).sqrt() * (num_output_samples as f32).sqrt());
    let out = output.mapv(|v| v.re * norm);

    // write
    let out_spec = hound::WavSpec {
        channels: 1,
        sample_rate: sample_rate,
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
