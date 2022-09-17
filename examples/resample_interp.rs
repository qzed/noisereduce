use sspse::wave::WavReaderExt;

use clap::{value_t_or_exit, Arg, Command};
use hound::{Error, SampleFormat, WavReader, WavSpec, WavWriter};
use dasp::signal::Signal;


fn app() -> Command<'static> {
    Command::new("Example: Re-sample Audio Signal using Sinc Interpolation")
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
    let target_sample_rate = value_t_or_exit!(matches, "samplerate", u32);

    // load wave file and extract first channel
    let mut reader = WavReader::open(path_in)?;
    let spec = reader.spec();

    // iterator for loading samples
    let samples = reader
        .samples_convert_dyn::<f32>()           // convert fo 32-bit float
        .step_by(spec.channels as usize)        // select first channel
        .map(|s| s.unwrap());                   // unwrap Result

    // create interpolator for sample-rate conversion
    let buffer = dasp::ring_buffer::Fixed::from([[0.0]; 64]);
    let interp = dasp::interpolate::sinc::Sinc::new(buffer);

    // convert sample rate
    let signal = dasp::signal::from_interleaved_samples_iter(samples);
    let signal = signal.from_hz_to_hz(interp, spec.sample_rate as f64, target_sample_rate as f64);

    // write result
    let target_spec = WavSpec {
        channels: 1,
        sample_rate: target_sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let mut writer = WavWriter::create(path_out, target_spec)?;
    for frame in signal.until_exhausted() {
        writer.write_sample(frame[0])?;
    }

    Ok(())
}
