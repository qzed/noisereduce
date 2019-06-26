use sspse::wave::WavReaderExt;

use hound::{WavReader, WavWriter, WavSpec, SampleFormat, Error};
use sample::Signal;


fn main() -> Result<(), Error> {
    let path_in = std::env::args_os().nth(1).expect("missing input file argument");
    let path_out = std::env::args_os().nth(2).expect("missing output file argument");

    let target_sample_rate = 8_000;

    // load wave file and extract first channel
    let mut reader = WavReader::open(path_in)?;
    let spec = reader.spec();

    // iterator for loading samples
    let samples = reader.samples_convert_dyn::<f32>()   // convert fo 32-bit float
        .step_by(spec.channels as usize)                // select first channel
        .map(|s| s.unwrap());                           // unwrap Result

    // create interpolator for sample-rate conversion
    let buffer = sample::ring_buffer::Fixed::from([[0.0]; 64]);
    let interp = sample::interpolate::Sinc::new(buffer);

    // convert sample rate
    let signal = sample::signal::from_interleaved_samples_iter(samples);
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
