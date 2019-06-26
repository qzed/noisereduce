use sspse::wave::WavReaderExt;

use hound::{WavReader, Error};
use num::Complex;
use ndarray::{s, Axis, Array1};


fn main() -> Result<(), Error> {
    let path_in = std::env::args_os().nth(1).expect("missing input file argument");
    let path_out = std::env::args_os().nth(2).expect("missing output file argument");

    // load first channel of wave file
    let (samples, samples_spec) = WavReader::open(path_in)?.collect_convert_dyn::<f32>()?;
    let samples = samples.index_axis_move(Axis(1), 0);

    // convert real to complex
    let mut input = samples.mapv(|v| Complex { re: v, im: 0.0 });

    // compute parameters based on new sample rate
    let new_sample_rate = 16_000;

    let num_input_samples = input.len();
    let num_output_samples = (num_input_samples as f64 * new_sample_rate as f64 / samples_spec.sample_rate as f64) as usize;

    // compute fft
    let mut spectrum = Array1::zeros(num_input_samples);
    let mut output = Array1::zeros(num_output_samples);

    let fft = rustfft::FFTplanner::new(false).plan_fft(num_input_samples);
    fft.process(input.as_slice_mut().unwrap(), spectrum.as_slice_mut().unwrap());

    // adapt spectrum
    let n = num_input_samples.min(num_output_samples);

    let mut spectrum_new = Array1::zeros(num_output_samples);
    spectrum_new.slice_mut(s![..n/2]).assign(&spectrum.slice(s![..n/2]));
    spectrum_new.slice_mut(s![spectrum_new.len()-n/2..]).assign(&spectrum.slice(s![spectrum.len()-n/2..]));

    if num_output_samples < num_input_samples {                 // downsampling
        if num_output_samples % 2 == 1 {                        // handle odd target center
            spectrum_new[n/2+1] = (spectrum[n/2+1] + spectrum[spectrum.len() - (n/2+1)]) / 2.0;
        }
    } else if num_output_samples > num_input_samples {          // upsampling
        if num_input_samples % 2 == 1 {                         // handle odd source center
            spectrum_new[n/2+1] = spectrum[n/2+1] / 2.0;
            spectrum_new[spectrum.len() - (n/2+1)] = spectrum[n/2+1] / 2.0;
        }
    }

    let mut spectrum = spectrum_new;

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
        sample_rate: new_sample_rate,
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
