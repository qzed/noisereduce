use sspse::wave::WavReaderExt;

use clap::{Arg, Command};
use gnuplot::{AutoOption, AxesCommon, Figure};
use hound::{Error, WavReader};
use ndarray::Axis;


fn app() -> Command<'static> {
    Command::new("Example: Plot Signal Waveform")
        .author(clap::crate_authors!())
        .arg(Arg::with_name("input")
                .help("The input file to use (wav)")
                .value_name("INPUT")
                .required(true)
                .allow_invalid_utf8(true))
}

fn main() -> Result<(), Error> {
    let matches = app().get_matches();
    let path_in = matches.value_of_os("input").unwrap();

    // load wave file and extract first channel
    let (samples, _samples_spec) = WavReader::open(path_in)?.collect_convert_dyn::<f32>()?;
    let samples = samples.index_axis_move(Axis(1), 0);

    // plot
    let mut fig = Figure::new();
    let ax = fig.axes2d();
    ax.set_x_range(AutoOption::Fix(0.0), AutoOption::Fix(samples.len() as f64 - 1.0));
    ax.lines(0..samples.len(), samples.iter(), &[]);
    fig.show().unwrap();

    Ok(())
}
