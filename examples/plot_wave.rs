use sspse::wave::WavReaderExt;

use hound::{WavReader, Error};
use ndarray::Axis;
use gnuplot::{Figure, AxesCommon, AutoOption};


fn main() -> Result<(), Error> {
    let path = std::env::args_os().nth(1).expect("missing file argument");

    // load wave file and extract first channel
    let (samples, _samples_spec) = WavReader::open(path)?.collect_convert_dyn::<f32>()?;
    let samples = samples.index_axis_move(Axis(1), 0);

    // plot
    let mut fig = Figure::new();
    let ax = fig.axes2d();
    ax.set_x_range(AutoOption::Fix(0.0), AutoOption::Fix(samples.len() as f64 - 1.0));
    ax.lines(0..samples.len(), samples.iter(), &[]);
    fig.show();

    Ok(())
}
