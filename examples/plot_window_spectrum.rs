use sspse::window::WindowFunction;

use ndarray::{s, Array1};
use num::{Complex, traits::Float, traits::Zero};
use rustfft::FFTplanner;
use gnuplot::{Figure, Axes2D, AxesCommon, AutoOption, Caption};


fn magnitude_to_db<F: Float>(m: F) -> F {
    F::from(20.0).unwrap() * F::log10(m)
}

fn fftshift<T: Copy + Zero>(input: Array1<T>) -> Array1<T> {
    let mut output = Array1::zeros(input.len());
    output.slice_mut(s![0..input.len()/2]).assign(&input.slice(s![(input.len()+1)/2..]));
    output.slice_mut(s![input.len()/2..]).assign(&input.slice(s![..(input.len()+1)/2]));
    output
}


fn plot_spectrum<W>(ax: &mut Axes2D, window: W, name: &'static str)
where
    W: WindowFunction<f64>,
{
    let len = window.len();
    let window = window.to_array();

    let mut planner = FFTplanner::new(false);
    let fft = planner.plan_fft(len);

    let mut buf_in  = Array1::zeros(len);
    let mut buf_out = Array1::zeros(len);

    for i in 0..len {
        buf_in[i] = Complex { re: window[i], im: 0.0 };
    }

    fft.process(buf_in.as_slice_mut().unwrap(), buf_out.as_slice_mut().unwrap());

    let mut freq = Array1::zeros(len);
    for i in 0..len {
        freq[i] = buf_out[i].norm();
    }

    let freq = freq.mapv_into(magnitude_to_db);
    let freq = fftshift(freq);

    ax.lines(0..freq.len(), freq.iter(), &[Caption(name)]);
}


fn main() {
    let len = 1023;

    let mut fig = Figure::new();
    let mut ax = fig.axes2d();
    ax.set_x_range(AutoOption::Fix(0.0), AutoOption::Fix(len as f64 - 1.0));
    plot_spectrum(&mut ax, sspse::window::rectangular(len), "Rectangular");
    plot_spectrum(&mut ax, sspse::window::bartlett(len), "Bartlett");
    plot_spectrum(&mut ax, sspse::window::hann(len), "Hann");
    plot_spectrum(&mut ax, sspse::window::hamming(len), "Hamming");
    plot_spectrum(&mut ax, sspse::window::blackman_nuttall(len), "Blackman-Nuttall");
    plot_spectrum(&mut ax, sspse::window::blackman_harris(len), "Blackman-Harris");
    fig.show();
}
