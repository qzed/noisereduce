use sspse::window;
use sspse::window::WindowFunction;

use gnuplot::Axes2D;


fn plot<W: WindowFunction<f64>>(ax: &mut Axes2D, window: W, name: &'static str) {
    let buf: Vec<f64> = window.iter().collect();
    ax.lines(0..window.len(), buf, &[gnuplot::Caption(name)]);
}


fn main() {
    use gnuplot::{Figure, AxesCommon, AutoOption};

    let len = 129;

    let mut fig = Figure::new();
    let mut ax = fig.axes2d();
    ax.set_x_range(AutoOption::Fix(0.0), AutoOption::Fix(len as f64 - 1.0));

    plot(&mut ax, window::rectangular(len), "Rectangular");
    plot(&mut ax, window::bartlett(len), "Bartlett");
    plot(&mut ax, window::hann(len), "Hann");
    plot(&mut ax, window::hamming(len), "Hamming");
    plot(&mut ax, window::blackman(len), "Blackman");
    plot(&mut ax, window::blackman_exact(len), "Blackman (exact)");
    plot(&mut ax, window::nuttall(len), "Nuttall");
    plot(&mut ax, window::blackman_nuttall(len), "Blackman-Nuttall");
    plot(&mut ax, window::blackman_harris(len), "Blackman-Harris");
    plot(&mut ax, window::flat_top(len), "Flat-Top");
    plot(&mut ax, window::gaussian(len, 0.4), "Gaussian (s=0.4)");
    plot(&mut ax, window::gaussian_confined(len, 0.1), "Confined Gaussian (s_t=0.1)");
    plot(&mut ax, window::generalized_normal(len, 0.4, 1.5), "Generalized Normal (s=0.4, p=1.5)");
    plot(&mut ax, window::tukey(len, 0.5), "Tukey (a=0.5)");

    fig.show();
}
