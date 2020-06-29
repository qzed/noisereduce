# Noisereduce

Modular implementation of real-time capable adaptive noise reduction algorithms.
Incorporates MMSE, log-MMSE, OM-LSA and MCRA algorithms based on _Speech Enhancement Using a Minimum Mean-Square Error Short-Time Spectral Amplitude Estimator_ (Ephraim and Malah), _Speech Enhancement Using a Minimum Mean-Square Error Log-Spectral Amplitude Estimator_ (Ephraim and Malah), and _Speech Enhancement for Non-Stationary Noise Environments_ (Cohen and Berdugo).
Project for the Speech Signal Processing and Speech Enhancement course, summer term 2019, University of Stuttgart.

Algorithms can be specified via parameter-files (see `params/` for examples). The main utility can be run via
```
cargo run --release --bin noisereduce -- -p <parameter-file.yml> <input.wav> <output.wav>
```

Feel free to have a look at the corresponding [paper][paper].

[paper]: https://nbviewer.jupyter.org/github/qzed/noisereduce/blob/master/paper/Real-Time%20capable%20Noise%20Reduction%20Methods.pdf
