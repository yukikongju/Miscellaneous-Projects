import matplotlib.pyplot as plt
import numpy as np

# --- generate signal
Fs = 300
N = 1024
Ts = 1 / Fs
t = Ts * np.arange(N)
x = np.exp(1j * 2 * np.pi * 50 * t)  # sinusoidal wave at 50 Hz
x = x * np.hamming(len(x))

# --- add noise
n = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)
noise_power = 2
r = x + n * np.sqrt(noise_power)

# --- power spectral density (PSD)
psd = np.abs(np.fft.fft(r)) ** 2 / (N * Fs)
psd_log = 10.0 * np.log10(psd)
psd_shifted = np.fft.fftshift(psd_log)
f = np.arange(-Fs / 2.0, Fs / 2.0, Fs / N)

# --- plot
plt.plot(f, psd_shifted)
plt.show()
