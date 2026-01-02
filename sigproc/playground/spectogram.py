import matplotlib.pyplot as plt
import numpy as np

N = 1024  # fft_size
Fs = 50e3
sample_rate = 1e6

# ---
fig, axs = plt.subplots(3, 1)
t = np.arange(N * 1000) / sample_rate
x = np.sin(2 * np.pi * Fs * t) + 0.2 * np.random.randn(len(t))
x = x * np.hamming(len(x))
axs[0].plot(t[:200], x[:200])

# --- spectogram => convert to decibels: 10 * np.log10(x) w/ power spectral density (PSD)
num_rows = len(x) // N
spectrogram = np.zeros((num_rows, N))
for i in range(num_rows):
    psd = np.abs(np.fft.fft(x[i * N : (i + 1) * N])) ** 2 / (N * sample_rate)
    psd_log = 10.0 * np.log10(psd)
    spectrogram[i, :] = np.fft.fftshift(psd_log)

# -- Note: /1e6 converts Hz to MHz
f = np.arange(-Fs / 2.0, Fs / 2.0, Fs / N)
axs[1].imshow(
    spectrogram,
    aspect="auto",
    extent=[-sample_rate / 2 / 1e6, sample_rate / 2 / 1e6, len(x) / sample_rate, 0],
)
#  axs[2].plot(f, spectrogram)
plt.show()
