import matplotlib.pyplot as plt
import numpy as np

# --- sin(2 * pi * f * t)
f = 0.15
Fs = 1  # Hz
N = 100
t = np.arange(N)
noise = N * np.random.randint(1)
s = np.sin(2 * np.pi * f * t)
s = s * np.hamming(N)  # make signal periodic with windowing

#  S = np.fft.fft(s)
f = np.arange(-Fs / 2, Fs / 2, Fs / N)
S = np.fft.fftshift(np.fft.fft(s))  # make centered around 0 Hz
s_magnitude = np.abs(S)
s_phase = np.angle(S)

# --
fig, axs = plt.subplots(3, 1)
axs[0].plot(f, s)
axs[1].plot(f, s_magnitude)
axs[2].plot(f, s_phase)
plt.show()
