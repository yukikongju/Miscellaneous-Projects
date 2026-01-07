import numpy as np
import matplotlib.pyplot as plt


# --- generate Quadrature Amplitude Motion (QAM) symbols
num_symbols = 1000
x_int = np.random.randint(0, 4, num_symbols)
x_degrees = x_int * 360.0 / 4.0 + 45  # 45, 135, 225, 315 degrees
#  x_rad = np.deg2rad(x_degrees)
x_radians = x_degrees * np.pi / 180.0
x_symbols = np.cos(x_radians) + 1j * np.sin(x_radians)

# --- add noise
n = (np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols)) / np.sqrt(2)
noise_power = 0.01
r = x_symbols + n * np.sqrt(noise_power)

# --- additive white gaussian noise (AWGN)
phase_noise = np.random.randn(num_symbols) * 0.1  # "strength" of phase noise
r2 = x_symbols * np.exp(1j * phase_noise)

# --- add noise and phase noise
r3 = x_symbols * np.exp(1j * phase_noise) + n * np.sqrt(noise_power)

# --- plot
fig, axs = plt.subplots(4, 1, figsize=(4, 6))
axs[0].plot(np.real(x_symbols), np.imag(x_symbols), ".")
axs[0].grid(True)
axs[1].plot(np.real(r), np.imag(r), ".")
axs[1].grid(True)
axs[2].plot(np.real(r2), np.imag(r2), ".")
axs[2].grid(True)
axs[3].plot(np.real(r3), np.imag(r3), ".")
axs[3].grid(True)
plt.show()
