import matplotlib.pyplot as plt
import numpy as np

# --- generate signal
N = 100
Fs_og = 0.1
t_og = np.arange(N)
s = np.sin(2 * np.pi * Fs_og * t_og)

# --- sampling
Fs_new = 0.2
D = Fs_og / Fs_new  # decimal factor
t_new = np.arange(int(N * Fs_new / Fs_og)) * D
#  s_resampled = np.interp(t_new, t_og, s)
s_resampled = s[t_new.astype(int)]

# --- plot
fig, axs = plt.subplots(2, 1)
axs[0].plot(t_og, s, "--.")
axs[1].plot(t_new, s_resampled, "--.")
plt.show()
