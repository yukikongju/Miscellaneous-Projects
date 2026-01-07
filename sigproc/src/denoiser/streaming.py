"""
Script which saves sounds snippets of mic in real-time
"""

import audioop
import math
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from scipy.ndimage import gaussian_filter1d
from typing import Generator, List


def configure_mic(
    pyaudio_format: int, n_channels: int, rate: int, chunk_length: int
) -> pyaudio.Stream:
    """
    Configure mic to process audio in chunk using pyaudio stream

    Parameters
    ----------
    pyaudio_format: int
        Audio format of pyaudio microphone reader
    n_channels: int
        number of inputs audio channels
    rate: int
        sampling rate
    chunk_length: int
        number of frames per buffer

    Returns
    -------
    stream: pyaudio.Stream
    """
    return pyaudio.PyAudio().open(
        format=pyaudio_format,
        channels=n_channels,
        rate=rate,
        input=True,
        output=True,
        frames_per_buffer=chunk_length,
    )


def fourier_transform(data: np.ndarray, smoothing_coefficient: int = 3) -> np.ndarray:
    """
    Call FFT on data and add smoothing for visualization.

    Parameters
    ----------
    data: np.ndarray
        Data to run FFT on.
    smoothing_coefficient: int
        Gaussian kernel smoothing width.

    Returns
    -------
        np.ndarray of FFT returns.
    """

    return gaussian_filter1d(
        np.abs(np.fft.fft(np.frombuffer(data, np.int16))), smoothing_coefficient
    )


def plot_temporal_domain(ax: plt.Axes, temporal_data: np.ndarray, temporal_vlim: float) -> None:
    """
    Plot the raw waveform.

    Parameters
    ----------
    ax: plt.Axes
        Axis to plot to.
    temporal_data: np.ndarray
        Temporal data to plot.
    temporal_vlim: float
        Min/max for plot.
    """

    ax.plot(temporal_data)
    ax.axhline(y=0.0, color="gray", linestyle="-")
    ax.set_xlim(0, len(temporal_data))
    ax.set_ylim(-temporal_vlim, temporal_vlim)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel("Signal waveform")


def plot_frequency_domain(
    ax: plt.Axes,
    frequency_data: np.ndarray,
    waveform_vlim: float,
    rate: int,
    chunk_length: int,
    text_label_offset_x: int,
) -> None:
    """
    Plot the magnitude of the positive frequency components of the discrete FFT.

    Parameters
    ----------
    ax: plt.Axes
        Axis to plot to.
    frequency_data: np.ndarray
        Frequency data to plot.
    """

    max_x = np.argmax(frequency_data)
    max_y = np.max(frequency_data)

    rescale_factor: float = rate / chunk_length

    ax.scatter(rescale_factor * max_x, max_y, color="red", marker="x")
    ax.text(
        rescale_factor * (max_x + text_label_offset_x),
        max_y,
        max_x * rescale_factor,
        color="red",
        backgroundcolor="white",
        alpha=0.8,
    )
    ax.set_xlim(0, rescale_factor * len(frequency_data) // 2)
    ax.set_ylim(1, waveform_vlim)
    ax.plot(rescale_factor * np.arange(len(frequency_data)), frequency_data)
    ax.set_yscale("log")
    ax.set_yticks([])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Log-scaled magnitude")


def get_chunk_generator(
    stream: pyaudio.Stream, chunk_length: int
) -> Generator[np.ndarray, None, None]:
    """
    Get generator for microphone audio data in increments.

    Parameters
    ----------
    stream: pyaudio.Stream
        PyAudio stream to collect chunks from.
    chunk_length: int
        Specifies the number of frames per buffer.

    Returns
    -------
    Generator yielding microphone np.ndarray chunk.
    """
    while True:
        data = stream.read(chunk_length, exception_on_overflow=False)
        data = np.frombuffer(data, np.int16)
        yield data


def get_chunk_decibel(data: np.ndarray, width: int) -> float:
    """
    Get decibel from audio chunk

    Parameters
    ----------
    data: np.ndarray
        Audio data
    width: int
        Bytes per sample

    Returns
    -------
    decibel: float
    """
    rms = audioop.rms(data, width)

    # Normalize and convert to decibels
    # 32767 is the maximum value for 16-bit audio
    normalized_rms = rms / 32767.0
    decibel = (
        20.0 * math.log10(normalized_rms) if normalized_rms > 0 else -999
    )  # -999 indicates silence
    return decibel


def plot_decibel_levels(ax: plt.Axes, decibels: List[float]):
    """
    Plot the Decibel levels

    Parameters
    ----------
    decibels: List[float]
        Array of previous decibel levels
    """
    n = len(decibels)
    ax.plot(range(n), decibels)
    ax.set_yticks([])
    ax.set_ylabel("Decibel (dB)")


def run():
    CHUNK_LENGTH = 1024
    FORMAT = pyaudio.paInt16  # Note: pyaudio.paInt32 yields lots of variance, why?
    N_CHANNELS = 1
    RATE = 10000
    TEMPORAL_VLIM = 35000.0
    WAVEFORM_VLIM = 1e6
    WAIT_DURATION = 1e-10
    BYTE_PER_SAMPLE = 2  # Bytes per sample for paInt16
    PREV_DECIBEL_CACHE_SIZE = 100
    NUM_SUBPLOTS = 3
    prev_decibel_levels = []

    stream = configure_mic(FORMAT, N_CHANNELS, RATE, CHUNK_LENGTH)
    chunk_generator = get_chunk_generator(stream, CHUNK_LENGTH)

    while True:
        plt.clf()

        data = next(chunk_generator)
        fft_data = fourier_transform(data, 4)
        decibel = get_chunk_decibel(data, BYTE_PER_SAMPLE)
        print(decibel)

        ax1 = plt.subplot(NUM_SUBPLOTS, 1, 1)
        plot_temporal_domain(ax1, data, TEMPORAL_VLIM)

        ax2 = plt.subplot(NUM_SUBPLOTS, 1, 2)
        plot_frequency_domain(ax2, fft_data, WAVEFORM_VLIM, RATE, CHUNK_LENGTH, 20)

        ax3 = plt.subplot(NUM_SUBPLOTS, 1, 3)
        prev_decibel_levels.insert(0, decibel)
        prev_decibel_levels = prev_decibel_levels[:PREV_DECIBEL_CACHE_SIZE]
        plot_decibel_levels(ax3, prev_decibel_levels)

        plt.pause(WAIT_DURATION)
