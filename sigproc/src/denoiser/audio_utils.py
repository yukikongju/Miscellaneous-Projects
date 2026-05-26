import numpy as np
import wave
import pyaudio
from typing import List
from pydub import AudioSegment


def save_audio_frames_to_wav(
    frames: List[np.ndarray], output_file_path: str, channels: int, format: int, rate: int
) -> None:
    """
    Saving audio frames to .wav file

    Parameters
    ----------
    frames: List[np.ndarray]
        List of audio chunks comming from pyaudio.Stream.read()
    output_file_path: str
        Output path of the generated .wav file
    channels: int
        number of channels in a single frame
    format: int
        Pyaudio format
    rate: int
        Sample rate of a chunk
    """
    with wave.open(output_file_path, "wb") as wf:
        wf.setnchannels(channels)
        sample_size = pyaudio.PyAudio().get_sample_size(format)
        wf.setsampwidth(sample_size)
        wf.setframerate(rate)
        wf.writeframes(b"".join(frames))


def convert_wav_to_audio_format(wav_path: str, audio_file_path: str, audio_type: str) -> None:
    """
    Converting .wav file to new audio format

    Parameters
    ----------
    wav_path: str
        Absolute/Relative path of audio file
    audio_file_path: str
        Absolute/relative path of output file
    audio_type: str
        Audio type to convert .wav file to. Should be one of the following: mp3, ogg, flac, mp4
    """
    if audio_type not in ["mp3", "ogg", "flac", "mp4"]:
        raise ValueError(f"Audio type {audio_type} not supported.")

    wav_file_extension = wav_path.split(".")[-1]
    if wav_file_extension != "wav":
        raise ValueError(f"Invalid wav file.")

    output_file_extension = audio_file_path.split(".")[-1]
    if output_file_extension != audio_type:
        raise ValueError(f"Audio file extension doesn't match audio type.")

    # load wav file
    audio = AudioSegment.from_wav(wav_path)

    # reduce file size and lower sample rate audio (use mono instead of stereo)
    audio = audio.set_frame_rate(22050)
    audio.set_channels(1)

    # Convert to MP3/OGG/FLAC/MP4
    audio.export(audio_file_path, format=audio_type)


#  if __name__ == "__main__":
#      wav_file = "outputs/denoised/wav12.wav"
#      audio_path = "outputs/denoised/output12.mp4"
#      convert_wav_to_audio_format(wav_file, audio_path, 'mp4')
