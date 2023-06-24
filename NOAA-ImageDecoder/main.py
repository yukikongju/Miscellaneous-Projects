import scipy.io.wavfile as wav
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class SignalDecoder:

    def __init__(self, wav_file_path, resample_rate = 4):
        self.resample_rate = resample_rate
        self.wav_file_path = wav_file_path
        self.fs, self.data = wav.read(self.wav_file_path)
        self._resampling_data()

    def _resampling_data(self):
        self.data = self.data[::self.resample_rate]
        self.fs = self.fs // self.resample_rate
        

    def plot_signal(self):
        data_crop = self.data[20*self.fs:21*self.fs]
        plt.plot(data_crop)
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.title('Signal')
        plt.show()

    def _hilbert_transform(self):
        analytical_signal = signal.hilbert(self.data)
        amplitude_envelope = np.abs(analytical_signal)
        return amplitude_envelope

    def get_image(self):
        frame_width = int(0.5*self.fs)
        data_am = self._hilbert_transform()

        w, h = frame_width, data_am.shape[0]//frame_width
        image = Image.new('RGB', (w, h))

        px, py = 0, 0
        for p in range(data_am.shape[0]):
            lum = int(data_am[p]//32 - 32)
            if lum < 0: lum = 0
            if lum > 255: lum = 255
            image.putpixel((px, py), (0, lum, 0))
            px += 1
            if px >= w:
                if (py % 50) == 0:
                    print(f"Line saved {py} of {h}")
                px = 0
                py += 1
                if py >= h:
                    break

        image = image.resize((w, self.resample_rate*h))
        plt.imshow(image)
        plt.show()
        

        
    

def main():
    wav_file_path = 'NOAA-ImageDecoder/samples/noaa18_27072012_try1.wav'
    decoder = SignalDecoder(wav_file_path)
    #  decoder.plot_signal()
    decoder.get_image()
    


if __name__ == "__main__":
    main()
