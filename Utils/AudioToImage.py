import os
import shutil
import wave
import numpy as np
import scipy.signal
import librosa
import matplotlib.pyplot as plt


class AudioToImage:
    def __init__(self, audio_path):
        self.audio_path = audio_path
        self.audio_data, self.audio_array = self.load_audio()
        self.f, self.t, self.Zxx = self.get_stft()
        self.mel_spectrogram = self.get_mel_spectrogram()
        self.mel_spectrogram_db = self.get_mel_spectrogram_db()

    def load_audio(self):
        with wave.open(self.audio_path, "rb") as audio_file:
            audio_data = audio_file.readframes(audio_file.getnframes())
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        return audio_data, audio_array

    def get_stft(self):
        window = scipy.signal.hann(1024)
        f, t, Zxx = scipy.signal.stft(self.audio_array, fs=44100, window=window, nperseg=1024)
        return f, t, Zxx

    def get_mel_spectrogram(self):
        mel_spectrogram = librosa.feature.melspectrogram(y=None, S=np.abs(self.Zxx), sr=44100, n_fft=1024,
                                                         hop_length=512)
        return mel_spectrogram

    def get_mel_spectrogram_db(self):
        mel_spectrogram_db = np.log10(self.mel_spectrogram + 1e-9)
        return mel_spectrogram_db

    def plot_mel_spectrogram(self, file_name):
        fig = plt.figure(figsize=(2, 2))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(self.mel_spectrogram_db, origin="lower", aspect="auto", cmap="gist_heat")
        fig.savefig(file_name)
        plt.close(fig)


def convert_all_wav_to_png(root_dir):
    image_dir = root_dir + "_IMAGES"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".wav"):
                audio_path = os.path.join(root, file)
                try:
                    print(f"Processing file {file}")
                    AtoI = AudioToImage(audio_path)
                    png_path = os.path.splitext(audio_path)[0] + ".png"
                    AtoI.plot_mel_spectrogram(png_path)
                except Exception as e:
                    print(f"Error processing file {file}: {e}")
                    continue

                rel_path = os.path.relpath(root, root_dir)
                destination_dir = os.path.join(image_dir, rel_path)
                if not os.path.exists(destination_dir):
                    os.makedirs(destination_dir)
                shutil.copy(png_path, destination_dir)
                os.remove(png_path)


def convert_wav_to_png(audio_path):
    audio_dir = os.path.dirname(audio_path)
    root, _ = os.path.splitext(audio_path)
    png_path = root + "_img.png"

    try:
        AtoI = AudioToImage(audio_path)
        AtoI.plot_mel_spectrogram(png_path)
    except Exception as e:
        print(f"Error processing file {png_path}: {e}")
        return
    return png_path

