import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

RAW_AUDIO_DIR = "data/audio/raw"
OUTPUT_DIR = "data/audio/spectrograms"
#
SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
IMG_SIZE = (224, 224)

CLASSES = ["fake", "real"]

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def audio_to_mel_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def save_spectrogram(spec, output_path):
    plt.figure(figsize=(4, 4))
    plt.axis("off")
    librosa.display.specshow(spec, cmap="magma")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    # Resize to 224x224
    img = Image.open(output_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img.save(output_path)


def process_all():
    print("Starting audio to spectrogram conversion...")

    for label in CLASSES:
        input_dir = os.path.join(RAW_AUDIO_DIR, label)
        output_dir = os.path.join(OUTPUT_DIR, label)

        ensure_dir(output_dir)

        files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]

        print(f"Processing {label}: {len(files)} files")

        for filename in files:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(
                output_dir,
                filename.replace(".wav", ".png")
            )

            try:
                spec = audio_to_mel_spectrogram(input_path)
                save_spectrogram(spec, output_path)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print("Spectrogram generation finished.")


if __name__ == "__main__":
    process_all()