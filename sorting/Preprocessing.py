import librosa
import librosa.display
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import soundfile as sf

def preprocess_audio(input_path, output_path, target_sr=16000, duration=5):
    # Load audio
    audio, sr = librosa.load(input_path, sr=None, mono=True)
    
    # Resample to target sample rate
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    # Trim or pad to fixed duration
    target_length = target_sr * duration
    if len(audio) > target_length:
        audio = audio[:target_length]
    else:
        audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')

    # Apply noise reduction using spectral gating
    audio = scipy.signal.wiener(audio)

    # Save the processed audio
    sf.write(output_path, audio, target_sr)

    return output_path

# Test with a sample file
preprocess_audio("C:/EEE/sorting/clip1.mp3", "C:/EEE/sorting/clip3.mp3")
