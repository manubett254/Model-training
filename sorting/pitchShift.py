import os
import librosa
import soundfile as sf
from pydub import AudioSegment

# Define paths
source_folder = "C:/EEE/en/en/finalSortedFiles/male/adult"  # Original files
output_folder = "C:/EEE/en/en/finalSortedFiles/male/child"  # Target folder

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get the first 100 MP3 files
files = sorted([f for f in os.listdir(source_folder) if f.endswith(".mp3")])[:100]

# Check if any files are found
if not files:
    print("❌ No MP3 files found in", source_folder)
    exit()

print(f"✅ Found {len(files)} files. Processing...")

# Pitch shift settings
PITCH_SHIFT_SEMITONES = 6  # Increase pitch by 6 semitones

for index, file in enumerate(files):
    try:
        # Print file info
        mp3_path = os.path.join(source_folder, file)
        print(f"\n🔍 Processing file {index+1}/{len(files)}: {mp3_path}")

        # Check if file exists before processing
        if not os.path.exists(mp3_path):
            print(f"⚠️ File not found: {mp3_path}")
            continue

        # Load MP3 file
        audio = AudioSegment.from_mp3(mp3_path)

        # Convert MP3 to WAV (Librosa needs WAV)
        wav_path = os.path.join(output_folder, f"temp_{index}.wav")
        audio.export(wav_path, format="wav")

        # Load WAV with Librosa
        y, sr = librosa.load(wav_path, sr=None)

        # Apply pitch shifting
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=PITCH_SHIFT_SEMITONES)

        # Save the pitch-shifted audio as WAV
        shifted_wav_path = os.path.join(output_folder, f"child_{index}.wav")
        sf.write(shifted_wav_path, y_shifted, sr)

        # Convert back to MP3
        shifted_audio = AudioSegment.from_wav(shifted_wav_path)
        final_mp3_path = os.path.join(output_folder, f"child_{index}.mp3")
        shifted_audio.export(final_mp3_path, format="mp3", bitrate="192k")

        # Remove temporary WAV files
        os.remove(wav_path)
        os.remove(shifted_wav_path)

        print(f"✅ Saved: {final_mp3_path}")

    except Exception as e:
        print(f"❌ Error processing {file}: {e}")

print("🎵✅ Pitch shifting complete! Check the 'child' folder for results.")
