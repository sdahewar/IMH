"""Test Vosk STT with Hindi model"""
import os
import wave
import json
import subprocess
from vosk import Model, KaldiRecognizer

# Model path
MODEL_PATH = "vosk-model-small-hi-0.22"
AUDIO_PATH = "audio/testaud.mp3"

print("🎤 Testing Vosk STT with Hindi model...")
print("="*60)

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found at: {MODEL_PATH}")
    exit(1)

print(f"✅ Model found: {MODEL_PATH}")

# Load model
print("\n⏳ Loading Vosk model...")
model = Model(MODEL_PATH)
print("✅ Model loaded!")

# Get ffmpeg path from imageio
try:
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"✅ ffmpeg found: {ffmpeg_exe}")
except:
    ffmpeg_exe = "ffmpeg"

# Convert MP3 to WAV using ffmpeg directly
wav_path = AUDIO_PATH.replace('.mp3', '.wav')

if not os.path.exists(wav_path):
    print(f"\n⚠️ Converting {AUDIO_PATH} to WAV...")
    
    cmd = [
        ffmpeg_exe,
        "-y",  # Overwrite
        "-i", AUDIO_PATH,
        "-ar", "16000",  # 16kHz sample rate
        "-ac", "1",  # Mono
        "-f", "wav",
        wav_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Converted to: {wav_path}")
        else:
            print(f"❌ ffmpeg error: {result.stderr}")
            exit(1)
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        exit(1)
else:
    print(f"✅ WAV file exists: {wav_path}")

# Transcribe
print(f"\n🎤 Transcribing: {wav_path}")

wf = wave.open(wav_path, "rb")
print(f"   Channels: {wf.getnchannels()}")
print(f"   Sample rate: {wf.getframerate()}")
print(f"   Duration: {wf.getnframes() / wf.getframerate():.1f} seconds")

# Create recognizer
rec = KaldiRecognizer(model, wf.getframerate())
rec.SetWords(True)

# Process audio
print("\n⏳ Processing audio...")
transcript = ""
while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        result = json.loads(rec.Result())
        transcript += result.get("text", "") + " "

# Get final result
final_result = json.loads(rec.FinalResult())
transcript += final_result.get("text", "")

wf.close()

print("\n" + "="*60)
print("📝 TRANSCRIPT (Hindi):")
print("="*60)
print(transcript.strip() if transcript.strip() else "[No speech detected]")
print("="*60)
