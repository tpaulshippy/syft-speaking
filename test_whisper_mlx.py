import pyaudio
import wave
import numpy as np
from lightning_whisper_mlx import LightningWhisperMLX
import torch
import time

def record_audio(filename="recording.wav", record_seconds=5, sample_rate=16000):
    """Record audio from the microphone and save it as a WAV file."""
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1
    
    p = pyaudio.PyAudio()
    
    print(f"Recording for {record_seconds} seconds...")
    
    stream = p.open(format=sample_format,
                   channels=channels,
                   rate=sample_rate,
                   frames_per_buffer=chunk,
                   input=True)
    
    frames = []
    
    for _ in range(0, int(sample_rate / chunk * record_seconds)):
        data = stream.read(chunk, exception_on_overflow=False)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    print("Finished recording.")
    
    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    return filename

def transcribe_audio(model, audio_file):
    """Transcribe the audio file using the Whisper model."""
    print("Transcribing audio...")
    start_time = time.time()
    
    # Transcribe the audio file
    result = model.transcribe(audio_file, language="en")
    
    end_time = time.time()
    print(f"Transcription completed in {end_time - start_time:.2f} seconds")
    print("\nTranscription:")
    print("-" * 50)
    print(result["text"])
    print("-" * 50)
    
    return result["text"]

def main():
    print("Initializing Whisper model...")
    # Initialize the model (this will download the model if it's not already cached)
    model = LightningWhisperMLX(
        model="distil-medium.en",  # You can change this to "tiny", "base", "small", "medium", or "large"
        batch_size=1,
        quant=None
    )
    
    try:
        while True:
            input("\nPress Enter to start recording (or Ctrl+C to exit)...")
            
            # Record audio
            audio_file = record_audio()
            
            # Transcribe the recorded audio
            transcription = transcribe_audio(model, audio_file)
            
            print("\nTranscription complete!")
            
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
