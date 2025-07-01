import asyncio
import numpy as np
import wave
from services.tts_service import KokoroTTSService

# TODO: Set these paths to your actual Kokoro model and voices files
MODEL_PATH = "kokoro/kokoro-v1.0.onnx"
VOICES_PATH = "kokoro/voices-v1.0.bin"
VOICE_ID = "af_sarah"  # Or another valid voice ID
SAMPLE_TEXT = "Hello, this is a test of the Kokoro TTS service."
OUTPUT_WAV = "kokoro_tts_test_output.wav"

async def main():
    tts = KokoroTTSService(
        model_path=MODEL_PATH,
        voices_path=VOICES_PATH,
        voice_id=VOICE_ID,
    )
    audio_chunks = []
    sample_rate = 16000  # Default, will be overwritten by actual output

    async for frame in tts.run_tts(SAMPLE_TEXT):
        if hasattr(frame, "audio") and hasattr(frame, "sample_rate"):
            audio = frame.audio
            sample_rate = frame.sample_rate
            # Convert bytes to numpy array for concatenation
            audio_np = np.frombuffer(audio, dtype=np.int16)
            audio_chunks.append(audio_np)
        elif hasattr(frame, "message"):
            print(f"Error: {frame.message}")

    if audio_chunks:
        audio_data = np.concatenate(audio_chunks)
        with wave.open(OUTPUT_WAV, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())
        print(f"Wrote output to {OUTPUT_WAV}")
    else:
        print("No audio data generated.")

if __name__ == "__main__":
    asyncio.run(main()) 