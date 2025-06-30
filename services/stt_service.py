import asyncio
import logging
import numpy as np
import torch
import wave
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import (
    AudioRawFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    TextFrame,
)
from lightning_whisper_mlx import LightningWhisperMLX

logging.basicConfig(level=logging.INFO)

class LocalWhisperSTT(FrameProcessor):
    """
    A Pipecat processor that uses a local LightningWhisperMLX model
    for speech-to-text transcription.
    """

    def __init__(self, model_name: str = "distil-medium.en", language: str = "en"):
        """
        Initializes the LocalWhisperSTT processor.

        Args:
            model_name (str): The name of the Whisper model to use.
            language (str): The language for transcription.
        """
        super().__init__()
        logging.info(f"Initializing Whisper model '{model_name}'...")
        self._model = LightningWhisperMLX(model=model_name, batch_size=12, quant=None)
        self._language = language
        self._audio_buffer = bytearray()
        logging.info("Whisper model initialized.")

    async def process_frame(self, frame, direction: FrameDirection):
        """
        Processes incoming audio frames and generates transcription frames.
        """
        if isinstance(frame, AudioRawFrame):
            # Append audio data to our buffer
            self._audio_buffer.extend(frame.audio)
            # We could do interim transcriptions here for faster feedback,
            # but for simplicity, we'll wait for a TextFrame to signal the end of an utterance.
            await self.push_frame(frame, direction)
        elif isinstance(frame, TextFrame) and frame.text.strip():
            # This is a signal that the user has finished speaking.
            # Now we process the buffered audio.
            if not self._audio_buffer:
                return

            logging.info(f"Transcribing {len(self._audio_buffer)} bytes of audio.")
            
            # Convert the byte buffer to a NumPy array
            audio_np = np.frombuffer(self._audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0

            # Transcribe the audio
            try:
                result = self._model.transcribe(audio_np, language=self._language)
                text = result.get("text", "").strip()
                if text:
                    logging.info(f"Transcription result: '{text}'")
                    await self.push_frame(TranscriptionFrame(text))
                else:
                    logging.warning("Transcription resulted in empty text.")
            except Exception as e:
                logging.error(f"Error during transcription: {e}")
            finally:
                # Clear the buffer for the next utterance
                self._audio_buffer = bytearray()
                
            # Pass the TextFrame along as it might be used by other processors
            await self.push_frame(frame, direction)
        else:
            # Pass through other frame types
            await self.push_frame(frame, direction)

# This processor is meant to be used within a Pipecat pipeline,
# so a direct test here is complex. The test will be part of the main application.