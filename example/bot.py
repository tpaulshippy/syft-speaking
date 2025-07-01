#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
import os
import sys
import time

# Add the parent directory to the Python path to import services
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import AudioRawFrame, TextFrame, TranscriptionFrame, SystemFrame
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.ollama.llm import OLLamaLLMService

# Import local services
from services.tts_service import KokoroTTSService

# Custom STT processor that works with VAD
class CustomWhisperSTT(FrameProcessor):
    """A custom STT processor that buffers audio and transcribes when enough data is collected."""
    def __init__(self, model_name: str = "distil-small.en", language: str = "en", buffer_threshold: int = 16000):
        super().__init__()
        self._model_name = model_name
        self._language = language
        self._audio_buffer = bytearray()
        self._buffer_threshold = buffer_threshold  # 1 second of audio at 16kHz
        self._model = None
        self._init_model()

    async def start(self):
        await super().start()

    def _init_model(self):
        """Initialize the Whisper model."""
        try:
            from lightning_whisper_mlx import LightningWhisperMLX
            logger.info(f"Initializing Whisper model '{self._model_name}'...")
            self._model = LightningWhisperMLX(model=self._model_name, batch_size=12, quant=None)
            logger.info("Whisper model initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Whisper model: {e}")
            self._model = None

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            # Buffer audio data
            self._audio_buffer.extend(frame.audio)
            
            # If we have enough audio data, transcribe it
            if len(self._audio_buffer) >= self._buffer_threshold and self._model:
                await self._transcribe_buffer()
            
            # Pass through audio frames
            await self.push_frame(frame, direction)
        
        else:
            # Pass through other frame types
            await self.push_frame(frame, direction)

    async def _transcribe_buffer(self):
        """Transcribe the buffered audio."""
        if not self._audio_buffer:
            return

        try:
            import numpy as np
            logger.info(f"Transcribing {len(self._audio_buffer)} bytes of audio.")
            
            # Convert the byte buffer to a NumPy array
            audio_np = np.frombuffer(self._audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0

            # Transcribe the audio
            result = self._model.transcribe(audio_np, language=self._language)
            text = result.get("text", "").strip()
            
            if text:
                logger.info(f"Transcription result: '{text}'")
                await self.push_frame(TranscriptionFrame(text=text, user_id=None, timestamp=time.time()))
            else:
                logger.warning("Transcription resulted in empty text.")
                
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
        finally:
            # Clear the buffer for the next utterance
            self._audio_buffer = bytearray()

load_dotenv(override=True)

SYSTEM_INSTRUCTION = f"""
"You are a friendly, helpful robot.

Your goal is to demonstrate your capabilities in a succinct way.

Your output will be converted to audio so don't include special characters in your answers.

Respond to what the user said in a creative and helpful way. Keep your responses brief. One or two sentences at most.
"""

KOKORO_MODEL_PATH = os.getenv("KOKORO_MODEL_PATH", "kokoro/kokoro-v1.0.onnx")
KOKORO_VOICES_PATH = os.getenv("KOKORO_VOICES_PATH", "kokoro/voices-v1.0.bin")
KOKORO_VOICE_ID = os.getenv("KOKORO_VOICE_ID", "af_sarah")

async def run_bot(webrtc_connection):
    pipecat_transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            audio_out_10ms_chunks=2,
        ),
    )

    # Create local services
    stt = CustomWhisperSTT(model_name="distil-small.en")
    llm = OLLamaLLMService(
        model="llama3.2",  # Must be pulled first: ollama pull llama3.2
        base_url="http://localhost:11434/v1",  # Default Ollama endpoint
    )

        
    tts = KokoroTTSService(
        model_path=KOKORO_MODEL_PATH,
        voices_path=KOKORO_VOICES_PATH,
        voice_id=KOKORO_VOICE_ID,
    )

    context = OpenAILLMContext(
        messages=[
            {
                "role": "system",
                "content": SYSTEM_INSTRUCTION
            }
        ],
    )

    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            pipecat_transport.input(),
            stt,  # Speech-to-Text
            context_aggregator.user(),
            llm,  # Local LLM
            tts,  # Text-to-Speech
            pipecat_transport.output(),
            context_aggregator.assistant()
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @pipecat_transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Pipecat Client connected")
        # Send initial greeting
        await task.queue_frames([TextFrame(text="Hello! I'm your local AI assistant. How can I help you today?")])

    @pipecat_transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Pipecat Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)
