import asyncio
import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocket
from pipecat.frames.frames import AudioRawFrame, TextFrame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketTransport
from pipecat.audio.vad.vad_analyzer import VADAnalyzer

from services.llm_service import OllamaService
from services.stt_service import LocalWhisperSTT
from services.tts_service import KokoroTTSClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------
# Custom Processors
# -----------------

class OllamaLLM(FrameProcessor):
    """A custom processor to handle conversation with the Ollama service."""
    def __init__(self, model: str = "llama3.2"):
        super().__init__()
        self.llm = OllamaService(model=model)
        self.messages = [{
            "role": "system",
            "content": "You are a helpful and concise voice assistant."
        }]

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            self.messages.append({"role": "user", "content": frame.text})
            full_response = ""
            async for chunk in self.llm.stream_chat(self.messages):
                full_response += chunk
                await self.push_frame(TextFrame(text=chunk))
            if full_response:
                self.messages.append({"role": "assistant", "content": full_response})

class KokoroTTS(FrameProcessor):
    """A custom processor to generate audio using the Kokoro TTS client."""
    def __init__(self):
        super().__init__()
        self.client = KokoroTTSClient()

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            audio_bytes = await self.client.generate_audio(frame.text)
            if audio_bytes:
                await self.push_frame(AudioRawFrame(audio=audio_bytes, sample_rate=16000, num_channels=1))

# -----------------
# FastAPI & Pipecat Setup
# -----------------

app = FastAPI()
app.mount("/static", StaticFiles(directory="web/static", html=True), name="static")

@app.websocket("/pipe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    transport = FastAPIWebsocketTransport(websocket)
    
    pipeline = Pipeline([
        transport.input(),
        VADAnalyzer(),
        LocalWhisperSTT(model_name="distil-small.en"),
        OllamaLLM(model="llama3.2"),
        KokoroTTS(),
        transport.output(),
    ])

    task = PipelineTask(pipeline)
    await task.run()

# Serve the main HTML file at the root
@app.get("/")
async def root():
    from fastapi.responses import FileResponse
    return FileResponse("web/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001, log_level="info")
