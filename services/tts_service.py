import aiohttp
import asyncio
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)

class KokoroTTSClient:
    """
    A client for an OpenAI-compatible TTS server like Kokoro-FastAPI.
    It sends text to the server and receives audio data in response.
    """

    def __init__(self, base_url: str = None, model: str = "tts-1"):
        """
        Initializes the KokoroTTSClient.

        Args:
            base_url (str, optional): The base URL of the TTS server.
                                      Defaults to KOKORO_TTS_URL or 'http://localhost:8000'.
            model (str, optional): The TTS model to use. Defaults to 'tts-1'.
        """
        self._base_url = base_url or os.getenv("KOKORO_TTS_URL", "http://localhost:8000")
        self._tts_url = f"{self._base_url}/v1/audio/speech"
        self._model = model
        logging.info(f"KokoroTTSClient initialized for server URL: {self._tts_url}")

    async def generate_audio(self, text: str) -> bytes:
        """
        Generates audio from the given text by making a POST request to the TTS server.

        Args:
            text (str): The text to be converted to speech.

        Returns:
            bytes: The raw audio data (e.g., WAV or MP3 file content).
                   Returns an empty bytes object if an error occurs.
        """
        if not text:
            return b""

        payload = {
            "input": text,
            "model": self._model,
            # You can add other parameters like 'voice' if your server supports them
            # "voice": "alloy"
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self._tts_url, json=payload) as response:
                    response.raise_for_status()
                    audio_data = await response.read()
                    logging.info(f"Successfully received {len(audio_data)} bytes of audio data.")
                    return audio_data
        except aiohttp.ClientError as e:
            logging.error(f"An error occurred while requesting TTS audio: {e}")
            return b""
        except Exception as e:
            logging.error(f"An unexpected error occurred in generate_audio: {e}")
            return b""

# Example usage for testing the service directly
async def main():
    """Main function to test the KokoroTTSClient."""
    print("Testing KokoroTTSClient...")
    
    client = KokoroTTSClient()

    text_to_speak = "Hello, this is a test of the OpenAI-compatible TTS client."
    print(f"Requesting audio for text: '{text_to_speak}' from {client._tts_url}")

    try:
        audio_bytes = await client.generate_audio(text_to_speak)
        if audio_bytes:
            # Save the audio to a file to verify it's working.
            # The format might be mp3 or wav depending on the server's default.
            output_filename = "tts_test_output.mp3"
            with open(output_filename, "wb") as f:
                f.write(audio_bytes)
            print(f"Successfully generated audio and saved it to '{output_filename}'")
            print("You can play this file to check the output.")
        else:
            print("\n[Error] Failed to generate audio.")
            print("Please check the console for error messages from the client.")

    except aiohttp.client_exceptions.ClientConnectorError as e:
        print(f"\n[Connection Error] Could not connect to the server at {client._tts_url}.")
        print("Please ensure the server is running and the URL is correct.")
        logging.debug(e)
    except Exception as e:
        print(f"\nAn unexpected error occurred during the test: {e}")

if __name__ == "__main__":
    # To run this test, ensure your TTS server is running.
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
