import asyncio
import logging
import ollama

logging.basicConfig(level=logging.INFO)

class OllamaService:
    """
    A service to interact with a local Ollama language model.
    It supports streaming responses for real-time interaction.
    """

    def __init__(self, model: str = "llama3.2"):
        """
        Initializes the OllamaService.

        Args:
            model (str): The name of the Ollama model to use.
        """
        self._model = model
        self._client = ollama.AsyncClient()
        logging.info(f"OllamaService initialized with model: {self._model}")

    async def stream_chat(self, messages: list[dict]) -> str:
        """
        Streams a chat response from the Ollama model.

        Args:
            messages (list[dict]): A list of messages in the conversation history,
                                   following the Ollama API format.

        Yields:
            str: Chunks of the response text as they are generated.
        """
        try:
            logging.info(f"Sending request to Ollama model {self._model} with messages: {messages}")
            async for part in await self._client.chat(model=self._model, messages=messages, stream=True):
                if part and "message" in part and "content" in part["message"]:
                    content = part["message"]["content"]
                    if content:
                        yield content
        except Exception as e:
            logging.error(f"An error occurred while streaming chat from Ollama: {e}")
            yield "I'm sorry, but I encountered an error while trying to respond."

# Example usage for testing the service directly
async def main():
    """Main function to test the OllamaService."""
    print("Testing OllamaService...")
    service = OllamaService(model="llama3.2")
    
    # Example conversation history
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "Why is the sky blue?",
        },
    ]

    print(f"\n--- Streaming response for model: {service._model} ---")
    full_response = ""
    try:
        async for chunk in service.stream_chat(messages):
            print(chunk, end="", flush=True)
            full_response += chunk
    except Exception as e:
        print(f"\nAn error occurred during the test: {e}")
    
    print("\n--- End of stream ---")
    print(f"\nFull response received:\n{full_response}")


if __name__ == "__main__":
    # To run this test, ensure your Ollama server is running with the specified model.
    # For example: `ollama run llama3.2`
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
