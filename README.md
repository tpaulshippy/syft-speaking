# Simple Chatbot Server

A Pipecat server-side bot that connects to a Pipecat client, enabling a user to talk to the bot through their browser or mobile device.


## Setup

1. Configure environment 

   Download kokoro files to the kokoro/ folder:
   - af_sarah.bin
   - kokoro-v1.0.onnx
   - voices-v1.0.bin

   Download Ollama and run llama3.2 model


2. Set up a virtual environment and install dependencies

   ```bash
   cd server
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

   > Using `uv`? Create your venv using: `uv sync`

3. Run the bot:

   ```bash
   python bot-openai.py --transport daily
   ```

   > Using `uv`? Run your bot using: `uv run bot-openai.py --transport daily`

## Troubleshooting

If you encounter this error:

```bash
aiohttp.client_exceptions.ClientConnectorCertificateError: Cannot connect to host api.daily.co:443 ssl:True [SSLCertVerificationError: (1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1000)')]
```

It's because Python cannot verify the SSL certificate from https://api.daily.co when making a POST request to create a room or token.

This issue occurs when the system doesn't have the proper CA certificates.

Install SSL Certificates (macOS): `/Applications/Python\ 3.12/Install\ Certificates.command`
