# Private Home AI Voice Assistant: Implementation Plan

This document outlines the implementation plan for a private, voice-to-voice AI assistant. The system is designed for home use, with features for homeschooling, Bible memory practice, and custom agentic workflows.

## 1. Core Technologies

-   [x] **Backend Framework:** Python
-   [ ] **Voice Orchestration:** [Pipecat](https://www.pipecat.ai/)
-   [ ] **Public Access:** Cloudflare Tunnel
-   [ ] **Local LLMs:** [Ollama](https://ollama.ai/)
-   [ ] **Speech-to-Text (STT):** [LightningWhisperMLX](https://github.com/mustafaaljadery/lightning-whisper-mlx)
-   [ ] **Text-to-Speech (TTS):** [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI)
-   [ ] **Database:** SQLite
-   [ ] **Initial LLMs:** Llama-3.2, Gemma3n (or other Mistral variants) via Ollama. The architecture will be model-agnostic.

## 2. Project Structure

```
syft-speaking/
├── .env
├── .gitignore
├── README.md
├── requirements.txt
├── main.py                 # Main Pipecat application
├── web/
│   ├── index.html
│   └── static/
│       ├── js/
│       │   └── app.js
│       └── css/
│           └── style.css
├── services/
│   ├── __init__.py
│   ├── llm_service.py      # Handles interaction with Ollama
│   ├── tts_service.py      # Kokoro-FastAPI client
│   └── stt_service.py      # LightningWhisperMLX client
├── agents/
│   ├── __init__.py
│   ├── assistant.py        # Core assistant logic
│   ├── homeschool.py       # Homeschooling agent (Tutor/Quiz)
│   ├── bible_memory.py     # Bible memory agent
│   └── email_agent.py      # Email summarization agent
└── data/
    └── syft_speaking.db    # SQLite database
```

## 3. Implementation Phases

### Phase 1: Core Infrastructure Setup

**Goal:** Establish a basic, working voice-to-voice loop.

1.  **Environment Setup:**
    *   [x] Create a Python virtual environment.
    *   [x] Install dependencies: `pipecat-ai`, `fastapi`, `uvicorn`, `requests`, `ollama`, `sqlalchemy`, `aiohttp`.
    *   [x] Set up `.gitignore` for `venv`, `.env`, `__pycache__`, etc.
    *   [x] Create project structure

2.  **Service Integration:**
    *   **Ollama:**
        *   [ ] Install and run Ollama.
        *   [ ] Pull the desired models: `ollama pull llama3.2`.
        *   [ ] Create `services/llm_service.py` to handle requests to the Ollama API.
    *   **Kokoro-FastAPI (TTS):**
        *   [ ] Set up and run the Kokoro-FastAPI server.
        *   [ ] Create `services/tts_service.py` to act as a client, sending text and receiving audio.
    *   **LightningWhisperMLX (STT):**
        *   [ ] Set up and run the LightningWhisperMLX server.
        *   [ ] Create `services/stt_service.py` to send audio data and receive text.

3.  **Pipecat Application (`main.py`):**
    *   [ ] Create a simple Pipecat application that:
        *   [ ] Uses a `WebRTC` transport.
        *   [ ] Integrates the STT, LLM, and TTS services.
        *   [ ] The LLM service will just do basic chat for now.

4.  **Web Frontend (`web/index.html`):**
    *   [ ] Create a simple HTML page with a "Start Talking" button.
    *   [ ] Write JavaScript (`web/static/js/app.js`) to handle the WebRTC connection to the Pipecat server.

5.  **Cloudflare Tunnel:**
    *   [ ] Install `cloudflared`.
    *   [ ] Configure a tunnel to expose the Pipecat server's port to a public URL for easy access from any device.

### Phase 2: Database and Initial Agentic Capabilities

**Goal:** Set up the database and create the foundation for agentic workflows.

1.  **Database Schema (`data/syft_speaking.db`):**
    *   [ ] Use SQLAlchemy to define the schema.
    *   [ ] Create tables for:
        *   [ ] `users` (for future multi-user support, can start with a single default user).
        *   [ ] `credentials` (for storing encrypted Gmail API tokens).
        *   [ ] `library_items` (to store URLs, file paths to PDFs, and photo paths).
        *   [ ] `homeschool_materials` (to store subject matter for tutoring/quizzes).
        *   [ ] `bible_verses` (to store verses and translations).

2.  **Email Agent (`agents/email_agent.py`):**
    *   [ ] Focus on Gmail integration first.
    *   [ ] Implement OAuth2 for authentication, storing refresh tokens in the `credentials` table.
    *   [ ] Create functions to:
        *   [ ] Fetch the latest emails.
        *   [ ] Summarize email content and attachments (initially, just text from the email body).

### Phase 3: Homeschooling and Bible Memory Features

**Goal:** Implement the educational and memory practice tools.

1.  **Homeschooling Agent (`agents/homeschool.py`):**
    *   **Tutor Mode:**
        *   [ ] Allow the user to specify a topic.
        *   [ ] The agent loads the relevant material from the `homeschool_materials` table.
        *   [ ] The agent uses the LLM to answer questions based *only* on the provided material.
    *   **Quiz Mode:**
        *   [ ] The agent generates questions based on the material.
        *   [ ] It asks the user questions and evaluates their spoken answers.

2.  **Bible Memory Agent (`agents/bible_memory.py`):**
    *   [ ] Allow the user to specify a Bible verse (e.g., "John 3:16").
    *   [ ] The agent fetches the verse text from an API or a local copy.
    *   [ ] The user recites the verse.
    *   [ ] The agent performs a strict, word-for-word comparison of the STT output against the verse text and provides feedback.

### Phase 4: Advanced Features and Refinements

**Goal:** Enhance the system's capabilities and user experience.

1.  **Digital Library Expansion:**
    *   [ ] Implement PDF text extraction (using a library like `PyMuPDF`).
    *   [ ] Integrate with a vector database (e.g., ChromaDB) for semantic search across all library items (web pages, PDFs, photo descriptions).
    *   [ ] For photos, use an LLM with vision capabilities to generate descriptions for searching.

2.  **Wake Word Activation:**
    *   [ ] Integrate a wake word engine like `picovoice` or a custom-trained model.
    *   [ ] This will require a persistent listening service on the client-side.

3.  **Multi-user Support:**
    *   [ ] Implement user profiles and authentication on the web interface.
    *   [ ] Associate credentials and library items with specific users.

4.  **LLM Tool-Use/Function-Calling:**
    *   [ ] As local models improve, refactor the agentic workflows to use native tool-calling capabilities. This will make the system more robust and extensible.
    *   [ ] The main assistant agent (`agents/assistant.py`) will act as a dispatcher, determining which specialized agent (homeschool, email, etc.) to invoke based on the user's request.

This phased approach allows for iterative development, ensuring a functional core is established quickly while providing a clear roadmap for adding more complex features over time.
