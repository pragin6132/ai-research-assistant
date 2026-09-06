# AI Research Assistant

An agentic research assistant pipeline (internship project).

> **Status:** Project skeleton only. No AI pipeline, no Tavily integration, no LLM
> calls, and no frontend are implemented yet. This is the foundation for future
> development steps.

## Project Structure

```
ai-research-assistant/
├── backend/
│   ├── app/
│   │   ├── api/        # FastAPI route definitions
│   │   ├── agents/     # Agent implementations (future)
│   │   ├── graph/      # Agentic pipeline / orchestration graph (future)
│   │   ├── services/   # External integrations: Tavily, LLM providers (future)
│   │   ├── schemas/    # Pydantic request/response models (future)
│   │   └── core/       # Configuration and shared core logic
│   └── main.py         # FastAPI app entrypoint
├── frontend/            # Frontend app (future)
├── tests/                # Test suite (future)
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy the environment template and fill in values as needed later:
   ```bash
   cp .env.example .env
   ```
   (No real API keys are required yet — this skeleton doesn't call any external APIs.)

## Running the Server

From the `backend/` directory:

```bash
cd backend
uvicorn main:app --reload
```

The server will start at `http://127.0.0.1:8000`.

## Testing the Health Endpoint

With the server running:

```bash
curl http://127.0.0.1:8000/api/health
```

Expected response:

```json
{"status": "ok"}
```

You can also open `http://127.0.0.1:8000/docs` in a browser to view the
interactive Swagger UI.

## Next Steps (not yet implemented)

- Wire up LLM providers (Gemini, NVIDIA, Groq) in `app/services/`
- Integrate Tavily for web search
- Build out the agentic pipeline in `app/graph/` and `app/agents/`
- Define request/response schemas in `app/schemas/`
- Build the frontend
- Add tests in `tests/`
