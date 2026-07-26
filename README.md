# CorrectiveRAG

An upgrade to a standard PDF chatbot: instead of always trusting its first
retrieval, it grades the relevance of what it finds, falls back to a live
web search when the document doesn't have the answer, and verifies its own
answer before returning it - regenerating if the check fails.

## Setup

1. **Create a virtual environment (recommended)**
   ```
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

3. **Get your free API keys**
   - Groq (LLM): https://console.groq.com — free tier, same one you used before.
   - Tavily (web search): https://tavily.com — free tier, 1,000 searches/month.

4. **Set up your `.env` file**
   Copy `.env.example` to `.env` and fill in your keys:
   ```
   cp .env.example .env
   ```

5. **Run the app**
   ```
   streamlit run app.py
   ```

## How it works (file by file)

- `ingest.py` — loads a PDF, splits it into chunks, embeds them locally
  (free, no API cost) with HuggingFace embeddings, and saves a FAISS index.
- `graph.py` — the actual Corrective RAG pipeline, built as a LangGraph
  state machine:
  1. `decompose_question` — splits complex questions into sub-questions
  2. `retrieve` — pulls relevant chunks from your FAISS index
  3. `grade_relevance` — LLM checks if what was retrieved is actually useful
  4. `web_search_fallback` — if not, searches the live web via Tavily
  5. `generate_answer` — answers using whichever context passed the check
  6. `verify_answer` — checks the answer is grounded and on-topic; loops
     back to retry (up to 2 times) if it fails either check
- `app.py` — Streamlit UI: upload a PDF, index it, ask questions, and see
  *how* the answer was produced (which source, how many retries).

## Notes on cost

Everything here runs on free tiers:
- Groq: free tier LLM calls
- HuggingFace embeddings: run locally, zero cost
- FAISS: local, zero cost
- Tavily: free tier web search

No credit card required for normal development/testing use.

## Suggested next steps

- Swap `all-MiniLM-L6-v2` for a stronger embedding model once you outgrow
  the free local one.
- Add a small eval script that runs a fixed set of test questions through
  the graph and logs retry rates - this is how real teams monitor RAG
  quality.
- Try LangGraph's built-in visualization (`graph.get_graph().draw_mermaid()`)
  to see your pipeline as a diagram - great for a portfolio README.