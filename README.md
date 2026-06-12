# Local AI Architecture Playground

A local, dependency-free Python learning project that evolves a terminal
assistant into a modular architecture assistant. It supports transparent
lexical retrieval and optional response generation through local Ollama.

## Current Capabilities

- Persistent conversation history and a recent-context window
- Local text knowledge base
- Overlapping document chunking
- Explainable weighted retrieval
- Filename and keyword-based topic metadata
- Deterministic ranked search results
- Optional grounded generation with a local Ollama model
- Reusable interfaces for future skills, tools, APIs, and agents

The detailed project documentation is available in
[`docs/README.md`](docs/README.md).

## Project Status

Demo 7 is the first published milestone:

- local lexical RAG with explainable ranking;
- optional grounded generation with Ollama and `qwen3:4b`;
- explicit grounded, general, retrieval, and fallback answer modes;
- a versioned 10-question retrieval evaluation;
- structured output ready for future tools, skills, and MCP adapters.

Run the milestone checks:

```bash
python -m unittest discover -v
python -m app.evaluate
python -m app.demo
```

The frozen baseline is available at
[`demo-7-local-rag`](https://github.com/jsc-cfb-ia/local-ai-architecture-playground/tree/demo-7-local-rag).

The current retrieval score favors terms in the active question, then adds
smaller boosts for recent context, source filenames, topic metadata, and query
coverage. This is intentionally simple enough to inspect before introducing
embeddings.

## Project Structure

```text
app/
  __init__.py
  context_manager.py
  main.py
  models.py
  retriever.py
  assistant.py
knowledge/
memory/
tests/
```

## Run

From the project root:

```bash
python -m app.main
```

The default mode uses Ollama with `qwen3:4b`. If Ollama is unavailable, a
grounded question falls back to lexical retrieval.

## Optional Ollama Mode

Install Ollama, then start with a small model:

```bash
ollama pull qwen3:4b
LOCAL_AI_PROVIDER=ollama OLLAMA_MODEL=qwen3:4b python -m app.main
```

The application uses `qwen3:4b` as its default Ollama model.
Model downloads are free, but require local disk space and an initial internet
connection. Once downloaded, inference and application traffic stay local.

If Ollama is stopped or the configured model is missing, the assistant reports
the problem and falls back to lexical retrieval.

When no relevant local document exists, Ollama can answer from its general
model knowledge. The terminal marks that response as `Answer mode: general`
and explicitly reports that no local source supports it.

To force retrieval-only mode:

```bash
LOCAL_AI_PROVIDER=retrieval python -m app.main
```

Available commands:

- `memory`: show persistent conversation history
- `status`: inspect the configured provider and model
- `clear context`: reset recent conversational context
- `exit`: close the assistant

## Test

```bash
python -m unittest discover -v
```

## Evaluate Retrieval

Run the versioned 10-question retrieval benchmark:

```bash
python -m app.evaluate
```

Demo 7 expects 100% top-1 source accuracy. The dataset is stored in
[`evaluation/questions.json`](evaluation/questions.json).

## Demo

Run the complete local RAG demo:

```bash
python -m app.demo
```

Run the same flow without an LLM:

```bash
python -m app.demo --mode retrieval
```

The presenter script and expected output are documented in
[`docs/demo-guide.md`](docs/demo-guide.md).

## Retrieval API

`app.retriever.search()` returns structured `SearchResult` objects for future
tools, local embedding models, or response synthesis. `search_best_match()`
formats the highest-ranked result for the current terminal interface.

## Reusable Assistant API

`app.assistant.ArchitectureAssistant` is the application core. It returns a
structured `AssistantResponse` containing content, sources, generation mode,
and an optional warning.

`app.models.LanguageModel` is a small protocol. A future local model runtime,
test double, agent framework, or remote provider can implement the same
`generate(system_prompt, user_prompt)` method without changing retrieval.

This separation supports several future adapters:

- CLI command
- Local HTTP API
- Agent tool
- MCP server tool
- Codex or Claude skill script
- Multi-agent architecture workflow

## Next Milestones

1. Add richer document metadata and paragraph-aware chunk boundaries.
2. Evaluate retrieval with a small architecture question dataset.
3. Add optional local embeddings with a lexical fallback.
4. Expose `ArchitectureAssistant` through a local tool or MCP server.
5. Introduce architecture-specific tools and agent workflows.
