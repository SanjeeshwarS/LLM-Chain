# LLM-Chain: Advanced AI Orchestration

A collection of modular LangChain implementations focusing on structured workflows, autonomous agents, and deterministic AI pipelines. This workspace serves as a reference for building robust systems using LangChain Expression Language (LCEL).

## Tech Stack
* **Framework:** LangChain (LCEL)
* **LLM Engine:** Ollama (Llama 3.1 & 3.2)
* **Data Validation:** Pydantic V2
* **Environment:** Python 3.12+ / UV

## Core Architectures
* **Structured Output (01):** Type-safe schema enforcement using Pydantic.
* **Sequential Chaining (02):** Linear data flow and prompt orchestration.
* **Parallel Execution (03):** Concurrent "Fan-out" processing for diverse analysis.
* **Dynamic Routing (04):** Sentiment-aware branching and persona switching.
* **Autonomous Tools (05):** ReAct-style agent execution via function calling.

## Getting Started

### Prerequisites
Ensure [Ollama](https://ollama.com/) is installed and the required models are pulled:
```bash
ollama pull llama3.1:8b
ollama pull llama3.2
```

### Installation
Clone the repository and install the core dependencies (**LangChain**, **Pydantic**, etc.) using `uv`:
```bash
git clone <repository-url>
cd LLM-Chain
uv sync
```

### Execution
Run any module directly:
```bash
python 01_structured_output_agent.py
```

## License
MIT
