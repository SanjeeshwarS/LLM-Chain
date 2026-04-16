# 🦙 LLM-Chain: Local AI Engineering Workspace

A comprehensive collection of production-ready AI workflows, autonomous agents, and structured pipelines built with **LangChain (LCEL)** and **Ollama**. This project focuses on high-performance, 100% local AI orchestration.

---

## 🚀 Overview

`LLM-Chain` is a sandbox for exploring advanced **LangChain Expression Language (LCEL)** patterns. From simple sequential chains to complex autonomous agents with tool-calling capabilities, this repository demonstrates how to build robust AI applications without relying on cloud-based APIs.

### Key Pillars:
- **Local-First:** Privacy and performance by running everything on Ollama.
- **Type Safety:** Leveraging Pydantic for structured, predictable LLM outputs.
- **Architectural Patterns:** Sequential, Parallel, and Conditional Routing.
- **Extensibility:** Easy-to-integrate tool-calling for deterministic tasks.

---

## 🛠️ Project Structure

| Module | Name | Core Concept | Description |
| :--- | :--- | :--- | :--- |
| `01` | [Structured Output Agent](./01_structured_output_agent.py) | **Pydantic Parsing** | Generates project blueprints with strict JSON schemas. |
| `02` | [Sequential Chain Pipeline](./02_sequential_chain_pipeline.py) | **Linear Chaining** | Orchestrates multi-step data flow between prompts. |
| `03` | [Parallel Execution Chain](./03_parallel_execution_chain.py) | **Fan-Out / Fan-In** | Concurrent processing for diverse output generation. |
| `04` | [Sentiment Router Agent](./04_sentiment_router_agent.py) | **Dynamic Routing** | Sentiment-aware persona switching via `RunnableBranch`. |
| `05` | [Tool Test Agent](./05_tool_test.py) | **Tool Calling** | Autonomous execution of Python functions for math/logic. |

---

## 🏗️ Getting Started

### 1. Prerequisites
- [Ollama](https://ollama.com/) installed and running.
- [Python 3.12+](https://www.python.org/downloads/).
- [uv](https://github.com/astral-sh/uv) (Recommended) for fast dependency management.

### 2. Model Installation
Pull the models used in the examples:
```bash
ollama pull llama3.1:8b
ollama pull llama3.2
```

### 3. Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd LLM-Chain

# Setup environment with uv
uv venv
source .venv/Scripts/activate  # Windows: .venv\Scripts\activate

# Install dependencies
uv sync
```

### 4. Running Examples
```bash
python 01_structured_output_agent.py
```

---

## 🧪 Advanced Features

### Structured Output (`01`)
Uses `llm.with_structured_output()` to force Llama into returning valid Pydantic objects. Perfect for frontend integration.

### Sentiment Routing (`04`)
Implements a decision-making engine that classifies input sentiment and routes it to specialized prompt handlers (Hype-Man, Critic, or Diplomat).

### Tool Augmentation (`05`)
Demonstrates how to bind Python functions to the LLM, allowing it to perform deterministic calculations instead of hallucinating results.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
