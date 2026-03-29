# 🦙 Llama Workspace: Local AI Engineering

A local AI engineering workspace dedicated to building advanced LangChain Expression Language (LCEL) pipelines, autonomous agents, and structured workflows. Everything in this repository runs 100% locally using Ollama and Llama 3.1/3.2 models.

## 🛠️ Tech Stack
* **Framework:** LangChain (LCEL)
* **Local LLM Engine:** Ollama
* **Alter LLM Engine:** Gemini, OpenAI, Anthropic
* **Models:** Llama 3.1 (8B), Llama 3.2 (1B/3B)
* **Data Parsing:** Pydantic (Structured Output)
* **Environment:** Python 3.12 

## 📂 Project Directory

| File Name | Core Concept | Description |
| :--- | :--- | :--- |
| `01_structured_output_agent.py` | **Pydantic Parsing** | Constrains LLM outputs into strict, predictable JSON schemas for a project blueprint generator. |
| `02_sequential_chain_pipeline.py` | **Sequential Chaining** | Uses `RunnableLambda` to intercept, reformat, and pass data between multiple LLM prompts. |
| `03_parallel_execution_chain.py` | **Fan-Out / Fan-In** | Processes a single context through multiple LLM prompts simultaneously to generate varying outputs. |
| `04_sentiment_router_engine.py` | **Conditional Routing** | A 3-way sentiment classifier that dynamically routes reviews to "Hype-Man," "Ruthless Critic," or "Neutral Diplomat" personas. |
| `05_math_tool_agent.py` | **Tool Calling** | Giving the LLM "hands" to trigger real Python functions for deterministic mathematical calculations. |

---

## 🏗️ Project Architectures Included

### 1. Structured Output Agent (`01`)
Demonstrates how to force an LLM to follow a strict schema. By using **Pydantic Data Models**, we ensure the AI returns a valid object that can be used directly in frontend applications or databases without parsing errors.

### 2. Sequential & Parallel Chaining (`02` & `03`)
Explores the power of **LCEL Pipes**. 
- **Sequential:** Shows data flowing from one LLM to another with custom Python logic in between.
- **Parallel:** Demonstrates "Fan-Out" architecture, where one input triggers multiple AI brains at once to save time and increase diversity of thought.

### 3. Semantic Sentiment Router (`04`)
An advanced implementation of **Conditional Execution (`RunnableBranch`)**. 
The pipeline first classifies user sentiment (Positive, Negative, or Neutral) and uses a "memory backpack" (`RunnablePassthrough.assign`) to carry the context forward. The router then dispatches the data to a specialized persona prompt, ensuring the AI's tone perfectly matches the user's vibe.

### 4. Autonomous Math Agent (`05`)
Moves beyond simple text generation into **Tool Use**. This agent uses a "Reasoning and Acting" (ReAct) loop. When asked a math problem, it doesn't "guess" the answer; it triggers real Python functions (Addition, Multiplication, Square Root) to provide 100% accurate results.

---
## 🚀 How to Run Locally

### 1. Prerequisites
* Install [Ollama](https://ollama.com/) and ensure it is running.
* Install [Python 3.12+](https://www.python.org/downloads/).
* (Recommended) Install [uv](https://github.com/astral-sh/uv) for fast package management.

### 2. Model Setup
Pull the required LLMs to your local machine:
```bash
ollama pull llama3.1:8b
ollama pull llama3.2

# Clone the repo
git clone <your-repo-link>
cd LangchainTrain

# Create and activate virtual environment (using uv)
uv venv
source .venv/Scripts/activate  # On Windows
# source .venv/bin/activate    # On Mac/Linux

# Install dependencies
uv pip install langchain langchain-ollama pydantic python-dotenv