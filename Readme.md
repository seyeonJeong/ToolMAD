# Tool-MAD: Multi-Agent Debate Framework with External Tools

This repository implements **Tool-MAD**, a multi-agent debate framework for fact verification that integrates Retrieval-Augmented Generation (RAG) and Search tools. The framework enables multiple agents to dynamically gather and argue over evidence, with a judge agent deciding the final verdict.

## 🔧 Features

- Dynamic multi-round debate framework
- RAG agent (e.g., Milvus + embedding)
- Web search agent
- LLM-based normal agent
- Judge agent using stability score
- Sample FEVER-style dataset evaluation

---

## 📁 Project Structure
├── Tool_MAD.py # Main script to run debate
├── rag_agent.py # Retrieval-based agent (e.g., Milvus RAG)
├── search_agent.py # Search engine-based agent
├── normal_agent.py # Judge LLM agent
├── debate.py # Debate flow and scoring
├── fever_sample_dataset(main_baseline).json # Sample dataset
└── requirements.txt # Dependency list

## 🚀 Quick Start

### 1. Install Dependencies

pip install -r requirements.txt

### 2. Select dataset

e.g., cd FEVER

### 3. Run Tool-MAD

python Tool_MAD.py
