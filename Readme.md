# Tool-MAD: A Multi-Agent Debate Framework for Fact Verification with Diverse Tool Augmentation and Adaptive Retrieval

## Abstract
Large Language Models (LLMs) suffer from hallucinations and factual inaccuracies, especially in complex reasoning and fact verification tasks. Multi-Agent Debate (MAD) systems aim to improve answer accuracy by enabling multiple LLM agents to engage in dialogue, promoting diverse reasoning and mutual verification. However, existing MAD frameworks primarily rely on internal knowledge or static documents, making them vulnerable to hallucinations. While MADKE introduces external evidence to mitigate this, its one-time retrieval mechanism limits adaptability to new arguments or emerging information during the debate. To address these limitations, We propose Tool-MAD, a multi-agent debate framework that enhances factual verification by assigning each agent a distinct external tool, such as a search API or RAG module. Tool-MAD introduces three key innovations: (1) a multi-agent debate framework where agents leverage heterogeneous external tools, encouraging diverse perspectives, (2) an adaptive query formulation mechanism that iteratively refines evidence retrieval based on the flow of the debate, and (3) the integration of Faithfulness and Answer Relevance scores into the final decision process, allowing the Judge agent to quantitatively assess the coherence and question alignment of each response and effectively detect hallucinations. Experimental results on four fact verification benchmarks demonstrate that Tool-MAD consistently outperforms state-of-the-art MAD frameworks, achieving up to 5.5% accuracy improvement. Furthermore, in medically specialized domains, Tool-MAD exhibits strong robustness and adaptability across various tool configurations and domain conditions, confirming its potential for broader real-world fact-checking applications.

## 🔧 Features

- Dynamic multi-round debate framework
- RAG agent (e.g., Milvus + embedding)
- Web search agent
- LLM-based normal agent
- Judge agent using stability score
- Sample FEVER-style dataset evaluation



## 📁 Project Structure
- Tool_MAD.py
- config.py
- rag_agent.py
- search_agent.py
- normal_agent.py
- debate.py
- fever_sample_dataset(main_baseline).json
- requirements.txt

## 🚀 Quick Start

### 1. Install Dependencies

``` sh
pip install -r requirements.txt

```

### 2. Configure API Keys and Settings

Before running Tool-MAD, you need to configure all API keys and settings in the `config.py` file located in each dataset folder.

#### 2.1 Edit config.py

Open the `config.py` file in each dataset folder (AVERITEC, FEVER, FAVIQ, FEVEROUS) and update the following values:

```python
OPENAI_API_KEY = "sk-your-openai-api-key-here"
TAVILY_API_KEY = "tvly-your-tavily-api-key-here"

MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
MILVUS_TOKEN = "your-token"
MILVUS_COLLECTION_NAME = "your-collection-name"

MODEL_FOLDER = "/path/to/your/model/folder/"
EMBED_MODEL_NAME = "Alibaba-NLP/gte-large-en-v1.5"
LLM_MODEL_NAME = "gpt-4o-mini"

RESULT_FILE_PATH = "results.json"
```

**Important**: 
- By editing only the `config.py` file, all agents (rag_agent, search_agent, normal_agent) and debate.py will automatically use the same API keys.
- Each dataset folder has its own `config.py` file, so edit the corresponding file when running experiments in each folder.

#### 2.2 Get API Keys

1. **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/)
2. **Tavily API Key**: Get from [Tavily](https://tavily.com/)
3. **Milvus**: Install and configure following the [Milvus documentation](https://milvus.io/docs)

#### 2.3 Model Download

SentenceTransformer models will be automatically downloaded on first use. Models will be saved to the path specified in `MODEL_FOLDER`.

### 3. Select dataset

``` sh
e.g., cd FEVER

```
### 4. Run Tool-MAD
``` sh
python Tool_MAD.py
```

## External Tools and Services

This project utilizes the following external tools and services as described in our paper:
- **RAG Database**: Used as the knowledge base for the RAG agent in our experiments. We use the Wikipedia dataset as described in the referenced paper. The database contains Wikipedia document embeddings stored in Milvus vector database, enabling efficient semantic search and retrieval during fact verification debates. [arXiv:2310.03714](https://arxiv.org/abs/2310.03714)

- **Tavily Search API**: Used for web search functionality in the search agent. Tavily provides real-time web search capabilities for retrieving up-to-date information during fact verification debates. [Tavily](https://tavily.com/)

- **Milvus Vector Database**: Used for RAG (Retrieval-Augmented Generation) functionality. Milvus serves as the vector database for storing and retrieving document embeddings in the RAG agent. [Milvus](https://milvus.io/)

- **SentenceTransformer**: Used for generating embeddings in the RAG agent. We employ the `Alibaba-NLP/gte-large-en-v1.5` model for document and query embeddings. Reimers & Gurevych (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. [arXiv:1908.10084](https://arxiv.org/abs/1908.10084)

- **CrossEncoder**: Used for reranking in the debate framework. We use the `cross-encoder/ms-marco-MiniLM-L-6-v2` model for cross-encoder reranking.

- **OpenAI API**: Used for LLM inference across all agents (RAG agent, search agent, normal agent, and judge agent). We use GPT-4o-mini as specified in the paper. [OpenAI](https://platform.openai.com/)

- **RAGAS**: Used for evaluating faithfulness and answer relevancy scores in the debate framework. Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. [arXiv:2309.15217](https://arxiv.org/abs/2309.15217)

## References

```
@misc{jeong2026toolmadmultiagentdebateframework,
      title={Tool-MAD: A Multi-Agent Debate Framework for Fact Verification with Diverse Tool Augmentation and Adaptive Retrieval}, 
      author={Seyeon Jeong and Yeonjun Choi and JongWook Kim and Beakcheol Jang},
      year={2026},
      eprint={2601.04742},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.04742}, 
}
```