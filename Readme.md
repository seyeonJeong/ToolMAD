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
- Tool_MAD.py # Main script to run debate
- rag_agent.py # Retrieval-based agent (e.g., Milvus RAG)
- search_agent.py # Search engine-based agent
- normal_agent.py # Judge LLM agent
- debate.py # Debate flow and scoring
- fever_sample_dataset(main_baseline).json # Sample dataset
- requirements.txt # Dependency list

## 🚀 Quick Start

### 1. Install Dependencies

``` sh
pip install -r requirements.txt

```

### 2. Select dataset

``` sh
e.g., cd FEVER

```
### 3. Run Tool-MAD
``` sh
python Tool_MAD.py
```

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