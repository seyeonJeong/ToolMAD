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