import os

OPENAI_API_KEY = "your-openai-api-key-here"
TAVILY_API_KEY = "your-tavily-api-key-here"
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
MILVUS_TOKEN = "your-token"
MILVUS_COLLECTION_NAME = "your-collection-name"

MODEL_FOLDER = "your-model-folder-path"
EMBED_MODEL_NAME = "Alibaba-NLP/gte-large-en-v1.5"
LLM_MODEL_NAME = "gpt-4o-mini"

RESULT_FILE_PATH = "your-result-file-path"
AGENT_SLEEP_TIME = 0.5
