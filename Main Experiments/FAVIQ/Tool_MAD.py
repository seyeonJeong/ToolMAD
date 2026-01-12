from debate import Dabate
from rag_agent import RAG_Agent
from search_agent import SearchAgent
from normal_agent import normal_llm
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

import os
import json

os.environ["TAVILY_API_KEY"] = "your-tavily-api-key"

# for RAG
model_folder = 'your-model-folder-path'
embed_name = model_folder + "Alibaba-NLP/gte-large-en-v1.5"
encoder = SentenceTransformer(embed_name, trust_remote_code= True)
connections.connect(host = "localhost", port = "your-port", token = "your-token")
collection_lists = 'your-collection-name'
collection = Collection(name = collection_lists)


rag_agent = RAG_Agent(
        agent_name="RAGAgent",
        debate_mode="init_debate",
        model_name="gpt-4o-mini",
        sleep_time=0.5,
        api_key="your-api-key",
        collection = collection,
        encoder = encoder
    )

search_agent =  SearchAgent(
        agent_name="Tavily_Agent",
        agent_role="Agent using Tavily API",
        debate_mode= "init_debate",
        model_name="gpt-4o-mini",
        sleep_time=0.5,
        api_key="your-api-key"
    )

normal_agent = normal_llm(
        debate_mode="initial debate",
        agent_name="Normal LLM",
        model_name="gpt-4o-mini",
        sleep_time=0.5,
        api_key="your-api-key"
    )

agents = [rag_agent, search_agent, normal_agent]

file_path = "Main Experiments\FAVIQ\experiment_dataset\faviq_sample_dataset(main_baseline).json"

with open(file_path, "r", encoding="utf-8") as f:
    sampled_items = json.load(f)
    
debate_history = []


for idx, item in enumerate(sampled_items, start = 1):
    claim = item.get("claim", "No claim provided")
    debate = Dabate(agents = agents, claim = claim, collection=collection, encoder=encoder)
    
    ground_truth = item.get("label", "UNKNOWN").upper().strip()
    debate_history.append(debate.init_store(ground_truth))
    
    
    for i in range(3):
        
        print(f"☀️Round {i+1} start☀️")
        
        if i == 0:
            print("☀️Start Init process☀️")
            debate.init_query_select()
            debate.init_debate()
        else:
            print("☀️Start debate process☀️")
            debate.query_select()
            debate.debate()
        
        print("☀️Check Consensus Process☀️")
        
        if debate.check_consensus():
            if debate.check_score():
                if debate.check_gt(ground_truth):
                    debate_history[idx-1]["debate_results"].append(debate.store_debate(ground_truth,True)) 
                else:
                    debate_history[idx-1]["debate_results"].append(debate.store_debate(debate.final_answer,False))
                break
            else:
                if i == 2:
                    debate_history[idx-1]["debate_results"].append(debate.store_debate(ground_truth,"Continue"))
                    debate.avoid_infinite_loop()
                    if ground_truth in debate.final_answer:
                        debate_history[idx-1]["debate_results"].append(debate.store_avoid_infinite_loop(True))
                    else:
                        debate_history[idx-1]["debate_results"].append(debate.store_avoid_infinite_loop(False))
                else:
                    debate_history[idx-1]["debate_results"].append(debate.store_debate(ground_truth,"Continue"))
                    debate.plus_round()
        else:
            if i == 2:
                debate_history[idx-1]["debate_results"].append(debate.store_debate(ground_truth,"Continue"))
                debate.avoid_infinite_loop()
                if ground_truth in debate.final_answer:
                    debate_history[idx-1]["debate_results"].append(debate.store_avoid_infinite_loop(True))
                else:
                    debate_history[idx-1]["debate_results"].append(debate.store_avoid_infinite_loop(False))
            else:
                debate_history[idx-1]["debate_results"].append(debate.store_debate(ground_truth,"Continue"))
                debate.plus_round()
            
        
        debate.add_history()
        
    match_flag = debate_history[idx-1]["debate_results"][-1]["match"]
    pred_label = debate.final_answer if hasattr(debate, "final_answer") else "UNKNOWN"
    current_match_count = sum(1 for r in debate_history if r["debate_results"][-1]["match"])


total = len(sampled_items)
supports_count = sum(1 for r in debate_history if r["ground_truth"] == "SUPPORTS")
refutes_count = sum(1 for r in debate_history if r["ground_truth"] == "REFUTES")
not_enough_info_count = sum(1 for r in debate_history if r["ground_truth"] == "NOT ENOUGH INFO")
match_count = sum(1 for r in debate_history if r["debate_results"][-1]["match"])

summary = {
    "total": total,
    "supports_count": supports_count,
    "refutes_count": refutes_count,
    "not_enough_info_count" : not_enough_info_count,
    "match_count": match_count,
    "accuracy": match_count / total if total > 0 else 0
}

output = {
    "summary": summary,
    "results": debate_history
}


with open("your-result-file-path", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)