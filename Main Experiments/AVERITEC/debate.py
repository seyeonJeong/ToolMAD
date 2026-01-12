from rag_agent import RAG_Agent
from search_agent import SearchAgent
from normal_agent import normal_llm
from langchain_community.tools.tavily_search import TavilySearchResults
from pymilvus import connections, Collection
from sentence_transformers import CrossEncoder
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas.evaluation import evaluate

import re

llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key="your-api-key",temperature = 0.0)

class Dabate:
    def __init__(self,agents,claim, collection, encoder, cross_encoder):
        self.agents = agents
        self.claim = claim
        self.rag_query = ""
        self.debate_round = 1
        self.rag_before_query = ""
        self.search_query = ""
        self.search_before_query = ""
        self.rag_documents = ""
        self.search_documents = ""
        self.rag_answer = ""
        self.search_answer = ""
        self.normal_answer = ""
        self.context = ""
        self.final_answer = ""
        self.collection = collection
        self.encoder = encoder
        self.debate_result = []
        self.debate_history = []
        self.search_retriever = TavilySearchResults(max_results=3, include_raw_content=True)
        self.cross_encoder = cross_encoder
        self.rag_faithful_score = 0.0
        self.rag_relevance_score = 0.0
        self.search_faithful_score = 0.0
        self.search_relevance_score = 0.0

        self.rag_faithful_score_sum = 0.0
        self.rag_relevance_score_sum = 0.0
        self.search_faithful_score_sum = 0.0
        self.search_relevance_score_sum = 0.0

        self.rag_low_score_alert = ""
        self.search_low_score_alert = ""
    
    def retrieve_documents(self, prompt : str, top_k: int = 1):
        prompt_embedding = self.encoder.encode([prompt])
        
        results = self.collection.search(
            data = prompt_embedding,
            anns_field = "vector",
            param = {"metric_type":"IP", "search_list" : 8},
            limit = top_k,
            output_fields = ["id","content"],
        )
        
        docs = []
        for hits in results:
            for hit in hits:
                docs.append(hit.entity.get("content"))
        return docs

    def format_docs(self, docs):

        if not isinstance(docs, list):
            print(f"Error: docs is not a list, but {type(docs)}")
            print(docs)
            return docs

        if not docs:
            print("Warning: docs is empty or None.")
            return ""

        formatted_docs = []
        for d in docs:
            if not isinstance(d, dict):
                print(f"Warning: Expected dict but got {type(d)}: {d}")
                continue
            if 'content' not in d:
                print(f"Warning: Missing 'content' key in document: {d}")
                formatted_docs.append("No content available")
            else:
                formatted_docs.append(d['content'])

        return "\n\n".join(formatted_docs)
    
    
    def parse_query(self,query: str) -> str:
        query = query.strip()
    
        match = re.search(r'\[(.*?)\]', query)
    
        return match.group(1).strip() if match else query

    def compute_ragas_scores(self, question, answer, contexts):

        context_list = contexts if isinstance(contexts, list) else [contexts]

        data = {
            "user_input": [question],
            "response": [answer],
            "contexts": [context_list],
        }
        dataset = Dataset.from_dict(data)

        results = evaluate(dataset, metrics=[faithfulness, answer_relevancy], llm=llm)

        return float(results["faithfulness"][0]), float(results["answer_relevancy"][0])
    
    def init_query_select(self):
        for i in self.agents:
            i.set_debate_mode("init_query_select")
            if i.tools == "RAG":
                self.rag_query = self.parse_query(i.query(prompt = self.claim))
            elif i.tools == "Web Search":
                self.search_query = self.parse_query(i.query(prompt = self.claim))
            else:
                break

    def init_debate(self):
        for i in self.agents:
            i.set_debate_mode("init_debate")
            if i.tools == "RAG":
                context_docs = self.retrieve_documents(self.rag_query, top_k = 3)
                self.rag_documents = "\n".join(context_docs) if context_docs else "No context found."
                self.rag_answer = i.query(prompt = self.claim, search_result = self.rag_documents)

                self.rag_faithful_score, self.rag_relevance_score = self.compute_ragas_scores(
                    question=self.claim,
                    answer=self.rag_answer,
                    contexts=self.rag_documents
                )
                self.set_score(self.rag_faithful_score,self.rag_relevance_score,"rag")


            elif i.tools == "Web Search":
                search_results = self.search_retriever.invoke(self.search_query)
                self.search_documents = self.format_docs(search_results)
                self.search_answer = i.query(prompt = self.claim, search_result = self.search_documents)

                self.search_faithful_score, self.search_relevance_score = self.compute_ragas_scores(
                    question=self.claim,
                    answer=self.search_answer,
                    contexts=self.search_documents
                )
                self.set_score(self.search_faithful_score, self.search_relevance_score,"search")

            else:
                break
            

    def query_select(self):
        for i in self.agents:
            i.set_debate_mode("query_select")
            if i.tools == "RAG":
                self.rag_before_query = self.rag_query
                query_result = self.parse_query(i.query(used_query = self.rag_query, before_answer = self.rag_answer, summerize = self.search_answer, prompt = self.claim))
                if "Continue" in query_result:
                    continue
                else:
                    self.rag_query = query_result

                
            elif i.tools == "Web Search":
                self.search_before_query = self.search_query
                query_result = self.parse_query(i.query(used_query = self.search_query, before_answer = self.search_answer, summerize = self.rag_answer, prompt = self.claim))
                if "Continue" in query_result:
                    continue
                else:
                    self.search_query = query_result

            else:
                break
            
    def debate(self):
        for i in self.agents:
            i.set_debate_mode("debate")
            if i.tools == "RAG":
                if self.rag_query != self.rag_before_query:
                    context_docs = self.retrieve_documents(self.rag_query, top_k = 3)
                    self.rag_documents = "\n".join(context_docs) if context_docs else "No context found."
                
                self.rag_answer = i.query(prompt = self.claim, summerize = self.search_answer, search_result = self.rag_documents, low_score_alert = self.rag_low_score_alert)

                if self.rag_low_score_alert != "":
                    self.rag_low_score_alert = ""

                self.rag_faithful_score, self.rag_relevance_score = self.compute_ragas_scores(
                    question=self.claim,
                    answer=self.rag_answer,
                    contexts=self.rag_documents
                )
                self.set_score(self.rag_faithful_score,self.rag_relevance_score,"rag")
            elif i.tools == "Web Search":
                if self.search_query != self.search_before_query:
                    search_results = self.search_retriever.invoke(self.search_query)
                    self.search_documents = self.format_docs(search_results)

                self.search_answer = i.query(prompt = self.claim, summerize = self.rag_answer,search_result = self.search_documents, low_score_alert = self.search_low_score_alert)

                if self.search_low_score_alert != "":
                    self.search_low_score_alert = ""
                    
                self.search_faithful_score, self.search_relevance_score = self.compute_ragas_scores(
                    question=self.claim,
                    answer=self.search_answer,
                    contexts=self.search_documents
                )
                self.set_score(self.search_faithful_score, self.search_relevance_score,"search")
            else:
                break
                
    def plus_round(self):
        self.debate_round += 1

    def check_score(self):
        if self.rag_faithful_score <= 0.7 or self.rag_relevance_score <= 0.8 or self.search_faithful_score <= 0.7 or self.search_relevance_score <= 0.8:
            if self.rag_faithful_score <= 0.7 or self.rag_relevance_score <= 0.8:
                if self.rag_faithful_score <= 0.7:
                    self.rag_low_score_alert += "My faithfulness score is below the threshold. This indicates that my previous response may not have been sufficiently grounded in the provided documents. I will revise my reasoning or refer back to the evidence more carefully in the next step."
                if self.rag_relevance_score <= 0.8:
                    self.rag_low_score_alert += "My answer relevance score is below the threshold. This suggests that my response may not have been sufficiently aligned with the user's question. I will refine my answer to better address the question using the most relevant parts of the evidence."

            elif self.search_faithful_score <= 0.7 or self.search_relevance_score <= 0.8:
                if self.search_faithful_score <= 0.7:
                    self.search_low_score_alert += "My faithfulness score is below the threshold. This indicates that my previous response may not have been sufficiently grounded in the provided documents. I will revise my reasoning or refer back to the evidence more carefully in the next step."
                if self.search_relevance_score <= 0.8:
                    self.search_low_score_alert += "My answer relevance score is below the threshold. This suggests that my response may not have been sufficiently aligned with the user's question. I will refine my answer to better address the question using the most relevant parts of the evidence."

            return False
        else:
            return True

    def check_consensus(self):
        if "Supported" in self.rag_answer:
            rag = "Supported"
        elif "Refuted" in self.rag_answer:
            rag = "Refuted"
        elif "Not Enough Evidence" in self.rag_answer:
            rag = "Not Enough Evidence"
        elif "Conflicting Evidence/Cherrypicking" in self.rag_answer:
            rag = "Conflicting Evidence/Cherrypicking"
        else:
            rag = ""
        
        if "Supported" in self.search_answer:
            search = "Supported"
        elif "Refuted" in self.search_answer:
            search = "Refuted"
        elif "Not Enough Evidence" in self.search_answer:
            search = "Not Enough Evidence"
        elif "Conflicting Evidence/Cherrypicking" in self.search_answer:
            search = "Conflicting Evidence/Cherrypicking"
        else:
            search = ""
            
        if rag == search:
            self.final_answer = rag
            return True
        else:
            return False
        
    def check_gt(self,gt):
        if self.final_answer == gt:
            return True
        else:
            return False
        
    def avoid_infinite_loop(self):
        judge = self.agents[-1]
        judge.set_debate_mode("judge")
        
        self.context = judge.query(prompt = self.claim, debate_history = self.debate_history, rag_faithfullness = self.rag_faithful_score_sum, rag_relevance = self.rag_relevance_score_sum, search_faithfullness = self.search_faithful_score_sum, search_relevance = self.search_relevance_score_sum)
        if "Supported" in self.context:
            self.final_answer = 'Supported'
        elif "Refuted" in self.context:
            self.final_answer = 'Refuted'
        elif "Not Enough Evidence" in self.context:
            self.final_answer = 'Not Enough Evidence'
        else:
            self.final_answer = "Conflicting Evidence/Cherrypicking"
    
    def check_judege_score(self):
        judge = self.agents[-1]
        judge.set_debate_mode("low_score")
        
        self.context = judge.query(prompt = self.claim, debate_history = self.debate_history, rag_faithfullness = self.rag_faithful_score, rag_relevance = self.rag_relevance_score, search_faithfullness = self.search_faithful_score, search_relevance = self.search_relevance_score)
        if "Supported" in self.context:
            self.final_answer = 'Supported'
        elif "Refuted" in self.context:
            self.final_answer = 'Refuted'
        elif "Not Enough Evidence" in self.context:
            self.final_answer = 'Not Enough Evidence'
        else:
            self.final_answer = "Conflicting Evidence/Cherrypicking"

    
    def get_final_answer(self):
        return self.final_answer
    
    def get_context(self):
        return self.context
    
    def set_final_answer(self,final_answer):
        self.final_answer =  final_answer
    
    def add_history(self):
        history = f"RAG Answer : {self.rag_answer}\n\nRAG documents : {self.rag_documents}\n\nSearch Answer : {self.search_answer}\n\n Search documents : {self.search_documents}\n\n"
        self.debate_history.append(history)
    
    def set_consistency_score(self,new_score,type):
        if type == "rag":
            self.rag_consistency_score += new_score
        else:
            self.search_consistency_score += new_score                               
        
    def set_score(self, faith, relevance ,type):
        if type == 'rag':
            self.rag_faithful_score_sum += faith
            self.rag_relevance_score_sum += relevance
        else:
            self.search_faithful_score_sum += faith
            self.search_relevance_score_sum += relevance
    
    def store_debate(self,gt,match):
        result_obj = {
            "debate_round" : self.debate_round,
            "RAG Query" : self.rag_query,
            "RAG Answer" : self.rag_answer,
            "RAG's score": {
                "faithfulness": self.rag_faithful_score,
                "relevance": self.rag_relevance_score
            },
            "Search's score":{
                "faithfulness" : self.search_faithful_score,
                "relevance" : self.search_relevance_score
            },
            "Search Query" : self.search_query,
            "Search Answer" : self.search_answer,
            "final_answer" : self.final_answer,
            "match" : match
        }
        
        return result_obj
    
    def store_avoid_infinite_loop(self,match):
        result_obj = {
            "debate_round" : "final",
            "RAG Query" : self.rag_query,
            "RAG Answer" : self.rag_answer,
            "Search Query" : self.search_query,
            "Search Answer" : self.search_answer,
            "Judge Answer" : self.final_answer,
            "RAG's score": {
                "final_faithfulness": self.rag_faithful_score_sum / 3.0,
                "final_relevance": self.rag_relevance_score_sum / 3.0
            },
            "Search's score":{
                "final_faithfulness" : self.search_faithful_score_sum / 3.0,
                "final_relevance" : self.search_relevance_score_sum / 3.0
            },
            "final_answer" : self.final_answer,
            "match" : match
        }
        
        return result_obj
        
    def init_store(self,gt):
        init_results = {
            "claim":self.claim,
            "ground_truth" : gt,
            "debate_results" : []
        }
        
        return init_results
        
