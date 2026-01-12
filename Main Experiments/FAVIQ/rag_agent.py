from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import openai
import time



class RAG_Agent:
    def __init__(self, agent_name: str, debate_mode : str, model_name : str, sleep_time : float, api_key : str, collection: Collection, encoder: SentenceTransformer):
        
        self.agent_name = agent_name
        self.debate_mode = debate_mode
        self.model_name = model_name
        self.tools = "RAG"
        self.sleep_time = sleep_time
        self.api_key = api_key
        self.collection = collection
        self.encoder = encoder
        
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

    def set_debate_mode(self, debate_mode : str):
        self.debate_mode = debate_mode
        
    def query(self, prompt : str, query : str = "", search_result : str = "", used_query : str = "", before_answer : str = "", summerize : str = "",low_score_alert : str = ""):

        
        client = openai.OpenAI(api_key=self.api_key)
        
        if self.debate_mode == "init_query_select":
            full_prompt = f'''
            Choose an approrpiate query based on the given claim.\n\n
            Only output the query, and wrap it in square brackets like this: [your query here]. Do not include anything else.\n\n
            Claim:{prompt}\n
            '''
            system_message = f"You are a participant in a fact-checking debate. Our goal is to reach a consensus with an accurate answer. You are an agent utilizing {self.tools}."
        
        elif self.debate_mode == "init_debate":
            full_prompt = f'''
            If the claim is correct, you must first explain why it is correct based on the document, then output **SUPPORTS**. 
            If the claim is incorrect, you must first explain why it is incorrect based on the document, then output **REFUTES**.

            Your final answer (**SUPPORTS** or **REFUTES**) must appear on the last line only, after your reasoning.

            Documents : {search_result}\n
            Claim : {prompt}\n
            '''    
            system_message = "You are a participant in a fact-checking debate. Based on the documents you have retrieved and the given claim, determine your response."
            
        elif self.debate_mode == "query_select":
            system_message = f"You are a participant in a fact-checking debate. Our goal is to reach a consensus with an accurate answer. You are an agent utilizing {self.tools}.\n\n"
            full_prompt = f'''
                You are in query select page, you can choose change query or continue use your query\n\n
                Only output the query, and wrap it in square brackets like this: [your query here]. Do not include anything else.\n\n
                Other debaters answer {summerize}\n
                Before you used query : {used_query}\n
                Claim : {prompt}\n
                '''
        
        elif self.debate_mode == "debate":
            system_message = "You should determine your answer based on the documents you have retrieved and the other debaters answer, and the given claim."
            full_prompt = f'''
            If the claim is correct, you must first explain why it is correct based on the document, then output **SUPPORTS**. 
            If the claim is incorrect, you must first explain why it is incorrect based on the document, then output **REFUTES**.

            Your final answer (**SUPPORTS** or **REFUTES**) must appear on the last line only, after your reasoning.

            Document = {search_result}
            Other debaters answer : {summerize}
            Claim : {prompt}
            '''
            
            if low_score_alert != "":
                full_prompt = low_score_alert + full_prompt
            
        messages = [
            {"role" : "system", "content" : system_message},
            {"role" : "user", "content": full_prompt}
        ]
        
        try:
            response = client.chat.completions.create(
                model = self.model_name,
                messages = messages,
                temperature = 0.0
            )
            
            answer = response.choices[0].message.content
            
        except Exception as e:
            answer = f"Error: {e}"
            
        time.sleep(self.sleep_time)
        return answer

        
        
        