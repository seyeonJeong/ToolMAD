import time
import openai

class normal_llm:
    def __init__(self, debate_mode : str, agent_name: str, model_name: str, sleep_time: float, api_key : str,prompt_template: str = "Context:\n{context}\n\nQuestion: {question}"):
        self.agent_name = agent_name
        self.debate_mode = debate_mode
        self.model_name = model_name
        self.sleep_time = sleep_time
        self.api_key = api_key
        self.tools = "Normal"
        
    def set_debate_mode(self, debate_mode : str):
        self.debate_mode = debate_mode
        
    def query(self, prompt : str = "", a_answer: str = "", b_answer: str="", c_answer: str="",rag_answer : str="", search_answer : str="", normal_answer : str="", summerize : str = "", rag_documents : str = "", search_documents : str = "", debate_history : list = [], rag_faithfullness : float = 0.0, rag_relevance : float = 0.0, search_faithfullness : float = 0.0, search_relevance : float = 0.0):
        
        client = openai.OpenAI(api_key=self.api_key)


        if self.debate_mode == "judge":
            full_prompt = (
                f"The faithfulness score measures how accurately the agent's answer reflects the retrieved documents, while the answer relevancy score indicates how well the answer addresses the original question.\n"
                f"A higher score means better alignment and greater reliability\n"
                f"Based on the debate history and faithfulness and answer relevancy score, please determine the correctness of the claim as follows:\n\nif the claim is correct, output **SUPPORTS**\n\n if the claim is incorrect, output **REFUTES**\n\nif it is uncertain whether the claim is correct, output **NOT ENOUGH INFO**\n\n"
                f"Debate History: {debate_history}\n\n"
                f"RAG Agent's Faithfullness Score: {rag_faithfullness / 3}\n\n"
                f"RAG Agent's Relevancy Score : {rag_relevance / 3}\n\n"
                f"Search Agent's Faithfullness Score : {search_faithfullness / 3}\n\n"
                f"Search Agent's Relevancy Score : {search_relevance / 3}\n\n"
                f"Claim: {prompt}\n\n")
            
            system_message = "You are the judge for the fact-checking debate. This is the final round of the debate. If no consensus has been reached, you must determine the correct answer"


        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": full_prompt}
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
    
