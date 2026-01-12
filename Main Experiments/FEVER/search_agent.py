import requests
import time
import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnablePassthrough


class SearchAgent:
    def __init__(self, agent_name: str, agent_role: str, debate_mode : str, model_name: str, sleep_time: float, api_key: str):

        self.agent_name = agent_name
        self.agent_role = agent_role
        self.debate_mode = debate_mode
        self.model_name = model_name
        self.tools = "Web Search"
        self.sleep_time = sleep_time
        self.api_key = api_key
    
    def set_debate_mode(self, debate_mode : str):
        self.debate_mode = debate_mode
        
    def format_docs(self, docs):
        if not isinstance(docs, list):
            print(f"Error: docs is not a list, but {type(docs)}")
            return ""

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

    
    def query(self, prompt: str, query : str = "", search_result : str = "", used_query : str = "", before_answer : str = "", summerize : str = "", low_score_alert : str = ""):

        
        client = OpenAI(api_key=self.api_key)
        
        
        
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
            If the evidence is unclear or inconsistent, you must first explain the uncertainty, then output **NOT ENOUGH INFO**.

            Your final answer (**SUPPORTS**, **REFUTES**, or **NOT ENOUGH INFO**) must appear on the last line only, after your reasoning.
            
            Claim : {prompt} 
            Document = {search_result}
            '''
            
            system_message = "You are a participant in a fact-checking debate. Based on the documents you have retrieved and the given claim, determine your response."
            
        elif self.debate_mode == "query_select":
                
            system_message = f"You are a participant in a fact-checking debate. Our goal is to reach a consensus with an accurate answer. You are an agent utilizing {self.tools}.\n\n"
            full_prompt = f'''
                You are in query select page, you can choose change query or continue use your query\n\n
                If your previous query was contradicted by the opponent's argument, or left parts of the claim unresolved, you are required to revise your query to directly address the disagreement.\n\n
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
            If the evidence is unclear or inconsistent, you must first explain the uncertainty, then output **NOT ENOUGH INFO**.

            Your final answer (**SUPPORTS**, **REFUTES**, or **NOT ENOUGH INFO**) must appear on the last line only, after your reasoning.
            
            Claim : {prompt} 
            Document = {search_result}
            Other debaters answer : {summerize}
            '''

            if low_score_alert != "":
                full_prompt = low_score_alert + full_prompt
            
            print(full_prompt)

        
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