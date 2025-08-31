import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


load_dotenv('.env')

class Googlellm:
    def __init__(self):
        load_dotenv('.env')
        self.api_key = os.getenv("GOOGLE_API_KEY")

    def get_llm_model(self):
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=self.api_key,
                temperature = 0,
                
            )
        except Exception as e:
            raise ValueError(f"Error Occured With Exception : {e}")
        
        return llm
        
        