from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

class RecommendationAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0,
        )
        self.db = self._initialize_database()
    
    def _initialize_database(self):
        """Initialize the vector database with insurance complaints data"""
        try:
            # Get the current file's directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(current_dir, "recommendation_agent_db")
            excel_path = os.path.join(current_dir, "home_insurance_complaints_full.xlsx")
            
            # Check if database already exists
            if os.path.exists(db_path):
                print("📚 Loading existing recommendation agent database...")
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                return Chroma(persist_directory=db_path, embedding_function=embeddings)
            
            # If database doesn't exist, create it
            print("🔨 Building new recommendation agent database...")
            if not os.path.exists(excel_path):
                print(f"❌ Excel file not found at: {excel_path}")
                return None
            
            loader = UnstructuredExcelLoader(excel_path)
            data = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            
            chunks = text_splitter.split_documents(data)
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            # Create and persist the database
            db = Chroma.from_documents(chunks, embeddings, persist_directory=db_path)
            db.persist()
            print(f"✅ Database created and saved to: {db_path}")
            
            return db
            
        except Exception as e:
            print(f"❌ Error initializing database: {e}")
            return None
    
    def process(self, state):
        """Main processing method that takes State and returns results"""
        try:
            # Extract user question from state
            messages = state.get("messages", [])
            if not messages:
                return {
                    "current_result": {
                        "status": "error",
                        "response": "No user question found",
                        "message": "No messages in state"
                    }
                }
            
            # Get the latest user message
            latest_message = messages[-1]
            if hasattr(latest_message, 'content'):
                user_question = latest_message.content
            elif isinstance(latest_message, dict):
                user_question = latest_message.get("content", "")
            else:
                user_question = str(latest_message)
            
            # Process the user question
            result = self._process_question(user_question)
            
            return {
                "current_result": {
                    "status": "success",
                    "response": result,
                    "message": "Recommendations generated successfully",
                    "query": user_question,
                    "agent_type": "Recommendation Agent"
                }
            }
            
        except Exception as e:
            return {
                "current_result": {
                    "status": "error",
                    "response": f"Error processing request: {str(e)}",
                    "message": f"Exception occurred: {str(e)}",
                    "agent_type": "Recommendation Agent"
                }
            }
    
    def _process_question(self, user_question):
        """Process a specific user question and return recommendations"""
        if not self.db:
            return "Sorry, I couldn't access the insurance complaints database. Please try again later."
        
        try:
            # Search for similar complaints
            output_retrieval = self.db.similarity_search(user_question, k=5)
            output_retrieval_merged = "\n".join([doc.page_content for doc in output_retrieval])
            
            # Generate response using LLM
            prompt = f"""
            You are an expert home insurance underwriter. 
            Your task is to draft a professional, empathetic email to a customer based on:

            1. **Context**: Previous similar complaints and their resolutions.  
            2. **Input**: The customer's current query/problem.  

            Guidelines for the email:
            - Be polite, empathetic, and professional.  
            - Acknowledge the customer's concern clearly.  
            - Refer to similar past cases from the context, 
            - Only mention about the resolution about similar cases dont mention any other details ex:Customer Name, Place etc 
            - Dont forget mention it varies to every individual.  
            - Suggest the next steps (e.g., inspection, documentation, claim submission).  
            - Keep the tone reassuring and customer-friendly.  
            - End with a courteous closing.
            - Make it short and consice

            Context:
            {output_retrieval_merged}

            Customer Query:
            {user_question}

            Draft Email:
            """
            
            response = self.llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            return f"Sorry, I encountered an error while processing your request: {str(e)}"

# For standalone testing (if needed)
if __name__ == "__main__":
    agent = RecommendationAgent()
    test_state = {
        "messages": [{"content": "my kitchen caught fire"}]
    }
    result = agent.process(test_state)
    print(result)