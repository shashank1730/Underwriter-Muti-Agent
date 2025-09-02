
from langchain.prompts import ChatPromptTemplate
import os 
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

class RAGAnalysisAgent:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        
        # Initialize embeddings and vector store
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Load the existing Chroma database
        self.db = Chroma(
            collection_name="policy_docs_hf",
            persist_directory="src/langgraph_agent/tools/rag_agent/chroma_db",
            embedding_function=self.embeddings
        )
        
        self.retriever = self.db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        
        # Setup LLM and chains
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=self.api_key,
            temperature=0
        )
        
        # Create the retrieval chain
        self.setup_chain()
    
    def setup_chain(self):
        """Setup the RAG chain"""
        template = """
        - You are employed at the company that provides home insurance.
        - The queries are related to clarification on policies.
        - Answers should be informative.
        - Do not answer questions that cannot be supported by a context.


        Context:
        {context}

        Question:
        {input}

        """
        
        chat_prompt = ChatPromptTemplate.from_template(template)
        document_chain = create_stuff_documents_chain(self.llm, chat_prompt)
        self.retrieval_chain = create_retrieval_chain(self.retriever, document_chain)
    
    def process_query(self, user_question: str):
        """Process a user question using RAG with real-time progress"""
        try:
            # Step 1: Search vector database
            search_results = self.retriever.get_relevant_documents(user_question)
            
            # Get search details
            search_details = []
            for i, doc in enumerate(search_results):
                search_details.append({
                    'rank': i + 1,
                    'content_preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    'metadata': doc.metadata if hasattr(doc, 'metadata') else {}
                })
            
            # Step 2: Create context from search results
            context = "\n\n".join([doc.page_content for doc in search_results])
            
            # Step 3: Process with LLM
            response = self.retrieval_chain.invoke({"input": user_question})
            
            return {
                "status": "success",
                "response": response.get("answer", "No answer generated"),
                "message": "RAG query processed successfully",
                "search_details": search_details,
                "documents_found": len(search_results),
                "context_length": len(context),
                "query": user_question
            }
        except Exception as e:
            return {
                "status": "error",
                "response": f"Error processing query: {str(e)}",
                "message": f"RAG processing failed: {str(e)}",
                "search_details": [],
                "documents_found": 0
            }