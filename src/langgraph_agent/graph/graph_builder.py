from langgraph.graph import StateGraph, START, END
from src.langgraph_agent.state.state import State
from dotenv import load_dotenv
from src.langgraph_agent.tools.Image_analysis_agent.image_analysis_agent import ImageAnalysisAgent

class GraphBuilder:
    def __init__(self, model):
        self.llm = model
        self.graph = StateGraph(State)
        self.image_agent = ImageAnalysisAgent()
    
    def orchestrator_node(self, state):
        """Uses LLM to intelligently classify the question and route accordingly"""
        messages = state.get("messages", [])
        if not messages:
            return {"next": "error", "reason": "No messages found"}
        
        # Get the latest user message
        latest_message = messages[-1]
        
        # Fix: Access content attribute directly, not with .get()
        if hasattr(latest_message, 'content'):
            user_question = latest_message.content
        elif isinstance(latest_message, dict):
            user_question = latest_message.get("content", "")
        else:
            user_question = str(latest_message)
        
        # Create classification prompt for LLM
        classification_prompt = f"""
        You are an intelligent question classifier for an insurance/real estate system.
        
        Analyze the following user question and classify it into ONE of these categories:
        
        1. "image_analysis" - If the question contains a property address, asks for property analysis, risk assessment, or image analysis
        2. "terms_conditions" - If the question is about terms, conditions, policies, agreements, rules, or guidelines
        3. "recommendation_agent" - If the question is a complaint, issue, problem, or asks for recommendations/solutions
        4. "general_response" - If it's a general question that doesn't fit the above categories
        
        IMPORTANT: Respond with ONLY the category name (e.g., "image_analysis", "terms_conditions", etc.)
        
        User Question: "{user_question}"
        
        Classification:
        """
        
        try:
            # Use invoke instead of predict and access .content
            classification = self.llm.invoke(classification_prompt).content.strip().lower()
            
            # Clean and validate the response
            valid_categories = ["image_analysis", "terms_conditions", "recommendation_agent", "general_response"]
            
            if classification in valid_categories:
                print(f"🤖 LLM classified question as: {classification}")
                return {"next": classification, "reason": f"LLM classified as {classification}"}
            else:
                print(f"⚠️ LLM returned invalid classification: {classification}, defaulting to general_response")
                return {"next": "general_response", "reason": "Invalid LLM classification, defaulting to general"}
                
        except Exception as e:
            print(f"❌ Error in LLM classification: {str(e)}, defaulting to general_response")
            return {"next": "general_response", "reason": f"LLM classification error: {str(e)}"}

    def terms_conditions_node(self, state):
        """Handles terms and conditions questions"""
        return {
            "status": "success",
            "response": "Here are the terms and conditions for our service...",
            "message": "Terms and conditions provided"
        }
    
    def recommendation_agent_node(self, state):
        """Handles complaints and provides recommendations"""
        return {
            "status": "success",
            "response": "I understand your concern. Here are some recommendations...",
            "message": "Recommendations provided for complaint"
        }
    
    def general_response_node(self, state):
        """Handles general questions"""
        return {
            "status": "success",
            "response": "I'm here to help! How can I assist you today?",
            "message": "General response provided"
        }
    
    def error_node(self, state):
        """Handles errors"""
        return {
            "status": "error",
            "response": "I encountered an error. Please try again.",
            "message": "Error occurred"
        }
    
    def build_graph(self):
        """Build the LangGraph with LLM-powered conditional routing"""
        
        # Add all nodes
        self.graph.add_node("orchestrator", self.orchestrator_node)
        self.graph.add_node("image_analysis", self.image_agent.process)
        self.graph.add_node("terms_conditions", self.terms_conditions_node)
        self.graph.add_node("recommendation_agent", self.recommendation_agent_node)
        self.graph.add_node("general_response", self.general_response_node)
        self.graph.add_node("error", self.error_node)
        
        # Start with orchestrator
        self.graph.add_edge(START, "orchestrator")
        
        # Conditional routing based on LLM classification
        self.graph.add_conditional_edges(
            "orchestrator",
            lambda x: x["next"],
            {
                "image_analysis": "image_analysis",
                "terms_conditions": "terms_conditions", 
                "recommendation_agent": "recommendation_agent",
                "general_response": "general_response",
                "error": "error"
            }
        )
        
        # All nodes go to END
        self.graph.add_edge("image_analysis", END)
        self.graph.add_edge("terms_conditions", END)
        self.graph.add_edge("recommendation_agent", END)
        self.graph.add_edge("general_response", END)
        self.graph.add_edge("error", END)
        
        # Compile the graph
        return self.graph.compile()