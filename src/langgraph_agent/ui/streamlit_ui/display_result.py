import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


class DisplayResultStreamlit:
    def __init__(self):
        pass
    
    def display_conversation(self, user_question: str, graph_result: dict):
        """Display user question and graph result"""
        
        # Display user question
        st.markdown(f"**You:** {user_question}")
        
        # Display assistant response
        if graph_result.get("status") == "success":
            st.markdown("**Assistant:** Here's your analysis:")
            
            # Show the result data
            if "image_analysis" in graph_result:
                st.json(graph_result["image_analysis"])
            
            if "risk_analysis" in graph_result:
                st.json(graph_result["risk_analysis"])
                
        else:
            st.error(f"**Assistant:** {graph_result.get('message', 'Error occurred')}")
    
    def simple_display(self, user_input: str, result: dict):
        """Simple display method"""
        st.write(f"**Question:** {user_input}")
        st.write(f"**Answer:** {result}")