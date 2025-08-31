import streamlit as st
from src.langgraph_agent.ui.streamlit_ui.load_ui import LoadStreamlitUI
from src.langgraph_agent.ui.streamlit_ui.display_result import DisplayResultStreamlit
from src.langgraph_agent.llm.llm import Googlellm
from src.langgraph_agent.graph.graph_builder import GraphBuilder
from src.langgraph_agent.state.state import State
from langchain_core.messages import HumanMessage, AIMessage

def load_langgraph_agenticai_app():
    """Loads and run the LanGraph AgenticAI application with streamlitUI."""
    
    st.title("🤖 Insurance Agent Assistant")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "display" not in st.session_state:
        st.session_state.display = DisplayResultStreamlit()
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])
    
    # Chat input
    user_message = st.chat_input("Enter your message: ")
    
    if user_message:
        # Add user message to chat
        st.chat_message("user").write(user_message)
        st.session_state.messages.append({"role": "user", "content": user_message})
        
        # Initialize LLM and Graph
        model = Googlellm()
        llm_model = model.get_llm_model()
        
        graphBuilder = GraphBuilder(llm_model)
        
        try:
            # Build and run graph
            graph = graphBuilder.build_graph()
            
            # Create initial state
            initial_state = State(
                messages=[HumanMessage(content=user_message)],
                current_result={},
                status="processing"
            )
            
            # Run the graph
            result = graph.invoke(initial_state)
            
            print(f"Debug - Full result object: {result}")
            print(f"Debug - Result type: {type(result)}")
            
            # Extract the result from the final state
            final_result = {}
            
            if isinstance(result, dict):
                final_result = result.get("current_result", {})
            elif hasattr(result, 'current_result'):
                final_result = result.current_result
            else:
                final_result = {}
            
            print(f"Debug - Final result: {final_result}")
            
            # Generate response based on result
            if final_result.get("status") == "success":
                if "image_analysis" in final_result:
                    response = f"✅ Property Analysis Complete!\n\n"
                    response += f"**Property ID:** {final_result.get('zpid', 'N/A')}\n\n"
                    response += f"**Image Analysis:**\n"
                    for key, value in final_result.get("image_analysis", {}).items():
                        response += f"- {key}: {value}\n"
                    response += f"\n**Risk Analysis:**\n"
                    if "risk_analysis" in final_result:
                        risk = final_result["risk_analysis"]
                        response += f"- Risk Score: {risk.get('Risk Score', 'N/A')}/100\n"
                        response += f"- Risk Factors: {', '.join(risk.get('Risk Factors', []))}\n"
                else:
                    response = final_result.get("response", "Analysis completed successfully!")
            else:
                response = final_result.get("message", "An error occurred during processing.")
            
            # Add assistant response to chat
            st.chat_message("assistant").write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Save results to JSON (your existing functionality)
            if final_result.get("status") == "success" and "zpid" in final_result:
                zpid = final_result["zpid"]
                import json
                with open(f"property_{zpid}_complete_analysis.json", "w") as f:
                    json.dump(final_result, f, indent=4)
                st.success(f"✅ Results saved to property_{zpid}_complete_analysis.json")
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            st.chat_message("assistant").write(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.error(f"Graph execution failed: {str(e)}")

