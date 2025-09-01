import streamlit as st
from src.langgraph_agent.ui.streamlit_ui.load_ui import LoadStreamlitUI
from src.langgraph_agent.ui.streamlit_ui.display_result import DisplayResultStreamlit
from src.langgraph_agent.llm.llm import Googlellm
from src.langgraph_agent.graph.graph_builder import GraphBuilder
from src.langgraph_agent.state.state import State
from langchain_core.messages import HumanMessage, AIMessage
import time
import random

# Add some randomness to make it feel more natural
def get_thinking_message(step):
    messages = {
        "analyzing": [
            "🔍 Analyzing your question...",
            " Understanding your request...",
            "📝 Processing your question..."
        ],
        "routing": [
            "🔄 Routing to appropriate agent...",
            "🎯 Finding the right specialist...",
            "🚀 Activating specialized agent..."
        ],
        "searching": [
            "🔎 Searching policy documents...",
            "📚 Looking through insurance terms...",
            "🔍 Finding relevant information..."
        ],
        "processing": [
            " Processing with AI model...",
            "🧠 AI is analyzing the context...",
            "💭 Generating comprehensive answer..."
        ]
    }
    return random.choice(messages.get(step, ["Processing..."]))

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
            
            # Show real-time thinking process using placeholders
            thinking_placeholder = st.empty()
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            # Step 1: Question Analysis
            thinking_placeholder.markdown("🤖 **Agent is thinking...**")
            status_placeholder.text("🔍 Analyzing your question...")
            progress_placeholder.progress(20)
            time.sleep(0.8)
            
            # Step 2: Agent Routing
            status_placeholder.text("🔄 Routing to appropriate agent...")
            progress_placeholder.progress(40)
            time.sleep(0.8)
            
            # Step 3: Determine which agent will be used
            # We need to run the orchestrator first to know which agent to activate
            orchestrator_result = None
            try:
                # Create a temporary state for orchestrator
                temp_state = State(
                    messages=[HumanMessage(content=user_message)],
                    current_result={},
                    status="processing"
                )
                
                # Run just the orchestrator to see which agent it chooses
                orchestrator_result = graphBuilder.orchestrator_node(temp_state)
                next_agent = orchestrator_result.get("next", "unknown")
                
                # Show the specific agent being activated
                if next_agent == "image_analysis":
                    status_placeholder.text("🔍 Activating Image Analysis Agent...")
                    progress_placeholder.progress(60)
                    time.sleep(0.8)
                    
                    status_placeholder.text("🌐 Searching Zillow for property...")
                    progress_placeholder.progress(80)
                    time.sleep(1.2)
                    
                    status_placeholder.text("🖼️ Analyzing property images...")
                    progress_placeholder.progress(90)
                    time.sleep(1.0)
                    
                elif next_agent == "terms_conditions":
                    status_placeholder.text("🔍 Activating RAG Agent...")
                    progress_placeholder.progress(60)
                    time.sleep(0.8)
                    
                    status_placeholder.text("🔎 Searching policy documents...")
                    progress_placeholder.progress(80)
                    time.sleep(1.2)
                    
                    status_placeholder.text("🧠 Processing with AI model...")
                    progress_placeholder.progress(90)
                    time.sleep(1.0)
                    
                elif next_agent == "recommendation_agent":
                    status_placeholder.text("💡 Activating Recommendation Agent...")
                    progress_placeholder.progress(60)
                    time.sleep(0.8)
                    
                    status_placeholder.text("🔍 Analyzing your concern...")
                    progress_placeholder.progress(80)
                    time.sleep(1.2)
                    
                    status_placeholder.text("💭 Generating recommendations...")
                    progress_placeholder.progress(90)
                    time.sleep(1.0)
                    
                else:  # general_response
                    status_placeholder.text("🤖 Activating General Response Agent...")
                    progress_placeholder.progress(60)
                    time.sleep(0.8)
                    
                    status_placeholder.text("💭 Processing your question...")
                    progress_placeholder.progress(80)
                    time.sleep(1.2)
                    
                    status_placeholder.text("✨ Generating response...")
                    progress_placeholder.progress(90)
                    time.sleep(1.0)
                
                # Step 4: Complete
                status_placeholder.text("✅ Generating response...")
                progress_placeholder.progress(100)
                time.sleep(0.5)
                
            except Exception as e:
                # Fallback to generic thinking if orchestrator fails
                status_placeholder.text("🤖 Processing your request...")
                progress_placeholder.progress(90)
                time.sleep(1.0)
                
                status_placeholder.text("✅ Generating response...")
                progress_placeholder.progress(100)
                time.sleep(0.5)
            
            # Run the actual graph
            result = graph.invoke(initial_state)
            
            # Clear ALL placeholders - this will remove the thinking messages completely
            thinking_placeholder.empty()
            progress_placeholder.empty()
            status_placeholder.empty()
            
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
                    # Handle RAG and other responses - ONLY show the actual answer
                    response = final_result.get("response", "Analysis completed successfully!")
                    
                    # Add citation showing which document chunks were used
                    if "search_details" in final_result and final_result["search_details"]:
                        response += f"\n\n---\n\n**📚 Source Citations:**\n"
                        response += f"*This answer was generated based on the following policy document sections:*\n\n"
                        
                        # Show all chunks that were used (not just the first one)
                        for chunk in final_result["search_details"][:3]:  # Show top 3 chunks
                            response += f"**Chunk {chunk['rank']}:**\n"
                            response += f"```\n{chunk['content_preview']}\n```\n\n"
                        
                        response += f"**Total Documents Used:** {final_result.get('documents_found', 0)} policy sections"
                    
                    # Add agent type if available
                    if "agent_type" in final_result:
                        response += f"\n\n**Agent:** {final_result['agent_type']}"
            else:
                response = final_result.get("message", "An error occurred during processing.")
                
                # Don't show process flow for errors either
                # Just show the error message
            
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
            # Clear thinking container on error
            if 'thinking_container' in locals():
                thinking_container.empty()
            error_msg = f"❌ Error: {str(e)}"
            st.chat_message("assistant").write(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.error(f"Graph execution failed: {str(e)}")


