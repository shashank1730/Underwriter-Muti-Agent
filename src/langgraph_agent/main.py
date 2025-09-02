import streamlit as st
from src.langgraph_agent.ui.streamlit_ui.load_ui import LoadStreamlitUI
from src.langgraph_agent.ui.streamlit_ui.display_result import DisplayResultStreamlit
from src.langgraph_agent.llm.llm import Googlellm
from src.langgraph_agent.graph.graph_builder import GraphBuilder
from src.langgraph_agent.state.state import State
from langchain_core.messages import HumanMessage, AIMessage
import time
import random
import asyncio
import threading
import re
import json

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

async def show_progress_animation(status_placeholder, progress_placeholder, next_agent, stop_event):
    """Show progress animation without blocking main execution"""
    loop_count = 0
    while not stop_event.is_set():
        loop_count += 1
        
        if next_agent == "image_analysis":
            # Continuous loop for image analysis
            status_placeholder.text("🔍 Activating High Value Assessment Agent...")
            progress_placeholder.progress(60)
            await asyncio.sleep(1.5)
            
            if stop_event.is_set():
                break
                
            status_placeholder.text("🌐 Searching Zillow for property...")
            progress_placeholder.progress(70)
            await asyncio.sleep(1.5)
            
            if stop_event.is_set():
                break
                
            status_placeholder.text("🖼️ Analyzing property images...")
            progress_placeholder.progress(80)
            await asyncio.sleep(1.5)
            
            if stop_event.is_set():
                break
                
            status_placeholder.text("🔄 Processing continues...")
            progress_placeholder.progress(75)
            await asyncio.sleep(1.0)
            
        elif next_agent == "terms_conditions":
            # Continuous loop for RAG agent
            status_placeholder.text("🔍 Activating Q&A UnderWriter Agent...")
            progress_placeholder.progress(60)
            await asyncio.sleep(1.5)
            
            if stop_event.is_set():
                break
                
            status_placeholder.text("🔎 Searching policy documents...")
            progress_placeholder.progress(70)
            await asyncio.sleep(1.5)
            
            if stop_event.is_set():
                break
                
            status_placeholder.text(" Processing with AI model...")
            progress_placeholder.progress(80)
            await asyncio.sleep(1.5)
            
            if stop_event.is_set():
                break
                
            status_placeholder.text("🔄 Processing continues...")
            progress_placeholder.progress(75)
            await asyncio.sleep(1.0)
            
        elif next_agent == "recommendation_agent":
            # Continuous loop for recommendation agent
            status_placeholder.text("💡 Activating UnderWriter Recommendation Agent...")
            progress_placeholder.progress(60)
            await asyncio.sleep(1.5)
            
            if stop_event.is_set():
                break
                
            status_placeholder.text("🔍 Analyzing your concern...")
            progress_placeholder.progress(60)
            await asyncio.sleep(1.5)
            
            if stop_event.is_set():
                break
                
            status_placeholder.text("💭 Generating recommendations...")
            progress_placeholder.progress(80)
            await asyncio.sleep(1.5)
            
            if stop_event.is_set():
                break
                
            status_placeholder.text("🔄 Processing continues...")
            progress_placeholder.progress(75)
            await asyncio.sleep(1.0)
            
        else:  # general_response
            # Continuous loop for general response
            status_placeholder.text(" Activating General Response Agent...")
            progress_placeholder.progress(60)
            await asyncio.sleep(1.5)
            
            if stop_event.is_set():
                break
                
            status_placeholder.text("💭 Processing your question...")
            progress_placeholder.progress(70)
            await asyncio.sleep(1.5)
            
            if stop_event.is_set():
                break
                
            status_placeholder.text("✨ Generating response...")
            progress_placeholder.progress(80)
            await asyncio.sleep(1.5)
            
            if stop_event.is_set():
                break
                
            status_placeholder.text("🔄 Processing continues...")
            progress_placeholder.progress(75)
            await asyncio.sleep(1.0)
        
        # Check if we should break the loop
        if loop_count >= 10:  # Show more loops for better effect
            break

def run_async_animation(status_placeholder, progress_placeholder, next_agent, stop_event):
    """Run async animation in a separate thread"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(show_progress_animation(status_placeholder, progress_placeholder, next_agent, stop_event))
    finally:
        loop.close()

def load_langgraph_agenticai_app():
    """Loads and run the LanGraph AgenticAI application with streamlitUI."""
    
    st.title("🤖 Insurance Underwriter AI Agents")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "display" not in st.session_state:
        st.session_state.display = DisplayResultStreamlit()
    
    # Show welcome suggestion box in sidebar instead of main chat
    with st.sidebar:
        st.markdown("## 🤖 Suggetion BOX")
        st.markdown("---")

        with st.expander("💡 Click to see examples", expanded=True):
            st.markdown("**🔍 High Value Assessment:** Does Image analysis & risk scoring (drop address)")
            st.markdown("**📚 Q&A UnderWriter:** Will Answer Policy related questions")
            st.markdown("**💡 UnderWriter Recommendation:** Help in Drafting client emails from previous cases")
            
            st.markdown("---")
            st.markdown("**💬 Example Questions:**")
            st.markdown("**Property:** `46 Creekstone Ln, Dawsonville, GA 30534`")
            st.markdown("**Terms:** `Does insurance cover war damage?`")
            st.markdown("**Recommendation:** `My kitchen caught fire, need help`")

    
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
            status_placeholder.text("🔍 Analyzing your question...")
            progress_placeholder.progress(20)
            time.sleep(0.8)
            
            # Step 2: Agent Routing
            status_placeholder.text("🔄 Routing to appropriate agent...")
            progress_placeholder.progress(40)
            time.sleep(0.8)
            
            # Step 3: Determine which agent will be used
            # We need to run the orchestrator first to know which agent it chooses
            orchestrator_result = None
            stop_event = threading.Event()
            animation_thread = None
            
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
                
                # Update the thinking placeholder with the routing info
                agent_names = {
                    "image_analysis": "High Value Assessment",
                    "terms_conditions": "Q&A UnderWriter", 
                    "recommendation_agent": "UnderWriter Recommendation",
                    "general_response": "General Response"
                }
                
                agent_display_name = agent_names.get(next_agent, next_agent.replace('_', ' ').title())
                thinking_placeholder.markdown(f"🤖 **Transaction routed to {agent_display_name} Agent**")
                
                # Start progress animation in background thread
                animation_thread = threading.Thread(
                    target=run_async_animation,
                    args=(status_placeholder, progress_placeholder, next_agent, stop_event)
                )
                animation_thread.start()
                
                # Let animation run for a moment to show it's working
                time.sleep(2)
                
                # Final step before running graph
                status_placeholder.text("✅ Generating response...")
                progress_placeholder.progress(80)
                time.sleep(1.0)
                
            except Exception as e:
                # Fallback to generic thinking if orchestrator fails
                status_placeholder.text("🤖 Processing your request...")
                progress_placeholder.progress(80)
                time.sleep(1.0)
                
                status_placeholder.text("✅ Generating response...")
                progress_placeholder.progress(80)
                time.sleep(0.5)
            
            # Run the actual graph (this happens while animation is running)
            result = graph.invoke(initial_state)
            
            # Stop the progress animation
            if stop_event:
                stop_event.set()
                if animation_thread:
                    animation_thread.join(timeout=1)
            
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
                    
                    # Person's Report Section (NOW AT TOP)
                    if "person_report" in final_result:
                        response += f"**📋 Exclusive Insurance Agent Property Assessment:**\n"
                        person_data = final_result["person_report"].get("Exclusive Insurance Agent Property Assessment", {})
                        for field, value in person_data.items():
                            response += f"- **{field}:** **{value}**\n"
                    
                    response += f"\n"  # Add spacing
                    
                    # Image Analysis Section
                    response += f"**🔍 AI Image Analysis from Zillow :**\n"
                    image_analysis = final_result.get("image_analysis", {})
                    for key, value in image_analysis.items():
                        response += f"- **{key}:** {value}\n"
                    
                    # Risk Analysis Section
                    response += f"\n**⚠️ Risk Analysis:**\n"
                    if "risk_analysis" in final_result:
                        risk = final_result["risk_analysis"]
                        response += f"- **Risk Score:** {risk.get('Risk Score', 'N/A')}/5\n"
                        
                        # Show Risk Factors
                        risk_factors = risk.get("Risk Factors", [])
                        if risk_factors:
                            response += f"- **Risk Factors:** "
                            for factor in risk_factors:
                                response += f"• {factor} "
                            response += f"\n"
                        else:
                            response += f"- **Risk Factors:** None identified\n"
                        
                        # Show Risk Reasoning
                        risk_reasoning = risk.get("Risk Reasoning", {})
                        if risk_reasoning:
                            response += f"\n**📝 Risk Reasoning:**\n"
                            for factor, reason in risk_reasoning.items():
                                response += f"- **{factor}:** {reason}\n"
                        
                        # Show Overall Assessment
                        overall_assessment = risk.get("Overall Risk Assessment", "")
                        if overall_assessment:
                            response += f"\n**📝 Overall Assessment:**\n{overall_assessment}\n"
                    
                    # Comparison Report Section
                    if "comparison_report" in final_result:
                        response += f"\n**📋 Comparison Report:**\n"
                        response += f"*What Person Provided vs What AI Detected*\n\n"
                        
                        comparison = final_result["comparison_report"].get("comparison", {})
                        for field, data in comparison.items():
                            actual = data.get("actual", "N/A")
                            predicted = data.get("predicted", "N/A")
                            match_icon = "✅" if str(actual).lower() == str(predicted).lower() else "❌"
                            response += f"{match_icon} **{field}:**\n"
                            response += f"  - Person: {actual}\n"
                            response += f"  - AI: {predicted}\n\n"
                    

                    
                else:
                    # Handle RAG and other responses - ONLY show the actual answer
                    response = final_result.get("response", "Analysis completed successfully!")
                    
                    # Add citation showing which document chunks were used
                    if "search_details" in final_result and final_result["search_details"]:
                        response += f"\n\n---\n\n**📚 Source Citations:**\n"
                        response += f"*This answer was generated based on the following policy document sections:*\n\n"
                        
                        # Show all chunks that were used (not just the first one)
                        for chunk in final_result["search_details"][:3]:  # Show top 3 chunks
                            response += f"**Citation {chunk['rank']}:**\n"
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
            # Clear thinking placeholders on error
            if 'thinking_placeholder' in locals():
                thinking_placeholder.empty()
            if 'progress_placeholder' in locals():
                progress_placeholder.empty()
            if 'status_placeholder' in locals():
                status_placeholder.empty()
            
            # Stop progress flow if it's running
            if 'stop_event' in locals() and stop_event:
                stop_event.set()
            
            error_msg = f"❌ Error: {str(e)}"
            st.chat_message("assistant").write(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.error(f"Graph execution failed: {str(e)}")

# Main execution
if __name__ == "__main__":
    load_langgraph_agenticai_app()


