import streamlit as st
from src.langgraph_agent.ui.streamlit_ui.load_ui import LoadStreamlitUI
from src.langgraph_agent.ui.streamlit_ui.display_result import DisplayResultStreamlit
from src.langgraph_agent.llm.llm import Googlellm
from src.langgraph_agent.graph.graph_builder import GraphBuilder
from src.langgraph_agent.state.state import State
from langchain_core.messages import HumanMessage, AIMessage
import time
import random
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

# Removed async animation functions to fix threading issues

def main_page():
    """Main analysis page"""
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
            st.markdown("**📋 Claims Summarizer:** Summarize claim information and notes")
            
            st.markdown("---")
            st.markdown("**💬 Example Questions:**")
            st.markdown("**Property:** `46 Creekstone Ln, Dawsonville, GA 30534`")
            st.markdown("**Terms:** `Does insurance cover war damage?`")
            st.markdown("**Recommendation:** `My kitchen caught fire, need help`")
            st.markdown("**Claims:** `Summarize CLM-2024-001` or `What happened with claim CLM-2024-002?`")

    
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
            
            # Show simple progress indicators
            thinking_placeholder = st.empty()
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            # Step 1: Question Analysis
            status_placeholder.text("🔍 Analyzing your question...")
            progress_placeholder.progress(20)
            time.sleep(0.5)
            
            # Step 2: Agent Routing
            status_placeholder.text("🔄 Routing to appropriate agent...")
            progress_placeholder.progress(40)
            time.sleep(0.5)
            
            # Step 3: Determine which agent will be used
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
                    "claims_summary": "Claims Summarizer",
                    "general_response": "General Response"
                }
                
                agent_display_name = agent_names.get(next_agent, next_agent.replace('_', ' ').title())
                thinking_placeholder.markdown(f"🤖 **Transaction routed to {agent_display_name} Agent**")
                
                # Show appropriate status based on agent
                if next_agent == "image_analysis":
                    status_placeholder.text("🌐 Searching Zillow for property...")
                    progress_placeholder.progress(60)
                elif next_agent == "terms_conditions":
                    status_placeholder.text("🔎 Searching policy documents...")
                    progress_placeholder.progress(60)
                elif next_agent == "recommendation_agent":
                    status_placeholder.text("💭 Generating recommendations...")
                    progress_placeholder.progress(60)
                elif next_agent == "claims_summary":
                    status_placeholder.text("📋 Loading claim data...")
                    progress_placeholder.progress(60)
                else:
                    status_placeholder.text("🤖 Processing your request...")
                    progress_placeholder.progress(60)
                
                time.sleep(1.0)
                
                # Final step
                status_placeholder.text("✅ Generating response...")
                progress_placeholder.progress(80)
                time.sleep(0.5)
                
            except Exception as e:
                # Fallback to generic thinking if orchestrator fails
                status_placeholder.text("🤖 Processing your request...")
                progress_placeholder.progress(80)
                time.sleep(1.0)
            
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
                    '''
                    # Person's Report Section (NOW AT TOP)
                    if "person_report" in final_result:
                        response += f"**📋 Exclusive Insurance Agent Property Assessment:**\n"
                        person_data = final_result["person_report"].get("Exclusive Insurance Agent Property Assessment", {})
                        for field, value in person_data.items():
                            response += f"- **{field}:** **{value}**\n"
                    
                    response += f"\n"  # Add spacing
                    '''
                    # Image Analysis Section
                    response += f"**🔍 AI Image Analysis from Zillow:**\n"
                    image_analysis = final_result.get("image_analysis", {})
                    for key, data in image_analysis.items():
                        if isinstance(data, dict) and "value" in data:
                            response += f"- **{key}:** {data['value']}\n"
                        else:
                            response += f"- **{key}:** {data}\n"
                    
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
            
            # Add review button for image analysis
            if final_result.get("status") == "success" and "image_analysis" in final_result:
                # Store analysis data in session state for review page
                st.session_state.analysis_data = final_result
                
                # Add Review Button
                st.write("---")
                st.write("### 🔍 Human Review Required")
                st.write("Please review the AI analysis against the actual property images:")
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("🔍 Open Review Panel", type="primary", use_container_width=True):
                        st.switch_page("pages/Review Panel")
                
                with col2:
                    if st.button("💾 Save Analysis", type="secondary", use_container_width=True):
                        zpid = final_result["zpid"]
                        with open(f"property_{zpid}_complete_analysis.json", "w", encoding="utf-8") as f:
                            json.dump(final_result, f, indent=4)
                        st.success(f"✅ Results saved to property_{zpid}_complete_analysis.json")
            
            # Save results to JSON (your existing functionality)
            elif final_result.get("status") == "success" and "zpid" in final_result:
                zpid = final_result["zpid"]
                import json
                with open(f"property_{zpid}_complete_analysis.json", "w", encoding="utf-8") as f:
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
    main_page()


