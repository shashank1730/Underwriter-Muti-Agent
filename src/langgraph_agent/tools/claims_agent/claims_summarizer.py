import os
import json
import re
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from src.langgraph_agent.llm.llm import Googlellm

class ClaimsSummarizerAgent:
    def __init__(self):
        load_dotenv('.env')
        self.llm = Googlellm()
        self.llm = self.llm.get_llm_model()
        
        # Load claims data from history.html
        self.claims_data = self._load_claims_data()
    
    def _load_claims_data(self) -> Dict[str, Any]:
        """Load claims data from history.html file"""
        try:
            # Read the history.html file
            with open('src/langgraph_agent/tools/claim_summarizer/history.html', 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract the claimsData from the JavaScript
            # Look for the claimsData object in the script
            pattern = r'const claimsData = ({.*?});'
            match = re.search(pattern, content, re.DOTALL)
            
            if match:
                # Get the JavaScript object string
                js_str = match.group(1)
                
                # More robust JavaScript to JSON conversion
                js_str = self._convert_js_to_json(js_str)
                
                # Parse the JSON
                claims_data = json.loads(js_str)
                print(f"✅ Loaded {len(claims_data)} claims from history.html")
                return claims_data
            else:
                print("❌ Could not find claimsData in history.html")
                return self._get_fallback_claims_data()
                
        except Exception as e:
            print(f"❌ Error loading claims data: {str(e)}")
            # Return hardcoded data as fallback
            return self._get_fallback_claims_data()
    
    def _convert_js_to_json(self, js_str: str) -> str:
        """Convert JavaScript object to valid JSON"""
        # Remove JavaScript comments
        js_str = re.sub(r'//.*?\n', '\n', js_str)  # Remove single line comments
        js_str = re.sub(r'/\*.*?\*/', '', js_str, flags=re.DOTALL)  # Remove multi-line comments
        
        # Replace unquoted property names with quoted ones
        js_str = re.sub(r'(\w+):', r'"\1":', js_str)
        
        # Handle single quotes in strings - convert to double quotes
        # This is a more complex regex to handle strings properly
        def replace_string_quotes(match):
            content = match.group(1)
            # Escape any existing double quotes in the content
            content = content.replace('"', '\\"')
            return f'"{content}"'
        
        # Replace single-quoted strings with double-quoted strings
        js_str = re.sub(r"'([^']*)'", replace_string_quotes, js_str)
        
        # Handle trailing commas (not allowed in JSON)
        js_str = re.sub(r',(\s*[}\]])', r'\1', js_str)
        
        # Handle undefined values
        js_str = re.sub(r':\s*undefined', ': null', js_str)
        
        return js_str
    
    def _get_fallback_claims_data(self) -> Dict[str, Any]:
        """Fallback claims data if HTML parsing fails"""
        return {
            "CLM-2024-001": {
                "title": "Water Damage - Kitchen Fire",
                "status": "investigation",
                "statusText": "Under Investigation",
                "date": "2024-01-15",
                "policyNumber": "POL-2024-789456",
                "insured": "John Smith",
                "adjuster": "Sarah Johnson",
                "estimatedAmount": "$15,000",
                "notes": [
                    {
                        "date": "2024-01-15",
                        "author": "Sarah Johnson",
                        "content": "Initial claim received at 2:30 PM today. Customer John Smith called our emergency hotline reporting a kitchen fire that occurred at approximately 1:45 PM. The fire started from an electrical outlet behind the refrigerator in the kitchen area.",
                        "tags": "urgent, initial, emergency, electrical, safety"
                    },
                    {
                        "date": "2024-01-16",
                        "author": "Sarah Johnson",
                        "content": "On-site inspection completed at 9:15 AM. Upon arrival, I conducted a thorough walkthrough of the affected areas. The fire damage is primarily concentrated in the kitchen area, specifically around the electrical outlet where the fire originated.",
                        "tags": "inspection, documentation, damage assessment, electrical, smoke damage"
                    }
                ]
            },
            "CLM-2024-002": {
                "title": "Auto Accident - Vehicle Damage",
                "status": "processing",
                "statusText": "Processing Payment",
                "date": "2024-01-20",
                "policyNumber": "POL-2024-123789",
                "insured": "Maria Garcia",
                "adjuster": "David Brown",
                "estimatedAmount": "$8,500",
                "notes": [
                    {
                        "date": "2024-01-20",
                        "author": "David Brown",
                        "content": "Auto accident claim filed at 4:15 PM. Customer Maria Garcia called to report a rear-end collision that occurred at 3:45 PM on Highway 101 near the downtown exit.",
                        "tags": "accident, rear-end collision, no injury, police report, comprehensive coverage"
                    }
                ]
            },
            "CLM-2024-003": {
                "title": "Property Theft - Home Burglary",
                "status": "open",
                "statusText": "New Claim",
                "date": "2024-01-25",
                "policyNumber": "POL-2024-456123",
                "insured": "Robert Wilson",
                "adjuster": "Amy Davis",
                "estimatedAmount": "$5,200",
                "notes": [
                    {
                        "date": "2024-01-25",
                        "author": "Amy Davis",
                        "content": "Theft claim filed at 6:30 PM. Customer Robert Wilson called to report a home burglary that occurred while he was at work.",
                        "tags": "theft, burglary, forced entry, electronics, jewelry, police investigation"
                    }
                ]
            }
        }
    
    def process(self, state) -> Dict[str, Any]:
        """Process claims summarization request"""
        messages = state.get("messages", [])
        if not messages:
            return {
                "current_result": {
                    "status": "error",
                    "response": "No messages found",
                    "message": "No user question to process"
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
        
        # Extract claim number from the question
        claim_number = self._extract_claim_number(user_question)
        
        if not claim_number:
            return {
                "current_result": {
                    "status": "error",
                    "response": "No claim number found in your question. Please provide a claim number like 'CLM-2024-001'",
                    "message": "Claim number not found"
                }
            }
        
        # Get claim data
        claim_data = self.claims_data.get(claim_number)
        
        if not claim_data:
            available_claims = list(self.claims_data.keys())
            return {
                "current_result": {
                    "status": "error",
                    "response": f"Claim {claim_number} not found. Available claims: {', '.join(available_claims)}",
                    "message": "Claim not found"
                }
            }
        
        # Summarize the claim
        summary = self._summarize_claim(claim_data, claim_number)
        
        return {
            "current_result": {
                "status": "success",
                "response": summary,
                "message": f"Successfully summarized claim {claim_number}",
                "claim_number": claim_number,
                "agent_type": "Claims Summarizer Agent"
            }
        }
    
    def _extract_claim_number(self, text: str) -> Optional[str]:
        """Extract claim number from text using regex"""
        # Look for patterns like CLM-2024-001, CLM-2024-002, etc.
        pattern = r'CLM-\d{4}-\d{3}'
        match = re.search(pattern, text.upper())
        return match.group(0) if match else None
    
    def _summarize_claim(self, claim_data: Dict[str, Any], claim_number: str) -> str:
        """Summarize a specific claim using LLM"""
        
        # Prepare the claim information for the LLM
        claim_info = f"""
        CLAIM INFORMATION:
        - Claim ID: {claim_number}
        - Title: {claim_data.get('title', 'N/A')}
        - Status: {claim_data.get('statusText', 'N/A')}
        - Date: {claim_data.get('date', 'N/A')}
        - Policy Number: {claim_data.get('policyNumber', 'N/A')}
        - Insured: {claim_data.get('insured', 'N/A')}
        - Adjuster: {claim_data.get('adjuster', 'N/A')}
        - Estimated Amount: {claim_data.get('estimatedAmount', 'N/A')}
        
        CLAIM NOTES ({len(claim_data.get('notes', []))} entries):
        """
        
        # Add all notes
        for i, note in enumerate(claim_data.get('notes', []), 1):
            claim_info += f"""
        Note {i} - {note.get('date', 'N/A')} by {note.get('author', 'N/A')}:
        {note.get('content', 'N/A')}
        Tags: {note.get('tags', 'N/A')}
        ---
        """
        
        # Create summarization prompt
        prompt = f"""
        You are an expert insurance claims analyst. Analyze the following claim data and provide a comprehensive summary.
        
        {claim_info}
        
        Please provide a detailed summary that includes:
        
        1. **Claim Overview**: Brief description of what happened
        2. **Timeline**: Key events and dates in chronological order
        3. **Key Issues**: Main problems or concerns identified
        4. **Resolution Status**: Current status and next steps
        5. **Financial Impact**: Estimated costs and payments
        6. **Key Personnel**: Who's involved and their roles
        7. **Recommendations**: Any suggested actions or improvements
        
        Format your response in a clear, professional manner suitable for insurance professionals.
        Use bullet points and clear headings for easy reading.
        """
        
        try:
            print(f"🔍 Summarizing claim {claim_number}...")
            response = self.llm.invoke(prompt).content
            print(f"✅ Successfully summarized claim {claim_number}")
            return response
            
        except Exception as e:
            print(f"❌ Error summarizing claim {claim_number}: {str(e)}")
            return f"Error summarizing claim {claim_number}: {str(e)}"
    
    def get_available_claims(self) -> list:
        """Get list of available claim numbers"""
        return list(self.claims_data.keys())
    
    def get_claim_summary(self, claim_number: str) -> Dict[str, Any]:
        """Get basic claim information without full summarization"""
        claim_data = self.claims_data.get(claim_number)
        if not claim_data:
            return None
        
        return {
            "claim_id": claim_number,
            "title": claim_data.get('title', 'N/A'),
            "status": claim_data.get('statusText', 'N/A'),
            "date": claim_data.get('date', 'N/A'),
            "policy_number": claim_data.get('policyNumber', 'N/A'),
            "insured": claim_data.get('insured', 'N/A'),
            "adjuster": claim_data.get('adjuster', 'N/A'),
            "estimated_amount": claim_data.get('estimatedAmount', 'N/A'),
            "notes_count": len(claim_data.get('notes', []))
        }
