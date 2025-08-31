import os 
import json
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from .zillow_scraper import ZillowScraper
from typing import Dict, Any, Tuple, Optional
from src.langgraph_agent.state.state import State

class ImageAnalysisAgent:
    def __init__(self):
        load_dotenv('.env')
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=self.api_key,
            temperature=0,
        )
        self.scraper = ZillowScraper()
        
        # Define analysis fields
        self.fields = [
            "Roof Type",
            "Exterior Material",
            "Pool",
            "Garage",
            "Number of Stories",
            "General Condition / Renovation Indicators",
            "Lot Size / Backyard Area",
            "Driveway Type / Paved Area",
            "Solar Panels / External Installations"
        ]

    def _analyze_property_images(self, property_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze property images using Gemini LLM"""
        images = property_data.get("images", [])
        
        prompt = f"""
        You are an expert home underwriter AI. Analyze the following property images.
        Fill ONLY the following fields in STRICT JSON format (no reasoning, no extra text):

        Fields: {', '.join(self.fields)}

        Images: {images}

        IMPORTANT: Respond with ONLY valid JSON, no markdown formatting, no extra text.
        Example output:
        {{
        "Roof Type": "Asphalt Shingle",
        "Exterior Material": "Brick",
        "Pool": "Yes",
        "Garage": "2-car",
        "Number of Stories": "2",
        "General Condition / Renovation Indicators": "Good",
        "Lot Size / Backyard Area": "Medium",
        "Driveway Type / Paved Area": "Concrete",
        "Solar Panels / External Installations": "None"
        }}
        """

        try:
            # Use invoke instead of predict (fixes deprecation warning)
            response = self.llm.invoke(prompt).content
            cleaned_response = re.sub(r"^```json|```$", "", response.strip())

            try:
                return json.loads(cleaned_response)
            except json.JSONDecodeError:
                # Try to extract JSON from the response
                print(f"❌ Raw LLM response: {response}")
                
                # Look for JSON pattern in the response
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group())
                    except:
                        pass
                
                # If still fails, return default values
                print("⚠️ Using default values due to JSON parsing failure")
                return {
                    "Roof Type": "Unknown",
                    "Exterior Material": "Unknown",
                    "Pool": "Unknown",
                    "Garage": "Unknown",
                    "Number of Stories": "Unknown",
                    "General Condition / Renovation Indicators": "Unknown",
                    "Lot Size / Backyard Area": "Unknown",
                    "Driveway Type / Paved Area": "Unknown",
                    "Solar Panels / External Installations": "Unknown"
                }
                
        except Exception as e:
            print(f"❌ Error in LLM call: {str(e)}")
            return None

    def _calculate_risk_score(self, image_analysis_json: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk score based on image analysis"""
        risk_score_prompt = f"""
        You are an expert insurance underwriter AI. 
        Given the following property details in JSON format, calculate a numeric Risk Score between 0 and 100.
        Higher = riskier property. Lower = safer property.

        Rules to follow:
        - Use ONLY the key and its associated value when evaluating risk. Do not assume meaning from the key alone.
        - Consider Roof, Exterior, Pool, Garage, Stories, Condition, Lot Size, Driveway, and Solar Panels.
        - Mention in Risk factors only that if value associated with the key is from the property details and spikes the score heavily.
        - make sure to combine key and value in the output

        IMPORTANT: Respond with ONLY valid JSON, no markdown formatting, no extra text.

        Schema (follow exactly):
        {{
        "Risk Score": <number between 0 and 100>,
        "Risk Factors": [ "Pool", "Asphalt Roof", "Excellent Condition", ... ]
        }}

        Given property details:
        {image_analysis_json}

        Fill ONLY the following fields in STRICT JSON format (no reasoning, no extra text).
        """

        try:
            calculated_risk_score = self.llm.invoke(risk_score_prompt).content
            cleaned_response = re.sub(r"^```json|```$", "", calculated_risk_score.strip())

            try:
                return json.loads(cleaned_response)
            except json.JSONDecodeError:
                print(f"❌ Raw risk score response: {calculated_risk_score}")
                
                # Try to extract JSON from the response
                json_match = re.search(r'\{.*\}', calculated_risk_score, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group())
                    except:
                        pass
                
                # If still fails, return default values
                print("⚠️ Using default risk values due to JSON parsing failure")
                return {
                    "Risk Score": 50,
                    "Risk Factors": ["Unable to analyze - using default score"]
                }
                
        except Exception as e:
            print(f"❌ Error in risk score calculation: {str(e)}")
            return {
                "Risk Score": 50,
                "Risk Factors": ["Error in analysis"]
            }

    def process(self, state: State) -> State:
        """
        Main process function to be called as a node in the graph
        """
        try:
            # Extract address from state messages
            messages = state.get("messages", [])
            
            # Find the latest user message with address
            user_address = None
            for message in reversed(messages):
                if hasattr(message, 'content') and message.content:
                    user_address = message.content
                    break
                elif isinstance(message, dict) and message.get('content'):
                    user_address = message['content']
                    break
            
            if not user_address:
                # Return updated state with error
                state["current_result"] = {
                    "status": "error",
                    "message": "No address found in messages"
                }
                return state
            
            # Split address at first comma
            address_parts = user_address.split(',', 1)
            
            if len(address_parts) < 2:
                state["current_result"] = {
                    "status": "error",
                    "message": "Address format should be: 'Street Address, City, State ZIP'"
                }
                return state
            
            # Extract parts
            full_address = address_parts[0].strip()
            citystatezip = address_parts[1].strip()
            
            print(f"🔍 Starting image analysis for: {full_address}, {citystatezip}")
            
            # Step 1: Scrape property data
            property_data = self.scraper.scrape_property_and_save(full_address, citystatezip)
            if not property_data:
                state["current_result"] = {
                    "status": "error",
                    "message": "Failed to get property data from Zillow"
                }
                return state
            
            zpid = property_data.get("zpid")
            print(f"✅ Got property data for ZPID: {zpid}")
            
            # Step 2: Analyze images
            print(" Analyzing property images...")
            image_analysis_result = self._analyze_property_images(property_data)
            if not image_analysis_result:
                state["current_result"] = {
                    "status": "error", 
                    "message": "Failed to analyze property images"
                }
                return state
            
            # Step 3: Calculate risk score
            print("📊 Calculating risk score...")
            risk_score_result = self._calculate_risk_score(image_analysis_result)
            
            # Step 4: Save results
            if zpid:
                with open(f"property_{zpid}_image_analysis.json", "w") as f:
                    json.dump(image_analysis_result, f, indent=4)
                
                with open(f"property_{zpid}_risk_analysis.json", "w") as f:
                    json.dump(risk_score_result, f, indent=4)
                
                print(f"✅ Results saved for ZPID: {zpid}")
            
            # Update state with results
            state["current_result"] = {
                "status": "success",
                "zpid": zpid,
                "property_data": property_data,
                "image_analysis": image_analysis_result,
                "risk_analysis": risk_score_result,
                "message": f"Successfully analyzed property {zpid}"
            }
            
            # Also update status
            state["status"] = "completed"
            
            print(f"Debug - State updated with result: {state['current_result']}")
            
            return state
            
        except Exception as e:
            print(f"❌ Error in image analysis: {str(e)}")
            state["current_result"] = {
                "status": "error",
                "message": f"Error in image analysis: {str(e)}"
            }
            return state
