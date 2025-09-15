import os 
import json
import re
import time
import base64
import requests
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
from .zillow_scraper import ZillowScraper
from typing import Dict, Any, Optional
from src.langgraph_agent.state.state import State
from src.langgraph_agent.llm.llm import Googlellm


class ImageAnalysisAgent:
    def __init__(self):
        load_dotenv('.env')
        self.api_key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.5-pro')
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
        self.llm = Googlellm()
        self.llm = self.llm.get_llm_model()

    def _download_and_convert_images(self, image_urls: list) -> list:
        """Download images from URLs and convert to base64 for Gemini"""
        converted_images = []
        
        print(f"📥 Downloading {len(image_urls)} images...")
        
        for i, url in enumerate(image_urls):
            try:
                print(f"  Downloading image {i+1}/{len(image_urls)}: {url[:50]}...")
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                # Convert to base64
                image_base64 = base64.b64encode(response.content).decode('utf-8')
                converted_images.append({
                    "url": url,
                    "base64": image_base64,
                    "index": i + 1
                })
                print(f"  ✅ Image {i+1} downloaded and converted")
                
            except Exception as e:
                print(f"  ❌ Failed to download image {i+1}: {str(e)}")
                continue
        
        print(f"📸 Successfully converted {len(converted_images)} images")
        return converted_images

    def _analyze_property_images(self, property_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze property images using efficient single-call approach"""
        images = property_data.get("images", [])
        
        if not images:
            print("❌ No images available for analysis")
            return self._get_default_analysis()
        
        print(f"🔍 Analyzing {len(images)} images with efficient single-call approach...")
        print(f"📸 Using {len(images)} images for comprehensive analysis")
        
        # Use ALL images in a single, comprehensive analysis call
        results = self._analyze_all_fields_single_call(images, property_data)
        
        return results

    def _analyze_all_fields_single_call(self, images: list, property_data: Dict[str, Any]) -> Dict[str, str]:
        """Analyze all fields using ALL images directly - no categorization needed"""
        
        if not images:
            return self._get_default_analysis()
        
        print(f"🔍 Analyzing ALL {len(images)} images directly...")
        
        # Download and convert images to base64 for Gemini
        converted_images = self._download_and_convert_images(images)
        
        if not converted_images:
            print("❌ No images could be downloaded and converted")
            return self._get_default_analysis()
        
        # Use converted images for analysis
        analysis_results = self._analyze_comprehensive_single_call(converted_images)
        
        # Create final result with source tracking
        final_result = {}
        for field in self.fields:
            if field in analysis_results:
                final_result[field] = analysis_results[field]
            else:
                final_result[field] = {
                    "value": "Not visible",
                    "source_image": "None",
                    "image_index": 0,
                    "confidence": "none"
                }
        
        print(f"✅ Analysis complete. Found {len([f for f in analysis_results if analysis_results[f].get('value', 'Not visible') != 'Not visible'])} fields")
        
        # Add all images to the result for review page
        final_result["relevant_images"] = images
        
        # Save results immediately
        zpid = property_data.get("zpid", "unknown")
        if zpid and zpid != "unknown":
            with open(f"property_{zpid}_image_analysis.json", "w", encoding="utf-8") as f:
                json.dump(final_result, f, indent=4)
            print(f"💾 Image analysis results saved to property_{zpid}_image_analysis.json")
        
        return final_result


    def _analyze_comprehensive_single_call(self, converted_images: list) -> Dict[str, Any]:
        """Analyze all fields using ALL images for better accuracy"""
        
        print(f"🔍 Comprehensive Analysis Debug:")
        print(f"  📸 Analyzing {len(converted_images)} converted images...")
        for i, img_data in enumerate(converted_images):
            print(f"    {i+1}. {img_data['url'][:50]}...")
        
        # For now, let's use a simpler approach - analyze images one by one
        # This is because LangChain's ChatGoogleGenerativeAI might not support multiple images in one call
        all_results = {}
        
        for field in self.fields:
            print(f"  🔍 Analyzing field: {field}")
            field_result = self._analyze_field_with_images(field, converted_images)
            all_results[field] = field_result
        
        return all_results

    def _analyze_field_with_images(self, field: str, converted_images: list) -> Dict[str, Any]:
        """Analyze a specific field using all images"""
        
        try:
            print(f"    🔍 Analyzing {field} with {len(converted_images)} images...")
            
            # Prepare images for Gemini
            images_for_gemini = []
            for img_data in converted_images:
                try:
                    # Convert base64 back to bytes
                    image_bytes = base64.b64decode(img_data['base64'])
                    # Create PIL Image from bytes
                    from PIL import Image
                    pil_image = Image.open(BytesIO(image_bytes))
                    images_for_gemini.append(pil_image)
                except Exception as e:
                    print(f"      ❌ Failed to process image: {str(e)}")
                    continue
            
            if not images_for_gemini:
                print(f"    ❌ No valid images for {field}")
                return {
                    "value": "Not visible",
                    "source_image": "None",
                    "image_index": 0,
                    "confidence": "none"
                }
            
            prompt = f"""
            You are an expert home underwriter AI. Analyze the provided property images specifically for: {field}
            
            CRITICAL INSTRUCTIONS for {field}:
            1. Look at ALL {len(images_for_gemini)} images provided - examine each one carefully
            2. Check ALL images to find the best evidence for {field}
            3. If {field} is clearly visible in ANY image, report it accurately
            4. If unsure, say "Not visible" rather than guessing
            5. Use the image number where you found the clearest evidence
            6. Respond with ONLY valid JSON, no markdown formatting
            
            Field-specific guidance:
            - For pools: Look carefully at backyard, aerial, and ground-level shots - pools are often blue/rectangular
            - For roof type: Look at front, aerial, and side views for roof material
            - For garage: Look at front views and driveway areas for garage doors
            - For exterior material: Look at siding, brick, stone, stucco on walls
            - For stories: Count visible floors from front and side views
            - For condition: Look for modern updates, new paint, well-maintained appearance
            - For lot size: Look at aerial views and backyard shots
            - For driveway: Look for concrete, asphalt, or gravel driveways
            - For solar panels: Look for panels on roof or ground installations
            
            Example output format:
            {{
            "value": "Asphalt Shingle",
            "source_image": "{converted_images[0]['url'] if converted_images else 'https://example.com/image1.jpg'}",
            "image_index": 1,
            "confidence": "high"
            }}
            
            CRITICAL RULES FOR SOURCE IMAGES:
            - If value is "Not visible", "No", or "None" → source_image must be "None" and image_index must be 0
            - If value is "Not visible", "No", or "None" → confidence must be "none"
            - Only assign source_image and image_index when you can actually SEE the feature
            - For "Not visible" cases: source_image = "None", image_index = 0, confidence = "none"
            - Use the FULL IMAGE URL for source_image
            
            Your analysis for {field}:
            """
            
            time.sleep(1)  # Rate limiting
            
            # Use Gemini with actual images
            response = self.model.generate_content([prompt] + images_for_gemini)
            cleaned_response = re.sub(r"^```json|```$", "", response.text.strip())
            
            try:
                result = json.loads(cleaned_response)
                print(f"    ✅ Successfully analyzed {field}")
                return result
            except json.JSONDecodeError:
                print(f"    ❌ JSON parsing failed for {field}. Raw response: {response.text[:200]}...")
                return {
                    "value": "Not visible",
                    "source_image": "None",
                    "image_index": 0,
                    "confidence": "none"
                }
                
        except Exception as e:
            print(f"    ❌ Error analyzing {field}: {str(e)}")
            return {
                "value": "Not visible",
                "source_image": "None",
                "image_index": 0,
                "confidence": "none"
            }

    def _get_default_comprehensive_analysis(self) -> Dict[str, Any]:
        """Return default analysis for comprehensive call"""
        result = {}
        for field in self.fields:
            result[field] = {
                "value": "Not visible",
                "source_image": "None",
                "image_index": 0,
                "confidence": "none"
            }
        return result


    def _get_default_analysis(self) -> Dict[str, str]:
        """Return default analysis when no images are available"""
        default_result = {}
        for field in self.fields:
            default_result[field] = {
                "value": "Unknown (no images)",
                "source_image": "None",
                "image_index": 0,
                "confidence": "none"
            }
        return default_result

    def _calculate_risk_score(self, image_analysis_json: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk score based on image analysis with detailed reasoning"""
        
        # Extract just the values for risk calculation
        property_values = {}
        for field, data in image_analysis_json.items():
            if isinstance(data, dict) and "value" in data:
                property_values[field] = data["value"]
            else:
                property_values[field] = str(data)
        
        risk_score_prompt = f"""
        You are an expert insurance underwriter AI. 
        Given the following property details in JSON format, calculate a numeric Risk Score between 0 and 5.
        Higher or 5 = riskier property. Lower or 0 = safer property.

        Rules to follow:
        - Use ONLY the key and its associated value when evaluating risk. Do not assume meaning from the key alone.
        - Consider Roof, Exterior, Pool, Garage, Stories, Condition, Lot Size, Driveway, and Solar Panels.
        - Mention in Risk factors only that if value associated with the key is from the property details and spikes the score heavily.
        - make sure to combine key and value in the output
        - Provide detailed reasoning for each risk factor
        - Risk Score must be a plain number between 0 and 5. Do not include any extra text, symbols, or percentages.
        - Do NOT add %, do NOT write /100, do NOT include text

        IMPORTANT: Respond with ONLY valid JSON, no markdown formatting, no extra text.

        Schema (follow exactly):
        {{
        "Risk Score": a numeric value between 0 and 5, ONLY the number, do NOT add %, do NOT write /100, do NOT include text
        "Risk Factors": [ "Pool: No", "Asphalt Roof: New", "Excellent Condition: Modern", ... ],
        "Risk Reasoning": {{
            "Pool: No": "No pool reduces liability risk",
            "Asphalt Roof: New": "New roof reduces maintenance risk",
            "Excellent Condition: Modern": "Modern condition reduces overall risk profile"
        }},
        "Overall Risk Assessment": "Short and Concise explanation of why this score was given"
        }}

        Given property details:
        {property_values}

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
                    "Risk Score": 0,
                    "Risk Factors": ["Unable to analyze - using default score"],
                    "Risk Reasoning": {},
                    "Overall Risk Assessment": "Default assessment due to parsing error"
                }
                
        except Exception as e:
            print(f"❌ Error in risk score calculation: {str(e)}")
            return {
                "Risk Score": 0,
                "Risk Factors": ["Error in analysis"],
                "Risk Reasoning": {},
                "Overall Risk Assessment": "Error occurred during analysis"
            }

    def _create_person_report(self) -> Dict[str, Any]:
        """Create a simple report showing what the person provided"""
        # Create person's provided values (you can replace these with real data)
        person_report = {
            "Exclusive Insurance Agent Property Assessment": {
                "Roof Type": "Asphalt Shingle",
                "Exterior Material": "Stone Veneer and Siding",
                "Pool": "No",
                "Garage": "2-car",
                "Number of Stories": "2",
                "General Condition / Renovation Indicators": "Excellent / Modern",
                "Lot Size / Backyard Area": "Medium",
                "Driveway Type / Paved Area": "Concrete",
                "Solar Panels / External Installations": "None"
            }
        }
        
        return person_report

    def process(self, state: State) -> State:
        """Main process function to be called as a node in the graph"""
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
            print("🔍 Analyzing property images...")
            image_analysis_result = self._analyze_property_images(property_data)
            if not image_analysis_result:
                state["current_result"] = {
                    "status": "error", 
                    "message": "Failed to analyze property images"
                }
                return state
            
            # Step 3: Calculate risk score with reasoning
            print("📊 Calculating risk score with reasoning...")
            risk_score_result = self._calculate_risk_score(image_analysis_result)
            
            # Step 4: Create person's report
            print("📋 Creating person's report...")
            person_report = self._create_person_report()
            
            # Step 5: Save results
            if zpid:
                with open(f"property_{zpid}_image_analysis.json", "w") as f:
                    json.dump(image_analysis_result, f, indent=4)
                
                with open(f"property_{zpid}_risk_analysis.json", "w") as f:
                    json.dump(risk_score_result, f, indent=4)
                
                with open(f"property_{zpid}_person_report.json", "w") as f:
                    json.dump(person_report, f, indent=4)
                
                print(f"✅ Results saved for ZPID: {zpid}")
            
            # Update state with results
            state["current_result"] = {
                "status": "success",
                "zpid": zpid,
                "property_data": property_data,
                "image_analysis": image_analysis_result,
                "risk_analysis": risk_score_result,
                "person_report": person_report,
                "message": f"Successfully analyzed property {zpid}"
            }
            
            # Also update status
            state["status"] = "completed"
            
            print(f"✅ Analysis complete for property {zpid}")
            
            return state
            
        except Exception as e:
            print(f"❌ Error in image analysis: {str(e)}")
            state["current_result"] = {
                "status": "error",
                "message": f"Error in image analysis: {str(e)}"
            }
            return state
