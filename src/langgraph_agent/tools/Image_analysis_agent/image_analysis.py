import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import requests
import json
import re
from zillow_scraper import ZillowScraper

load_dotenv('../../../../.env')

def recursive_chunk_safe(data, max_chunk_size, meta_stack=None):
    if meta_stack is None:
        meta_stack = []

    result_chunks = []

    def helper(d, meta_stack):
        # Leaf node
        if not isinstance(d, (dict, list)):
            chunk = {"meta": list(meta_stack), "value": d}
            return [chunk]

        chunks = []

        if isinstance(d, dict):
            current_chunk = {}
            for k, v in d.items():
                meta_stack.append(k)
                sub_chunks = helper(v, meta_stack)
                meta_stack.pop()
                
                for sub in sub_chunks:
                    temp_chunk = current_chunk.copy()
                    temp_chunk[k] = sub
                    if len(json.dumps(temp_chunk)) > max_chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = {k: sub}
                    else:
                        current_chunk[k] = sub
            if current_chunk:
                chunks.append(current_chunk)

        elif isinstance(d, list):
            current_list = []
            for item in d:
                sub_chunks = helper(item, meta_stack)
                for sub in sub_chunks:
                    temp_list = current_list + [sub]
                    if len(json.dumps(temp_list)) > max_chunk_size:
                        if current_list:
                            chunks.append(current_list)
                        current_list = [sub]
                    else:
                        current_list.append(sub)
            if current_list:
                chunks.append(current_list)

        return chunks

    # wrap each chunk with meta for top level
    top_level_chunks = helper(data, meta_stack)
    for c in top_level_chunks:
        result_chunks.append({"meta": list(meta_stack), "data": c})

    return result_chunks

def analyze_property_images(property_data):
    """
    Analyze property images using Gemini LLM
    """
    images = property_data.get("images", [])
    
    # ------------------- Initialize Gemini LLM -------------------
    api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0,
    )

    # ------------------- Define Fields -------------------
    fields = [
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

    # ------------------- Construct Prompt -------------------
    prompt = f"""
    You are an expert home underwriter AI. Analyze the following property images.
    Fill ONLY the following fields in STRICT JSON format (no reasoning, no extra text):

    Fields: {', '.join(fields)}

    Images: {images}

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

    # ------------------- Call Gemini -------------------
    response = llm.predict(prompt)

    # ------------------- Clean & Parse JSON -------------------
    # Remove Markdown or extra characters
    cleaned_response = re.sub(r"^```json|```$", "", response.strip())

    try:
        image_analysis_json = json.loads(cleaned_response)
        return image_analysis_json
    except json.JSONDecodeError:
        print("❌ Failed to parse JSON from LLM output:")
        print(response)
        return None

def calculate_risk_score(image_analysis_json, llm):
    """
    Calculate risk score based on image analysis with detailed reasoning
    """
    risk_score_calculation_prompt = f"""
        You are an expert insurance underwriter AI. 
        Given the following property details in JSON format, calculate a numeric Risk Score between 0 and 5.
        Higher = riskier property. Lower = safer property.

        Rules to follow:
        - Use ONLY the key and its associated value when evaluating risk. Do not assume meaning from the key alone.
        - Consider Roof, Exterior, Pool, Garage, Stories, Condition, Lot Size, Driveway, and Solar Panels.
        - Mention in Risk factors only that if value associated with the key is from the property details and spikes the score heavily.
        - make sure to combine key and value in the output
        - Provide detailed reasoning for each risk factor

        Schema (follow exactly):
        {{
        "Risk Score": <number between 0 and 5>,
        "Risk Factors": [ "Pool: No", "Asphalt Roof: New", "Excellent Condition: Modern", ... ],
        "Risk Reasoning": {{
            "Pool: No": "No pool reduces liability risk",
            "Asphalt Roof: New": "New roof reduces maintenance risk",
            "Excellent Condition: Modern": "Modern condition reduces overall risk profile"
        }},
        "Overall Risk Assessment": "Detailed explanation of why this score was given"
        }}

        Given property details:
        {image_analysis_json}

        Fill ONLY the following fields in STRICT JSON format (no reasoning, no extra text).
    """
    calculated_risk_score = llm.predict(risk_score_calculation_prompt)

    # Clean & Parse JSON
    cleaned_calculated_risk_score = re.sub(r"^```json|```$", "", calculated_risk_score.strip())

    try:
        risk_json = json.loads(cleaned_calculated_risk_score)  
        return risk_json
    except json.JSONDecodeError:
        print("❌ Failed to parse Risk Score JSON")
        print(calculated_risk_score)
        return {}

def create_comparison_report(image_analysis_json):
    """
    Create a simple comparison report between actual and predicted values
    """
    # Create dummy actual values (you can replace these with real data)
    actual_values = {
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
    
    comparison_report = {
        "comparison": {}
    }
    
    # Compare each field
    for field, actual_value in actual_values.items():
        predicted_value = image_analysis_json.get(field, "Not Detected")
        
        comparison_report["comparison"][field] = {
            "actual": actual_value,
            "predicted": predicted_value
        }
    
    return comparison_report

def save_comparison_report(zpid, comparison_report):
    """
    Save comparison report to JSON file
    """
    filename = f"property_{zpid}_comparison_report.json"
    with open(filename, "w") as f:
        json.dump(comparison_report, f, indent=4)
    print(f"✅ Comparison report saved to {filename}")
    return filename

def main():
    """
    Main function to run the complete workflow
    """
    # Initialize Zillow scraper
    scraper = ZillowScraper()
    
    # Example usage - you can modify these parameters
    full_address = "46 Creekstone Ln"
    citystatezip = "Dawsonville, GA 30534"
    
    # Get property data using Zillow scraper
    print("🔍 Scraping property data from Zillow...")
    property_data = scraper.scrape_property_and_save(full_address, citystatezip)
    
    if not property_data:
        print("❌ Failed to get property data")
        return
    
    # Get ZPID for file naming
    zpid = property_data.get("zpid")
    print(f"✅ Got property data for ZPID: {zpid}")
    
    # Example usage of chunking (optional)
    max_size = 1000  # max chunk size in characters
    chunks = recursive_chunk_safe(property_data, max_size)
    print(f"📊 Created {len(chunks)} chunks from property data")
    
    # Analyze images
    print("🔍 Analyzing property images...")
    image_analysis_result = analyze_property_images(property_data)
    
    if not image_analysis_result:
        print("❌ Failed to analyze images")
        return
    
    # Initialize LLM for risk calculation
    api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0,
    )
    
    # Calculate risk score with reasoning
    print("📊 Calculating risk score with reasoning...")
    risk_score_result = calculate_risk_score(image_analysis_result, llm)
    
    # Create comparison report
    print("📋 Creating comparison report...")
    comparison_report = create_comparison_report(image_analysis_result)
    
    # Save results
    if zpid:
        # Save image analysis
        with open(f"property_{zpid}_image_analysis.json", "w") as f:
            json.dump(image_analysis_result, f, indent=4)
        print(f"✅ Image analysis saved to property_{zpid}_image_analysis.json")
        
        # Save risk analysis with reasoning
        with open(f"property_{zpid}_risk_analysis.json", "w") as f:
            json.dump(risk_score_result, f, indent=4)
        print(f"✅ Risk analysis saved to property_{zpid}_risk_analysis.json")
        
        # Save comparison report
        save_comparison_report(zpid, comparison_report)
    
    # Print results
    print("\n✅ Image Analysis Result:")
    print(json.dumps(image_analysis_result, indent=2))
    
    print("\n✅ Risk Score Result with Reasoning:")
    print(json.dumps(risk_score_result, indent=2))
    
    print("\n✅ Comparison Report:")
    print(json.dumps(comparison_report, indent=2))

if __name__ == "__main__":
    main()
