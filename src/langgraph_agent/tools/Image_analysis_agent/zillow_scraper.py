import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import requests
import json


class ZillowScraper:
    def __init__(self):
        # -------------------- Load Environment --------------------
        load_dotenv('.env')  # Adjust path to your .env
        self.rapidapi_key = os.getenv("RAPIDAPI_KEY")
        
        print(f" API Key loaded: {'Yes' if self.rapidapi_key else 'No'}")
        print(f"🔑 API Key length: {len(self.rapidapi_key) if self.rapidapi_key else 0}")
        
        self.headers = {
            "x-rapidapi-host": "zillow-com1.p.rapidapi.com",
            "x-rapidapi-key": self.rapidapi_key
        }
    
    def get_zillow_id(self, full_address, citystatezip):
        """
        Get Zillow property ID (ZPID) from address and city/state/zip
        """
        print(f"🔍 Searching for: {full_address}, {citystatezip}")

        address = f"{full_address}, {citystatezip}"
        
        # Try different search strategies including sold properties
        search_strategies = [
            # Strategy 1: All properties (including sold)
            {
                "location": address,
                "home_type": "Houses",
                "status_type": "ForSale"  # Include sold, for sale, for rent, etc.
            },

            {
                "location": address,
                "home_type": "Houses",
                "status_type": "ForRent"  # Include sold, for sale, for rent, etc.
            },

            {
                "location": citystatezip,
                "home_type": "Houses",
                "status_type": "ForSale"  # Include sold, for sale, for rent, etc.
            },
            # Strategy 2: Just sold properties
            {
                "location": citystatezip,
                "home_type": "Houses",
                "status_type": "RecentlySold"
            },
            # Strategy 3: Broader search without status filter
            {
                "location": citystatezip,
                "status_type": "ForRent",  # All home types
                
            },
            {
                "location": full_address,
                "status_type": "ForRent", 
            },
        ]
        
        for i, strategy in enumerate(search_strategies):
            print(f"🔍 Trying search strategy {i+1}: {strategy}")
            
            url = "https://zillow-com1.p.rapidapi.com/propertyExtendedSearch"
            
            try:
                response = requests.get(url, headers=self.headers, params=strategy)
                print(f" Response status: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Handle both list and dictionary responses
                    if isinstance(data, list):
                        # If response is a list, use it directly
                        props = data
                        print(f"🏠 Found {len(props)} properties (list response)")
                    elif isinstance(data, dict):
                        # If response is a dictionary, get props from it
                        props = data.get("props", [])
                        print(f"🏠 Found {len(props)} properties (dict response)")
                    else:
                        print(f"❌ Unexpected response type: {type(data)}")
                        props = []
                    
                    # Print first few properties for debugging
                    for j, prop in enumerate(props[:5]):
                        if isinstance(prop, dict):
                            print(f" Property {j+1}: {prop.get('address', 'No address')} - Status: {prop.get('statusText', 'Unknown')} - ZPID: {prop.get('zpid', 'No ZPID')}")
                        else:
                            print(f" Property {j+1}: {prop} (not a dict)")
                    
                    # Look for exact address match first
                    for prop in props:
                        if isinstance(prop, dict):
                            prop_addr = prop.get("address", "").lower()
                            search_addr = full_address.lower()
                            
                            # Try exact match first
                            if search_addr in prop_addr or prop_addr in search_addr:
                                print(f"✅ Found exact match: {prop.get('address')} - Status: {prop.get('statusText', 'Unknown')} - ZPID: {prop.get('zpid')}")
                                return prop.get("zpid")
                    
                    # If no exact match, try partial match
                    address_parts = full_address.lower().split()
                    for prop in props:
                        if isinstance(prop, dict):
                            prop_addr = prop.get("address", "").lower()
                            matches = sum(1 for part in address_parts if part in prop_addr)
                            if matches >= len(address_parts) * 0.7:  # 70% match
                                print(f"✅ Found partial match: {prop.get('address')} - Status: {prop.get('statusText', 'Unknown')} - ZPID: {prop.get('zpid')}")
                                return prop.get("zpid")
                            
                else:
                    print(f"❌ Strategy {i+1} failed: {response.status_code} - {response.text[:200]}")
                    
            except Exception as e:
                print(f"❌ Strategy {i+1} error: {str(e)}")
                continue
        
        print("❌ No matching property found with any strategy")
        return None

    def get_full_property_info(self, zpid):
        """
        Get full property information including images using ZPID
        """
        if not zpid:
            print("❌ No ZPID provided")
            return None

        # Property details
        url = f"https://zillow-com1.p.rapidapi.com/property"
        params = {"zpid": zpid}
        resp = requests.get(url, headers=self.headers, params=params)
        if resp.status_code != 200:
            print("❌ Failed to get property details")
            return None
        details = resp.json()

        # Images
        url_img = f"https://zillow-com1.p.rapidapi.com/images"
        resp_img = requests.get(url_img, headers=self.headers, params={"zpid": zpid})
        images = resp_img.json().get("images", [])

        details["images"] = images
        return details

    def scrape_property_and_save(self, full_address, citystatezip, output_filename=None):
        """
        Complete workflow: get ZPID, fetch property info, and save to JSON
        """
        # Get ZPID
        zpid = self.get_zillow_id(full_address, citystatezip)
        if not zpid:
            return None
        
        # Get full property info
        full_info = self.get_full_property_info(zpid)
        if not full_info:
            return None
        
        # Save to file
        if not output_filename:
            output_filename = f"property_{zpid}.json"
        
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(full_info, f, indent=4)
        
        print(f"✅ Property data saved to {output_filename}")
        return full_info

