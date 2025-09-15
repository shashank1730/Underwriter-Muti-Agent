import streamlit as st
import json
import pandas as pd
from src.langgraph_agent.tools.Image_analysis_agent.image_analysis_agent import ImageAnalysisAgent
def review_page():
    st.title("🔍 Human Review Panel")
    st.write("Review and update AI analysis results")
    
    # Check if there's analysis data to review
    if 'analysis_data' not in st.session_state:
        st.warning("No analysis data available. Please run an analysis first.")
        return
    
    final_result = st.session_state.analysis_data
    
    # Get property data and image analysis
    property_data = final_result.get("property_data", {})
    image_analysis = final_result.get("image_analysis", {})
    images = property_data.get("images", [])
    
    # Get relevant images used in analysis
    relevant_images = final_result.get("relevant_images", [])

    
    # Get person report data
    person_report = final_result.get("person_report", {})

    # Extract person assessment data
    person_assessment = {}
    if person_report and "Exclusive Insurance Agent Property Assessment" in person_report:
        person_assessment = person_report["Exclusive Insurance Agent Property Assessment"]

    # Define fields to review
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
    
    # Initialize session state for updated fields
    if 'updated_fields' not in st.session_state:
        st.session_state.updated_fields = {}
    
    # Property info header
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write(f"**Property ID:** {final_result.get('zpid', 'Unknown')}")
        address = property_data.get('address', {})
        if isinstance(address, dict):
            street = address.get('streetAddress', 'Unknown')
            city = address.get('city', 'Unknown')
            state = address.get('state', 'Unknown')
            zipcode = address.get('zipcode', 'Unknown')
            full_address = f"{street}, {city}, {state} {zipcode}"
        else:
            full_address = str(address)
        st.write(f"**Address:** {full_address}")
    with col2:
        if st.button("🔄 Refresh Data"):
            st.rerun()
    
    st.write("---")
    
    # Show relevant images used in analysis
    if relevant_images:
        st.write("### 📸 Images Used in Analysis")
        st.write(f"**Total images analyzed:** {len(relevant_images)}")
        
        # Display images in a grid
        cols = st.columns(3)
        for i, img_url in enumerate(relevant_images):
            with cols[i % 3]:
                try:
                    st.image(img_url, caption=f"Image {i+1}", width=200)
                except Exception as e:
                    st.write(f"Image {i+1}: {img_url}")
                    st.write(f"Error loading: {str(e)}")
    
    # Add Person Assessment table
    
    person_data = []
    for field in fields:
        person_value = person_assessment.get(field, 'Not provided')
        person_data.append({
            'Field': field,
            'Person Assessment': person_value
        })
    
    # Create summary table
    
    summary_data = []
    for field in fields:
        # Check if field has been updated
        if field in st.session_state.updated_fields:
            value = st.session_state.updated_fields[field]
            confidence = "human_updated"
            source_img = "Human Updated"
        else:
            current_value = image_analysis.get(field, {})
            if isinstance(current_value, dict):
                value = current_value.get('value', 'Unknown')
                confidence = current_value.get('confidence', 'unknown')
                source_img = current_value.get('source_image', 'None')
            else:
                value = str(current_value)
                confidence = 'unknown'
                source_img = 'None'
        
        summary_data.append({
            'Field': field,
            'Current Value': value,
            'Confidence': confidence,
            'Source Image': source_img
        })


    st.write("### 📋 Person Assessment")
    person_df = pd.DataFrame(person_data)
    st.dataframe(person_df, use_container_width=True)
    
    st.write("### 📊 AI Analysis Summary")
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Show update count and changes
    if st.session_state.updated_fields:
        st.info(f"📝 {len(st.session_state.updated_fields)} field(s) have been updated")
        
        # Show what changed
        with st.expander("🔄 View Changes", expanded=False):
            for field, new_value in st.session_state.updated_fields.items():
                original_data = image_analysis.get(field, {})
                if isinstance(original_data, dict):
                    original_value = original_data.get('value', 'Unknown')
                else:
                    original_value = str(original_data)
                
                st.write(f"**{field}:**")
                st.write(f"  Original: {original_value}")
                st.write(f"  Updated: {new_value}")
                st.write("---")
    
    st.write("---")
    
    # Show each field for review
    for field in fields:
        with st.expander(f"🔍 {field}", expanded=True):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Show current value properly
                current_data = image_analysis.get(field, {})
                if isinstance(current_data, dict):
                    current_value = current_data.get('value', 'Unknown')
                    confidence = current_data.get('confidence', 'unknown')
                    source_image = current_data.get('source_image', 'None')
                    image_index = current_data.get('image_index', 0)
                else:
                    current_value = str(current_data)
                    confidence = 'unknown'
                    source_image = 'None'
                    image_index = 0
                
                st.write(f"**Current AI Analysis:** {current_value}")
                st.write(f"**Confidence:** {confidence}")
                st.write(f"**Source:** {source_image}")
                
                # Show the specific source image if available
                if images and image_index > 0 and image_index <= len(images):
                    try:
                        st.write("**Source Image:**")
                        st.image(images[image_index-1], caption=f"Image {image_index}", width=300)
                    except Exception as e:
                        st.write(f"Could not load Image {image_index}: {str(e)}")
                elif source_image and source_image != 'None' and source_image.startswith('http'):
                    # Handle case where source_image is a URL
                    try:
                        st.write("**Source Image:**")
                        st.image(source_image, caption="Source Image", width=300)
                    except Exception as e:
                        st.write(f"Could not load source image: {str(e)}")
                        st.write(f"URL: {source_image}")
                



            
            with col2:
                # Get current value for the input
                current_display_value = st.session_state.updated_fields.get(field, current_value)
                
                # Update field value with better input
                new_value = st.text_area(
                    f"Update {field}:",
                    value=current_display_value,
                    key=f"update_{field}",
                    height=100
                )
                
                # Add some common options for certain fields
                if field == "Pool":
                    pool_options = ["No", "Yes", "Not visible", "In-ground", "Above-ground"]
                    selected_pool = st.selectbox(
                        "Quick select:",
                        options=pool_options,
                        index=pool_options.index(new_value) if new_value in pool_options else 0,
                        key=f"select_{field}"
                    )
                    if selected_pool != new_value:
                        new_value = selected_pool
                
                elif field == "Roof Type":
                    roof_options = ["Asphalt Shingle", "Metal", "Tile", "Slate", "Wood Shake", "Not visible", "Other"]
                    selected_roof = st.selectbox(
                        "Quick select:",
                        options=roof_options,
                        index=roof_options.index(new_value) if new_value in roof_options else 0,
                        key=f"select_{field}"
                    )
                    if selected_roof != new_value:
                        new_value = selected_roof
                
                elif field == "Garage":
                    garage_options = ["No garage", "1-car", "2-car", "3-car", "4-car", "Attached", "Detached", "Not visible"]
                    selected_garage = st.selectbox(
                        "Quick select:",
                        options=garage_options,
                        index=garage_options.index(new_value) if new_value in garage_options else 0,
                        key=f"select_{field}"
                    )
                    if selected_garage != new_value:
                        new_value = selected_garage
                
                elif field == "Number of Stories":
                    stories_options = ["1 story", "2 stories", "3 stories", "Split-level", "Not visible"]
                    selected_stories = st.selectbox(
                        "Quick select:",
                        options=stories_options,
                        index=stories_options.index(new_value) if new_value in stories_options else 0,
                        key=f"select_{field}"
                    )
                    if selected_stories != new_value:
                        new_value = selected_stories
                
                if st.button(f"Update {field}", key=f"btn_{field}"):
                    st.session_state.updated_fields[field] = new_value
                    st.success(f"Updated {field} to: {new_value}")
                    st.rerun()
    
    # Show updated values summary
    if st.session_state.updated_fields:
        st.write("---")
        st.write("### 📝 Updated Values Summary")
        for field, value in st.session_state.updated_fields.items():
            st.write(f"**{field}:** {value}")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("💾 Save Updated Analysis", type="primary"):
                # Update the final result with new values
                updated_analysis = image_analysis.copy()
                
                # Update each field with the new value while preserving the structure
                for field, new_value in st.session_state.updated_fields.items():
                    if field in updated_analysis:
                        # Preserve the original structure but update the value
                        if isinstance(updated_analysis[field], dict):
                            updated_analysis[field]["value"] = new_value
                            updated_analysis[field]["confidence"] = "human_updated"
                        else:
                            updated_analysis[field] = {
                                "value": new_value,
                                "source_image": "Human Updated",
                                "image_index": 0,
                                "confidence": "human_updated"
                            }
                    else:
                        updated_analysis[field] = {
                            "value": new_value,
                            "source_image": "Human Updated",
                            "image_index": 0,
                            "confidence": "human_updated"
                        }
                
                final_result["image_analysis"] = updated_analysis
                
                # Save updated analysis
                zpid = final_result["zpid"]
                with open(f"property_{zpid}_updated_analysis.json", "w", encoding="utf-8") as f:
                    json.dump(final_result, f, indent=4)
                
                st.success("✅ Updated analysis saved!")
                st.session_state.updated_fields = {}
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset Changes"):
                st.session_state.updated_fields = {}
                st.rerun()
    
    # Add export options
    if st.session_state.updated_fields:
        st.write("---")
        st.write("### 📤 Export Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Export as JSON"):
                zpid = final_result.get("zpid", "unknown")
                filename = f"property_{zpid}_updated_analysis.json"
                with open(filename, "r", encoding="utf-8") as f:
                    file_content = f.read()
                st.download_button(
                    label="Download JSON",
                    data=file_content,
                    file_name=filename,
                    mime="application/json"
                )
        
        with col2:
            if st.button("📊 Export Summary as CSV"):
                csv_data = summary_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"property_{final_result.get('zpid', 'unknown')}_summary.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("📋 Copy to Clipboard"):
                st.code(summary_df.to_string(index=False), language="text")
