import streamlit as st
from src.langgraph_agent.main import main_page
from src.langgraph_agent.review_page import review_page

# Configure the app
st.set_page_config(
    page_title="Underwriter Multi-Agent AI",
    page_icon="🤖",
    layout="wide"
)

# Define pages
pages = {
    "🤖 Main Analysis": [
        st.Page(main_page, title="Property Analysis", icon="🔍")
    ],
    "👥 Human Review": [
        st.Page(review_page, title="Review Panel", icon="✅")
    ]
}

# Create navigation
pg = st.navigation(pages, position="top")
pg.run()