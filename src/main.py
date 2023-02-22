import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from modules.introduction import create_intro
from modules.resources import create_resource_page
from modules.xai import create_explainable_ai_page

if __name__ == '__main__':
    st.set_page_config(layout="wide")
    PAGES = {
        "Introduction": create_intro,
        "Resources": create_resource_page,
        "XAI": create_explainable_ai_page
    }

    # Set sidebar header
    st.sidebar.header("Explainable AI Demo")

    # Create sidebar menu
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    PAGES[selection]()
