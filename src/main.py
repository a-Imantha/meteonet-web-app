
import streamlit as st

from modules.introduction import create_intro
from modules.resources import create_resource_page
from modules.xai import create_explainable_ai_page
# from modules.xai import create_explainable_ai_page

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
    # with st.sidebar:
    #     choose = option_menu("Content", ["Introduction", "Resources", "Explainable AI Demo"],
    #                          icons=['house', 'kanban', 'person lines fill'],
    #                          menu_icon="app-indicator", default_index=0,
    #                          styles={
    #         "container": {"padding": "5!important", "background-color": "#fafafa"},
    #         "icon": {"color": "blue", "font-size": "25px", "font-family": "Cooper Black"},
    #         "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
    #         "nav-link-selected": {"background-color": "#7692c2"},
    #     }
    #     )
    # if choose == "Introduction":
    #     create_intro()
    # elif choose == "Resources":
    #     create_resource_page()
    # elif choose == "Explainable AI Demo":
    #     create_explainable_ai_page()
