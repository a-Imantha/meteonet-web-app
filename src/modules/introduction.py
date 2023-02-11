import streamlit as st

def create_intro():
    st.write("""
    # Explainable AI for Precipitation Nowcasting

    This application predicts precipitaion for the next 30 minutes, given a processed image sequence of,
    - Rain Radar Images
    - Satellite Images
    - Wind Speed U Component Images
    - Wind Speed V Component Images
    are provided. 
    """)