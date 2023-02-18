import streamlit as st
from PIL import Image
import numpy as np
def create_explainable_ai_page():

    st.write("""
    # Explainable AI for Precipitation Nowcasting - Demo

    Dataset Used for this is the Meteonet Dataset by Meteo France: [Meteonet](https://meteofrance.github.io/meteonet)
    """)

    expected_files = ['rr_0.png', 'rr_15.png', 'rr_30.png', 'rr_45.png', 'rr_60.png', 'wu_0.png', 'wu_60.png',
                      'wv_0.png', 'wv_60.png', 'sat_0.png', 'sat_60.png']

    uploaded_files = st.file_uploader("Upload your files", accept_multiple_files=True, type="png")

    images = {}
    if len(uploaded_files)>0:
        for file in expected_files:
            if not any(file in f.name for f in uploaded_files):
                st.warning(f"File {file} not found in uploaded data. Please check your files and try again.")

        for file in uploaded_files:
            image = Image.open(file)
            if image.size != (128, 128):
                image = image.resize((128, 128))
            image_array = np.array(image)
            images[file.name] = image_array

    with st.expander("Input Sequence", expanded=True):
        if images:
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.image(images.get("rr_0.png"), width=128, caption="Rain Radar at t = 0")
            col2.image(images.get("rr_15.png"), width=128, caption="Rain Radar at t = 15")
            col3.image(images.get("rr_30.png"), width=128, caption="Rain Radar at t = 30")
            col4.image(images.get("rr_45.png"), width=128, caption="Rain Radar at t = 45")
            col5.image(images.get("rr_60.png"), width=128, caption="Rain Radar at t = 60")
            col6.image(images.get("wu_0.png"), width=128, caption="Wind U Component at t = 0")
            col1.image(images.get("wu_60.png"), width=128, caption="Wind U Component at t = 60")
            col2.image(images.get("wv_0.png"), width=128, caption="Wind U Component at t = 0")
            col3.image(images.get("wv_60.png"), width=128, caption="Wind U Component at t = 60")
            col4.image(images.get("sat_0.png"), width=128, caption="Satellite at t = 0")
            col5.image(images.get("sat_60.png"), width=128, caption="Satellite at t = 60")