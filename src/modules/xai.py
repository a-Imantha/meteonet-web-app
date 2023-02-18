import streamlit as st
from PIL import Image
import numpy as np
from processing.utils import process_input_seq
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

SAMPLE_SEQUENCES = {
    "2018_08_12_08_00": "src/samples/Inp_Seq_2018_08_12_08_00.zip"
}

def create_explainable_ai_page():

    st.write("""
    # Explainable AI for Precipitation Nowcasting - Demo

    Dataset Used for this is the Meteonet Dataset by Meteo France: [Meteonet](https://meteofrance.github.io/meteonet)
    
    The inputs should be 11 images with size 128*128. Contents of the images are explained below with the file name format they should bear.
    """)

    st.write("""
        You can select a sample datetime stamp on which the data is recorded and click download. After downloading extract the downloaded zip folder and re-upload the list of images using the file uploader.
        """)

    selected_sequence = st.selectbox("Select a sequence", list(SAMPLE_SEQUENCES.keys()))
    zip_file_path = SAMPLE_SEQUENCES[selected_sequence]
    with open(zip_file_path, "rb") as f:
        bytes_data = f.read()
        st.download_button("Download Sequence", data=bytes_data, file_name=zip_file_path, mime="application/zip")


    expected_files = ['rr_0.png', 'rr_15.png', 'rr_30.png', 'rr_45.png', 'rr_60.png', 'wu_0.png', 'wu_60.png',
                      'wv_0.png', 'wv_60.png', 'sat_0.png', 'sat_60.png']

    uploaded_files = st.file_uploader("Upload your files", accept_multiple_files=True, type="png")

    images = {}
    if len(uploaded_files)>0:
        for file in expected_files:
            if not any(file in f.name for f in uploaded_files):
                st.warning(f"File {file} not found in uploaded data. Please check your files and try again.")

        for file in uploaded_files:
            image = Image.open(file).convert('L')
            image = np.array(image)
            image = remove_zero_pad(image)
            image = Image.fromarray(image)
            if image.size != (128, 128):
                image = image.resize((128, 128), resample=Image.LANCZOS)

            images[file.name] = np.array(image)

    with st.expander("Input Sequence", expanded=True):
        if images:
            plot_seq(images)

    processed_input = 0
    if images:
        images, processed_input = process_input_seq(images)

    with st.expander("Processed Sequence", expanded=True):
        if processed_input == 1:
            plot_seq(images)

def fig2img(img):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig, ax = plt.subplots()
    ax.set_axis_off()  # remove axis ticks and labels
    fig.tight_layout(pad=0)
    ax.imshow(img, cmap='viridis')

    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img
def plot_seq(seq):
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.image(fig2img(seq.get("rr_0.png")), use_column_width=True, caption="Rain Radar at t = 0", )
    col2.image(fig2img(seq.get("rr_15.png")), use_column_width=True, caption="Rain Radar at t = 15")
    col3.image(fig2img(seq.get("rr_30.png")), use_column_width=True, caption="Rain Radar at t = 30")
    col4.image(fig2img(seq.get("rr_45.png")), use_column_width=True, caption="Rain Radar at t = 45")
    col5.image(fig2img(seq.get("rr_60.png")), use_column_width=True, caption="Rain Radar at t = 60")
    col6.image(fig2img(seq.get("wu_0.png")), use_column_width=True, caption="Wind U Component at t = 0")
    col1.image(fig2img(seq.get("wu_60.png")), use_column_width=True, caption="Wind U Component at t = 60")
    col2.image(fig2img(seq.get("wv_0.png")), use_column_width=True, caption="Wind U Component at t = 0")
    col3.image(fig2img(seq.get("wv_60.png")), use_column_width=True, caption="Wind U Component at t = 60")
    col4.image(fig2img(seq.get("sat_0.png")), use_column_width=True, caption="Satellite at t = 0")
    col5.image(fig2img(seq.get("sat_60.png")), use_column_width=True, caption="Satellite at t = 60")

def remove_zero_pad(image):
    dummy = np.argwhere(image < 245) # assume blackground is zero
    max_y = dummy[:, 0].max()
    min_y = dummy[:, 0].min()
    min_x = dummy[:, 1].min()
    max_x = dummy[:, 1].max()
    crop_image = image[min_y:max_y, min_x:max_x]

    return crop_image