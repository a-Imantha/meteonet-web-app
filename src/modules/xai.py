import streamlit as st
import torch
from PIL import Image
import numpy as np
from src.processing.utils import process_input_seq, get_precision, get_recall, get_f1, get_csi, plot_seq, fig2img, \
    plot_seq_with_overlap, remove_zero_pad
from src.models import unet_with_backbone_resnet18, unet_with_backbone_vgg16
import matplotlib.pyplot as plt
import torch.nn as nn
import seaborn as sns
from torch import optim

SAMPLE_SEQUENCES = {
    "2018_08_02_15_00": "src/samples/SEQ_2018_08_02_15_00.zip",
    "2018_08_10_09_00": "src/samples/SEQ_2018_08_10_09_00.zip",
    "2018_08_11_22_00": "src/samples/SEQ_2018_08_11_22_00.zip",
    "2018_08_12_03_00": "src/samples/SEQ_2018_08_12_03_00.zip",
    "2018_08_12_08_00": "src/samples/SEQ_2018_08_12_08_00.zip",
    "2018_08_12_10_00": "src/samples/SEQ_2018_08_12_10_00.zip"

}



sns.set()
sns.set(font_scale=1.5)


def create_explainable_ai_page():
    expected_files = ['rr_0.png', 'rr_15.png', 'rr_30.png', 'rr_45.png', 'rr_60.png', 'wu_0.png', 'wu_60.png',
                      'wv_0.png', 'wv_60.png', 'sat_0.png', 'sat_60.png', 'target_90.png']
    st.write("""
    # Explainable AI for Precipitation Nowcasting - Demo
    
    This demonstration showcases the application of explainable AI for precipitation nowcasting, using the comprehensive [Meteonet Dataset](https://meteofrance.github.io/meteonet) by Meteo France. The system uses a sequence of 11 images collected over an hour, comprising data from rain radar, wind, and satellite sources, to predict rainfall for the next 30 minutes.
    
    For accurate results, it is recommended to upload 12 grayscale images that represent a surface area of 100 km x 150 km, as the model has been trained on a similar area. Each image should correspond to filename, as outlined below:     
    """)
    with st.expander("Input Contents"):
        input_spec = {
            'File Name': expected_files,
            'Description': [
                'Rain Radar Image at t = t',
                'Rain Radar Image at t = t + 15 mins',
                'Rain Radar Image at t = t + 30 mins',
                'Rain Radar Image at t = t + 45 mins',
                'Rain Radar Image at t = t + 60 mins',
                'Wind Speed in U direction at t = t + 0 mins',
                'Wind Speed in U direction at t = t + 60 mins',
                'Wind Speed in V direction at t = t + 0 mins',
                'Wind Speed in V direction at t = t + 60 mins',
                'Satellite Image at t = t + 0 mins',
                'Satellite Image at t = t + 60 mins',
                'Binarized Rain Radar Image(Observed Rain Radar/Target) at t = t + 90 mins'
            ],
            'Comments': ['required', 'required', 'required', 'required', 'required', 'required', 'required', 'required',
                         'required', 'required', 'required', 'optional']
        }
        st.table(input_spec)
    st.write("""
        The inclusion of a target image is optional. However, if provided, the application generates prediction performance scores by comparing the predicted output with the provided target image.
        
        ## 1. Download input samples
        
        To evaluate the system, You can download sample input sequences by selecting an item from the below picker and clicking "Download Sequence". The timestamp provided corresponds to the recorded time of the input sequence, obtained from the Meteonet dataset. More sample sequences can be found [here](https://drive.google.com/drive/folders/1fsMh0pTH4JK9uCTtdgUnmA1bwyl8qIo_?usp=sharing).
        
        After downloading, **extract** the ZIP folder and upload the list of images(12 images/ 11 images excluding target_90.png) using the file uploader.    
        """)

    selected_sequence = st.selectbox("**Select a sample sequence to download**", list(SAMPLE_SEQUENCES.keys()))
    zip_file_path = SAMPLE_SEQUENCES[selected_sequence]
    with open(zip_file_path, "rb") as f:
        bytes_data = f.read()
        st.download_button("Download Sequence", data=bytes_data, file_name=zip_file_path, mime="application/zip")

    st.write("""
        ## 2. Upload input sequence and target images
    """
             )
    uploaded_files = st.file_uploader("**Upload your files**", accept_multiple_files=True, type="png")

    images = {}
    if len(uploaded_files) > 0:
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

    if images:
        st.write("""
            ## 3. Results for the Uploaded Sequence
            
            **Uploaded Input sequence is displayed below.**""")
        with st.expander("Input Sequence", expanded=True):
            plot_seq(images)

    processed_input = 0
    if images:
        with st.spinner("Pre-Processing the Input sequence..."):
            input_seq_images, processed_input = process_input_seq(images)

    if processed_input == 1:
        st.write("""
                 **Below sequence shows the input sequence pre-processed to be uploaded to the neural network for evaluating** 
                 """)
        with st.expander("Processed Sequence", expanded=True):
            plot_seq(input_seq_images)

    if processed_input == 1:
        device = torch.device('cpu')
        inp_seq = []
        with st.spinner("Predicting the Precipitation ahead of 30 minutes(t = 90 mins)..."):
            for key in expected_files[:-1]:
                inp_seq.append(input_seq_images.get(key))
            inp_seq = torch.from_numpy(np.array(inp_seq).astype('float32'))[np.newaxis, :].to(device)

            model = unet_with_backbone_vgg16.get_model()
            pred = model(inp_seq)
            pred = torch.sigmoid(pred)
            pred = pred.cpu().detach()[0, 0, :, :]
            pred_bin = pred > 0.5

        st.write("""
                 ### Prediction from the Model
                 
                 The prediction from the neural network is displayed here. This shows how the precipitation should look like in the next 30 minutes(or t = 90 mins). 
                 """)
        with st.expander("Prediction ahead of 30 minutes", expanded=True):
            col1, col2, col3 = st.columns(3)
            col1.image(fig2img(pred), use_column_width=True, caption="Predicted Precipitation at t = 90 mins")
            col2.image(fig2img(pred_bin), use_column_width=True,
                       caption="Binarized Predicted Precipitation at t = 90 mins")
            target_img = images.get('target_90.png') > 100
            col3.image(fig2img(target_img), use_column_width=True,
                       caption="Observed(Target) Precipitation at t = 90 mins")

            st.write("""
                **Prediction Performance:**
            """)
            scores_dict = {'Metric': ['Precision', 'Recall', 'F1 Score', 'CSI Score'],
                           'Score': [get_precision(pred_bin, target_img), get_recall(pred_bin, target_img),
                                     get_f1(pred_bin, target_img), get_csi(pred_bin, target_img)]}
            st.table(scores_dict)

        ig_sum_arr = []
        with st.spinner("Computing the Integrated Gradients..."):
            target = torch.from_numpy(np.array(images.get('target_90.png') > 50).astype('float32'))[np.newaxis, :].to(
                device)
            print("target_shape:", target.shape)
            ig_, pred_ = integrated_gradients(inp_seq, target, model)
            ig_seq = {}
            print('ig_shape:', ig_.shape)
            i = 0
            for key in expected_files[:-1]:
                ig_seq[key] = ig_[i, :, :]
                ig_sum_arr.append(np.sum(ig_[i, :, :]))
                i += 1
            print('expected_files:',expected_files[:-1])
            print('ig_sum_arr:', ig_sum_arr)
        st.write("""
                 ### Integrated Gradients as a XAI Method
                 
                 Integrated Gradients plotted on top of the input sequence is displayed below. 
                 """)
        with st.expander("Integrated Gradients Results", expanded=True):
            plot_seq_with_overlap(input_seq_images, ig_seq)
            st.write("""---""")
            col1, col2 = st.columns(2)

            # Log Sum of pixels
            fig, ax = plt.subplots(figsize=(15, 10))

            with plt.style.context('seaborn-darkgrid'):
                plt.plot(expected_files[:-1], list(np.log(ig_sum_arr)), marker='o')
            plt.xticks(rotation=90)
            plt.xlabel('Input sequence time')
            plt.ylabel('log(Gradient value sum)')
            plt.title('Integrated Gradient Sum in log scale for each image in input sequence')
            print("Just before pyplot")

            col1.write("**Integrated Gradient Sum in log scale for each image in input sequence**")
            col1.pyplot(fig)
            col2.write("Plot to be added!")


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        #         BCE = nn.functional.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = dice_loss

        return Dice_BCE


def integrated_gradients(inputs, target, model, baseline=None, steps=50):
    device = torch.device('cpu')
    if baseline is None:
        baseline = 0.0 * np.random.random(inputs.shape)
        baseline = torch.tensor(baseline, dtype=torch.float32, device=device)
        print("baseline:", baseline)
    # scale inputs and compute gradients
    scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in range(0, steps + 1)]
    grads, pred_, target_ = predict_and_gradients(scaled_inputs, target, model)
    #     grads, _ = predict_and_gradients(scaled_inputs, model, target_label_idx, cuda)
    avg_grads = np.average(grads[:-1], axis=0)
    baseline = baseline.detach().squeeze(0).cpu().numpy()
    delta_x = (avg_grads - baseline)
    int_grads = delta_x * avg_grads
    #     avg_grads = np.transpose(avg_grads, (1, 2, 0))
    #     delta_X = (pre_processing(inputs, cuda) - pre_processing(baseline, cuda)).detach().squeeze(0).cpu().numpy()
    #     delta_X = np.transpose(delta_X, (1, 2, 0))
    #     integrated_grad = delta_X * avg_grads
    return int_grads, pred_


def predict_and_gradients(inputs, target, model):
    device = torch.device('cpu')
    gradients = []
    x = 0
    for inp in inputs:
        inp = torch.tensor(inp, dtype=torch.float32, device=device, requires_grad=True)
        epoch_loss = 0
        pred = model(inp)
        optimizer = optim.Adam(model.parameters(), lr=0.0008, weight_decay=1e-5)
        criterion = DiceBCELoss()
        loss = criterion(pred, target)
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        gradient = inp.grad.detach().cpu().numpy()[0]
        gradients.append(gradient)
    return gradients, torch.sigmoid(pred), pred
