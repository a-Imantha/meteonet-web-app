import streamlit as st
import torch
from PIL import Image
import numpy as np
from processing.utils import process_input_seq
import matplotlib.pyplot as plt
import torch.nn as nn
import segmentation_models_pytorch as smp
from matplotlib.colors import ListedColormap
from torch import optim
SAMPLE_SEQUENCES = {
    "2018_08_02_15_00": "src/samples/SEQ_2018_08_02_15_00.zip",
    "2018_08_10_09_00": "src/samples/SEQ_2018_08_10_09_00.zip",
    "2018_08_11_22_00": "src/samples/SEQ_2018_08_11_22_00.zip",
    "2018_08_12_03_00": "src/samples/SEQ_2018_08_12_03_00.zip",
    "2018_08_12_08_00": "src/samples/SEQ_2018_08_12_08_00.zip",
    "2018_08_12_10_00": "src/samples/SEQ_2018_08_12_10_00.zip"

}
MODEL_PATH_VGG16 = "src/models/model-vgg16-1-state-dict.pt"
MODEL_PATH_RESNET18 = "src/models/model-resnet18-1-state-dict.pt"
MODEL_PATH_ENSEMBLED = "src/models/model-ensembled-state-dict.pt"

def create_explainable_ai_page():

    st.write("""
    # Explainable AI for Precipitation Nowcasting - Demo

    Dataset Used for this is the Meteonet Dataset by Meteo France: [Meteonet](https://meteofrance.github.io/meteonet)
    
    The inputs should be 12 images with size 128*128. Contents of the images are explained below with the file name format they should bear.
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
                      'wv_0.png', 'wv_60.png', 'sat_0.png', 'sat_60.png', 'target_90.png']

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

    if images:
        st.write("""
                Uploaded Input sequence is displayed below.
                """)
        with st.expander("Input Sequence", expanded=True):
            plot_seq(images)

    processed_input = 0
    if images:
        with st.spinner("Pre-Processing the Input sequence..."):
            input_seq_images, processed_input = process_input_seq(images)

    if processed_input == 1:
        st.write("""
                 Below sequence shows the input sequence pre-processed to be uploaded to the neural network for evaluating 
                 """)
        with st.expander("Processed Sequence", expanded=True):
            plot_seq(input_seq_images)

    if processed_input == 1:
        device = torch.device('cpu')
        inp_seq = []
        with st.spinner("Predicting the Precipitation ahead of 30 minutes(t = 90 mins)..."):
            for key in expected_files[:-1]:
                inp_seq.append(input_seq_images.get(key))
            inp_seq = torch.from_numpy(np.array(inp_seq).astype('float32'))[np.newaxis,:].to(device)
            print("inp_seq Shape:", inp_seq.shape)
            model = get_model()
            pred = model(inp_seq)
            pred = torch.sigmoid(pred)
            pred = pred.cpu().detach()[0,0,:,:]
            pred_bin = pred>0.5

        st.write("""
                 The prediction from the neural network is displayed here. This shows how the precipitation should look like in the next 30 minutes(or t = 90 mins). 
                 """)
        with st.expander("Processed Sequence", expanded=True):
            col1, col2, col3 = st.columns(3)
            col1.image(fig2img(pred), use_column_width=True, caption="Predicted Precipitation at t = 90 mins")
            col2.image(fig2img(pred_bin), use_column_width=True, caption="Binarized Predicted Precipitation at t = 90 mins")
            col3.image(fig2img(images.get('target_90.png')>100), use_column_width=True,
                       caption="Observed(Target) Precipitation at t = 90 mins")

        ig_sum_arr = []
        with st.spinner("Computing the Integrated Gradients..."):
            target = torch.from_numpy(np.array(images.get('target_90.png')>50).astype('float32'))[np.newaxis,:].to(device)
            print("target_shape:", target.shape)
            ig_, pred_ = integrated_gradients(inp_seq, target, model)
            ig_seq = {}

            i = 0
            for key in expected_files[:-1]:
                ig_seq[key] = ig_[i,:,:]
                ig_sum_arr.append(np.sum(ig_[i,:,:]))
                i += 1
        data = {"x": expected_files[:-1], "y":list(np.log(ig_sum_arr))}
        # print("data", data)
        print("x:", expected_files)
        print("y:", list(np.log(ig_sum_arr)))

        st.write("""
                 Integrated Gradients plotted on top of the input sequence is displayed below. 
                 """)
        with st.expander("IG Results", expanded=True):
            plot_seq_with_overlap(input_seq_images, ig_seq)

            # Log Sum of pixels
            fig, ax = plt.subplots(figsize=(15, 10))
            plt.plot(expected_files[:-1], list(np.log(ig_sum_arr)), marker = 'o')
            plt.xticks(rotation=90)
            plt.xlabel('Input sequence time')
            plt.ylabel('log(Gradient value sum)')
            plt.title('Integrated Gradient Sum in log scale for each image in input sequence')
            print("Just before pyplot")
            st.pyplot(fig)
def fig2img_overlap(img, overlap):
    import io
    buf = io.BytesIO()
    fig, ax = plt.subplots()
    ax.set_axis_off()  # remove axis ticks and labels
    fig.tight_layout(pad=0)
    ax.pcolormesh(img, alpha=0.3)
    ax.pcolormesh(overlap, cmap="hot", alpha=0.7)

    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def plot_seq_with_overlap(seq, overlap_seq):
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.image(fig2img_overlap(seq.get("rr_0.png"), overlap_seq.get("rr_0.png")), use_column_width=True, caption="Rain Radar at t = 0", )
    col2.image(fig2img_overlap(seq.get("rr_15.png"), overlap_seq.get("rr_15.png")), use_column_width=True, caption="Rain Radar at t = 15")
    col3.image(fig2img_overlap(seq.get("rr_30.png"), overlap_seq.get("rr_30.png")), use_column_width=True, caption="Rain Radar at t = 30")
    col4.image(fig2img_overlap(seq.get("rr_45.png"), overlap_seq.get("rr_45.png")), use_column_width=True, caption="Rain Radar at t = 45")
    col5.image(fig2img_overlap(seq.get("rr_60.png"), overlap_seq.get("rr_60.png")), use_column_width=True, caption="Rain Radar at t = 60")
    col6.image(fig2img_overlap(seq.get("wu_0.png"), overlap_seq.get("wu_0.png")), use_column_width=True, caption="Wind U Component at t = 0")
    col1.image(fig2img_overlap(seq.get("wu_60.png"), overlap_seq.get("wu_60.png")), use_column_width=True, caption="Wind U Component at t = 60")
    col2.image(fig2img_overlap(seq.get("wv_0.png"), overlap_seq.get("wv_0.png")), use_column_width=True, caption="Wind V Component at t = 0")
    col3.image(fig2img_overlap(seq.get("wv_60.png"), overlap_seq.get("wv_60.png")), use_column_width=True, caption="Wind V Component at t = 60")
    col4.image(fig2img_overlap(seq.get("sat_0.png"), overlap_seq.get("sat_0.png")), use_column_width=True, caption="Satellite at t = 0")
    col5.image(fig2img_overlap(seq.get("sat_60.png"), overlap_seq.get("sat_60.png")), use_column_width=True, caption="Satellite at t = 60")

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
    col2.image(fig2img(seq.get("wv_0.png")), use_column_width=True, caption="Wind V Component at t = 0")
    col3.image(fig2img(seq.get("wv_60.png")), use_column_width=True, caption="Wind V Component at t = 60")
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


class EnsembleModel(nn.Module):
    def __init__(self, models, train_type):
        super().__init__()
        if train_type == 0:
            n_channels = 5
        elif train_type == 1:
            n_channels = 7
        elif train_type == 2:
            n_channels = 9
        elif train_type == 3:
            n_channels = 11
        self.models = models
        self.ens_model = nn.Sequential(
            nn.Conv2d(len(models), 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):

        with torch.no_grad():
            x_ = torch.cat((self.models[0](x),), dim=1)
            for model in self.models[1:]:
                x_ = torch.cat((x_, model(x)), dim=1)

        out = self.ens_model(x_)
        return out

def get_model():
    models = []
    models.append(smp.Unet('vgg16', in_channels=11, classes=1, encoder_weights=None))
    models[0].load_state_dict(torch.load(MODEL_PATH_VGG16, map_location=torch.device('cpu')))
    models.append(smp.Unet('resnet18', in_channels=11, classes=1, encoder_weights=None))
    models[1].load_state_dict(torch.load(MODEL_PATH_RESNET18, map_location=torch.device('cpu')))
    ens = EnsembleModel([model for model in models], 3)
    ens.load_state_dict(torch.load(MODEL_PATH_ENSEMBLED, map_location=torch.device('cpu')))
    return models[0]


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
        baseline = 0.0 *np.random.random(inputs.shape)
        baseline = torch.tensor(baseline, dtype=torch.float32, device=device)
        print("baseline:",baseline)
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
    x =0
    for inp in inputs:
        inp = torch.tensor(inp, dtype=torch.float32, device=device, requires_grad=True)
        epoch_loss = 0
        pred = model(inp)
        optimizer = optim.Adam(model.parameters(), lr=0.0008, weight_decay=1e-5)
        criterion = DiceBCELoss()
        loss = criterion(pred,  target)
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        gradient = inp.grad.detach().cpu().numpy()[0]
        gradients.append(gradient)
    return gradients, torch.sigmoid(pred), pred