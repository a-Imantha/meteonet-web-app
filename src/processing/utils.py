import src.constants as const
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
import io


def __pre_process_rain_radar_image(image):
    image = ((image * 549.0) / 255.0)

    image = np.clip(image, const.RAIN_RADAR_LEFT_CUTOFF, const.RAIN_RADAR_RIGHT_CUTOFF)
    image = np.log(np.add(image, 1)) / np.log(const.RAIN_RADAR_RIGHT_CUTOFF + 1)
    return image


def __pre_process_sat_image(image):
    # Correct the pixel conversions
    image = ((image * 89.0) / 255.0) - 69

    image = np.clip(image, -69, 20)

    image = -((image - const.SAT_MEAN) / const.SAT_STD)
    image = (image + abs(const.SAT_MIN)) / (abs(const.SAT_MIN) + const.SAT_MAX)
    return image


def __pre_process_wind_u(image):
    # Correct the pixel conversions
    image = ((image * 48.0) / 255.0) - 16

    image = np.clip(image, -16, 32)

    image = ((image - const.WIND_U_MEAN) / const.WIND_U_STD)
    image = (image + abs(const.WIND_U_MIN)) / (abs(const.WIND_U_MIN) + const.WIND_U_MAX)
    return image


def __pre_process_wind_v(image):
    # Correct the pixel conversions
    image = ((image * 45.0) / 255.0) - 22

    image = np.clip(image, -22, 23)

    image = ((image - const.WIND_V_MEAN) / const.WIND_V_STD)
    image = (image + abs(const.WIND_V_MIN)) / (abs(const.WIND_V_MIN) + const.WIND_V_MAX)
    return image


def process_input_seq(images_dic):
    images_dic['rr_0.png'] = __pre_process_rain_radar_image(images_dic['rr_0.png'])
    images_dic['rr_15.png'] = __pre_process_rain_radar_image(images_dic['rr_15.png'])
    images_dic['rr_30.png'] = __pre_process_rain_radar_image(images_dic['rr_30.png'])
    images_dic['rr_45.png'] = __pre_process_rain_radar_image(images_dic['rr_45.png'])
    images_dic['rr_60.png'] = __pre_process_rain_radar_image(images_dic['rr_60.png'])
    images_dic['sat_0.png'] = __pre_process_sat_image(images_dic['sat_0.png'])
    images_dic['sat_60.png'] = __pre_process_sat_image(images_dic['sat_60.png'])
    images_dic['wu_0.png'] = __pre_process_wind_u(images_dic['wu_0.png'])
    images_dic['wu_60.png'] = __pre_process_wind_u(images_dic['wu_60.png'])
    images_dic['wv_0.png'] = __pre_process_wind_v(images_dic['wv_0.png'])
    images_dic['wv_60.png'] = __pre_process_wind_v(images_dic['wv_60.png'])

    return images_dic, 1


def remove_zero_pad(image):
    dummy = np.argwhere(image < 245)  # assume blackground is zero
    max_y = dummy[:, 0].max()
    min_y = dummy[:, 0].min()
    min_x = dummy[:, 1].min()
    max_x = dummy[:, 1].max()
    crop_image = image[min_y:max_y, min_x:max_x]

    return crop_image


def fig2img(img):
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


def fig2img_overlap(img, overlap):
    buf = io.BytesIO()
    fig, ax = plt.subplots()
    ax.set_axis_off()  # remove axis ticks and labels
    fig.tight_layout(pad=0)
    ax.imshow(img, alpha=0.3)
    ax.imshow(overlap, cmap="hot", alpha=0.7)

    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def plot_seq_with_overlap(seq, overlap_seq):
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.image(fig2img_overlap(seq.get("rr_0.png"), overlap_seq.get("rr_0.png")), use_column_width=True,
               caption="Rain Radar at t = 0", )
    col2.image(fig2img_overlap(seq.get("rr_15.png"), overlap_seq.get("rr_15.png")), use_column_width=True,
               caption="Rain Radar at t = 15")
    col3.image(fig2img_overlap(seq.get("rr_30.png"), overlap_seq.get("rr_30.png")), use_column_width=True,
               caption="Rain Radar at t = 30")
    col4.image(fig2img_overlap(seq.get("rr_45.png"), overlap_seq.get("rr_45.png")), use_column_width=True,
               caption="Rain Radar at t = 45")
    col5.image(fig2img_overlap(seq.get("rr_60.png"), overlap_seq.get("rr_60.png")), use_column_width=True,
               caption="Rain Radar at t = 60")
    col6.image(fig2img_overlap(seq.get("wu_0.png"), overlap_seq.get("wu_0.png")), use_column_width=True,
               caption="Wind U Component at t = 0")
    col1.image(fig2img_overlap(seq.get("wu_60.png"), overlap_seq.get("wu_60.png")), use_column_width=True,
               caption="Wind U Component at t = 60")
    col2.image(fig2img_overlap(seq.get("wv_0.png"), overlap_seq.get("wv_0.png")), use_column_width=True,
               caption="Wind V Component at t = 0")
    col3.image(fig2img_overlap(seq.get("wv_60.png"), overlap_seq.get("wv_60.png")), use_column_width=True,
               caption="Wind V Component at t = 60")
    col4.image(fig2img_overlap(seq.get("sat_0.png"), overlap_seq.get("sat_0.png")), use_column_width=True,
               caption="Satellite at t = 0")
    col5.image(fig2img_overlap(seq.get("sat_60.png"), overlap_seq.get("sat_60.png")), use_column_width=True,
               caption="Satellite at t = 60")


def __calculate_tp_fp_fn_tn(pred, target):
    tp_fp_fn_tn = [0, 0, 0, 0]

    diff = 2 * pred - target
    diff = np.array(diff)

    print("diff:", diff)
    tp_fp_fn_tn[0] = (diff == 1).sum()
    tp_fp_fn_tn[1] = (diff == 2).sum()
    tp_fp_fn_tn[2] = (diff == -1).sum()
    tp_fp_fn_tn[3] = (diff == 0).sum()

    return tp_fp_fn_tn


def get_precision(pred, target):
    tp_fp_fn_tn = __calculate_tp_fp_fn_tn(pred, target)
    precision = tp_fp_fn_tn[0] / (tp_fp_fn_tn[0] + tp_fp_fn_tn[1])
    return precision


def get_recall(pred, target):
    tp_fp_fn_tn = __calculate_tp_fp_fn_tn(pred, target)
    recall = tp_fp_fn_tn[0] / (tp_fp_fn_tn[0] + tp_fp_fn_tn[2])
    return recall


def get_f1(pred, target):
    precision = get_precision(pred, target)
    recall = get_recall(pred, target)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def get_csi(pred, target):
    tp_fp_fn_tn = __calculate_tp_fp_fn_tn(pred, target)
    csi = tp_fp_fn_tn[0] / (tp_fp_fn_tn[0] + tp_fp_fn_tn[1] + tp_fp_fn_tn[2])
    return csi
