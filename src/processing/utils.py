import constants as const
import numpy as np


def __pre_process_rain_radar_image(image):
    image = np.clip(image, const.RAIN_RADAR_LEFT_CUTOFF, const.RAIN_RADAR_RIGHT_CUTOFF)
    image =  np.log(np.add(image, 1))/np.log(const.RAIN_RADAR_RIGHT_CUTOFF + 1)
    return image

def __pre_process_sat_image(image):

    # Correct the pixel conversions
    image = ((image*89.0)/255.0) - 69

    image = np.clip(image, -69, 20)

    image = -((image - const.SAT_MEAN) / const.SAT_STD)
    image = (image + abs(const.SAT_MIN)) / (abs(const.SAT_MIN) + const.SAT_MAX)
    return image

def __pre_process_wind_u(image):

    # Correct the pixel conversions
    image = ((image*48.0)/255.0) - 16

    image = np.clip(image, -16, 32)

    image = ((image - const.WIND_U_MEAN) / const.WIND_U_STD)
    image = (image + abs(const.WIND_U_MIN)) / (abs(const.WIND_U_MIN) + const.WIND_U_MAX)
    return image

def __pre_process_wind_v(image):

    # Correct the pixel conversions
    image = ((image*45.0)/255.0) - 22

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

# def __pre_process_wind_u_images():
