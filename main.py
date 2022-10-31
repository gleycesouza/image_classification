import streamlit as st
import base64
from PIL import Image, ImageOps
import keras
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import io


# background image to streamlit

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file) 
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/jpg;base64,%s");
    opacity: 0.65;
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: scroll; # doesn't work
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

header_html = "<img src='data:image/png;base64,{}' class='center' style='width: 130px; display: block;margin-left: auto; margin-right: auto;'>".format(
    get_base64_of_bin_file("imgs/benthic-logo.png")
)
st.markdown(
    header_html, unsafe_allow_html=True,
)

st.markdown('<br><h3 style="color:black; text-align: center; margin-bottom: .1em;">Image Quality Evaluation</h3>', unsafe_allow_html=True)
set_png_as_page_bg('imgs/background.jpg')


def cnn_luz_classification(img):
    cnn_luz_final = keras.models.load_model('weights/cnn_1.h5')

    test_image_1 = tf.keras.utils.load_img(img, target_size = (64, 64))
    test_image_1 = tf.keras.utils.img_to_array(test_image_1)
    test_image_1 = np.expand_dims(test_image_1, axis = 0)

    result_1 = cnn_luz_final.predict(test_image_1)
    prob_luz = 100*cnn_luz_final.predict(test_image_1/255)[0][0]
    if prob_luz<50:
        prob_luz = 100 - prob_luz
    # training_set_1.class_indices
    if round(result_1[0][0]) == 0:
        prediction_luz = 'Luz Ok'
    elif round(result_1[0][0]) == 1:
        prediction_luz = 'Luz Ruim'
    return prediction_luz,prob_luz

def cnn_foco_classification(img):
    cnn_foco_final = keras.models.load_model('weights/cnn_2.h5')

    test_image_2 = tf.keras.utils.load_img(img, target_size = (64, 64))
    test_image_2 = tf.keras.utils.img_to_array(test_image_2)
    test_image_2 = np.expand_dims(test_image_2, axis = 0)
    result_foco = cnn_foco_final.predict(test_image_2)
    prob_foco = 100*cnn_foco_final.predict(test_image_2/255)[0][0]
    if prob_foco<50:
        prob_foco = 100 - prob_foco
    if round(result_foco[0][0]) == 0:
        prediction_foco = 'Foco Ok'
    elif round(result_foco[0][0]) == 1:
        prediction_foco = 'Foco Ruim'

    return prediction_foco,prob_foco

def cnn_vibracao_classification(img):
    cnn_vibracao_final = keras.models.load_model('weights/cnn_3.h5')

    test_image_2 = tf.keras.utils.load_img(img, target_size = (64, 64))
    test_image_2 = tf.keras.utils.img_to_array(test_image_2)
    test_image_2 = np.expand_dims(test_image_2, axis = 0)
    result_vibracao = cnn_vibracao_final.predict(test_image_2)
    prob_vibracao = 100*cnn_vibracao_final.predict(test_image_2/255)[0][0]
    if prob_vibracao<50:
        prob_vibracao = 100 - prob_vibracao
    if round(result_vibracao[0][0]) == 0:
        prediction_vibracao = ' Vibracao Ok'
    elif round(result_vibracao[0][0]) == 1:
        prediction_vibracao = 'vibracao Ruim'

    return prediction_vibracao,prob_vibracao

def cnn_ring_light_classification(img):
    cnn_ring_light_final = keras.models.load_model('weights/cnn_4.h5')

    test_image_2 = tf.keras.utils.load_img(img, target_size = (64, 64))
    test_image_2 = tf.keras.utils.img_to_array(test_image_2)
    test_image_2 = np.expand_dims(test_image_2, axis = 0)
    result_ring_light = cnn_ring_light_final.predict(test_image_2)
    prob_ring_light = 100*cnn_ring_light_final.predict(test_image_2/255)[0][0]
    if prob_ring_light<50:
        prob_ring_light = 100 - prob_ring_light
    if round(result_ring_light[0][0]) == 0:
        prediction_ring_light = ' Ring Light Ok'
    elif round(result_ring_light[0][0]) == 1:
        prediction_ring_light = 'Ring Light Ruim'

    return prediction_ring_light,prob_ring_light

def cnn_led_classification(img):
    cnn_led_final = keras.models.load_model('weights/cnn_5.h5')

    test_image_2 = tf.keras.utils.load_img(img, target_size = (64, 64))
    test_image_2 = tf.keras.utils.img_to_array(test_image_2)
    test_image_2 = np.expand_dims(test_image_2, axis = 0)
    result_led = cnn_led_final.predict(test_image_2)
    prob_led = 100*cnn_led_final.predict(test_image_2/255)[0][0]
    if prob_led<50:
        prob_led = 100 - prob_led
    if round(result_led[0][0]) == 0:
        prediction_led = 'Led Ok'
    elif round(result_led[0][0]) == 1:
        prediction_led = 'Led Ruim'

    return prediction_led,prob_led

def cnn_etiqueta_classification(img):
    cnn_etiqueta_final = keras.models.load_model('weights/cnn_6.h5')

    test_image_2 = tf.keras.utils.load_img(img, target_size = (64, 64))
    test_image_2 = tf.keras.utils.img_to_array(test_image_2)
    test_image_2 = np.expand_dims(test_image_2, axis = 0)
    result_etiqueta = cnn_etiqueta_final.predict(test_image_2)
    prob_etiqueta = 100*cnn_etiqueta_final.predict(test_image_2/255)[0][0]
    if prob_etiqueta<50:
        prob_etiqueta = 100 - prob_etiqueta
    if round(result_etiqueta[0][0]) == 0:
        prediction_etiqueta = 'Etiqueta Ok'
    elif round(result_etiqueta[0][0]) == 1:
        prediction_etiqueta = 'Etiqueta Ruim'

    return prediction_etiqueta,prob_etiqueta

def cnn_altura_classification(img):
    cnn_altura_final = keras.models.load_model('weights/cnn_7.h5')

    test_image_2 = tf.keras.utils.load_img(img, target_size = (64, 64))
    test_image_2 = tf.keras.utils.img_to_array(test_image_2)
    test_image_2 = np.expand_dims(test_image_2, axis = 0)
    result_altura = cnn_altura_final.predict(test_image_2)
    prob_altura = 100*cnn_altura_final.predict(test_image_2/255)[0][0]
    if prob_altura<50:
        prob_altura = 100 - prob_altura
    if round(result_altura[0][0]) == 0:
        prediction_altura = 'Altura Ok'
    elif round(result_altura[0][0]) == 1:
        prediction_altura = 'Altura Ruim'

    return prediction_altura,prob_altura

uploaded_file = st.file_uploader("Select your picture...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # st.image(image, caption='Image uploaded', use_column_width=True)
    status = True
    buffer = io.BytesIO()     # create file in memory
    image.save(buffer, 'jpeg') # save in file in memory - it has to be `jpeg`, not `jpg`
    buffer.seek(0)            # move to the beginning of file
    bg_image = buffer         # use it without `open()`

    results = []
    results.append(cnn_luz_classification(bg_image))
    results.append(cnn_foco_classification(bg_image))
    results.append(cnn_vibracao_classification(bg_image))
    results.append(cnn_ring_light_classification(bg_image))
    results.append(cnn_led_classification(bg_image))
    results.append(cnn_etiqueta_classification(bg_image))
    results.append(cnn_altura_classification(bg_image))

    p_tag = '<p style="color:black; text-align: center; margin-bottom: .1em;">'

    results_items = [f'{p_tag} {r[0]} ({int(r[1])}%) </p>' for r in results]
    results_str = '\n'.join(results_items)

    content = f'''<div style="background-color: white; border-radius: 10px; padding: 10px 0">
        <h4 style="color:black; text-align: center; margin-bottom: .1em;">Results:</h4>
        {results_str}
    </div>'''

    st.markdown(content, unsafe_allow_html=True)