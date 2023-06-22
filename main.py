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
import cv2
from PIL import Image, ImageDraw, ImageFont
from google.protobuf import descriptor_pb2
import re
import pyzbar.pyzbar as pyzbar
import utils

@st.cache_data()
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
    opacity: 1;
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: scroll; # doesn't work
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

header_html = "<img src='data:image/png;base64,{}' class='center' style='width: 130px; display: block;margin-left: auto; margin-right: auto;'>".format(
    get_base64_of_bin_file("./pictures/benthic-logo.png")
)
st.markdown(
    header_html, unsafe_allow_html=True,
)

st.markdown('<br><h3 style="color:black; text-align: center;">Image Quality Assurance</h3>', unsafe_allow_html=True)
set_png_as_page_bg('./pictures/background-overlay2.jpg')

col1, col2= st.columns(2)

with col1:
    uploaded_file_jpg = st.file_uploader("Select your JPG file...", type='.jpg')

with col2:
    uploaded_file_cr2 = st.file_uploader("Select your CR2 file...", type='.cr2')

if uploaded_file_jpg is not None:
    image = Image.open(uploaded_file_jpg)
    # st.image(image, caption='Image uploaded', use_column_width=True)
    
    status = True
    buffer = io.BytesIO()     # create file in memory
    image.save(buffer, 'jpeg') # save in file in memory - it has to be `jpeg`, not `jpg`
    buffer.seek(0)            # move to the beginning of file
    bg_image = buffer         # use it without `open()`

    # Botão de seleção longitudinal ou topo
    option = st.radio('Select edit type:', ('Top', 'Longitudinal TRX','Longitudinal Bender'))

    def evaluation_t(bg_image):
        element = st.markdown('<p style="color:black; text-align: center; margin-bottom: .1em;">Analyzing the image...</p>', unsafe_allow_html=True)

        results = []
        #long_or_top(bg_image)
        results.append(utils.cnn_ring_light_classification(bg_image))
        results.append(utils.cnn_led_classification(bg_image))
        results.append(utils.cnn_etiqueta_classification(bg_image))
        results.append(utils.cnn_luz_classification(bg_image)) #mais ou menos
        results.append(utils.cnn_foco_classification(bg_image)) #mais ou menos
        results.append(utils.cnn_vibracao_classification(bg_image)) #mais ou menos
        #results.append(cnn_altura_classification(bg_image))
        
        p_tag = '<p style="color:black; text-align: center; margin-bottom: .1em;">'

        results_items = [f'{p_tag} {r[0]} ({int(r[1])}%) </p>' for r in results]

        results_first = results_items[0:3]
        results_str_1 = '\n'.join(results_first)
        results_second = results_items[3:]
        results_str_2 = '\n'.join(results_second)

        b64img = base64.b64encode(buffer.getvalue()).decode()
        
        content = f'''
        <div style="display: flex">
            <div style="flex: 1; position: relative;border-radius: 5px;">
                <img src='data:image/png;base64,{b64img}' class='center' style='padding-right:10px;display: block; margin: auto; max-width: 100%'>
            </div>
            <div style="flex: 1; background-color: white; border-radius: 5px; overflow: hidden; margin-bottom: .10em;">
                <p style="color:rgb(100,150,250); background-color: #eee; text-align: center; line-height:30px; margin-bottom: .1em;">Reliable rating:</p>
                {results_str_1}
                <p style="color:rgb(255,140,0); background-color: #eee; text-align: center; line-height:30px; margin-bottom: .1em;">Inaccurate rating:</p>
                {results_str_2}
            </div>
        </div>'''

        element.empty()
        st.markdown(content, unsafe_allow_html=True)

    def process_image():
        
        if "visibility" not in st.session_state:
            st.session_state.visibility = "visible"
            st.session_state.disabled = False

        st.markdown('<br><h3 style="color:black; text-align: center;">Image editing</h3>', unsafe_allow_html=True)

        predict_label = utils.read_qr_code(bg_image.getvalue())
        col1, col2 = st.columns((2,1))

        with col1:
            text_input = st.text_input(
                "Input the image id. Example: DGT-2152,2A1,TRX-CK0UE",
                predict_label,
                label_visibility=st.session_state.visibility,
                disabled=st.session_state.disabled,
            )

        furo, testemunho, amostra, secao, ensaio, TSA, num_image = utils.process_text_input(text_input)

        with col2:
            num_image = st.text_input(
                "Enter image number. Example: 504",
                num_image,
                label_visibility=st.session_state.visibility,
                disabled=st.session_state.disabled,
            )

        if not predict_label and text_input == 'None':
            st.warning('Invalid QR Code reading! Enter the data required for editing.', icon="⚠️")
        
        elif text_input:
            element = st.markdown('<p style="color:black; text-align: center; margin-bottom: .1em;">Editing image...</p>', unsafe_allow_html=True)
        
            # furo, testemunho, amostra, secao, ensaio, TSA, num_image = utils.process_text_input(text_input)
            
            if num_image == "":
                num_image = ensaio

            if option == 'Top':
                image_circle = utils.detect_circle_image(bg_image)
                resized_image = utils.resize_image_circle(image_circle, 7.5, 600)
                border_added_image = utils.add_border_circle(resized_image, 0.15,0.45,600)
                final_img = utils.add_text_and_scale_circle(border_added_image, furo, testemunho, secao, amostra, 600)
                image_edited_name = furo + "_" + TSA + "_T"

            elif option == 'Longitudinal TRX':
                image_rectangle = utils.detect_rectangle_image(bg_image, 7.61/0.95, 14.90/0.95, 600,2.54)
                resized_image = utils.resize_image_rectangle(image_rectangle, 0.95, 600)
                border_added_image = utils.add_border_rectangle(resized_image, 0.15, 0.65, 0.50, 600)
                final_img = utils.add_text_and_scale_rectangle(border_added_image, furo, testemunho, secao, amostra, 600)
                image_edited_name = furo + "_" + TSA + "_L"

            else:
                image_rectangle = utils.detect_rectangle_image_bender(bg_image, 14.90, 14.21, 600,1.80)
                resized_image = utils.resize_image_rectangle(image_rectangle, 1.46, 600)
                border_added_image = utils.add_border_rectangle(resized_image, 0.15, 0.50, 0.50, 600)
                final_img = utils.add_text_and_scale_rectangle_bender(border_added_image, furo, testemunho, secao, amostra, 600)
                image_edited_name = furo + "_" + TSA + "_L"   

            if uploaded_file_cr2 is None:
                col_cr2_1, col_cr2_2 = st.columns((2,1))
                
                with col_cr2_1:
                    st.download_button('↓ Renamed Image .jpg', bg_image, image_edited_name + ".JPG")

                with col_cr2_2:
                    st.download_button('↓ Edited image .jpg', final_img, image_edited_name + "_" + num_image + ".JPG")

            else:
                if uploaded_file_jpg.name[:-4] == uploaded_file_cr2.name[:-4]:
                    
                    col_cr2_1, col_cr2_2, col_cr2_3 = st.columns(3)
                
                    with col_cr2_1:
                        st.download_button('↓ Renamed image .jpg', bg_image, image_edited_name + ".JPG")

                    with col_cr2_2:
                        st.download_button('↓ Renamed image .cr2', uploaded_file_cr2, image_edited_name + "_" + num_image + ".CR2")

                    with col_cr2_3:
                        st.download_button('↓ Edited image .jpg', final_img, image_edited_name + "_" + num_image + ".JPG")


                else:
                    st.warning("Make sure the .cr2 and .jpg file are the same or remove the .cr2", icon="⚠️")
                
            b64img_final = base64.b64encode(final_img).decode()

            content = f'''
                <div style="display: flex">
                    <div style="flex: 1; position: relative;border-radius: 5px;">
                        <img src='data:image/png;base64,{b64img_final}' class='center' style='padding-right:10px;display: block; margin: auto; max-width: 40%'>
                    </div>
                </div>'''
            st.markdown(f'<p style="color:black; text-align: center; margin-bottom: .1em;">Edited image preview:</p>', unsafe_allow_html=True)
            st.markdown(content, unsafe_allow_html=True)
            st.markdown(f'<p style="color:black; text-align: center; margin-bottom: .1em;">{image_edited_name + "_" + num_image + ".JPG"}</p>', unsafe_allow_html=True)
            element.empty()

    if option == 'Top':
        evaluation_t(bg_image)
        process_image()
    else:
        process_image()

    

