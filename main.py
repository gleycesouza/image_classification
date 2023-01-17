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
    get_base64_of_bin_file("./benthic-logo.png")
)
st.markdown(
    header_html, unsafe_allow_html=True,
)

st.markdown('<br><h3 style="color:black; text-align: center;">Image Quality Assurance</h3>', unsafe_allow_html=True)
set_png_as_page_bg('./background-overlay2.jpg')


def cnn_luz_classification(img):
    cnn_luz_final = keras.models.load_model('./cnn_1.h5')

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
    cnn_foco_final = keras.models.load_model('./cnn_2.h5')

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
    cnn_vibracao_final = keras.models.load_model('./cnn_3.h5')

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
        prediction_vibracao = 'Vibracao Ruim'

    return prediction_vibracao,prob_vibracao

def cnn_ring_light_classification(img):
    cnn_ring_light_final = keras.models.load_model('./cnn_4.h5')

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
    cnn_led_final = keras.models.load_model('./cnn_5.h5')

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
    cnn_etiqueta_final = keras.models.load_model('./cnn_6.h5')

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
    cnn_altura_final = keras.models.load_model('./cnn_7.h5')

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

def detect_circle_image(image_bytes):
    # Load image
    #image = cv2.imread(image_path)
    #image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    nparr = np.frombuffer(image_bytes.getvalue(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                                  # Convert image to grayscale
    blur = cv2.medianBlur(gray, 11)                                                 # Apply median blur to the image
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]    # Apply Otsu's threshold to the image

    # Create a morphological kernel and apply opening operation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)

    # Detect circles in the image
    circles = cv2.HoughCircles(opening, cv2.HOUGH_GRADIENT, dp=1.1, minDist=10000, param1=200, param2=20, minRadius=1000, maxRadius=2000)

    circles = np.uint16(np.around(circles))                                         # Change data type of the circles to int
    x, y, r = circles[0][0][1], circles[0][0][0], circles[0][0][2]-20               # Select the first detected circle
    cropped_image = image[x-r:x+r, y-r:y+r]                                         # Crop the image to fit the circle

    # Create a white circle mask
    mask = np.zeros(cropped_image.shape, dtype=np.uint8)
    mask = cv2.circle(mask, (cropped_image.shape[0]-r, cropped_image.shape[1]-r), r, (255, 255, 255), -1) 
    result = cv2.bitwise_and(mask, cropped_image)                                   # Apply the mask to the image

    # Convert the result to PIL Image object 
    result_image = Image.fromarray(result)
    return result_image

def resize_image(result_image, size_cm, dpi):
    # Calculate the size in pixels
    pixels = size_cm * dpi / 1.9227
    new_image = result_image.resize((int(pixels), int(pixels))) # Resize the image
    
    with io.BytesIO() as output:
        new_image.save(output, format="JPEG")
        img_bytes = output.getvalue()
    return img_bytes

def add_border(result_image_resized, border_size_cm, border_size_top_cm, dpi):
    # Open the resized image as a PIL image
    #image = Image.open(image_path)
    image = Image.open(io.BytesIO(result_image_resized))

    # Set the desired border size in pixels
    right, left, bottom = (int(border_size_cm * dpi / 1.9227) for _ in range(3))
    top = int(border_size_top_cm * dpi / 1.9227)

    # Get the current image size
    width, height = image.size

    # Calculate the new size with the added borders
    new_width = width + right + left
    new_height = height + top + bottom

    # Create a new image with the new size and black background
    result = Image.new(image.mode, (new_width, new_height), (0, 0, 0))

    # Paste the original image on the new image with the added borders
    result.paste(image, (left, top))

    return result

def add_text_and_scale(result, furo, testemunho, secao, amostra, dpi):
    # Open an Image
    img = result

    # Create an ImageDraw object
    draw = ImageDraw.Draw(img)

    # Define custom font style and font size
    myFont_a = ImageFont.truetype('ARIAL.TTF', 86)
    myFont = ImageFont.truetype('ARIALBD.TTF', 86)

    # Define text to be added to the image
    texts = [
        (furo, 20, 20),
        (testemunho, 1790, 20),
        (secao, 2030, 20),
        (amostra, 2230, 20)
    ]
    # Add Text to an image
    for value, x, y in texts:
        draw.text((x, y), value, font=myFont, fill='white')

    # Draw a rectangle with the specified coordinates
    x1, y1, x2, y2 = 1939, 2400, 2250, 2500
    rec = ImageDraw.Draw(img)
    rec.rectangle((x1, y1, x2, y2), fill='white')

    # Scale
    draw.text((2000, 2406), "1 cm", font=myFont_a, fill='black')
 
    # Save the edited image
    buf = io.BytesIO()
    img.save(buf, format='jpeg', dpi=(dpi, dpi), quality=95)
    
    return buf.getvalue()

uploaded_file = st.file_uploader("Select your picture...", type="jpg")
#text_io = io.TextIOWrapper(uploaded_file)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # st.image(image, caption='Image uploaded', use_column_width=True)
    
    status = True
    buffer = io.BytesIO()     # create file in memory
    image.save(buffer, 'jpeg') # save in file in memory - it has to be `jpeg`, not `jpg`
    buffer.seek(0)            # move to the beginning of file
    bg_image = buffer         # use it without `open()`

    element = st.markdown('<p style="color:black; text-align: center; margin-bottom: .1em;">Analyzing the image...</p>', unsafe_allow_html=True)
    
    results = []
    results.append(cnn_ring_light_classification(bg_image))
    results.append(cnn_led_classification(bg_image))
    results.append(cnn_etiqueta_classification(bg_image))
    results.append(cnn_luz_classification(bg_image)) #mais ou menos
    results.append(cnn_foco_classification(bg_image)) #mais ou menos
    results.append(cnn_vibracao_classification(bg_image)) #mais ou menos
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
            <p style="color:rgb(100,150,250); background-color: #eee; text-align: center; line-height:30px; margin-bottom: .1em;">Avaliação confiável:</p>
            {results_str_1}
            <p style="color:rgb(255,140,0); background-color: #eee; text-align: center; line-height:30px; margin-bottom: .1em;">Avaliação insegura:</p>
            {results_str_2}
        </div>
    </div>'''

    element.empty()
    st.markdown(content, unsafe_allow_html=True)

    def process_image():
        
        if "visibility" not in st.session_state:
            st.session_state.visibility = "visible"
            st.session_state.disabled = False

        st.markdown('<br><h3 style="color:black; text-align: center;">Image editing (Under construction, do not use!)</h3>', unsafe_allow_html=True)
        # st.markdown(content_text, unsafe_allow_html=True)

        text_input = st.text_input(
            "Scan or enter the information needed for editing. Example: DGT-2152,2A1,THI",
            label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,
        )
        if text_input:
            element = st.markdown('<p style="color:black; text-align: center; margin-bottom: .1em;">Editing image...</p>', unsafe_allow_html=True)
        
            text_input_list = text_input.split(",")
            furo = text_input_list[0]
            testemunho = 'T-0'+ text_input_list[1][0]
            secao = 'S-'+ text_input_list[1][1]
            amostra ='A-0'+ text_input_list[1][2]

            image_circle = detect_circle_image(bg_image)
            resized_image = resize_image(image_circle, 7.5, 600)
            border_added_image = add_border(resized_image, 0.15,0.45,600)
            final_img = add_text_and_scale(border_added_image, furo, testemunho, secao, amostra, 600)
            image_name = furo+"_"+text_input_list[1]+"_T_num.jpeg"

            element.empty()

            st.download_button('Download Edited Image', final_img, image_name)
            # st.download_button('Download Edited Image', final_img, file_name='imagem_gerada.jpeg')

    process_image()
        
    

    

