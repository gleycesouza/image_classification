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

def long_or_top(img):
    long_or_top = keras.models.load_model('./long_or_top.h5')

    test_image_1 = tf.keras.utils.load_img(img, target_size=(64, 64))
    test_image_1 = tf.keras.utils.img_to_array(test_image_1)
    test_image_1 = np.expand_dims(test_image_1, axis=0)

    result = long_or_top.predict(test_image_1)

    if round(result[0][0], 0) == 1:
        result = 'Topo'
        print('------------------------------TOPO----------------------------------')
    else:
        result = 'Longitudinal'
        print('--------------------------LONGITUDINAL-----------------------------')

    return result

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
        prediction_luz = 'Light Ok'
    elif round(result_1[0][0]) == 1:
        prediction_luz = 'Bad Light'
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
        prediction_foco = 'Focus Ok'
    elif round(result_foco[0][0]) == 1:
        prediction_foco = 'Bad Focus'

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
        prediction_vibracao = ' Vibration Ok'
    elif round(result_vibracao[0][0]) == 1:
        prediction_vibracao = 'Bad Vibration'

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
        prediction_ring_light = 'Bad Ring Light'

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
        prediction_led = 'LED Ok'
    elif round(result_led[0][0]) == 1:
        prediction_led = 'Bad LED'

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
        prediction_etiqueta = 'Label Ok'
    elif round(result_etiqueta[0][0]) == 1:
        prediction_etiqueta = 'Bad Label'

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
        prediction_altura = 'Height Ok'
    elif round(result_altura[0][0]) == 1:
        prediction_altura = 'Bad Height'

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

def detect_rectangle_image(image_bytes, centimeters_x, centimeters_y, dpi):
    # Load image
    #image = cv2.imread(image_path)
    #image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    nparr = np.frombuffer(image_bytes.getvalue(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)

    height, width = image.shape[:2]                                                 
    if width > height:                                                             
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)                            

    x, y, w, h = 600, 400, 3500, 3700                                               
    cropped_image = image[y:y+h, x:x+w] 

    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)                                  # Convert image to grayscale
    blur = cv2.medianBlur(gray, 11)                                                 # Apply median blur to the image
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]    # Apply Otsu's threshold to the image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
    thresh_image = cv2.adaptiveThreshold(opening, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Encontra os contornos na imagem
    contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = contours[0]

    # Aproxima o contorno para um círculo
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    center = (int(x), int(y))
    radius = int(radius)

    # Cálculo das coordenadas de corte
    pixels_x = int(centimeters_x * dpi / 2.54)  # 2.54 cm por polegada
    pixels_y = int(centimeters_y * dpi / 2.54)  # 2.54 cm por polegada
    
    #center = (int(x), int(y))
    left = int(center[0] - pixels_x / 2)
    top = int(center[1] - pixels_y / 2)
    right = int(center[0] + pixels_x / 2)
    bottom = int(center[1] + pixels_y / 2)

    # Corta a região de interesse da imagem nas dimensões desejadas
    cropped_image = cropped_image[top:bottom, left:right]                                       # Crop the image to fit the circle

    # with io.BytesIO() as output:
    #     result_image.save(output, format="JPEG")
    #     img_bytes = output.getvalue()
    # return img_bytes

    # Convert the result to PIL Image object 
    result_image = Image.fromarray(cropped_image)
    return result_image

def resize_image_circle(result_image, size_cm, dpi):
    # Calculate the size in pixels
    pixels = size_cm * dpi / 1.9227
    new_image = result_image.resize((int(pixels), int(pixels))) # Resize the image
    
    with io.BytesIO() as output:
        new_image.save(output, format="JPEG")
        img_bytes = output.getvalue()
    return img_bytes

def resize_image_rectangle(result_image, percent, dpi):
   # Obter o DPI original
    dpi = result_image.info.get("dpi", (dpi, dpi))

    # Calcular as novas dimensões em pixels
    new_width = int(result_image.width * percent)
    new_height = int(result_image.height * percent)

    # Redimensionar a imagem mantendo o DPI original
    resized_image = result_image.resize((new_width, new_height), resample=Image.LANCZOS)

    # Definir o DPI da imagem redimensionada
    resized_image.info["dpi"] = dpi

    with io.BytesIO() as output:
        resized_image.save(output, format="JPEG")
        img_bytes = output.getvalue()
    return img_bytes

def add_border_circle(result_image_resized, border_size_cm, border_size_top_cm, dpi):
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

def add_border_rectangle(result_image_resized, border_size_side_cm, border_top_cm, border_bot_cm, dpi):
    # Open the resized image as a PIL image
    #image = Image.open(image_path)
    image = Image.open(io.BytesIO(result_image_resized))

    # Set the desired border size in pixels
    right = int(border_size_side_cm * dpi / 2.54)
    left = int(border_size_side_cm * dpi / 2.54)
    top = int(border_top_cm * dpi / 2.54)
    bottom = int(border_bot_cm * dpi / 2.54)

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

def add_text_and_scale_circle(result, furo, testemunho, secao, amostra, dpi):
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

def add_text_and_scale_rectangle(result, furo, testemunho, secao, amostra, dpi):
    # Open an Image
    img = result

    # Create an ImageDraw object
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # Define custom font style and font size
    myFont_a = ImageFont.truetype('ARIAL.TTF', 76)
    myFont = ImageFont.truetype('ARIALBD.TTF', 86)

    # Define text to be added to the image
    texts = [
        (furo, 50, 30),
        (testemunho, 1130, 30),
        (secao, 1400, 30),
        (amostra, 1620, 30)
    ]
    # Add Text to an image
    for value, x, y in texts:
        draw.text((x, y), value, font=myFont, fill='white')

    # Draw a rectangle with the specified coordinates
    x1, y1, x2, y2 = (width/2)-118, (height)-110, (width/2)+118, (height)+110
    rec = ImageDraw.Draw(img)
    rec.rectangle((x1, y1, x2, y2), fill='white')
    #1797

    # Scale
    draw.text(((width/2)-80, height-90), "1 cm", font=myFont_a, fill='black')
 
    # Save the edited image
    buf = io.BytesIO()
    img.save(buf, format='jpeg', dpi=(dpi, dpi), quality=95)
    
    return buf.getvalue()

def cropp_image(image):
  cropped_image = image[2700:4500,1600:4500]
  return cropped_image

def read_qr_code(image_buffer):
    # Carrega a imagem
    #image = cv2.imread(image_path)
    image = cv2.imdecode(np.frombuffer(image_buffer, np.uint8), cv2.IMREAD_COLOR)
    image_cropped = cropp_image(image)
    gray = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 11)
    #cv2_imshow(blur)
    # Encontra os códigos QR na imagem
    codes = pyzbar.decode(blur)
    
    # Loop para ler todos os códigos QR encontrados
    for code in codes:
        result = code.data.decode("utf-8")
        return result

def process_text_input(text_input):
    text_input_list = text_input.split(",")

    if len(text_input_list) > 3:
        text_input_list = text_input_list[-3:]
        furo = text_input_list[0].split("\r\n")[1]
    else:
        furo = text_input_list[0]

    testemunho = re.split(re.compile(r'[ABC]'), text_input_list[1])[0]
    amostra = re.split(re.compile(r'[ABC]'), text_input_list[1]) [1]
    secao = re.findall(re.compile(r'[ABC]'), text_input_list[1])[0]

    ensaio = text_input_list[2]
    secao = 'S-' + secao

    if len(testemunho) == 1:
        testemunho = 'T-0' + testemunho
    else:
        testemunho = 'T-' + testemunho

    if len(amostra) == 1:
        amostra = 'A-0' + amostra
    else:
        amostra = 'A-' + amostra
    
    TSA = text_input_list[1]

    return (furo, testemunho, amostra, secao, ensaio,TSA)