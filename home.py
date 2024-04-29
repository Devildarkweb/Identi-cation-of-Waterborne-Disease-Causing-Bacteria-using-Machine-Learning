import numpy as np
from keras.models import load_model
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(
    page_title = "Home - Protozoan Parasite Detection Web App"
)
def protozoan_parasite_detection(model, input_image):
    img_rgb = input_image.convert('RGB')
    img = ImageOps.fit(img_rgb, (224,224), Image.Resampling.LANCZOS)
    
    img_array = np.asarray(img)
    img_array_reshape = img_array.reshape(1,224,224,3)

    if(model == 'DenseNet169'):
        loaded_model_one = load_model("DenseNet169_model.h5")
        a = loaded_model_one.predict(img_array_reshape)
        acc = np.amax(a) * 100
        acc = round(acc, 3)
        indices = a.argmax()
    if(model == 'MobileNet'):
        loaded_model_two = load_model("MobileNet_model.h5")
        b = loaded_model_two.predict(img_array_reshape)
        acc = np.amax(b) * 100
        acc = round(acc, 3)
        indices = b.argmax()
    if(model == 'InceptionV3'):
        loaded_model_three = load_model("InceptionV3_model.h5")
        c = loaded_model_three.predict(img_array_reshape)
        acc = np.amax(c) * 100
        acc = round(acc,3)
        indices = c.argmax()
    if(model == 'VGG19'):
        loaded_model_four = load_model("VGG19_model.h5")
        d = loaded_model_four.predict(img_array_reshape)
        acc = np.amax(d) * 100
        acc = round(acc, 3)
        indices = d.argmax()

    if(indices == 0):
        return f"Cryptosporidium cyst with an accuracy {acc} %"
    elif(indices == 1):
        return f"Entamoeba histolytica with an accuracy {acc} %"
    elif(indices == 2):
        return f"Giardia cyst with an accuracy {acc} %"

def main():
   
    # Giving a title
    st.title("Protozoan Parasite Detection Web App")
    st.write("**A web app which is used to detect the microscopic image of three types of protozoan parasites using microscopic images by four convolutional neural network models**")
    
    # Code for Protozoan Parasite Detection
    detect_protozoan_parasite = ''
    
    # Choosing of model
    option = st.selectbox('**Please choose the model**', ('DenseNet169', 'MobileNet',
    'InceptionV3', 'VGG19'))
    
    # Getting the input image from the user
    file = st.file_uploader("**Choose an image file**", type = ['jpg', 'png', 'jpeg'])
    st.write("**Please wait as it takes some time. Scroll down to see the image.**")
    if file is not None:
        img_open = Image.open(file)
        detect_protozoan_parasite = protozoan_parasite_detection(option, img_open)
    else:
        st.warning("Please refer to the Instructions in the Instruction Tab (Left Side) before use")
    if detect_protozoan_parasite:
        st.subheader(f"{option} model detects this microscopic image as:red[{detect_protozoan_parasite}]")
        st.image(img_open, use_column_width=True)

if __name__ == '__main__':
    main()
    