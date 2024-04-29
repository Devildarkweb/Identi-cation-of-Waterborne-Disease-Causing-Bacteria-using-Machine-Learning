import streamlit as st
st.set_page_config(
    page_title = "Instructions"
)
st.title("Instructions")

st.markdown(
    """
    To run this app some instruction you must know before this
    - At first, choose a convolutional neural network model from the dropdown menu
    - Upload an image or drag it to the file upload section
    - The image should be in JPG, JPEG and PNG format
    - The 4 convolutional neural network models used are dealt with the microscopic
    images
    - By uploading, you can see the type of the protozoan parasite the microscopic image
    represents
    - It is also shows the accuracy of chances of having that type of protozoan parasite
    """
)