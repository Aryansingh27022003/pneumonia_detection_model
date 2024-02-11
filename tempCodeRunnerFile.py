import streamlit as st
from  keras.models import load_model


st.title('Pneumonia classification')

st.header('Please upload a chest X-ray image')

st.file_uploader('',type=['jpeg','jpg','png'])

load_model('./pneumonia.h5')

with open('./labels.txt','r') as f:
    class_name=[a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()
    
print(class_name)    