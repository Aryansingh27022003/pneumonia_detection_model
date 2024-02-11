import streamlit as st
from  keras.models import load_model
from PIL import Image
from util import classify,set_background

set_background('doctorp.png')
#set title
st.title('Pneumonia classification')

#set header
st.header('Please upload a chest X-ray image')

#upload file
file=st.file_uploader('',type=['jpeg','jpg','png'])

#load model
model=load_model('./pneumonia.h5')

#load class name
with open('./labels.txt','r') as f:
    class_name=[a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()
    
print(class_name)    

#display image
if file is not None:
    image=Image.open(file).convert('RGB')
    st.image(image,use_column_width=True)
    
    #classify image
    class_name,conf_score=classify(image,model,class_name)

    #write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}".format(conf_score))