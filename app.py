import streamlit as st
import cv2
import keras
from keras.models import load_model
from keras import preprocessing
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

st.header('Mask Detector')
face_model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
mask_label = {0:'MASK',1:'NO MASK'}

def main():
    file_uploaded=st.file_uploader('Choose the file', type=['jpg','jpeg','jpg','png'])

    if file_uploaded is not None:
        image=Image.open(file_uploaded)
        image = image.convert('RGB')
        predict_class(image)
def predict_class(image):
    model=keras.models.load_model('model.h5',compile=False)
    # model=keras.models.load_model('modelnew.h5',compile=False)
    model.compile(loss='binary_crossentropy',metrics=['accuracy'], optimizer='adam')
    image_array = np.asarray(image)
    faces = face_model.detectMultiScale(image_array,scaleFactor=1.1, minNeighbors=4)
    img=image
    if(len(faces)==0):
        st.write("No Faces Detected")
        st.image([img])
    else:
        for i in range(len(faces)):
            (x,y,w,h) = faces[i]
            crop = image_array[y:y+h,x:x+w]
            crop = cv2.resize(crop,(64,64))
            crop = np.reshape(crop,[1,64,64,3])
            predictions = model.predict(crop)
            print(predictions[0][0])
            org = (x+20, y+40)
            score = float(predictions[0])
            arg=0
            if(score>0.5):
                arg=1
            cv2.putText(image_array,mask_label[arg],org,cv2.FONT_HERSHEY_SIMPLEX,color=(255,0,0), thickness=5,fontScale=1)
            cv2.rectangle(image_array,(x,y),(x+w,y+h),(255,0,0), thickness=3)
        st.image([image_array]) 

if __name__=="__main__":
    main()